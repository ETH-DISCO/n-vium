"""
Fine-tuning Script for Benchmark Improvement (Tulu 3 SFT Mixture)
=================================================================
Supports both BaselineLlamaModel and NviumModel.
All parameters are accepted via argparse.

Usage:
  python sft.py --checkpoint_path /path/to/checkpoint --output_dir /path/to/output

  # Or with torchrun for multi-GPU:
  torchrun --nproc_per_node=4 sft.py --checkpoint_path /path/to/checkpoint
"""

import os

# Set cache directories before any HF imports
_SCRATCH = os.environ.get("SCRATCH")
if not _SCRATCH:
    raise RuntimeError("SCRATCH environment variable is not set")
cache_dir = os.path.join(_SCRATCH, "cache", "sft")
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(cache_dir, "hf_home")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "hf_datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "hf_models")
os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hf_hub")
os.environ["XDG_CACHE_HOME"] = os.path.join(cache_dir, "xdg_cache")

import sys
import copy
import yaml
import json
import struct
import argparse
import torch
import numpy as np
from typing import Optional

import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file as safetensors_load_file
from trl import SFTTrainer, SFTConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "pretraining", "Nvium"))
from Nvium_model import NviumModel
from Nvium_baseline import BaselineLlamaModel


# Default ChatML template
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)

# SFT-specific loss overrides for multihead models
DEFAULT_SFT_EXP_OVERRIDES = {
    "loss": {
        "warmup_steps": 0,
        "main": {
            "mix_loss": True,
            "early_ce_loss": False,
            "final_ce_loss": False,
            "router_loss": False,
            "router_compute_loss": True,
            "distill_loss": False,
        },
        "alphas_main": {
            "mix_loss": 1.0,
            "router_loss": 0.1,
            # router_compute_loss alpha inherited from pretrain config via _deep_merge
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune baseline or multihead model on SFT data")

    # Required
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to pretrained checkpoint directory")

    # Model
    parser.add_argument("--model_type", type=str, default=None, choices=["baseline", "Nvium"],
                        help="Model type. Auto-detected from config.yaml if not provided.")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to experiment config.yaml. Auto-detected if not provided.")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="allenai/tulu-3-sft-mixture",
                        help="HuggingFace dataset name")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size (None = use full dataset)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Defaults to <experiment_dir>/ft_output")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 = use num_train_epochs instead)")

    # Loss mode (only relevant for multihead)
    parser.add_argument("--sft_loss_mode", type=str, default="full",
                        choices=["full", "final_only", "mix_only"],
                        help="Loss mode for multihead model")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="ForwardThinking-FT")
    parser.add_argument("--report_to", type=str, default="wandb")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _has_orig_mod_prefix(safetensors_path: str) -> bool:
    with open(safetensors_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
    keys = [k for k in header if k != "__metadata__"]
    return any(k.startswith("_orig_mod.") for k in keys)


def auto_detect_config(checkpoint_path: str) -> str:
    """Find config.yaml relative to checkpoint: checkpoint -> checkpoints -> experiment_dir/.hydra/config.yaml"""
    experiment_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(experiment_dir, ".hydra", "config.yaml")
    if os.path.exists(config_path):
        return config_path
    # Fallback: check for config.yaml directly in experiment dir
    config_path2 = os.path.join(experiment_dir, "config.yaml")
    if os.path.exists(config_path2):
        return config_path2
    raise FileNotFoundError(
        f"Could not find config.yaml. Searched:\n  {config_path}\n  {config_path2}\n"
        f"Please provide --config_path explicitly."
    )


# ---------------------------------------------------------------------------
# CUSTOM TRAINER — handles both baseline and multihead
# ---------------------------------------------------------------------------
class FTTrainer(SFTTrainer):
    def __init__(self, *args, model_type="baseline", sft_loss_mode="full",
                 sft_exp_config=None, num_exits=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.sft_loss_mode = sft_loss_mode
        self.sft_exp_config = sft_exp_config or {}
        self._global_forward_step = 0
        self._train_metrics_buffer = []
        self._eval_metrics_buffer = []
        self._in_eval = False
        self.histogram_logging_steps = 100
        self.num_exits = num_exits
        self.bins = np.linspace(0.0, 1.0, 11)
        self.bin_labels = [f"{round(self.bins[i],1)}-{round(self.bins[i+1],1)}" for i in range(10)]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.model_type == "Nvium":
            inputs["current_step"] = self._global_forward_step

        outputs = model(**inputs)

        if self.model_type == "Nvium":
            if self.sft_loss_mode == "full":
                loss = outputs.loss
            elif self.sft_loss_mode == "final_only":
                loss = outputs.loss_dict["final_ce_loss"]
            elif self.sft_loss_mode == "mix_only":
                loss = outputs.loss_dict["mix_loss"]
            else:
                loss = outputs.loss
        else:
            loss = outputs.loss

        # Buffer metrics
        if hasattr(outputs, "loss_dict") and outputs.loss_dict is not None:
            metrics = {}
            for key, val in outputs.loss_dict.items():
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        metrics[key] = val.detach().item()
                    else:
                        metrics[key] = val.detach().mean().item()
                else:
                    metrics[key] = val

            if getattr(outputs, "exit_probabilities", None) is not None:
                exit_probs = outputs.exit_probabilities.detach()
                for i in range(exit_probs.shape[-1]):
                    exit_name = f"exit_{i+1}" if i < exit_probs.shape[-1] - 1 else "exit_final"
                    p = exit_probs[..., i]
                    metrics[f"{exit_name}_mean"] = p.mean().item()
                    p_vals = p.float().cpu().numpy().flatten()
                    counts, _ = np.histogram(p_vals, bins=self.bins)
                    metrics[f"{exit_name}_hist_counts"] = counts

            if self._in_eval:
                self._eval_metrics_buffer.append(metrics)
            else:
                self._train_metrics_buffer.append(metrics)

        self._global_forward_step += 1
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        if self.model_type == "Nvium":
            inputs["current_step"] = self._global_forward_step

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss.detach() if outputs.loss is not None else None

        if hasattr(outputs, "loss_dict") and outputs.loss_dict is not None:
            metrics = {}
            for key, val in outputs.loss_dict.items():
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        metrics[key] = val.detach().item()
                    else:
                        metrics[key] = val.detach().mean().item()
                else:
                    metrics[key] = val

            if getattr(outputs, "exit_probabilities", None) is not None:
                exit_probs = outputs.exit_probabilities.detach()
                for i in range(exit_probs.shape[-1]):
                    exit_name = f"exit_{i+1}" if i < exit_probs.shape[-1] - 1 else "exit_final"
                    p = exit_probs[..., i]
                    metrics[f"{exit_name}_mean"] = p.mean().item()
                    p_vals = p.float().cpu().numpy().flatten()
                    counts, _ = np.histogram(p_vals, bins=self.bins)
                    metrics[f"{exit_name}_hist_counts"] = counts

            if self._in_eval:
                self._eval_metrics_buffer.append(metrics)

        return (loss, None, None)

    def _average_buffer(self, buffer):
        if not buffer:
            return {}

        all_keys = set().union(*(d.keys() for d in buffer))
        hist_keys = [k for k in all_keys if k.endswith("_hist_counts")]
        scalar_keys = [k for k in all_keys if k not in hist_keys]

        local_avgs = {}
        for k in scalar_keys:
            values = [m[k] for m in buffer if k in m]
            if values:
                local_avgs[k] = sum(values) / len(values)

        local_hists = {}
        for hk in hist_keys:
            hist_sum = sum(m[hk] for m in buffer if hk in m)
            if isinstance(hist_sum, int) and hist_sum == 0:
                hist_sum = np.zeros(len(self.bins) - 1)
            local_hists[hk] = hist_sum

        if dist.is_initialized():
            with torch.no_grad():
                sorted_keys = sorted(local_avgs.keys())
                if sorted_keys:
                    metrics_tensor = torch.tensor(
                        [local_avgs[k] for k in sorted_keys], device=self.model.device
                    )
                    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                    metrics_tensor /= dist.get_world_size()
                    synced_data = dict(zip(sorted_keys, metrics_tensor.tolist()))
                    local_avgs.update(synced_data)

                for hk, hist_arr in local_hists.items():
                    hist_tensor = torch.tensor(hist_arr, device=self.model.device, dtype=torch.float32)
                    dist.all_reduce(hist_tensor, op=dist.ReduceOp.SUM)
                    local_hists[hk] = hist_tensor.cpu().numpy()

        local_avgs.update(local_hists)
        return local_avgs

    def log(self, logs: dict, *args, **kwargs) -> None:
        is_eval = any(k.startswith("eval") for k in logs)
        mode = "eval" if is_eval else "train"

        custom_metrics = self._average_buffer(
            self._eval_metrics_buffer if is_eval else self._train_metrics_buffer
        )
        if is_eval:
            self._eval_metrics_buffer = []
        else:
            self._train_metrics_buffer = []

        super().log(logs, *args, **kwargs)

        if not self.is_world_process_zero():
            return

        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        custom_logs = {}

        if self.state.global_step % self.histogram_logging_steps == 0:
            for key, val in custom_metrics.items():
                if key.endswith("_hist_counts") and isinstance(val, np.ndarray):
                    exit_name = key.replace("_hist_counts", "")
                    total_tokens = val.sum()
                    if total_tokens > 0:
                        prop = val / total_tokens
                        custom_logs[f"histogram/{mode}/{exit_name}"] = prop.tolist()

        for k, v in custom_metrics.items():
            if k.endswith("_hist_counts"):
                continue
            if k.startswith("heads/"):
                metric_name = k.replace("heads/", "")
                custom_logs[f"heads/{mode}/{metric_name}"] = v
            elif "mean" in k:
                custom_logs[f"router/{mode}/{k}"] = v
            else:
                custom_logs[f"losses/{mode}/{k}"] = v

        if custom_logs:
            wandb.log(custom_logs, step=self.state.global_step)

    def evaluate(self, *args, **kwargs):
        self._in_eval = True
        self._eval_metrics_buffer = []
        try:
            result = super().evaluate(*args, **kwargs)
        finally:
            self._in_eval = False
        return result

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        save_dir = output_dir if output_dir is not None else self.args.output_dir
        cfg_to_save = dict(self.sft_exp_config)
        cfg_to_save["model_type"] = self.model_type
        cfg_path = os.path.join(save_dir, "ft_exp_config.yaml")
        if self.is_world_process_zero():
            with open(cfg_path, "w") as f:
                yaml.dump(cfg_to_save, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, exp_config: dict, model_type: str,
               sft_overrides: Optional[dict] = None, dtype=torch.bfloat16):
    """Load either a baseline or multihead model from a pretrained checkpoint."""
    if sft_overrides and model_type == "Nvium":
        exp_config = _deep_merge(exp_config, sft_overrides)

    print(f"[FT] Loading {model_type} model from {checkpoint_path}")

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    has_orig_mod = os.path.exists(safetensors_path) and _has_orig_mod_prefix(safetensors_path)

    ModelClass = NviumModel if model_type == "Nvium" else BaselineLlamaModel

    if has_orig_mod:
        print("[FT] Detected torch.compile checkpoint, stripping _orig_mod. prefix")
        config = AutoConfig.from_pretrained(checkpoint_path)
        model = ModelClass(config, exp_config=exp_config, attn_implementation="sdpa")
        state_dict = safetensors_load_file(safetensors_path)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[FT] Missing keys (expected if tied): {missing}")
        if unexpected:
            print(f"[FT] Unexpected keys: {unexpected}")
        model = model.to(dtype)
    else:
        model = ModelClass.from_pretrained(
            checkpoint_path,
            exp_config=exp_config,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        )

    # Re-apply weight tying
    model._apply_tying(exp_config.get("tying", {}))

    return model


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
def load_ft_dataset(dataset_name: str, split: str = "train",
                    max_samples: Optional[int] = None, seed: int = 42):
    print(f"[FT] Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
        print(f"[FT] Subsampled to {len(ds)} examples")

    # Normalize column names
    if "conversations" in ds.column_names and "messages" not in ds.column_names:
        ds = ds.rename_column("conversations", "messages")

    # Convert from/value format to standard role/content
    if "messages" in ds.column_names:
        sample_msg = ds[0]["messages"][0]
        if "from" in sample_msg and "role" not in sample_msg:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            def convert_messages(example):
                example["messages"] = [
                    {"role": role_map.get(m["from"], m["from"]), "content": m["value"]}
                    for m in example["messages"]
                ]
                return example
            ds = ds.map(convert_messages)
            print("[FT] Converted from/value format to role/content format")

    print(f"[FT] Dataset loaded: {len(ds)} examples, columns: {ds.column_names}")

    if "messages" in ds.column_names:
        sample = ds[0]["messages"]
        print(f"[FT] Sample conversation ({len(sample)} turns):")
        for msg in sample[:3]:
            role = msg.get("role", msg.get("from", "?"))
            content = msg.get("content", msg.get("value", ""))[:80]
            print(f"  [{role}]: {content}...")

    return ds


def _get_tokenized_cache_path(dataset_name: str, split: str, max_seq_length: int,
                               max_samples: Optional[int] = None) -> str:
    """Build a deterministic cache path for pre-tokenized datasets."""
    safe_name = dataset_name.replace("/", "__")
    samples_tag = f"_n{max_samples}" if max_samples else ""
    return os.path.join(cache_dir, f"tokenized_{safe_name}_{split}_L{max_seq_length}{samples_tag}")


def load_or_tokenize_dataset(dataset_name: str, split: str, tokenizer,
                              max_seq_length: int, max_samples: Optional[int] = None,
                              seed: int = 42):
    """Load pre-tokenized dataset from disk cache, or tokenize and save."""
    cache_path = _get_tokenized_cache_path(dataset_name, split, max_seq_length, max_samples)

    if os.path.exists(cache_path):
        print(f"[FT] Loading cached tokenized dataset from {cache_path}")
        ds = load_from_disk(cache_path)
        print(f"[FT] Loaded {len(ds)} examples from cache")
        return ds

    # Load raw dataset
    ds = load_ft_dataset(dataset_name, split=split, max_samples=max_samples, seed=seed)

    # Apply chat template to get text
    def apply_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return example

    print(f"[FT] Applying chat template to {split} dataset...")
    ds = ds.map(apply_template, num_proc=4)

    # Tokenize
    all_columns = ds.column_names

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    print(f"[FT] Tokenizing {split} dataset (max_seq_length={max_seq_length})...")
    ds = ds.map(tokenize_fn, num_proc=4, remove_columns=all_columns)

    # Save to disk
    print(f"[FT] Saving tokenized dataset to {cache_path}")
    ds.save_to_disk(cache_path)
    print(f"[FT] Cached {len(ds)} tokenized examples")

    return ds


def setup_tokenizer(tokenizer_path: str, chat_template: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"[FT] Set pad_token to eos_token: '{tokenizer.pad_token}'")

    if chat_template is not None:
        tokenizer.chat_template = chat_template
        print("[FT] Applied custom chat template")
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = CHATML_TEMPLATE
        print("[FT] Applied default ChatML template")

    return tokenizer


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Auto-detect config ---
    config_path = args.config_path or auto_detect_config(args.checkpoint_path)
    print(f"[FT] Using config: {config_path}")
    with open(config_path, "r") as f:
        exp_config = yaml.safe_load(f)

    # --- Determine model type ---
    model_type = args.model_type or exp_config.get("model_type", "baseline")
    print(f"[FT] Model type: {model_type}")

    # --- Output directory ---
    if args.output_dir is None:
        experiment_dir = os.path.dirname(os.path.dirname(args.checkpoint_path))
        ckpt_name = os.path.basename(args.checkpoint_path)
        output_dir = os.path.join(experiment_dir, f"ft_output_{ckpt_name}")
    else:
        output_dir = args.output_dir

    # --- Run name ---
    if args.run_name is None:
        experiment_name = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint_path)))
        ckpt_name = os.path.basename(args.checkpoint_path)
        run_name = f"ft-{experiment_name}-{ckpt_name}"
    else:
        run_name = args.run_name

    print("=" * 60)
    print(f"Fine-Tuning — {model_type.upper()} Model")
    print("=" * 60)

    # --- 1. Load model ---
    sft_overrides = DEFAULT_SFT_EXP_OVERRIDES if model_type == "Nvium" else None
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        exp_config=exp_config,
        model_type=model_type,
        sft_overrides=sft_overrides,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if model_type == "Nvium":
        print(f"  Early layer indices: {model.early_layer_indices}")
        print(f"  Num exits: {model.num_exits}")

    # --- 2. Load tokenizer ---
    tokenizer = setup_tokenizer(args.checkpoint_path)

    # Add special tokens if needed
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    tokens_to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
        # Resize early_lm_heads for multihead models
        if model_type == "Nvium":
            for i, early_head in enumerate(model.early_lm_heads):
                if early_head.weight.shape[0] != len(tokenizer):
                    new_head = torch.nn.Linear(
                        model.config.hidden_size, len(tokenizer), bias=False
                    ).to(device=early_head.weight.device, dtype=early_head.weight.dtype)
                    new_head.weight.data[:early_head.weight.shape[0]] = early_head.weight.data
                    model.early_lm_heads[i] = new_head
                    print(f"  Resized early_lm_heads[{i}] to {len(tokenizer)}")
        print(f"  Added special tokens: {tokens_to_add}")
        print(f"  New vocab size: {len(tokenizer)}")

    # --- 3. Load dataset (with disk caching) ---
    train_dataset = load_or_tokenize_dataset(
        dataset_name=args.dataset_name,
        split="train",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    eval_max = min(1000, args.max_samples or 1000)
    eval_dataset = None
    for eval_split in ("test", "validation"):
        try:
            eval_dataset = load_or_tokenize_dataset(
                dataset_name=args.dataset_name,
                split=eval_split,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                max_samples=eval_max,
                seed=args.seed,
            )
            break
        except Exception:
            continue

    if eval_dataset is None:
        print("[FT] No eval split found, using 5% of train")
        split = train_dataset.train_test_split(test_size=0.05, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    # --- 4. Training config ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_seq_length,
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=args.report_to,
        run_name=run_name,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # --- 5. Create trainer ---
    num_exits = getattr(model, "num_exits", 1)
    merged_exp_config = _deep_merge(exp_config, DEFAULT_SFT_EXP_OVERRIDES) if model_type == "Nvium" else exp_config

    trainer = FTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        model_type=model_type,
        sft_loss_mode=args.sft_loss_mode,
        sft_exp_config=merged_exp_config,
        num_exits=num_exits,
    )

    print(f"\n  Output dir: {output_dir}")
    print(f"  Run name: {run_name}")
    print(f"  Loss mode: {args.sft_loss_mode}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset) if eval_dataset else 0}")
    print(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Max seq length: {args.max_seq_length}")

    # --- Save full FT config for reproducibility ---
    os.makedirs(output_dir, exist_ok=True)
    ft_run_config = {
        "checkpoint_path": args.checkpoint_path,
        "config_path": config_path,
        "model_type": model_type,
        "dataset_name": args.dataset_name,
        "max_samples": args.max_samples,
        "sft_loss_mode": args.sft_loss_mode,
        "training": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": args.lr_scheduler_type,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
        },
        "logging": {
            "logging_steps": args.logging_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "save_total_limit": args.save_total_limit,
            "run_name": run_name,
            "wandb_project": args.wandb_project,
        },
        "loss_config": merged_exp_config.get("loss", {}),
    }
    ft_config_path = os.path.join(output_dir, "ft_run_config.yaml")
    with open(ft_config_path, "w") as f:
        yaml.dump(ft_run_config, f, default_flow_style=False)
    print(f"  FT run config saved to: {ft_config_path}")

    # --- 6. Train ---
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    # --- 7. Save ---
    print("\n[DONE] Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

    return trainer


if __name__ == "__main__":
    main()
