# layerskip_main.py
#
# Training entry-point for LayerSkip.
# Forked from multihead_main.py; shares the same Hydra config structure,
# dataset loading, and Trainer so that results are directly comparable.
#
# Key differences vs multihead_main.py:
#   - Uses LayerSkipModel (no routers / adapters / early heads)
#   - Uses LayerSkipTrainer (simpler metric logging — no router histograms)
#   - Config key "model_type: layerskip" routes here

import os
import sys
import json
import torch
import wandb
import shutil

from layerskip_model import LayerSkipModel
from layerskip_trainer import LayerSkipTrainer
from transformers.integrations import WandbCallback
from transformers import set_seed

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    AutoConfig,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Nvium"))
from Nvium_utils import get_dataset, print_run, print_gpu_debug, TokensPerSecondCallback, CustomWandbCallback

import hydra
from omegaconf import OmegaConf


#######################################################################
# DDP + HYDRA SAFETY  (identical to multihead_main)
#######################################################################

rank = int(os.environ.get("RANK", "0"))
if rank != 0:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["HYDRA_MAIN_PROCESS_ONLY"] = "1"

SCRATCH = os.environ.get("SCRATCH")
if not SCRATCH:
    raise RuntimeError("SCRATCH environment variable is not set")
cache_dir = os.path.join(SCRATCH, "cache")
os.makedirs(cache_dir, exist_ok=True)

print_run("INFO", f"Using cache dir: {cache_dir}")
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_TOKENIZERS_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir


#######################################################################
# BUILD MODEL
#######################################################################

def get_dtype(load_args):
    if load_args.get("bf16", False):
        return torch.bfloat16
    if load_args.get("fp16", False):
        return torch.float16
    return torch.float32


def build_model(exp_cfg):
    base_model_name = exp_cfg["model_name_or_path"]
    ckpt_path = exp_cfg.get("init_checkpoint", None)

    load_args = exp_cfg["load_args"]
    torch_dtype = get_dtype(load_args)
    attn_impl = load_args.get("attn_implementation", None)

    print_run("MODEL", "Architecture selected: LayerSkipModel")

    hf_config = AutoConfig.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=False,
    )

    model_overrides = exp_cfg.get("model_overrides", {})
    if model_overrides:
        print_run("CONFIG", f"Applying model_overrides: {model_overrides}")
        hf_config.update(model_overrides)

    if exp_cfg.get("random_init_all", False):
        print_run("MODEL", "Building LayerSkipModel from scratch (random init)")
        model = LayerSkipModel(
            hf_config,
            exp_config=exp_cfg,
            attn_implementation=attn_impl,
        )
        model = model.to(dtype=torch_dtype)
    else:
        load_path = ckpt_path if ckpt_path else base_model_name
        print_run("MODEL", f"Loading LayerSkipModel weights from: {load_path}")
        model = LayerSkipModel.from_pretrained(
            load_path,
            config=hf_config,
            exp_config=exp_cfg,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )

    # Print parameter summary
    model.print_parameter_summary()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print_run("INFO", f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.4f}%)")

    json_str = json.dumps(hf_config.to_dict(), indent=4)
    indented = "\n".join(["    " + l for l in json_str.splitlines()])
    print_run("CONFIG", f"HF config:\n{indented}")

    first_layer = model.model.layers[0]
    attn0 = first_layer.self_attn
    print_run("INFO", "self_attn class:", type(attn0))
    print_run("INFO", "config._attn_implementation:", getattr(model.config, "_attn_implementation", None))
    print_run("INFO", "Loaded precision:", torch_dtype)

    return model


#######################################################################
# MAIN
#######################################################################

@hydra.main(version_base=None, config_path="../../config", config_name="layerskip_config")
def main(cfg):
    seed = int(cfg.seed)
    set_seed(seed)
    print_run("INFO", f"Global seed set to {seed}")

    run_dir = os.getcwd()
    print_run("HYDRA", "Hydra run dir:", run_dir)

    stdout_path = os.path.join(run_dir, "stdout.log")
    stderr_path = os.path.join(run_dir, "stderr.log")
    sys.stdout = open(stdout_path, "a")
    sys.stderr = open(stderr_path, "a")

    exp_config = OmegaConf.to_container(cfg, resolve=True)
    base_model_name = exp_config["model_name_or_path"]

    # Wandb dir inside Hydra folder
    wandb_dir = os.path.join(run_dir, "wandb")
    os.environ["WANDB_DIR"] = wandb_dir
    os.makedirs(wandb_dir, exist_ok=True)

    # Copy source for reproducibility
    src_folder = os.path.dirname(__file__)
    dst_folder = os.path.join(run_dir, "source_snapshot")
    print_run("INFO", f"Copying source → {dst_folder}")
    shutil.copytree(
        src_folder, dst_folder, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".git", "wandb", "outputs"),
    )

    yaml_str = OmegaConf.to_yaml(cfg)
    indented = "\n".join(["    " + l for l in yaml_str.splitlines()])
    print_run("CONFIG", f"Config:\n{indented}")

    print_run("CUDA", "torch.cuda.is_available():", torch.cuda.is_available())
    print_run("CUDA", "cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print_run("CUDA", "Current device:", torch.cuda.current_device())
        print_run("CUDA", "Device name:", torch.cuda.get_device_name(0))
    print_gpu_debug(prefix="GPU-INIT")

    ###################################################################
    # Tokenizer
    ###################################################################
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, cache_dir=cache_dir, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ###################################################################
    # Model
    ###################################################################
    model = build_model(exp_config)
    print_run("ATTENTION", "flash_sdp enabled:", torch.backends.cuda.flash_sdp_enabled())
    print_run("ATTENTION", "mem_efficient_sdp enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())

    ###################################################################
    # Dataset
    ###################################################################
    tok = get_dataset(exp_config, cache_dir, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ###################################################################
    # Dynamic step calculations (identical to multihead_main)
    ###################################################################
    ta_cfg = cfg.training_args
    output_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = (
        ta_cfg.per_device_train_batch_size
        * ta_cfg.gradient_accumulation_steps
        * world_size
    )
    num_training_sequences = len(tok["train"])
    epoch_steps = (num_training_sequences // global_batch_size) * ta_cfg.num_train_epochs
    cfg_max_steps = int(getattr(ta_cfg, "max_steps", -1))
    total_optimization_steps = cfg_max_steps if cfg_max_steps > 0 else epoch_steps
    save_interval = max(ta_cfg.save_steps, int(total_optimization_steps * ta_cfg.eval_save_interval_percent))

    print_run("INFO", f"Global Batch Size:          {global_batch_size}")
    print_run("INFO", f"Total Optimization Steps:   {total_optimization_steps}")
    print_run("INFO", f"Dynamic Eval/Save Interval: {save_interval}")

    ###################################################################
    # TrainingArguments
    ###################################################################
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=ta_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=ta_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=ta_cfg.gradient_accumulation_steps,
        gradient_checkpointing=ta_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=ta_cfg.gradient_checkpointing_kwargs,
        learning_rate=float(ta_cfg.learning_rate),
        num_train_epochs=ta_cfg.num_train_epochs,
        max_steps=cfg_max_steps,
        optim=ta_cfg.optim,
        adam_beta1=ta_cfg.adam_beta1,
        adam_beta2=ta_cfg.adam_beta2,
        adam_epsilon=float(ta_cfg.adam_epsilon),
        lr_scheduler_type=ta_cfg.lr_scheduler_type,
        warmup_ratio=ta_cfg.warmup_ratio,
        lr_scheduler_kwargs={"min_lr": ta_cfg.min_lr},
        max_grad_norm=ta_cfg.max_grad_norm,
        weight_decay=ta_cfg.weight_decay,
        fp16=ta_cfg.fp16,
        bf16=ta_cfg.bf16,
        dataloader_num_workers=ta_cfg.dataloader_num_workers,
        dataloader_pin_memory=ta_cfg.dataloader_pin_memory,
        dataloader_prefetch_factor=ta_cfg.dataloader_prefetch_factor,
        logging_steps=ta_cfg.logging_steps,
        save_steps=save_interval,
        eval_steps=save_interval,
        eval_strategy=ta_cfg.eval_strategy,
        report_to=ta_cfg.report_to,
        run_name=ta_cfg.run_name,
        torch_compile=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,   # layer dropout skips layers → their params get no grad that step
    )

    ###################################################################
    # Trainer
    ###################################################################
    seq_len = exp_config.get("model_overrides", {}).get("max_position_embeddings", 1024)

    trainer = LayerSkipTrainer(
        model=model,
        args=args,
        train_dataset=tok["train"],
        eval_dataset=tok.get("validation"),
        data_collator=collator,
        callbacks=[
            TokensPerSecondCallback(seq_len=seq_len, global_batch_size=global_batch_size),
        ],
    )

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(CustomWandbCallback())
        if wandb.run is None:
            wandb.init(
                project=ta_cfg.project_name,
                name=args.run_name,
                settings=wandb.Settings(start_method="thread"),
            )
    else:
        trainer.remove_callback(WandbCallback)

    # Inject total_training_steps so the exponential S(t) schedule is correct.
    # Do this before training starts (model already built, but the value is
    # read dynamically from self.total_training_steps each forward pass).
    if hasattr(model, "total_training_steps"):
        model.total_training_steps = total_optimization_steps
        print_run("LAYERSKIP", f"Set model.total_training_steps = {total_optimization_steps}")

    resume_ckpt = exp_config.get("resume_from_checkpoint", None)
    if resume_ckpt:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(resume_ckpt, "model.safetensors"))
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print_run("RESUME", f"Missing keys: {missing}")
        if unexpected:
            print_run("RESUME", f"Unexpected keys: {unexpected}")
        print_run("RESUME", f"Loaded {len(cleaned)} keys from {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)


if __name__ == "__main__":
    main()
