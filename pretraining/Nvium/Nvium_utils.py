import warnings
warnings.filterwarnings("ignore")
import os
import sys
import yaml
import math
import json
import pickle
import shutil
import itertools
import random
import torch
import time
import copy
import numpy as np
import torch.nn.functional as F
from transformers import TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.integrations import WandbCallback
from typing import Optional, Tuple, List, Dict
import torch.nn as nn
from pathlib import Path
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from collections import defaultdict

import wandb
import torch
import warnings

def print_run(label: str, *args, sep=" ", end="\n", file=None, **kwargs):
    # Default to current sys.stdout if no file passed
    if file is None:
        file = sys.stdout
    # Only print on global rank 0
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    # Normal print behavior with a [LABEL] prefix
    msg = sep.join(str(a) for a in args)
    print(f"[{label}] {msg}", end=end, file=file, **kwargs)


def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file does not exist: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_yaml(path):
    if not os.path.exists(path):
        print(f"[load_yaml] File not found: {path}")
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[load_yaml] Error reading {path}: {e}")
        return {}

def save_yaml(data, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False
        )



def _maybe_cache_locally(network_path: str, dataset_name: str) -> str:
    """Copy dataset to node-local storage. Rank 0 copies, others wait."""
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    local_path = os.path.join(tmpdir, dataset_name)
    sentinel = local_path + ".done"

    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        if not os.path.exists(sentinel):
            # Remove partial copy if exists
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            print_run("DATA", f"Caching to local: {network_path} -> {local_path}")
            shutil.copytree(network_path, local_path)
            # Mark as complete
            open(sentinel, "w").close()
            print_run("DATA", "Cache copy done.")
        else:
            print_run("DATA", f"Local cache exists: {local_path}")

    # File-based barrier — works even if DDP isn't initialized
    while not os.path.exists(sentinel):
        time.sleep(2)

    return local_path


def get_dataset(exp_config: dict, cache_dir: str, tokenizer):

    tok = None

    if exp_config["dataset"] in ("c4-scaling-30B", "c4-30B-scaling"):
        dataset_name = exp_config["dataset"]
        network_paths = {
            "c4-scaling-30B": f"{cache_dir}/hf_c4_datasets/scaling/c4_30B_5M_split_30B/hf_dataset",
            "c4-30B-scaling": f"{cache_dir}/hf_c4_datasets/scaling_width/c4_30B_5M_split_30B/hf_dataset",
        }
        path = network_paths[dataset_name]
        if exp_config.get("cache_locally", False):
            path = _maybe_cache_locally(path, dataset_name)
            print_run("DATA", f"Loaded {dataset_name} from local storage: {path}")
        else:
            print_run("DATA", f"Loaded {dataset_name} from network storage: {path}")
        tok = load_from_disk(path)

    else:
        tok = None
        print_run("DATA", "Datset not found...")

    return tok


def print_gpu_debug(prefix="GPU"):
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")

    print_run(prefix, f"RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size}")
    print_run(prefix, f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    print_run(prefix, f"torch.cuda.is_available()={torch.cuda.is_available()}")
    print_run(prefix, f"torch.cuda.device_count()={torch.cuda.device_count()}")

    if torch.cuda.is_available():
        cur = torch.cuda.current_device()
        print_run(prefix, f"current_device={cur}")
        print_run(prefix, f"device_name={torch.cuda.get_device_name(cur)}")

        # touch GPU to be 100% sure
        x = torch.randn(1, device=f"cuda:{cur}")
        torch.cuda.synchronize()
        print_run(prefix, f"allocation OK on cuda:{cur}")


class TokensPerSecondCallback(TrainerCallback):
    def __init__(self, seq_len: int, global_batch_size: int, log_every: int = 20, warmup_steps: int = 50):
        self.seq_len = int(seq_len)
        self.gbs = int(global_batch_size)
        self.log_every = int(log_every)
        self.warmup_steps = int(warmup_steps)

        self._t0 = None
        self._step0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()
        self._step0 = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step < self.warmup_steps:
            return

        if state.global_step % self.log_every != 0:
            return

        t1 = time.perf_counter()
        dt = max(t1 - self._t0, 1e-9)
        steps = max(state.global_step - self._step0, 1)

        tokens_per_step = self.gbs * self.seq_len
        toks = steps * tokens_per_step
        toks_per_s = toks / dt

        # Print only on rank 0
        try:
            import torch.distributed as dist
            is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        except Exception:
            is_rank0 = True

        if is_rank0:
            print(
                f"[THROUGHPUT] step={state.global_step} "
                f"gbs={self.gbs} seqlen={self.seq_len} "
                f"tokens/step={tokens_per_step:,} "
                f"avg_tokens/s={toks_per_s:,.0f}",
                flush=True,
            )

        # reset window
        self._t0 = t1
        self._step0 = state.global_step


class CustomWandbCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not state.is_world_process_zero or wandb.run is None or logs is None:
            return

        # Pop the raw data so it doesn't clutter standard logs
        hist_step = logs.pop("_hist_data_step", None)
        hist_mean = logs.pop("_hist_data_mean", None)
        
        payload = dict(logs)

        if hist_step is not None and hist_mean is not None:
            bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
            
            # --- Prepare tables ---
            # Table for the current step snapshot
            step_data = [[lbl, float(p)] for lbl, p in zip(bin_labels, hist_step)]
            step_table = wandb.Table(data=step_data, columns=["bin", "proportion"])
            
            # Table for the running mean
            mean_data = [[lbl, float(p)] for lbl, p in zip(bin_labels, hist_mean)]
            mean_table = wandb.Table(data=mean_data, columns=["bin", "proportion"])
            
            # --- Existing metrics ---
            payload["router/dist_snapshot"] = wandb.plot.bar(
                step_table, "bin", "proportion", title="p(early) Dist - Last Batch"
            )
            payload["router/dist_mean"] = wandb.plot.bar(
                mean_table, "bin", "proportion", title="p(early) Dist - Running Mean"
            )

            # --- History ---
            if state.global_step % 500 == 0:
                history_key = f"router_history/step_{state.global_step}"
                
                payload[history_key] = wandb.plot.bar(
                    step_table, "bin", "proportion", 
                    title=f"p(early) Distribution - Step {state.global_step}"
                )

        # Log everything
        wandb.run.log(payload, step=state.global_step)


def freeze_model_parameters(model, config):
    """
    Freeze/unfreeze parameters based on config.
    Works with both single-exit and multi-exit models.
    """
    fo = config["freeze_options"]
    print_run("FREEZE", f"Applying freeze options: {fo}")

    # Start with everything frozen
    for p in model.parameters():
        p.requires_grad = False

    # 1. Backbone (transformer layers)
    if fo.get("train_backbone", False):
        for _, p in model.model.named_parameters():
            p.requires_grad = True
        if fo.get("freeze_embeddings", False):
            model.model.embed_tokens.weight.requires_grad = False

    # 2. Final lm head
    if fo.get("train_lm_head", False):
        for name, p in model.named_parameters():
            if name.startswith("lm_head") or name.endswith("model.embed_tokens.weight"):
                p.requires_grad = True

    # 3. Early lm heads
    if fo.get("train_early_lm_head", False):
        if hasattr(model, 'early_lm_heads'):
            # Multi-exit: ModuleList of heads
            for early_head in model.early_lm_heads:
                for p in early_head.parameters():
                    p.requires_grad = True
        elif hasattr(model, 'early_lm_head'):
            # Single-exit: backward compatibility
            for p in model.early_lm_head.parameters():
                p.requires_grad = True

    # 4. Early adapters
    if fo.get("train_early_adapter", False):
        if hasattr(model, 'early_adapters') and model.early_adapters is not None:
            # Multi-exit: ModuleList of adapters
            for early_adapter in model.early_adapters:
                for p in early_adapter.parameters():
                    p.requires_grad = True
        elif hasattr(model, 'early_adapter') and model.early_adapter is not None:
            # Single-exit: backward compatibility
            for p in model.early_adapter.parameters():
                p.requires_grad = True
    
    # 5. Early norms
    if fo.get("train_early_norm", False):
        if hasattr(model, 'early_norms'):
            # Multi-exit: ModuleList of norms
            for early_norm in model.early_norms:
                for p in early_norm.parameters():
                    p.requires_grad = True
        elif hasattr(model, 'early_norm'):
            # Single-exit: backward compatibility
            for p in model.early_norm.parameters():
                p.requires_grad = True

    # 6. Final adapter
    if fo.get("train_final_adapter", False) and model.final_adapter is not None:
        for p in model.final_adapter.parameters():
            p.requires_grad = True

    # 7. Routers
    if fo.get("train_router", False):
        # Router layers
        if hasattr(model, 'router_layers'):
            for router_layer in model.router_layers:
                for p in router_layer.parameters():
                    p.requires_grad = True
        elif hasattr(model, 'router_layer'):
            # Single-exit: backward compatibility
            for p in model.router_layer.parameters():
                p.requires_grad = True
        
        # Router final norms
        if hasattr(model, 'router_final_norms'):
            for router_norm in model.router_final_norms:
                for p in router_norm.parameters():
                    p.requires_grad = True
        elif hasattr(model, 'router_final_norm'):
            # Single-exit: backward compatibility
            for p in model.router_final_norm.parameters():
                p.requires_grad = True
        
        # Router heads
        if hasattr(model, 'router_heads'):
            for router_head in model.router_heads:
                for p in router_head.parameters():
                    p.requires_grad = True
        elif hasattr(model, 'router_head'):
            # Single-exit: backward compatibility
            for p in model.router_head.parameters():
                p.requires_grad = True

    # 8. Freeze specific layers
    for i, layer in enumerate(model.model.layers):
        if i in fo.get("frozen_layers", []):
            for p in layer.parameters():
                p.requires_grad = False

    # 9. Logging
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(numel for _, numel in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    
    print_run("FREEZE", f"{len(trainable)} parameter groups are trainable.")
    print_run("FREEZE", f"Total trainable params: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")
    
    # Detailed breakdown
    print_run("FREEZE", "Trainable parameter breakdown:")
    for name, numel in sorted(trainable, key=lambda x: -x[1])[:20]:  # Top 20 by size
        print_run("FREEZE", f"  {name}: {numel:,}")
    
    if len(trainable) > 20:
        print_run("FREEZE", f"  ... and {len(trainable) - 20} more")


def compute_distribution_metrics(early_logits, final_logits, k=5, p=0.95):
    """
    Computes overlap, coverage, and count metrics for Top-K and Top-P.
    Expects flattened logits: [num_valid_tokens, vocab_size]
    """
    # 1. Convert logits to probabilities
    probs_e = F.softmax(early_logits, dim=-1)
    probs_f = F.softmax(final_logits, dim=-1)
    vocab_size = probs_e.shape[-1]

    # --- Top-k metrics ---
    # Get top-k values and indices
    topk_vals_e, topk_inds_e = torch.topk(probs_e, k, dim=-1)
    topk_vals_f, topk_inds_f = torch.topk(probs_f, k, dim=-1)

    # A. Top-k Coverage (Mass)
    # What % of probability mass is held in the top k tokens?
    # e.g., if top 5 tokens sum to 0.8, the model is very confident.
    k_coverage_e = topk_vals_e.sum(dim=-1).mean()
    k_coverage_f = topk_vals_f.sum(dim=-1).mean()

    # --- Top-1 metrics ---
    # 1. Top-1 Mass (Confidence): The probability of the single most likely token
    # We take the 0-th index of the topk values
    top1_mass_e = topk_vals_e[:, 0].mean()
    top1_mass_f = topk_vals_f[:, 0].mean()

    # 2. Top-1 Overlap (Agreement): Does the early exit predict the exact same token as the final layer?
    # We compare the 0-th index of the indices
    top1_match = (topk_inds_e[:, 0] == topk_inds_f[:, 0])
    top1_overlap = top1_match.float().mean() # Returns % of matching predictions

    # B. Top-k Overlap
    # Create boolean masks for the top-k positions
    # We scatter 1s into a zero-tensor of shape [N, Vocab] at the top-k indices
    mask_e_k = torch.zeros_like(probs_e, dtype=torch.bool).scatter_(1, topk_inds_e, True)
    mask_f_k = torch.zeros_like(probs_f, dtype=torch.bool).scatter_(1, topk_inds_f, True)

    # Intersection count (how many tokens are in both sets?)
    overlap_k_count = (mask_e_k & mask_f_k).sum(dim=-1).float().mean()
    
    # --- Top-p (Nucleus) metrics ---
    # We must sort to calculate Top-P
    # Note: sorting is expensive (O(N log N)), use sparingly!
    sorted_probs_e, sorted_indices_e = torch.sort(probs_e, descending=True, dim=-1)
    sorted_probs_f, sorted_indices_f = torch.sort(probs_f, descending=True, dim=-1)

    # Calculate Cumulative Sum
    cumsum_e = torch.cumsum(sorted_probs_e, dim=-1)
    cumsum_f = torch.cumsum(sorted_probs_f, dim=-1)

    # Identify tokens strictly INSIDE the nucleus
    # We want the smallest set S s.t. sum(S) >= p
    # Mask is True for all tokens that contribute to the top-p sum
    # We shift the mask right by 1 to ensure the token that crosses the threshold is included
    mask_sorted_e = cumsum_e < p
    mask_sorted_e[..., 1:] = mask_sorted_e[..., :-1].clone()
    mask_sorted_e[..., 0] = True # Always include at least the top 1 token
    
    mask_sorted_f = cumsum_f < p
    mask_sorted_f[..., 1:] = mask_sorted_f[..., :-1].clone()
    mask_sorted_f[..., 0] = True

    # C. Top-p Count
    # How many tokens constitute the nucleus? (Lower = sharper distribution)
    p_count_e = mask_sorted_e.sum(dim=-1).float().mean()
    p_count_f = mask_sorted_f.sum(dim=-1).float().mean()

    # D. Top-p Overlap (Jaccard Index)
    # We need to map the sorted masks back to the original vocab indices to compare them
    mask_e_p = torch.zeros_like(probs_e, dtype=torch.bool).scatter_(1, sorted_indices_e, mask_sorted_e)
    mask_f_p = torch.zeros_like(probs_f, dtype=torch.bool).scatter_(1, sorted_indices_f, mask_sorted_f)

    intersection_p = (mask_e_p & mask_f_p).sum(dim=-1).float()
    union_p = (mask_e_p | mask_f_p).sum(dim=-1).float()
    
    # Jaccard overlap: Intersection / Union
    # Add epsilon to avoid div/0
    p_jaccard_overlap = (intersection_p / (union_p + 1e-6)).mean()

    return {
        # New Top-1 Metrics
        "top1_overlap_acc": top1_overlap,    # % of time Early == Final
        "top1_mass_early": top1_mass_e,      # How confident is Early?
        "top1_mass_final": top1_mass_f,      # How confident is Final?
        
        f"top{k}_overlap_count": overlap_k_count,
        f"top{k}_mass_early": k_coverage_e,
        f"top{k}_mass_final": k_coverage_f,
        f"top{p}_count_early": p_count_e,
        f"top{p}_count_final": p_count_f,
        f"top{p}_jaccard_overlap": p_jaccard_overlap
    }


# -----------------------------------------------------------------------------
# 2. Simple GELU Router Components (New)
# -----------------------------------------------------------------------------
class SimpleMLPBlock(nn.Module):
    """
    Standard MLP Block:
    x -> Linear -> GELU -> Linear -> out
    """
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class RouterSimpleMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, intermediate_dim, num_layers=1):
        super().__init__()
        layers = []
        
        layers.append(SimpleMLPBlock(input_dim, intermediate_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(SimpleMLPBlock(hidden_dim, intermediate_dim, hidden_dim))
            
        self.layers = nn.Sequential(*layers)

    def forward(self, hidden_states, **kwargs):
        out = self.layers(hidden_states)
        return (out,)


def create_router(model_config, router_cfg, input_dim=None):
    router_type = router_cfg.get("router_type", "llama")
    
    # Resolve input dimension
    actual_input_dim = input_dim if input_dim is not None else model_config.hidden_size
        

    target_hidden = int(router_cfg.get("hidden_size", 512))
    intermediate_dim = int(router_cfg.get("intermediate_size", target_hidden * 2))

    if router_type in ["simple_mlp", "simple_deep_mlp"]:
        num_layers = 1 if router_type == "simple_mlp" else int(router_cfg.get("num_layers", 2))
        module = RouterSimpleMLP(actual_input_dim, target_hidden, intermediate_dim, num_layers)
        return module, target_hidden
    
    else:
        raise ValueError(f"Unknown router_type: {router_type}")