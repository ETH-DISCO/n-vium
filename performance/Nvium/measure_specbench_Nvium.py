#!/usr/bin/env python3
"""
Spec-Bench speedup evaluation for Nvium early-exit models.

Downloads Spec-Bench prompts (cached), runs a grid of (head_temp, router_temp)
settings, and reports walltime speedup with mean ± std across multiple runs.

Usage:
    python measure_specbench_Nvium.py \
        --forward_checkpoint /path/to/Nvium_h256_l24/checkpoint-XXXX \
        --baseline_checkpoint /path/to/baseline_h256_l24/checkpoint-XXXX

    # Custom settings:
    python measure_specbench_Nvium.py \
        --forward_checkpoint ... --baseline_checkpoint ... \
        --tokens 256 --runs 5 --prompts_per_category 20 \
        --head_temps 0.0 0.6 1.0 --router_temps 0.0 0.6 1.0
"""

import os
import re
import gc
import sys
import json
import subprocess
import random
import argparse
import time
import yaml
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

# ============================================================
# Imports
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "pretraining", "Nvium"))
from Nvium_baseline import BaselineLlamaModel
from Nvium_model import NviumModel
sys.path.pop(0)

from model_inference_piggyback_Nvium import get_Nvium_inference_model_class
from model_inference_baseline import get_baseline_model_class

# ============================================================
# CONFIG
# ============================================================
TORCH_DTYPE = torch.bfloat16

def fmt_duration(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"

SPECBENCH_REPO = "https://github.com/hemingkx/Spec-Bench.git"
DEFAULT_CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "."), "cache", "SpecDec")

DEFAULT_HEAD_TEMPS = [0.0, 0.6, 1.0]
DEFAULT_ROUTER_TEMPS = [0.0, 0.6, 1.0]


# ============================================================
# SPEC-BENCH DATA
# ============================================================
def download_specbench(cache_dir):
    """Clone Spec-Bench repo (shallow) if not already cached."""
    repo_dir = os.path.join(cache_dir, "Spec-Bench")
    question_file = os.path.join(repo_dir, "data", "spec_bench", "question.jsonl")

    if os.path.exists(question_file):
        print(f"[Cache] Using cached Spec-Bench: {question_file}")
        return question_file

    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Download] Cloning Spec-Bench to {repo_dir}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", SPECBENCH_REPO, repo_dir],
        check=True,
    )

    if not os.path.exists(question_file):
        raise FileNotFoundError(
            f"Expected question.jsonl at {question_file} but not found. "
            "Check Spec-Bench repo structure."
        )

    print(f"[Download] Done. Cached at: {question_file}")
    return question_file


def load_prompts(question_file, limit=None, seed=42):
    """
    Load Spec-Bench prompts, take first turn only.
    If limit is set, subsample up to `limit` prompts per category.
    Otherwise, use all prompts (full benchmark).
    Returns list of {"id": int, "category": str, "text": str}.
    """
    all_prompts = []
    with open(question_file) as f:
        for line in f:
            item = json.loads(line.strip())
            all_prompts.append({
                "id": item["question_id"],
                "category": item.get("category", "unknown"),
                "text": item["turns"][0],
            })

    by_cat = {}
    for p in all_prompts:
        by_cat.setdefault(p["category"], []).append(p)

    if limit is not None:
        rng = random.Random(seed)
        sampled = []
        for cat in sorted(by_cat):
            items = by_cat[cat]
            if len(items) <= limit:
                sampled.extend(items)
            else:
                sampled.extend(rng.sample(items, limit))
        label = f"sampled {len(sampled)} (limit={limit}/cat)"
    else:
        sampled = all_prompts
        label = f"all {len(sampled)}"

    print(f"[Data] Loaded {len(all_prompts)} prompts, using {label} "
          f"({len(by_cat)} categories)")
    for cat in sorted(by_cat):
        n_total = len(by_cat[cat])
        n_sampled = sum(1 for p in sampled if p["category"] == cat)
        print(f"  {cat}: {n_sampled}/{n_total}")

    return sampled


def tokenize_prompts(prompts, tokenizer, max_input_tokens=256, model_max_context=None,
                     tokens_to_gen=0):
    """
    Tokenize prompts and truncate to max_input_tokens.
    Also tokenizes without truncation to detect prompts that exceed the model's
    context window (input_len + tokens_to_gen > model_max_context).

    Returns (tokenized, overflow_info) where overflow_info is a dict:
        {
          "global_overflow": int,
          "per_category": {cat: int, ...},
          "model_max_context": int,
          "orig_lengths": {cat: [int, ...], ...},
        }
    """
    tokenized = []
    orig_lens_by_cat = {}
    overflow_by_cat = {}

    for p in prompts:
        # Untruncated length for overflow check
        full_tokens = tokenizer(p["text"], return_tensors="pt", truncation=False)
        orig_len = full_tokens.input_ids.shape[1]
        orig_lens_by_cat.setdefault(p["category"], []).append(orig_len)

        # Check if full prompt + generation exceeds model context
        if model_max_context is not None:
            if orig_len + tokens_to_gen > model_max_context:
                overflow_by_cat[p["category"]] = overflow_by_cat.get(p["category"], 0) + 1

        # Truncated copy for actual generation
        tokens = tokenizer(p["text"], return_tensors="pt", truncation=True,
                           max_length=max_input_tokens)
        tokenized.append({
            "id": p["id"],
            "category": p["category"],
            "input_ids": tokens.input_ids.to("cuda"),
            "orig_length": orig_len,
        })

    lengths = [t["input_ids"].shape[1] for t in tokenized]
    print(f"[Tokenize] Input lengths (post-trunc): min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, truncated to max {max_input_tokens}")

    global_overflow = sum(overflow_by_cat.values())
    if model_max_context is not None:
        print(f"[Tokenize] Model context = {model_max_context}, "
              f"generation budget = {tokens_to_gen}")
        print(f"[Tokenize] Prompts exceeding context (input_len + gen > {model_max_context}): "
              f"{global_overflow}/{len(prompts)}")
        for cat in sorted(orig_lens_by_cat):
            cnt = overflow_by_cat.get(cat, 0)
            if cnt > 0:
                print(f"    {cat}: {cnt}/{len(orig_lens_by_cat[cat])}")

    overflow_info = {
        "global_overflow": global_overflow,
        "per_category": overflow_by_cat,
        "model_max_context": model_max_context,
        "tokens_to_gen": tokens_to_gen,
        "per_category_total": {c: len(v) for c, v in orig_lens_by_cat.items()},
    }
    return tokenized, overflow_info


# ============================================================
# MODEL LOADING
# ============================================================
def load_config(model_path):
    """Load experiment config saved by Hydra.

    Supports three layouts:
      1. Flat HF model dir:     {model_path}/.hydra/config.yaml
      2. Sibling checkpoint:    {model_path}/../.hydra/config.yaml
         (run_dir/checkpoint-N/ with .hydra/ next to checkpoint dir)
      3. Nested checkpoint:     {model_path}/../../.hydra/config.yaml
         (run_dir/checkpoints/checkpoint-N/ with .hydra/ at run_dir)
    """
    candidates = [
        os.path.join(model_path, ".hydra", "config.yaml"),
        os.path.join(os.path.dirname(model_path), ".hydra", "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(model_path)), ".hydra", "config.yaml"),
    ]
    for config_path in candidates:
        if os.path.exists(config_path):
            print(f"CONFIG PATH: {config_path}")
            with open(config_path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Config not found for {model_path}. Tried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


def load_forward_model(model_path):
    """Load multi-exit model wrapped for inference."""
    cfg = load_config(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)
    # Training configs save use_cache=False; force True for inference so HF
    # generate() slices input_ids to 1-token decode steps (otherwise the
    # custom is_decoding branch never fires and routers never run).
    hf_config.use_cache = True

    # Tokenizer: try model_path first (HF downloads include it), fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    InferenceModel = get_Nvium_inference_model_class(NviumModel)
    model = InferenceModel(hf_config, exp_config=cfg, attn_implementation="sdpa")
    model = model.to(dtype=TORCH_DTYPE)

    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    if hasattr(model, "_apply_tying"):
        model._apply_tying(cfg.get("tying", {}))
    return model, tokenizer, cfg


def load_baseline_model(model_path):
    """Load trained baseline model wrapped for inference."""
    cfg = load_config(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)
    hf_config.use_cache = True  # override training-time use_cache=False

    InferenceBaseline = get_baseline_model_class(BaselineLlamaModel)
    model = InferenceBaseline.from_pretrained(
        model_path, config=hf_config, exp_config=cfg,
        torch_dtype=TORCH_DTYPE, attn_implementation="sdpa",
    )
    if hasattr(model, "_apply_tying"):
        model._apply_tying(cfg.get("tying", {}))
    return model


# ============================================================
# BENCHMARKING
# ============================================================
def build_temperature_grid(head_temps, router_temps):
    """Build grid of (head_temp, router_temp) settings."""
    grid = []
    for ht in head_temps:
        for rt in router_temps:
            grid.append({"head_temp": ht, "router_temp": rt})
    return grid


def run_single_pass(model, tokenized_prompts_by_cat, tokens_to_gen, head_temp, is_forward,
                    save_responses=False):
    """
    Run one complete pass over all prompts, measuring per-category timing.
    Returns (global_stats, per_category_stats, responses) tuple.
    `responses` is None unless save_responses=True, in which case it is a list of
    {"category", "prompt_id", "input_len", "output_ids"} dicts populated AFTER the
    timed CUDA region (so detokenization has zero impact on timing).
    """
    is_greedy = (head_temp == 0.0)

    # Allow natural stopping: model may emit EOS and produce variable-length
    # sequences.  Speedup is then measured per-token via tps (tokens/second),
    # which is invariant to total sequence length.
    gen_kwargs = dict(
        max_new_tokens=tokens_to_gen,
        pad_token_id=model.generation_config.pad_token_id,
    )
    if is_greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = head_temp
        gen_kwargs["top_k"] = 50
        gen_kwargs["top_p"] = 1.0

    # Reset state
    if hasattr(model, "history"):
        model.history = {"decisions": [], "layers_skipped": []}
    if hasattr(model, "deferred_queues"):
        model.deferred_queues = [[] for _ in range(model.num_early_exits)]
    if hasattr(model, "deferred_queue"):
        model.deferred_queue = []

    total_tokens = 0
    total_wall_time = 0.0
    cat_stats = {}
    responses = [] if save_responses else None

    with torch.no_grad():
        for cat, items in tokenized_prompts_by_cat.items():
            # Reset deferred queues between categories
            if hasattr(model, "deferred_queues"):
                model.deferred_queues = [[] for _ in range(model.num_early_exits)]
            if hasattr(model, "deferred_queue"):
                model.deferred_queue = []

            # Track decisions per category for forward model
            if is_forward and hasattr(model, "history"):
                decisions_before = len(model.history["decisions"])

            cat_tokens = 0
            # Holds (prompt_id, input_len, output_tensor_ref) during the timed
            # region — just GPU tensor refs, no copies, no detokenization.
            cat_raw_outputs = [] if save_responses else None

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for item in items:
                if hasattr(model, "deferred_queues"):
                    model.deferred_queues = [[] for _ in range(model.num_early_exits)]
                if hasattr(model, "deferred_queue"):
                    model.deferred_queue = []

                if not is_greedy:
                    torch.manual_seed(42 + item["id"])
                    torch.cuda.manual_seed(42 + item["id"])

                outputs = model.generate(input_ids=item["input_ids"], **gen_kwargs)
                cat_tokens += outputs.shape[1] - item["input_ids"].shape[1]
                if save_responses:
                    cat_raw_outputs.append((item["id"], item["input_ids"].shape[1], outputs))
            end.record()
            torch.cuda.synchronize()

            # Extract output token ids OUTSIDE the timed region
            if save_responses:
                for prompt_id, in_len, out_tensor in cat_raw_outputs:
                    out_ids = out_tensor[0, in_len:].detach().cpu().tolist()
                    responses.append({
                        "category": cat,
                        "prompt_id": prompt_id,
                        "input_len": in_len,
                        "output_ids": out_ids,
                    })

            cat_wall = start.elapsed_time(end) / 1000.0
            cat_tps = cat_tokens / cat_wall if cat_wall > 0 else 0

            cat_result = {
                "wall_time_s": cat_wall,
                "total_tokens": cat_tokens,
                "tps": cat_tps,
                "num_prompts": len(items),
            }

            # Per-category exit stats for forward model
            if is_forward and hasattr(model, "history"):
                cat_decisions = model.history["decisions"][decisions_before:]
                cat_skipped = model.history["layers_skipped"][decisions_before:]
                n = len(cat_decisions)
                if n > 0:
                    exit_counts = {}
                    for d in cat_decisions:
                        exit_counts[d] = exit_counts.get(d, 0) + 1
                    cat_result["exit_breakdown"] = {k: v / n for k, v in sorted(exit_counts.items())}
                    cat_result["early_exit_ratio"] = 1.0 - (exit_counts.get("final", 0) / n)
                    cat_result["avg_layers_skipped"] = sum(cat_skipped) / n

            cat_stats[cat] = cat_result
            total_tokens += cat_tokens
            total_wall_time += cat_wall

            # Progress print — AFTER timing, between categories
            avg_len = cat_tokens / len(items) if items else 0
            exit_str = ""
            if "early_exit_ratio" in cat_result:
                exit_str = (f", exit={cat_result['early_exit_ratio']:.1%}"
                            f", skipL={cat_result['avg_layers_skipped']:.1f}")
            label_str = "fwd" if is_forward else "base"
            print(f"      [{label_str}] {cat:>20s}: {cat_tps:.1f} tok/s, "
                  f"{cat_wall:.2f}s, {len(items)} prompts, "
                  f"avg_len={avg_len:.1f}{exit_str}")

    # Global stats
    global_tps = total_tokens / total_wall_time if total_wall_time > 0 else 0
    stats = {
        "wall_time_s": total_wall_time,
        "total_tokens": total_tokens,
        "tps": global_tps,
    }

    if is_forward and hasattr(model, "history") and model.history["decisions"]:
        decisions = model.history["decisions"]
        skipped = model.history["layers_skipped"]
        n = len(decisions)

        exit_counts = {}
        for d in decisions:
            exit_counts[d] = exit_counts.get(d, 0) + 1

        stats["exit_breakdown"] = {k: v / n for k, v in sorted(exit_counts.items())}
        stats["early_exit_ratio"] = 1.0 - (exit_counts.get("final", 0) / n) if n > 0 else 0
        stats["avg_layers_skipped"] = sum(skipped) / n if n > 0 else 0

    return stats, cat_stats, responses


def benchmark_model_at_setting(model, tokenized_prompts_by_cat, tokens_to_gen,
                                head_temp, router_temp, num_runs, label, is_forward):
    """
    Benchmark one model at one temperature setting across multiple runs.
    First run is burn-in (discarded).
    Returns (global_result, per_category_result) tuple.
    """
    if is_forward:
        model.sample_router = (router_temp > 0.0)
        model.router_sample_temp = max(router_temp, 1e-5) if router_temp > 0 else 1.0

    ht_str = "greedy" if head_temp == 0.0 else f"T={head_temp}"
    rt_str = "greedy" if router_temp == 0.0 else f"T={router_temp}"
    print(f"\n  [{label}] head={ht_str}, router={rt_str}, {num_runs} runs (+1 burn-in)")

    setting_start = time.time()
    run_stats = []
    run_cat_stats = []  # list of per-category dicts per run
    final_responses = None
    for i in range(num_runs + 1):  # +1 for burn-in
        # Save responses only on the last measurement run
        save_responses = (i == num_runs)
        stats, cat_stats, responses = run_single_pass(
            model, tokenized_prompts_by_cat,
            tokens_to_gen, head_temp, is_forward,
            save_responses=save_responses,
        )
        if save_responses:
            final_responses = responses

        if i == 0:
            print(f"    Burn-in: {stats['tps']:.1f} tok/s [discarded]")
            if "exit_breakdown" in stats:
                bd = stats["exit_breakdown"]
                parts = [f"{k}={v:.1%}" for k, v in bd.items()]
                print(f"    Burn-in exits: {', '.join(parts)}")
            continue

        run_stats.append(stats)
        run_cat_stats.append(cat_stats)
        print(f"    Run {i}: {stats['tps']:.1f} tok/s, {stats['wall_time_s']:.2f}s", end="")
        if "early_exit_ratio" in stats:
            print(f", exit={stats['early_exit_ratio']:.2%}, skip={stats['avg_layers_skipped']:.1f}L", end="")
        print()

    # Aggregate global stats across runs
    tps_arr = np.array([s["tps"] for s in run_stats])
    lat_arr = np.array([s["wall_time_s"] for s in run_stats])
    # Global average response length: total_tokens / num_prompts_total
    total_prompts = sum(run_cat_stats[0][c]["num_prompts"] for c in run_cat_stats[0])
    resp_len_arr = np.array([s["total_tokens"] / total_prompts for s in run_stats])

    result = {
        "mean_tps": float(np.mean(tps_arr)),
        "std_tps": float(np.std(tps_arr)),
        "mean_latency_s": float(np.mean(lat_arr)),
        "std_latency_s": float(np.std(lat_arr)),
        "mean_response_length": float(np.mean(resp_len_arr)),
        "std_response_length": float(np.std(resp_len_arr)),
    }

    if is_forward and "early_exit_ratio" in run_stats[0]:
        exit_arr = np.array([s["early_exit_ratio"] for s in run_stats])
        skip_arr = np.array([s["avg_layers_skipped"] for s in run_stats])
        result["mean_early_exit_ratio"] = float(np.mean(exit_arr))
        result["std_early_exit_ratio"] = float(np.std(exit_arr))
        result["mean_layers_skipped"] = float(np.mean(skip_arr))
        result["std_layers_skipped"] = float(np.std(skip_arr))

        all_exits = set()
        for s in run_stats:
            all_exits.update(s.get("exit_breakdown", {}).keys())
        exit_breakdown_mean = {}
        for key in sorted(all_exits):
            vals = [s.get("exit_breakdown", {}).get(key, 0.0) for s in run_stats]
            exit_breakdown_mean[key] = float(np.mean(vals))
        result["exit_breakdown"] = exit_breakdown_mean

    # Aggregate per-category stats across runs
    categories = sorted(run_cat_stats[0].keys())
    cat_result = {}
    for cat in categories:
        cat_tps = np.array([rcs[cat]["tps"] for rcs in run_cat_stats])
        cat_lat = np.array([rcs[cat]["wall_time_s"] for rcs in run_cat_stats])
        n_prompts_cat = run_cat_stats[0][cat]["num_prompts"]
        cat_resp_len = np.array([rcs[cat]["total_tokens"] / n_prompts_cat
                                 for rcs in run_cat_stats])
        cat_agg = {
            "mean_tps": float(np.mean(cat_tps)),
            "std_tps": float(np.std(cat_tps)),
            "mean_latency_s": float(np.mean(cat_lat)),
            "num_prompts": n_prompts_cat,
            "mean_response_length": float(np.mean(cat_resp_len)),
            "std_response_length": float(np.std(cat_resp_len)),
        }
        if is_forward and "early_exit_ratio" in run_cat_stats[0][cat]:
            cat_exit = np.array([rcs[cat]["early_exit_ratio"] for rcs in run_cat_stats])
            cat_skip = np.array([rcs[cat]["avg_layers_skipped"] for rcs in run_cat_stats])
            cat_agg["mean_early_exit_ratio"] = float(np.mean(cat_exit))
            cat_agg["mean_layers_skipped"] = float(np.mean(cat_skip))

            # Aggregate per-exit breakdown for this category across runs
            all_keys = set()
            for rcs in run_cat_stats:
                all_keys.update(rcs[cat].get("exit_breakdown", {}).keys())
            cat_bd = {}
            for key in sorted(all_keys):
                vals = [rcs[cat].get("exit_breakdown", {}).get(key, 0.0)
                        for rcs in run_cat_stats]
                cat_bd[key] = float(np.mean(vals))
            cat_agg["exit_breakdown"] = cat_bd
        cat_result[cat] = cat_agg

    print(f"  => {label}: {result['mean_tps']:.1f} ± {result['std_tps']:.1f} tok/s")
    if "exit_breakdown" in result:
        parts = [f"{k}={v:.1%}" for k, v in result["exit_breakdown"].items()]
        print(f"  => Exits: {', '.join(parts)}")

    setting_elapsed = time.time() - setting_start
    result["benchmark_time_s"] = round(setting_elapsed, 1)
    print(f"  => Time: {fmt_duration(setting_elapsed)}")

    return result, cat_result, final_responses


# ============================================================
# MAIN BENCHMARK
# ============================================================
def _save_per_setting_yaml(path, entry, forward_path, baseline_path,
                            forward_name, baseline_name, tokens_to_gen,
                            max_input_tokens, num_runs, prompts, limit,
                            forward_cfg, exit_indices, overflow_info,
                            head_temps, router_temps, grid, setting_idx):
    """Save a single setting's result as a standalone YAML with full context.

    Used in --results_subdir mode so intermediate progress survives job crashes
    and can be resumed without re-running completed settings.
    """
    entry_clean = {k: v for k, v in entry.items()
                   if k not in ("_forward_responses", "_baseline_responses")}
    data = {
        "config": {
            "forward_path": forward_path,
            "baseline_path": baseline_path,
            "forward_name": forward_name,
            "baseline_name": baseline_name,
            "tokens_per_generation": tokens_to_gen,
            "max_input_tokens": max_input_tokens,
            "num_runs_per_setting": num_runs,
            "num_prompts": len(prompts),
            "limit_per_category": limit,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "benchmark_source": "Spec-Bench (Xia et al., 2024)",
        },
        "context_overflow": {
            "model_max_context": overflow_info["model_max_context"],
            "tokens_to_gen": overflow_info["tokens_to_gen"],
            "global_overflow_count": overflow_info["global_overflow"],
            "global_total": len(prompts),
            "per_category": {
                cat: {
                    "overflow": overflow_info["per_category"].get(cat, 0),
                    "total": overflow_info["per_category_total"].get(cat, 0),
                }
                for cat in sorted(overflow_info["per_category_total"])
            },
        },
        "forward_model_config": {
            "num_layers": forward_cfg.get("num_hidden_layers", None),
            "hidden_size": forward_cfg.get("hidden_size", None),
            "early_layer_indices": exit_indices,
            "num_exits": len(exit_indices) + 1,
        },
        "temperature_grid": {
            "head_temps": [t if t > 0 else "greedy" for t in head_temps],
            "router_temps": [t if t > 0 else "greedy" for t in router_temps],
            "total_settings": len(grid),
            "this_setting_index": setting_idx,
        },
        "result": entry_clean,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _write_per_setting_responses(path, forward_path, baseline_path, tokens_to_gen,
                                  setting_dict, forward_responses, baseline_responses,
                                  tokenizer, prompts):
    """Write a per-setting responses text file. Mirrors the format of the
    non-subdir-mode combined responses file, but scoped to one setting."""
    prompt_by_id = {p["id"]: p for p in prompts}
    fwd_resps = forward_responses or []
    base_resps = baseline_responses or []
    base_by_id = {x["prompt_id"]: x for x in base_resps}
    by_cat = {}
    for x in fwd_resps:
        by_cat.setdefault(x["category"], []).append(x)

    with open(path, "w") as rf:
        rf.write("Spec-Bench responses\n")
        rf.write(f"Forward:  {forward_path}\n")
        rf.write(f"Baseline: {baseline_path}\n")
        rf.write(f"Tokens to gen (max): {tokens_to_gen}\n")
        rf.write("=" * 80 + "\n\n")
        rf.write(f"### Setting: head_temp={setting_dict['head_temp']}, "
                 f"router_temp={setting_dict['router_temp']} ###\n\n")
        for cat in sorted(by_cat):
            rf.write(f"---- Category: {cat} ----\n\n")
            for fx in by_cat[cat]:
                pid = fx["prompt_id"]
                prompt_text = prompt_by_id.get(pid, {}).get("text", "<?>")
                fwd_text = tokenizer.decode(fx["output_ids"], skip_special_tokens=True)
                base_x = base_by_id.get(pid)
                base_text = (tokenizer.decode(base_x["output_ids"], skip_special_tokens=True)
                             if base_x else "<missing>")
                rf.write(f"[prompt_id={pid}, input_len={fx['input_len']}]\n")
                rf.write(f"PROMPT: {prompt_text}\n\n")
                rf.write(f"FORWARD  ({len(fx['output_ids'])} tok): {fwd_text}\n\n")
                rf.write(f"BASELINE ({len(base_x['output_ids']) if base_x else 0} tok): "
                         f"{base_text}\n")
                rf.write("-" * 80 + "\n")
            rf.write("\n")


def save_report(output_dir, report_data):
    """Save benchmark report with auto-incrementing version."""
    os.makedirs(output_dir, exist_ok=True)
    existing = os.listdir(output_dir)
    versions = [int(m.group(1)) for f in existing
                for m in [re.search(r'specbench_Nvium_v(\d+)\.yaml', f)] if m]
    version = max(versions) + 1 if versions else 1

    path = os.path.join(output_dir, f"specbench_Nvium_v{version}.yaml")
    with open(path, 'w') as f:
        yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)
    print(f"\n[Report] Saved to: {path}")
    return path


def benchmark_pair(forward_path, baseline_path, cache_dir,
                   tokens_to_gen=128, num_runs=5, limit=None,
                   max_input_tokens=256, head_temps=None, router_temps=None,
                   results_subdir=None):
    """Full benchmark pipeline for one multi-exit forward / baseline pair.

    If `results_subdir` is provided, per-setting intermediate YAMLs + responses
    are written to `{forward_path}/benchmarks/{results_subdir}/` as each setting
    completes, enabling resume-from-crash. The final combined report is written
    as `combined.yaml` in the same subdir (overwritten on each run).
    """

    if head_temps is None:
        head_temps = DEFAULT_HEAD_TEMPS
    if router_temps is None:
        router_temps = DEFAULT_ROUTER_TEMPS

    forward_name = os.path.basename(os.path.normpath(forward_path))
    baseline_name = os.path.basename(os.path.normpath(baseline_path))

    grid = build_temperature_grid(head_temps, router_temps)

    # --- Resume support: compute report_dir early and scan for per-setting files ---
    report_dir = os.path.join(forward_path, "benchmarks")
    if results_subdir:
        report_dir = os.path.join(report_dir, results_subdir)
    os.makedirs(report_dir, exist_ok=True)

    def _setting_tag(ht, rt):
        ht_tag = "greedy" if ht == 0.0 else f"t{ht:g}"
        rt_tag = "greedy" if rt == 0.0 else f"t{rt:g}"
        return ht_tag, rt_tag

    def _setting_filepath(ht, rt):
        ht_tag, rt_tag = _setting_tag(ht, rt)
        return os.path.join(report_dir, f"setting_head_{ht_tag}_router_{rt_tag}.yaml")

    def _setting_responses_path(ht, rt):
        ht_tag, rt_tag = _setting_tag(ht, rt)
        return os.path.join(report_dir, f"responses_head_{ht_tag}_router_{rt_tag}.txt")

    precomputed = {}
    if results_subdir:
        for _idx, _s in enumerate(grid):
            _fp = _setting_filepath(_s["head_temp"], _s["router_temp"])
            if os.path.exists(_fp):
                try:
                    with open(_fp) as _f:
                        _data = yaml.safe_load(_f)
                    if isinstance(_data, dict) and "result" in _data:
                        precomputed[_idx] = _data
                        _ht_tag, _rt_tag = _setting_tag(_s["head_temp"], _s["router_temp"])
                        print(f"[Resume] Found existing: head={_ht_tag}, "
                              f"router={_rt_tag} → {os.path.basename(_fp)}")
                except Exception as _e:
                    print(f"[Resume] Failed to parse {_fp}: {_e}")
        if precomputed:
            print(f"[Resume] Will skip {len(precomputed)}/{len(grid)} already-computed settings")

    limit_str = f"limit={limit}/cat" if limit else "full"
    print(f"\n{'#'*60}")
    print(f"  Spec-Bench Evaluation (Multi-Exit)")
    print(f"  Forward:  {forward_name}")
    print(f"  Baseline: {baseline_name}")
    print(f"  Prompts:  {limit_str} | Gen: {tokens_to_gen} tokens")
    print(f"  Grid:     {len(grid)} settings × {num_runs} runs")
    print(f"{'#'*60}")

    timings = {}
    total_start = time.time()

    # --- 1. Download / load prompts ---
    t0 = time.time()
    question_file = download_specbench(cache_dir)
    prompts = load_prompts(question_file, limit=limit)
    timings["data_loading_s"] = round(time.time() - t0, 1)
    print(f"[Timer] Data loading: {fmt_duration(timings['data_loading_s'])}")

    # --- 2. Load models ---
    t0 = time.time()
    print("\n--- Loading Multi-Exit Forward Model ---")
    forward_model, tokenizer, forward_cfg = load_forward_model(forward_path)
    forward_model.to("cuda").eval()
    forward_model.generation_config.pad_token_id = tokenizer.pad_token_id

    exit_indices = forward_cfg.get("early_layer_idx", [])
    if isinstance(exit_indices, int):
        exit_indices = [exit_indices]
    print(f"[Model] Exit points: {exit_indices}")

    timings["forward_model_loading_s"] = round(time.time() - t0, 1)
    print(f"[Timer] Forward model loading: {fmt_duration(timings['forward_model_loading_s'])}")

    t0 = time.time()
    print("\n--- Loading Baseline Model ---")
    baseline_model = load_baseline_model(baseline_path)
    baseline_model.to("cuda").eval()
    baseline_model.generation_config.pad_token_id = tokenizer.pad_token_id
    timings["baseline_model_loading_s"] = round(time.time() - t0, 1)
    print(f"[Timer] Baseline model loading: {fmt_duration(timings['baseline_model_loading_s'])}")

    # --- 3. Tokenize & group by category ---
    t0 = time.time()
    model_max_context = getattr(forward_model.config, "max_position_embeddings", None)
    if model_max_context is not None and max_input_tokens + tokens_to_gen > model_max_context:
        print(f"[WARN] max_input_tokens ({max_input_tokens}) + tokens_to_gen ({tokens_to_gen}) "
              f"= {max_input_tokens + tokens_to_gen} > model context ({model_max_context}). "
              f"Long prompts may fail during generation. "
              f"Consider --max_input_tokens {model_max_context - tokens_to_gen}.")
    tokenized, overflow_info = tokenize_prompts(
        prompts, tokenizer, max_input_tokens,
        model_max_context=model_max_context, tokens_to_gen=tokens_to_gen,
    )
    timings["tokenization_s"] = round(time.time() - t0, 1)
    print(f"[Timer] Tokenization: {fmt_duration(timings['tokenization_s'])}")

    tokenized_by_cat = {}
    for item in tokenized:
        tokenized_by_cat.setdefault(item["category"], []).append(item)

    # --- 4. Run grid ---
    all_results = []
    grid_start = time.time()

    for setting_idx, setting in enumerate(grid):
        ht = setting["head_temp"]
        rt = setting["router_temp"]

        ht_str = "greedy" if ht == 0.0 else f"{ht}"
        rt_str = "greedy" if rt == 0.0 else f"{rt}"

        if setting_idx in precomputed:
            saved_result = precomputed[setting_idx]["result"]
            for _k in ("_forward_responses", "_baseline_responses"):
                saved_result.pop(_k, None)
            all_results.append(saved_result)
            print(f"\n{'='*60}")
            print(f"  [Resumed] Setting {setting_idx+1}/{len(grid)}: "
                  f"head={ht_str}, router={rt_str}")
            print(f"  Speedup: {saved_result.get('speedup', 0):.3f}x "
                  f"(loaded from {os.path.basename(_setting_filepath(ht, rt))})")
            print(f"{'='*60}")
            continue

        setting_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Setting {setting_idx+1}/{len(grid)}: head={ht_str}, router={rt_str}")
        print(f"{'='*60}")

        # Benchmark forward model
        forward_result, forward_cat, forward_responses = benchmark_model_at_setting(
            forward_model, tokenized_by_cat, tokens_to_gen,
            ht, rt, num_runs, "Forward", is_forward=True,
        )

        # Benchmark baseline model
        baseline_result, baseline_cat, baseline_responses = benchmark_model_at_setting(
            baseline_model, tokenized_by_cat, tokens_to_gen,
            ht, rt, num_runs, "Baseline", is_forward=False,
        )

        # Global speedup
        speedup = forward_result["mean_tps"] / baseline_result["mean_tps"] \
            if baseline_result["mean_tps"] > 0 else 0

        # Per-category speedup
        per_category = {}
        for cat in sorted(tokenized_by_cat.keys()):
            fwd_cat_tps = forward_cat[cat]["mean_tps"]
            base_cat_tps = baseline_cat[cat]["mean_tps"]
            cat_speedup = fwd_cat_tps / base_cat_tps if base_cat_tps > 0 else 0
            per_category[cat] = {
                "forward_tps": fwd_cat_tps,
                "baseline_tps": base_cat_tps,
                "speedup": float(cat_speedup),
                "num_prompts": forward_cat[cat]["num_prompts"],
                "forward_response_length": forward_cat[cat]["mean_response_length"],
                "baseline_response_length": baseline_cat[cat]["mean_response_length"],
            }
            if "mean_early_exit_ratio" in forward_cat[cat]:
                per_category[cat]["early_exit_ratio"] = forward_cat[cat]["mean_early_exit_ratio"]
                per_category[cat]["avg_layers_skipped"] = forward_cat[cat]["mean_layers_skipped"]
            if "exit_breakdown" in forward_cat[cat]:
                per_category[cat]["exit_breakdown"] = forward_cat[cat]["exit_breakdown"]

        print(f"\n  Global speedup: {speedup:.3f}x ({(speedup-1)*100:+.1f}%)")
        print(f"  Global avg response length: "
              f"fwd={forward_result['mean_response_length']:.1f} tok, "
              f"base={baseline_result['mean_response_length']:.1f} tok")
        if "exit_breakdown" in forward_result:
            bd_parts = [f"{k}={v:.1%}" for k, v in forward_result["exit_breakdown"].items()]
            print(f"  Global exits: {', '.join(bd_parts)}  "
                  f"(avg layers skipped={forward_result.get('mean_layers_skipped', 0):.2f})")

        print(f"  Per-category:")
        for cat in sorted(per_category):
            pc = per_category[cat]
            print(f"    {cat:>20s}: {pc['speedup']:.3f}x  "
                  f"(fwd={pc['forward_tps']:.1f}, base={pc['baseline_tps']:.1f} tok/s)  "
                  f"len: fwd={pc['forward_response_length']:.1f}, "
                  f"base={pc['baseline_response_length']:.1f}")
            if "exit_breakdown" in pc:
                bd_parts = [f"{k}={v:.1%}" for k, v in pc["exit_breakdown"].items()]
                print(f"    {'':>20s}  exits: {', '.join(bd_parts)}  "
                      f"skipL={pc.get('avg_layers_skipped', 0):.2f}")

        setting_elapsed = time.time() - setting_start
        remaining = (len(grid) - setting_idx - 1) * (time.time() - grid_start) / (setting_idx + 1)
        print(f"  Setting time: {fmt_duration(setting_elapsed)} | "
              f"ETA remaining: {fmt_duration(remaining)}")

        all_results.append({
            "setting": {
                "head_temp": ht if ht > 0 else "greedy",
                "router_temp": rt if rt > 0 else "greedy",
            },
            "forward": forward_result,
            "baseline": baseline_result,
            "speedup": float(speedup),
            "speedup_pct": f"{(speedup-1)*100:+.1f}%",
            "per_category": per_category,
            "setting_time_s": round(setting_elapsed, 1),
            "_forward_responses": forward_responses,   # consumed after loop, not in yaml
            "_baseline_responses": baseline_responses,
        })

        # --- Save per-setting file immediately (resume checkpoint) ---
        if results_subdir:
            _per_path = _setting_filepath(ht, rt)
            _save_per_setting_yaml(
                path=_per_path,
                entry=all_results[-1],
                forward_path=forward_path, baseline_path=baseline_path,
                forward_name=forward_name, baseline_name=baseline_name,
                tokens_to_gen=tokens_to_gen, max_input_tokens=max_input_tokens,
                num_runs=num_runs, prompts=prompts, limit=limit,
                forward_cfg=forward_cfg, exit_indices=exit_indices,
                overflow_info=overflow_info,
                head_temps=head_temps, router_temps=router_temps,
                grid=grid, setting_idx=setting_idx,
            )
            print(f"[Save] Per-setting result → {_per_path}")

            _resp_path = _setting_responses_path(ht, rt)
            _write_per_setting_responses(
                path=_resp_path,
                forward_path=forward_path, baseline_path=baseline_path,
                tokens_to_gen=tokens_to_gen,
                setting_dict=all_results[-1]["setting"],
                forward_responses=forward_responses,
                baseline_responses=baseline_responses,
                tokenizer=tokenizer, prompts=prompts,
            )
            print(f"[Save] Per-setting responses → {_resp_path}")

            # Free tensor refs from the in-memory entry (already persisted)
            all_results[-1].pop("_forward_responses", None)
            all_results[-1].pop("_baseline_responses", None)

    # --- 5. Summary ---
    total_elapsed = time.time() - total_start
    timings["grid_benchmark_s"] = round(time.time() - grid_start, 1)
    timings["total_s"] = round(total_elapsed, 1)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {forward_name} (exits: {exit_indices})")
    print(f"  Total time: {fmt_duration(total_elapsed)}")
    print(f"{'='*60}")
    print(f"  {'Head':>8s}  {'Router':>8s}  {'Fwd TPS':>10s}  {'Base TPS':>10s}  {'Speedup':>8s}  {'Exit%':>8s}  {'SkipL':>6s}  {'Time':>8s}")
    print(f"  {'-'*74}")
    for r in all_results:
        s = r["setting"]
        ht_s = str(s["head_temp"])
        rt_s = str(s["router_temp"])
        fwd_tps = f"{r['forward']['mean_tps']:.1f}"
        base_tps = f"{r['baseline']['mean_tps']:.1f}"
        spd = f"{r['speedup']:.3f}x"
        exit_r = f"{r['forward'].get('mean_early_exit_ratio', 0):.1%}" \
            if "mean_early_exit_ratio" in r["forward"] else "N/A"
        skip_l = f"{r['forward'].get('mean_layers_skipped', 0):.1f}" \
            if "mean_layers_skipped" in r["forward"] else "N/A"
        t_s = fmt_duration(r.get("setting_time_s", 0))
        print(f"  {ht_s:>8s}  {rt_s:>8s}  {fwd_tps:>10s}  {base_tps:>10s}  {spd:>8s}  {exit_r:>8s}  {skip_l:>6s}  {t_s:>8s}")

    # Per-category summary for best setting
    best = max(all_results, key=lambda r: r["speedup"])
    print(f"\n  Per-category breakdown (best setting, speedup={best['speedup']:.3f}x):")
    print(f"  {'Category':>20s}  {'Speedup':>8s}  {'Fwd TPS':>10s}  {'Base TPS':>10s}  {'Exit%':>8s}")
    print(f"  {'-'*62}")
    for cat in sorted(best["per_category"]):
        pc = best["per_category"][cat]
        exit_str = f"{pc['early_exit_ratio']:.1%}" if "early_exit_ratio" in pc else "N/A"
        print(f"  {cat:>20s}  {pc['speedup']:.3f}x  {pc['forward_tps']:>10.1f}  "
              f"{pc['baseline_tps']:>10.1f}  {exit_str:>8s}")

    if "exit_breakdown" in best["forward"]:
        print(f"\n  Exit distribution:")
        for k, v in best["forward"]["exit_breakdown"].items():
            print(f"    {k}: {v:.1%}")

    # --- 6. Build report ---
    cat_counts = {}
    for p in prompts:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    report = {
        "config": {
            "forward_path": forward_path,
            "baseline_path": baseline_path,
            "forward_name": forward_name,
            "baseline_name": baseline_name,
            "tokens_per_generation": tokens_to_gen,
            "max_input_tokens": max_input_tokens,
            "num_runs_per_setting": num_runs,
            "num_prompts": len(prompts),
            "limit_per_category": limit,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "benchmark_source": "Spec-Bench (Xia et al., 2024)",
        },
        "context_overflow": {
            "model_max_context": overflow_info["model_max_context"],
            "tokens_to_gen": overflow_info["tokens_to_gen"],
            "global_overflow_count": overflow_info["global_overflow"],
            "global_total": len(prompts),
            "per_category": {
                cat: {
                    "overflow": overflow_info["per_category"].get(cat, 0),
                    "total": overflow_info["per_category_total"].get(cat, 0),
                }
                for cat in sorted(overflow_info["per_category_total"])
            },
        },
        "forward_model_config": {
            "num_layers": forward_cfg.get("num_hidden_layers", None),
            "hidden_size": forward_cfg.get("hidden_size", None),
            "early_layer_indices": exit_indices,
            "num_exits": len(exit_indices) + 1,
        },
        "timings": timings,
        "prompt_distribution": cat_counts,
        "temperature_grid": {
            "head_temps": [t if t > 0 else "greedy" for t in head_temps],
            "router_temps": [t if t > 0 else "greedy" for t in router_temps],
            "total_settings": len(grid),
        },
        "results": all_results,
    }

    # --- Subdir mode: write combined.yaml and skip the versioned responses block ---
    if results_subdir:
        # Strip any lingering tensor refs (already stripped per-setting, defensive)
        for r in all_results:
            r.pop("_forward_responses", None)
            r.pop("_baseline_responses", None)
        combined_path = os.path.join(report_dir, "combined.yaml")
        with open(combined_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)
        print(f"\n[Report] Saved combined to: {combined_path}")

        del forward_model, baseline_model
        torch.cuda.empty_cache()
        gc.collect()
        return report

    # Save in the forward model directory
    report_dir = os.path.join(forward_path, "benchmarks")

    # ---- Write responses text file (detokenization happens here, OUTSIDE timing) ----
    # Build a lookup from prompt id -> original text for readability
    prompt_by_id = {p["id"]: p for p in prompts}

    os.makedirs(report_dir, exist_ok=True)
    existing = os.listdir(report_dir)
    resp_versions = [int(m.group(1)) for f in existing
                     for m in [re.search(r'responses_v(\d+)\.txt', f)] if m]
    resp_version = max(resp_versions) + 1 if resp_versions else 1
    resp_path = os.path.join(report_dir, f"responses_v{resp_version}.txt")

    with open(resp_path, "w") as rf:
        rf.write(f"Spec-Bench responses\n")
        rf.write(f"Forward:  {forward_path}\n")
        rf.write(f"Baseline: {baseline_path}\n")
        rf.write(f"Tokens to gen (max): {tokens_to_gen}\n")
        rf.write("=" * 80 + "\n\n")

        for r in all_results:
            setting = r["setting"]
            rf.write(f"### Setting: head_temp={setting['head_temp']}, "
                     f"router_temp={setting['router_temp']} ###\n\n")

            fwd_resps = r.pop("_forward_responses", None) or []
            base_resps = r.pop("_baseline_responses", None) or []

            # Index baseline responses by prompt_id for side-by-side alignment
            base_by_id = {x["prompt_id"]: x for x in base_resps}

            # Group forward responses by category, then iterate
            by_cat = {}
            for x in fwd_resps:
                by_cat.setdefault(x["category"], []).append(x)

            for cat in sorted(by_cat):
                rf.write(f"---- Category: {cat} ----\n\n")
                for fx in by_cat[cat]:
                    pid = fx["prompt_id"]
                    prompt_text = prompt_by_id.get(pid, {}).get("text", "<?>")
                    fwd_text = tokenizer.decode(fx["output_ids"], skip_special_tokens=True)
                    base_x = base_by_id.get(pid)
                    base_text = (tokenizer.decode(base_x["output_ids"], skip_special_tokens=True)
                                 if base_x else "<missing>")

                    rf.write(f"[prompt_id={pid}, input_len={fx['input_len']}]\n")
                    rf.write(f"PROMPT: {prompt_text}\n\n")
                    rf.write(f"FORWARD  ({len(fx['output_ids'])} tok): {fwd_text}\n\n")
                    rf.write(f"BASELINE ({len(base_x['output_ids']) if base_x else 0} tok): "
                             f"{base_text}\n")
                    rf.write("-" * 80 + "\n")
                rf.write("\n")
            rf.write("\n")
    print(f"[Report] Saved responses to: {resp_path}")

    save_report(report_dir, report)

    # Cleanup
    del forward_model, baseline_model
    torch.cuda.empty_cache()
    gc.collect()

    return report


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Spec-Bench speedup evaluation for multi-exit early-exit models"
    )
    parser.add_argument("--forward_path", type=str, required=True,
                        help="Path to multi-exit model directory")
    parser.add_argument("--baseline_path", type=str, required=True,
                        help="Path to baseline model directory")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tokens", type=int, default=128,
                        help="Number of tokens to generate per prompt (default: 128)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of measurement runs per setting (default: 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max prompts per category (default: None = all)")
    parser.add_argument("--max_input_tokens", type=int, default=256,
                        help="Truncate input prompts to this length (default: 256)")
    parser.add_argument("--head_temps", type=float, nargs="+",
                        default=DEFAULT_HEAD_TEMPS,
                        help="Head temperatures to test (0.0 = greedy)")
    parser.add_argument("--router_temps", type=float, nargs="+",
                        default=DEFAULT_ROUTER_TEMPS,
                        help="Router temperatures to test (0.0 = greedy)")
    parser.add_argument("--results_subdir", type=str, default=None,
                        help="If set, write per-setting intermediate YAMLs + "
                             "responses under benchmarks/<subdir>/ and a final "
                             "combined.yaml. Enables resume-from-crash.")
    args = parser.parse_args()

    benchmark_pair(
        forward_path=args.forward_path,
        baseline_path=args.baseline_path,
        cache_dir=args.cache_dir,
        tokens_to_gen=args.tokens,
        num_runs=args.runs,
        limit=args.limit,
        max_input_tokens=args.max_input_tokens,
        head_temps=args.head_temps,
        router_temps=args.router_temps,
        results_subdir=args.results_subdir,
    )


if __name__ == "__main__":
    main()
