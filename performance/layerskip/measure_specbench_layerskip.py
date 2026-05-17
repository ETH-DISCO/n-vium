#!/usr/bin/env python3
"""
Spec-Bench speedup evaluation for LayerSkip self-speculative decoding.

Loads ONE LayerSkip checkpoint, runs a grid of (exit_layer, draft_length)
settings, and reports walltime speedup vs dense generation using the same
model. The dense baseline is the underlying LlamaForCausalLM.generate()
(no speculative decoding), bypassing the inference wrapper.

Usage:
    python measure_specbench_layerskip.py \
        --checkpoint /path/to/outputs_layerskip/.../checkpoints/checkpoint-17284 \
        --exit_layers 2 6 10 \
        --draft_lengths 5 10

    # Full Spec-Bench (480 prompts), single setting:
    python measure_specbench_layerskip.py \
        --checkpoint ... --exit_layers 6 --draft_lengths 5 \
        --prompts_per_category 80 --runs 3
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pretraining", "layerskip"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pretraining", "Nvium"))

from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from safetensors.torch import load_file

from layerskip_model import LayerSkipModel
from model_inference_layerskip import get_layerskip_inference_class

# ============================================================
# CONFIG
# ============================================================
TORCH_DTYPE = torch.bfloat16
SPECBENCH_REPO = "https://github.com/hemingkx/Spec-Bench.git"
DEFAULT_CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "."), "cache", "SpecDec")


def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"


# ============================================================
# SPEC-BENCH DATA (mirrors measure_specbench.py)
# ============================================================
def download_specbench(cache_dir):
    repo_dir = os.path.join(cache_dir, "Spec-Bench")
    question_file = os.path.join(repo_dir, "data", "spec_bench", "question.jsonl")
    if os.path.exists(question_file):
        print(f"[Cache] Using cached Spec-Bench: {question_file}")
        return question_file
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[Download] Cloning Spec-Bench to {repo_dir}...")
    subprocess.run(["git", "clone", "--depth", "1", SPECBENCH_REPO, repo_dir], check=True)
    if not os.path.exists(question_file):
        raise FileNotFoundError(f"Expected question.jsonl at {question_file} but not found.")
    return question_file


def load_prompts(question_file, prompts_per_category=20, seed=42):
    all_prompts = []
    with open(question_file) as f:
        for line in f:
            item = json.loads(line.strip())
            all_prompts.append({
                "id": item["question_id"],
                "category": item.get("category", "unknown"),
                "text": item["turns"][0],
            })

    rng = random.Random(seed)
    by_cat = {}
    for p in all_prompts:
        by_cat.setdefault(p["category"], []).append(p)

    sampled = []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        if len(items) <= prompts_per_category:
            sampled.extend(items)
        else:
            sampled.extend(rng.sample(items, prompts_per_category))

    print(f"[Data] Loaded {len(all_prompts)} prompts, sampled {len(sampled)} "
          f"({len(by_cat)} categories, up to {prompts_per_category} each)")
    for cat in sorted(by_cat):
        n_total = len(by_cat[cat])
        n_sampled = sum(1 for p in sampled if p["category"] == cat)
        print(f"  {cat}: {n_sampled}/{n_total}")
    return sampled


def tokenize_prompts(prompts, tokenizer, max_input_tokens=256):
    tokenized = []
    for p in prompts:
        tokens = tokenizer(p["text"], return_tensors="pt", truncation=True,
                           max_length=max_input_tokens)
        tokenized.append({
            "id": p["id"],
            "category": p["category"],
            "input_ids": tokens.input_ids.to("cuda"),
        })
    lengths = [t["input_ids"].shape[1] for t in tokenized]
    print(f"[Tokenize] Input lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, truncated to max {max_input_tokens}")
    return tokenized


# ============================================================
# MODEL LOADING
# ============================================================
def load_layerskip_model(checkpoint_path):
    """
    Load the LayerSkip model with the speculative-decoding inference wrapper.
    The same instance serves both roles:
      - speculative: model.generate_with_stats(...)
      - dense:       LlamaForCausalLM.generate(model, ...)   (MRO bypass)
    """
    experiment_root = os.path.dirname(checkpoint_path)
    config_path = os.path.join(experiment_root, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        # checkpoint inside a checkpoints/ subdir — try one level up
        experiment_root = os.path.dirname(experiment_root)
        config_path = os.path.join(experiment_root, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found (tried .hydra/config.yaml)")
    with open(config_path) as f:
        exp_cfg = yaml.safe_load(f)
    print(f"[CONFIG] {config_path}")

    CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "."), "cache")
    tokenizer = AutoTokenizer.from_pretrained(
        exp_cfg["model_name_or_path"],
        cache_dir=CACHE_DIR, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_config = AutoConfig.from_pretrained(
        exp_cfg["model_name_or_path"],
        cache_dir=CACHE_DIR, trust_remote_code=True, local_files_only=True,
    )
    if exp_cfg.get("model_overrides"):
        hf_config.update(exp_cfg["model_overrides"])

    InferenceClass = get_layerskip_inference_class(LayerSkipModel)
    model = InferenceClass(
        hf_config,
        exp_config=exp_cfg,
        attn_implementation=exp_cfg.get("load_args", {}).get("attn_implementation", "sdpa"),
    )
    model = model.to(dtype=TORCH_DTYPE)

    sf_path = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(sf_path):
        raise FileNotFoundError(f"No model.safetensors at {checkpoint_path}")
    state_dict = load_file(sf_path)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    print(f"[LOAD] Weights from {sf_path}")

    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, exp_cfg, experiment_root


# ============================================================
# BENCHMARKING
# ============================================================
def run_speculative_pass(model, tokenized_prompts, tokens_to_gen,
                          exit_layer, draft_length):
    """One full pass in speculative mode across all prompts. Returns stats."""
    model.history = {
        "n_accepted_per_step": [],
        "n_draft_per_step": [],
        "decisions": [],
        "layers_skipped": [],
    }

    total_tokens = 0
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for item in tokenized_prompts:
            out, _ = model.generate_with_stats(
                item["input_ids"],
                max_new_tokens=tokens_to_gen,
                exit_layer=exit_layer,
                draft_length=draft_length,
                do_sample=False,
            )
            total_tokens += out.shape[1] - item["input_ids"].shape[1]
    end.record()
    torch.cuda.synchronize()

    wall = start.elapsed_time(end) / 1000.0
    tps = total_tokens / wall if wall > 0 else 0.0

    n_acc = sum(model.history["n_accepted_per_step"])
    n_draft = sum(model.history["n_draft_per_step"])
    n_steps = len(model.history["n_accepted_per_step"])

    return {
        "wall_time_s": wall,
        "total_tokens": total_tokens,
        "tps": tps,
        "acceptance_rate": (n_acc / n_draft) if n_draft > 0 else 0.0,
        "mean_accepted_per_step": (n_acc / n_steps) if n_steps > 0 else 0.0,
    }


def run_dense_pass(model, tokenized_prompts, tokens_to_gen, tokenizer):
    """
    One full pass in dense mode (no speculative).  Bypasses the inference
    wrapper's generate() override by calling LlamaForCausalLM.generate directly.
    """
    total_tokens = 0
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for item in tokenized_prompts:
            # MRO bypass: call LlamaForCausalLM.generate directly on `model`.
            out = LlamaForCausalLM.generate(
                model,
                input_ids=item["input_ids"],
                min_new_tokens=tokens_to_gen,
                max_new_tokens=tokens_to_gen,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=-1,
            )
            total_tokens += out.shape[1] - item["input_ids"].shape[1]
    end.record()
    torch.cuda.synchronize()

    wall = start.elapsed_time(end) / 1000.0
    tps = total_tokens / wall if wall > 0 else 0.0
    return {"wall_time_s": wall, "total_tokens": total_tokens, "tps": tps}


def benchmark_setting(model, tokenizer, tokenized, tokens_to_gen,
                       exit_layer, draft_length, num_runs, is_speculative):
    label = f"Spec(exit={exit_layer},K={draft_length})" if is_speculative else "Dense"
    print(f"\n  [{label}] {num_runs} runs (+1 burn-in)")

    run_stats = []
    t0 = time.time()
    for i in range(num_runs + 1):
        if is_speculative:
            stats = run_speculative_pass(
                model, tokenized, tokens_to_gen, exit_layer, draft_length
            )
        else:
            stats = run_dense_pass(model, tokenized, tokens_to_gen, tokenizer)

        if i == 0:
            print(f"    Burn-in: {stats['tps']:.1f} tok/s [discarded]")
            continue

        run_stats.append(stats)
        extras = ""
        if "acceptance_rate" in stats:
            extras = f", accept={stats['acceptance_rate']:.1%}"
        print(f"    Run {i}: {stats['tps']:.1f} tok/s, {stats['wall_time_s']:.2f}s{extras}")

    tps_arr = np.array([s["tps"] for s in run_stats])
    lat_arr = np.array([s["wall_time_s"] for s in run_stats])
    result = {
        "mean_tps": float(np.mean(tps_arr)),
        "std_tps": float(np.std(tps_arr)),
        "mean_latency_s": float(np.mean(lat_arr)),
        "std_latency_s": float(np.std(lat_arr)),
    }
    if is_speculative:
        acc_arr = np.array([s["acceptance_rate"] for s in run_stats])
        macc_arr = np.array([s["mean_accepted_per_step"] for s in run_stats])
        result["mean_acceptance_rate"] = float(np.mean(acc_arr))
        result["mean_accepted_per_step"] = float(np.mean(macc_arr))

    result["benchmark_time_s"] = round(time.time() - t0, 1)
    print(f"  => {label}: {result['mean_tps']:.1f} ± {result['std_tps']:.1f} tok/s "
          f"({fmt_duration(result['benchmark_time_s'])})")
    return result


# ============================================================
# REPORT
# ============================================================
def save_report(output_dir, report_data):
    os.makedirs(output_dir, exist_ok=True)
    existing = os.listdir(output_dir)
    versions = [int(m.group(1)) for f in existing
                for m in [re.search(r'specbench_layerskip_v(\d+)\.yaml', f)] if m]
    version = max(versions) + 1 if versions else 1
    path = os.path.join(output_dir, f"specbench_layerskip_v{version}.yaml")
    with open(path, 'w') as f:
        yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)
    print(f"\n[Report] Saved to: {path}")
    return path


def benchmark_layerskip(checkpoint, cache_dir, tokens_to_gen, num_runs,
                         prompts_per_category, max_input_tokens,
                         exit_layers, draft_lengths):

    grid = [(el, k) for el in exit_layers for k in draft_lengths]
    ckpt_name = os.path.basename(os.path.dirname(checkpoint))

    print(f"\n{'#'*60}")
    print(f"  Spec-Bench LayerSkip Evaluation")
    print(f"  Checkpoint: {ckpt_name}")
    print(f"  Grid: exit_layers={exit_layers}  draft_lengths={draft_lengths}  "
          f"({len(grid)} settings × {num_runs} runs)")
    print(f"  Prompts: {prompts_per_category}/cat | Gen: {tokens_to_gen} tokens")
    print(f"{'#'*60}")

    timings = {}
    total_start = time.time()

    t0 = time.time()
    question_file = download_specbench(cache_dir)
    prompts = load_prompts(question_file, prompts_per_category)
    timings["data_loading_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    model, tokenizer, exp_cfg, experiment_root = load_layerskip_model(checkpoint)
    model.to("cuda")
    timings["model_loading_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    tokenized = tokenize_prompts(prompts, tokenizer, max_input_tokens)
    timings["tokenization_s"] = round(time.time() - t0, 1)

    # --- Dense baseline: run ONCE, reuse for all grid points ---
    print(f"\n{'='*60}")
    print(f"  Dense baseline (LlamaForCausalLM.generate, no speculative)")
    print(f"{'='*60}")
    dense_result = benchmark_setting(
        model, tokenizer, tokenized, tokens_to_gen,
        exit_layer=None, draft_length=None,
        num_runs=num_runs, is_speculative=False,
    )

    # --- Grid over (exit_layer, draft_length) ---
    all_results = []
    grid_start = time.time()
    for idx, (exit_layer, K) in enumerate(grid):
        s_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Setting {idx+1}/{len(grid)}: exit_layer={exit_layer}, draft_length={K}")
        print(f"{'='*60}")

        spec_result = benchmark_setting(
            model, tokenizer, tokenized, tokens_to_gen,
            exit_layer=exit_layer, draft_length=K,
            num_runs=num_runs, is_speculative=True,
        )

        speedup = spec_result["mean_tps"] / dense_result["mean_tps"] \
            if dense_result["mean_tps"] > 0 else 0.0
        print(f"\n  Speedup: {speedup:.3f}x ({(speedup-1)*100:+.1f}%)")

        setting_elapsed = time.time() - s_start
        remaining = (len(grid) - idx - 1) * (time.time() - grid_start) / (idx + 1)
        print(f"  Setting time: {fmt_duration(setting_elapsed)} | "
              f"ETA remaining: {fmt_duration(remaining)}")

        all_results.append({
            "setting": {"exit_layer": exit_layer, "draft_length": K},
            "speculative": spec_result,
            "baseline": dense_result,
            "speedup": float(speedup),
            "speedup_pct": f"{(speedup-1)*100:+.1f}%",
            "setting_time_s": round(setting_elapsed, 1),
        })

    # --- Summary ---
    total_elapsed = time.time() - total_start
    timings["grid_benchmark_s"] = round(time.time() - grid_start, 1)
    timings["total_s"] = round(total_elapsed, 1)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {ckpt_name}")
    print(f"  Dense baseline: {dense_result['mean_tps']:.1f} tok/s")
    print(f"  Total time: {fmt_duration(total_elapsed)}")
    print(f"{'='*60}")
    print(f"  {'exit':>5s}  {'K':>3s}  {'Spec TPS':>10s}  {'Accept':>8s}  "
          f"{'Speedup':>8s}  {'Time':>8s}")
    print(f"  {'-'*55}")
    for r in all_results:
        s = r["setting"]
        spec_tps = f"{r['speculative']['mean_tps']:.1f}"
        accept = f"{r['speculative'].get('mean_acceptance_rate', 0):.1%}"
        spd = f"{r['speedup']:.3f}x"
        t_s = fmt_duration(r.get("setting_time_s", 0))
        print(f"  {s['exit_layer']:>5d}  {s['draft_length']:>3d}  "
              f"{spec_tps:>10s}  {accept:>8s}  {spd:>8s}  {t_s:>8s}")

    best = max(all_results, key=lambda r: r["speedup"])
    print(f"\n  Best: exit_layer={best['setting']['exit_layer']}  "
          f"draft_length={best['setting']['draft_length']}  "
          f"({best['speedup']:.3f}x speedup)")

    # --- Build report ---
    cat_counts = {}
    for p in prompts:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1

    report = {
        "config": {
            "checkpoint": checkpoint,
            "checkpoint_name": ckpt_name,
            "tokens_per_generation": tokens_to_gen,
            "max_input_tokens": max_input_tokens,
            "num_runs_per_setting": num_runs,
            "num_prompts": len(prompts),
            "prompts_per_category": prompts_per_category,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "benchmark_source": "Spec-Bench (Xia et al., 2024)",
        },
        "model_config": {
            "num_layers": exp_cfg.get("model_overrides", {}).get(
                "num_hidden_layers", exp_cfg.get("num_hidden_layers", None)),
            "hidden_size": exp_cfg.get("model_overrides", {}).get(
                "hidden_size", exp_cfg.get("hidden_size", None)),
        },
        "grid": {
            "exit_layers": exit_layers,
            "draft_lengths": draft_lengths,
            "total_settings": len(grid),
        },
        "timings": timings,
        "prompt_distribution": cat_counts,
        "dense_baseline": dense_result,
        "results": all_results,
    }

    report_dir = os.path.join(checkpoint, "benchmarks")
    save_report(report_dir, report)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return report


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Spec-Bench evaluation for LayerSkip models")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a LayerSkip checkpoint directory")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--tokens", type=int, default=128,
                   help="Tokens to generate per prompt (default: 128)")
    p.add_argument("--runs", type=int, default=3,
                   help="Measurement runs per setting (default: 3)")
    p.add_argument("--prompts_per_category", type=int, default=20,
                   help="Max prompts per Spec-Bench category (default: 20)")
    p.add_argument("--max_input_tokens", type=int, default=256,
                   help="Truncate input prompts to this length (default: 256)")
    p.add_argument("--exit_layers", type=int, nargs="+", default=[6],
                   help="Exit layers to sweep (default: [6])")
    p.add_argument("--draft_lengths", type=int, nargs="+", default=[5],
                   help="Draft lengths to sweep (default: [5])")
    args = p.parse_args()

    benchmark_layerskip(
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        tokens_to_gen=args.tokens,
        num_runs=args.runs,
        prompts_per_category=args.prompts_per_category,
        max_input_tokens=args.max_input_tokens,
        exit_layers=args.exit_layers,
        draft_lengths=args.draft_lengths,
    )


if __name__ == "__main__":
    main()
