#!/usr/bin/env python3
"""
Spec-Bench speedup evaluation for CALM adaptive early-exit decoding.

Loads ONE CALM checkpoint, sweeps a set of confidence thresholds, and
reports walltime speedup vs dense generation using the same model. The
dense baseline is the underlying LlamaForCausalLM.generate() (no early
exit), bypassing the inference wrapper via MRO.

Usage:
    python3.12 measure_specbench_calm.py \
        --checkpoint /path/to/outputs_calm/.../checkpoints/checkpoint-4321 \
        --thresholds 0.5 0.7 0.9
"""

import os
import re
import gc
import sys
import json
import glob
import time
import random
import argparse
import subprocess

import yaml
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pretraining", "calm"))

from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from safetensors.torch import load_file

from calm_model import CalmModel
from model_inference_calm import get_calm_inference_class

# ============================================================
# CONFIG
# ============================================================
TORCH_DTYPE = torch.bfloat16
SPECBENCH_REPO = "https://github.com/hemingkx/Spec-Bench.git"
DEFAULT_CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "."), "cache", "SpecDec")


def fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


# ============================================================
# SPEC-BENCH DATA
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
def load_calm_model(checkpoint_path, cache_dir, model_name_or_path):
    """
    Load the CALM model with the adaptive-exit inference wrapper.
    The same instance serves both roles:
      - adaptive:  model.generate_with_stats(...)
      - dense:     LlamaForCausalLM.generate(model, ...)   (MRO bypass)
    """
    # Try to find .hydra/config.yaml by walking up from the checkpoint dir.
    experiment_root = os.path.dirname(checkpoint_path)
    config_path = os.path.join(experiment_root, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        experiment_root = os.path.dirname(experiment_root)
        config_path = os.path.join(experiment_root, ".hydra", "config.yaml")
    exp_cfg = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            exp_cfg = yaml.safe_load(f)
        print(f"[CONFIG] {config_path}")

    if exp_cfg is None:
        # Fallback — minimal config sufficient for CalmModel.__init__
        print(f"[CONFIG] No .hydra/config.yaml found; using minimal fallback")
        exp_cfg = {
            "model_name_or_path": model_name_or_path,
            "early_layer_idx": [4, 8, 12, 16, 20],
            "calm": {
                "threshold": 0.9,
                "confidence_measure": "max_prob",
                "per_exit_norms": True,
                "loss_weighting": "linear",
            },
            "load_args": {"attn_implementation": "sdpa", "bf16": True},
        }

    # HF lookups need the main project cache, not the SpecBench subdir.
    hf_cache_dir = os.path.join(os.environ.get("SCRATCH", "."), "cache")
    fallback_name = exp_cfg.get("model_name_or_path", model_name_or_path)

    # Tokenizer: prefer the checkpoint dir, fall back to the HF hub ID.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, cache_dir=hf_cache_dir, local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            fallback_name, cache_dir=hf_cache_dir, trust_remote_code=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config: same order — checkpoint dir first, then HF hub fallback.
    try:
        hf_config = AutoConfig.from_pretrained(
            checkpoint_path,
            cache_dir=hf_cache_dir, trust_remote_code=True, local_files_only=True,
        )
    except Exception:
        hf_config = AutoConfig.from_pretrained(
            fallback_name, cache_dir=hf_cache_dir, trust_remote_code=True,
        )
        overrides = exp_cfg.get("model_overrides", {})
        if overrides:
            hf_config.update(overrides)

    attn_impl = exp_cfg.get("load_args", {}).get("attn_implementation", "sdpa")

    InferenceClass = get_calm_inference_class(CalmModel)
    model = InferenceClass(
        hf_config,
        exp_config=exp_cfg,
        attn_implementation=attn_impl,
    )
    model = model.to(dtype=TORCH_DTYPE)

    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
    if not ckpt_files:
        raise FileNotFoundError(f"No .safetensors at {checkpoint_path}")
    state_dict = {}
    for f in ckpt_files:
        state_dict.update(load_file(f))
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    calm_missing = [k for k in missing if "exit_norms" not in k]
    if calm_missing:
        print(f"[WARN] Missing backbone keys: {calm_missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    print(f"[LOAD] Weights from {checkpoint_path}")

    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, exp_cfg, experiment_root


# ============================================================
# BENCHMARKING
# ============================================================
def run_adaptive_pass(model, tokenized_prompts, tokens_to_gen, threshold,
                       do_sample=False, temperature=1.0):
    """One full pass in adaptive-exit mode across all prompts. Returns stats."""
    total_tokens = 0
    all_exits = []
    n_early = 0

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for item in tokenized_prompts:
            out, hist = model.generate_with_stats(
                item["input_ids"],
                max_new_tokens=tokens_to_gen,
                threshold=threshold,
                do_sample=do_sample,
                temperature=temperature,
                eos_token_id=-1,  # force exact length for timing fairness
            )
            total_tokens += out.shape[1] - item["input_ids"].shape[1]
            all_exits.extend(hist["exits"])
            n_early += hist["n_early_exits"]
    end.record()
    torch.cuda.synchronize()

    wall = start.elapsed_time(end) / 1000.0
    tps = total_tokens / wall if wall > 0 else 0.0
    avg_exit = float(np.mean(all_exits)) if all_exits else 0.0
    early_frac = n_early / len(all_exits) if all_exits else 0.0

    return {
        "wall_time_s": wall,
        "total_tokens": total_tokens,
        "tps": tps,
        "avg_exit_layer": avg_exit,
        "early_exit_frac": early_frac,
    }


def run_dense_pass(model, tokenized_prompts, tokens_to_gen, tokenizer,
                    do_sample=False, temperature=1.0):
    """
    One full pass in dense mode (no adaptive exit). Uses the inference
    wrapper's own `generate_dense` bare-loop so that prompt handling, KV
    cache management, and tokenize/append overhead are IDENTICAL to
    generate_with_stats. Only the per-token forward differs (full model
    vs early-exit + state-propagation), giving an apples-to-apples
    speedup measurement.
    """
    total_tokens = 0
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for item in tokenized_prompts:
            out = model.generate_dense(
                item["input_ids"],
                max_new_tokens=tokens_to_gen,
                do_sample=do_sample,
                temperature=temperature,
                eos_token_id=-1,
            )
            total_tokens += out.shape[1] - item["input_ids"].shape[1]
    end.record()
    torch.cuda.synchronize()

    wall = start.elapsed_time(end) / 1000.0
    tps = total_tokens / wall if wall > 0 else 0.0
    return {"wall_time_s": wall, "total_tokens": total_tokens, "tps": tps}


def benchmark_setting(model, tokenizer, tokenized, tokens_to_gen,
                      threshold, num_runs, is_adaptive,
                      do_sample=False, temperature=1.0):
    label = f"Adaptive(τ={threshold:.2f})" if is_adaptive else "Dense"
    print(f"\n  [{label}] {num_runs} runs (+1 burn-in)")

    run_stats = []
    t0 = time.time()
    for i in range(num_runs + 1):
        if is_adaptive:
            stats = run_adaptive_pass(
                model, tokenized, tokens_to_gen, threshold,
                do_sample=do_sample, temperature=temperature,
            )
        else:
            stats = run_dense_pass(
                model, tokenized, tokens_to_gen, tokenizer,
                do_sample=do_sample, temperature=temperature,
            )

        if i == 0:
            print(f"    Burn-in: {stats['tps']:.1f} tok/s [discarded]")
            continue

        run_stats.append(stats)
        extras = ""
        if "avg_exit_layer" in stats:
            extras = (f", avg_exit={stats['avg_exit_layer']:.1f}"
                      f", early_frac={stats['early_exit_frac']:.3f}")
        print(f"    Run {i}: {stats['tps']:.1f} tok/s, "
              f"{stats['wall_time_s']:.2f}s{extras}")

    tps_arr = np.array([s["tps"] for s in run_stats])
    lat_arr = np.array([s["wall_time_s"] for s in run_stats])
    result = {
        "mean_tps": float(np.mean(tps_arr)),
        "std_tps": float(np.std(tps_arr)),
        "mean_latency_s": float(np.mean(lat_arr)),
        "std_latency_s": float(np.std(lat_arr)),
    }
    if is_adaptive:
        avg_arr = np.array([s["avg_exit_layer"] for s in run_stats])
        ef_arr = np.array([s["early_exit_frac"] for s in run_stats])
        result["mean_avg_exit_layer"] = float(np.mean(avg_arr))
        result["mean_early_exit_frac"] = float(np.mean(ef_arr))

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
    versions = [int(m.group(1)) for fn in existing
                for m in [re.search(r'specbench_calm_v(\d+)\.yaml', fn)] if m]
    version = max(versions) + 1 if versions else 1
    path = os.path.join(output_dir, f"specbench_calm_v{version}.yaml")
    with open(path, 'w') as f:
        yaml.dump(report_data, f, default_flow_style=False, sort_keys=False)
    print(f"\n[Report] Saved to: {path}")
    return path


def benchmark_calm(checkpoint, cache_dir, tokens_to_gen, num_runs,
                   prompts_per_category, max_input_tokens,
                   thresholds, model_name_or_path,
                   do_sample=False, temperature=1.0, seed=42):

    ckpt_name = os.path.basename(os.path.dirname(checkpoint))

    mode = f"sample(T={temperature:.2f})" if do_sample else "greedy"
    print(f"\n{'#'*60}")
    print(f"  Spec-Bench CALM Evaluation")
    print(f"  Checkpoint: {ckpt_name}")
    print(f"  Thresholds: {thresholds}  ({len(thresholds)} settings × {num_runs} runs)")
    print(f"  Prompts: {prompts_per_category}/cat | Gen: {tokens_to_gen} tokens | Decoding: {mode}")
    print(f"{'#'*60}")

    timings = {}
    total_start = time.time()

    t0 = time.time()
    question_file = download_specbench(cache_dir)
    prompts = load_prompts(question_file, prompts_per_category)
    timings["data_loading_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    model, tokenizer, exp_cfg, experiment_root = load_calm_model(
        checkpoint, cache_dir, model_name_or_path,
    )
    model.to("cuda")
    timings["model_loading_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    tokenized = tokenize_prompts(prompts, tokenizer, max_input_tokens)
    timings["tokenization_s"] = round(time.time() - t0, 1)

    # --- Dense baseline: run ONCE, reuse for all thresholds ---
    print(f"\n{'='*60}")
    print(f"  Dense baseline (bare-loop, no early exit)")
    print(f"{'='*60}")
    torch.manual_seed(seed)
    dense_result = benchmark_setting(
        model, tokenizer, tokenized, tokens_to_gen,
        threshold=None, num_runs=num_runs, is_adaptive=False,
        do_sample=do_sample, temperature=temperature,
    )

    # --- Grid over thresholds ---
    all_results = []
    grid_start = time.time()
    for idx, tau in enumerate(thresholds):
        s_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Setting {idx+1}/{len(thresholds)}: threshold={tau:.2f}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        ada_result = benchmark_setting(
            model, tokenizer, tokenized, tokens_to_gen,
            threshold=tau, num_runs=num_runs, is_adaptive=True,
            do_sample=do_sample, temperature=temperature,
        )

        speedup = (ada_result["mean_tps"] / dense_result["mean_tps"]
                   if dense_result["mean_tps"] > 0 else 0.0)
        print(f"\n  Speedup: {speedup:.3f}x ({(speedup-1)*100:+.1f}%)")

        setting_elapsed = time.time() - s_start
        remaining = (len(thresholds) - idx - 1) * (time.time() - grid_start) / (idx + 1)
        print(f"  Setting time: {fmt_duration(setting_elapsed)} | "
              f"ETA remaining: {fmt_duration(remaining)}")

        all_results.append({
            "setting": {"threshold": float(tau)},
            "adaptive": ada_result,
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
    print(f"  {'tau':>5s}  {'TPS':>9s}  {'avg_exit':>9s}  {'early_frac':>11s}  "
          f"{'Speedup':>8s}  {'Time':>8s}")
    print(f"  {'-'*60}")
    for r in all_results:
        tau = r["setting"]["threshold"]
        tps = f"{r['adaptive']['mean_tps']:.1f}"
        avg_e = f"{r['adaptive'].get('mean_avg_exit_layer', 0):.1f}"
        ef = f"{r['adaptive'].get('mean_early_exit_frac', 0):.3f}"
        spd = f"{r['speedup']:.3f}x"
        t_s = fmt_duration(r.get("setting_time_s", 0))
        print(f"  {tau:>5.2f}  {tps:>9s}  {avg_e:>9s}  {ef:>11s}  "
              f"{spd:>8s}  {t_s:>8s}")

    best = max(all_results, key=lambda r: r["speedup"])
    print(f"\n  Best: threshold={best['setting']['threshold']:.2f}  "
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
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "seed": int(seed),
        },
        "model_config": {
            "num_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "exit_layers": list(model.exit_layer_indices),
            "per_exit_norms": bool(getattr(model, "per_exit_norms_enabled", False)),
        },
        "grid": {
            "thresholds": [float(t) for t in thresholds],
            "total_settings": len(thresholds),
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
    p = argparse.ArgumentParser(description="Spec-Bench evaluation for CALM models")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a CALM checkpoint directory")
    p.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--model_name_or_path", type=str, default="JackFram/llama-160m",
                   help="HF model ID for tokenizer fallback")
    p.add_argument("--tokens", type=int, default=128,
                   help="Tokens to generate per prompt (default: 128)")
    p.add_argument("--runs", type=int, default=3,
                   help="Measurement runs per setting (default: 3)")
    p.add_argument("--prompts_per_category", type=int, default=20,
                   help="Max prompts per Spec-Bench category (default: 20)")
    p.add_argument("--max_input_tokens", type=int, default=896,
                   help="Truncate input prompts to this length (default: 896 = 1024 - 128)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3],
                   help="Confidence thresholds to sweep (default: [0.3])")
    p.add_argument("--do_sample", action="store_true",
                   help="Sample from the softmax distribution instead of greedy.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (only used when --do_sample).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed — reset before dense and each adaptive run "
                        "so sampling is reproducible across settings.")
    args = p.parse_args()

    benchmark_calm(
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        tokens_to_gen=args.tokens,
        num_runs=args.runs,
        prompts_per_category=args.prompts_per_category,
        max_input_tokens=args.max_input_tokens,
        thresholds=args.thresholds,
        model_name_or_path=args.model_name_or_path,
        do_sample=args.do_sample,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
