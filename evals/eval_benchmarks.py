"""
Evaluate pretrained model checkpoints on standard benchmarks using lm-eval-harness.
Supports both baseline (BaselineLlamaModel) and Nvium (NviumModel) architectures.

Usage:
    python eval_benchmarks.py \
        --baseline_dir /path/to/baseline_160M_24L \
        --Nvium_dir /path/to/Nvium_160M_24L_exits_6_12_18_24 \
        --benchmarks mmlu arc_easy hellaswag piqa openbookqa winogrande \
        --limit 1000 \
        --batch_size 16
"""

import os

# ── Cache setup (MUST be before any HF/lm_eval imports) ──────────────────────
SCRATCH = os.environ.get("SCRATCH")
if not SCRATCH:
    raise RuntimeError("SCRATCH environment variable is not set")
CACHE_DIR = os.path.join(SCRATCH, "cache/evals")
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "hf_home")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "hf_datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "hf_models")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hf_hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hf_hub")
os.environ["XDG_CACHE_HOME"] = os.path.join(CACHE_DIR, "xdg_cache")
os.environ["XDG_DATA_HOME"] = os.path.join(CACHE_DIR, "xdg_data")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lm_eval
from transformers import AutoModelForCausalLM, AutoConfig

# Add model source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pretraining", "Nvium"))
from Nvium_model import NviumModel
from Nvium_baseline import BaselineLlamaModel


# ── Metric keys per benchmark ────────────────────────────────────────────────
BENCHMARK_METRIC_KEYS = {
    "arc_easy":     ["acc,none", "acc_norm,none"],
    "hellaswag":    ["acc,none", "acc_norm,none"],
    "mmlu":         ["acc,none"],
    "openbookqa":   ["acc,none", "acc_norm,none"],
    "piqa":         ["acc,none", "acc_norm,none"],
    "winogrande":   ["acc,none"],
}

# Primary metric to use for plotting (per benchmark)
PRIMARY_METRIC = {
    "arc_easy":     "acc_norm,none",
    "hellaswag":    "acc_norm,none",
    "mmlu":         "acc,none",
    "openbookqa":   "acc_norm,none",
    "piqa":         "acc_norm,none",
    "winogrande":   "acc,none",
}


def log(label: str, msg: str):
    print(f"[{label}] {msg}", flush=True)


# ── Checkpoint discovery ─────────────────────────────────────────────────────
def get_sorted_checkpoints(experiment_dir: str, only_step: Optional[int] = None) -> List[str]:
    """Find all checkpoint-* dirs sorted by step number. Optionally filter to a single step.
    Supports both pretraining layout ({dir}/checkpoints/checkpoint-*) and
    FT layout ({dir}/checkpoint-*) automatically."""
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        # FT layout: checkpoints are directly in experiment_dir
        ckpt_dir = experiment_dir

    ckpts = []
    for name in os.listdir(ckpt_dir):
        m = re.match(r"checkpoint-(\d+)", name)
        if m:
            step = int(m.group(1))
            if only_step is None or step == only_step:
                ckpts.append((step, os.path.join(ckpt_dir, name)))

    ckpts.sort(key=lambda x: x[0])
    return ckpts  # list of (step, path)


# ── Load hydra exp_config ────────────────────────────────────────────────────
def load_exp_config(experiment_dir: str, checkpoint_path: str = None) -> dict:
    """Load experiment config. Tries in order:
    1. {experiment_dir}/.hydra/config.yaml  (pretraining layout)
    2. {experiment_dir}/../../.hydra/config.yaml  (checkpoint subdir layout)
    3. {experiment_dir}/ft_exp_config.yaml  (FT final/ dir)
    4. {checkpoint_path}/ft_exp_config.yaml  (FT layout, per-checkpoint)
    """
    cfg_path = os.path.join(experiment_dir, ".hydra", "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f)

    # checkpoint subdir layout: experiment_dir is .../checkpoints/checkpoint-N
    parent_cfg_path = os.path.join(experiment_dir, "..", "..", ".hydra", "config.yaml")
    parent_cfg_path = os.path.normpath(parent_cfg_path)
    if os.path.exists(parent_cfg_path):
        log("INFO", f"Loading config from {parent_cfg_path}")
        with open(parent_cfg_path) as f:
            return yaml.safe_load(f)

    ft_cfg_path = os.path.join(experiment_dir, "ft_exp_config.yaml")
    if os.path.exists(ft_cfg_path):
        log("INFO", f"Loading config from {ft_cfg_path}")
        with open(ft_cfg_path) as f:
            return yaml.safe_load(f)

    if checkpoint_path is not None:
        ft_cfg_path2 = os.path.join(checkpoint_path, "ft_exp_config.yaml")
        if os.path.exists(ft_cfg_path2):
            log("INFO", f"Loading config from {ft_cfg_path2}")
            with open(ft_cfg_path2) as f:
                return yaml.safe_load(f)

    log("WARN", f"No config found for {experiment_dir}, using empty config")
    return {}


# ── Model loading via monkey-patch ───────────────────────────────────────────
def make_custom_loader(exp_cfg: dict):
    """Create a from_pretrained replacement that loads our custom model classes."""
    model_type = exp_cfg.get("model_type", "baseline")
    model_cache = {}

    def custom_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("decode_head", None)

        key = pretrained_model_name_or_path
        if key in model_cache:
            log("EVAL", f"Reusing cached model for {key}")
            return model_cache[key]

        log("EVAL", f"Loading {model_type} model from {key}")
        hf_config = AutoConfig.from_pretrained(key, trust_remote_code=True)

        if model_type == "Nvium":
            ModelClass = NviumModel
        else:
            ModelClass = BaselineLlamaModel

        # For loading from checkpoint, we don't want random init
        load_cfg = dict(exp_cfg)
        load_cfg["random_init_all"] = False
        if model_type == "Nvium":
            load_cfg["decode_head"] = "mix"

        model = ModelClass.from_pretrained(
            key,
            config=hf_config,
            exp_config=load_cfg,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            ignore_mismatched_sizes=True,
            **{k: v for k, v in kwargs.items() if k not in ("config", "exp_config", "torch_dtype", "local_files_only", "ignore_mismatched_sizes")},
        )

        # Re-apply tying after weight load
        if hasattr(model, "_apply_tying"):
            tie_cfg = load_cfg.get("tying", {})
            model._apply_tying(tie_cfg)

        model_cache[key] = model
        return model

    custom_from_pretrained.model_cache = model_cache
    return custom_from_pretrained


# ── Run lm-eval for a single checkpoint ──────────────────────────────────────
def evaluate_checkpoint(
    checkpoint_path: str,
    exp_cfg: dict,
    benchmarks: List[str],
    batch_size: int,
    limit: Optional[int],
    output_json: str,
) -> Dict[str, Any]:
    """Run lm-eval on one checkpoint, save results to JSON, return metrics dict."""

    # Check if results already exist
    if os.path.exists(output_json):
        log("EVAL", f"Found cached results: {output_json}")
        with open(output_json) as f:
            return json.load(f)

    log("EVAL", f"Evaluating {checkpoint_path} on {benchmarks}")

    # Monkey-patch AutoModelForCausalLM.from_pretrained
    loader = make_custom_loader(exp_cfg)
    orig = AutoModelForCausalLM.from_pretrained
    AutoModelForCausalLM.from_pretrained = loader

    try:
        eval_kwargs = dict(
            model="hf",
            model_args=f"pretrained={checkpoint_path},trust_remote_code=True",
            tasks=benchmarks,
            batch_size=batch_size,
            log_samples=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if limit is not None:
            eval_kwargs["limit"] = limit

        results = lm_eval.simple_evaluate(**eval_kwargs)
    finally:
        AutoModelForCausalLM.from_pretrained = orig
        # Free GPU memory
        for m in loader.model_cache.values():
            del m
        loader.model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Extract metrics
    metrics = {}
    for benchmark in benchmarks:
        task_results = (results.get("results", {}) or {}).get(benchmark, {})
        benchmark_metrics = {}
        keys = BENCHMARK_METRIC_KEYS.get(benchmark, ["acc,none"])
        for k in keys:
            if k in task_results:
                benchmark_metrics[k] = float(task_results[k])
        metrics[benchmark] = benchmark_metrics

    # Build output
    output = {
        "checkpoint_path": checkpoint_path,
        "step": int(re.search(r"checkpoint-(\d+)", checkpoint_path).group(1)),
        "benchmarks": benchmarks,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    log("EVAL", f"Saved results to {output_json}")
    for bm, md in metrics.items():
        parts = [f"{k.split(',')[0]}={v:.4f}" for k, v in md.items()]
        log("EVAL", f"  {bm}: {', '.join(parts)}")

    return output


# ── Evaluate all checkpoints for one experiment ──────────────────────────────
def evaluate_experiment(
    experiment_dir: str,
    benchmarks: List[str],
    batch_size: int,
    limit: Optional[int],
    output_base: str,
    label: str,
    only_step: Optional[int] = None,
) -> List[Dict]:
    """Evaluate all checkpoints in an experiment. Returns list of result dicts."""
    checkpoints = get_sorted_checkpoints(experiment_dir, only_step=only_step)
    first_ckpt = checkpoints[0][1] if checkpoints else None
    exp_cfg = load_exp_config(experiment_dir, checkpoint_path=first_ckpt)
    log("EVAL", f"[{label}] Found {len(checkpoints)} checkpoints in {experiment_dir}")

    all_results = []
    for step, ckpt_path in checkpoints:
        json_path = os.path.join(output_base, f"{label}_step_{step}.json")
        result = evaluate_checkpoint(
            ckpt_path, exp_cfg, benchmarks, batch_size, limit, json_path
        )
        all_results.append(result)

    return all_results


# ── Plotting ─────────────────────────────────────────────────────────────────
def make_plots(
    baseline_results: List[Dict],
    Nvium_results: List[Dict],
    benchmarks: List[str],
    output_dir: str,
    experiment_name: str,
):
    """Generate per-benchmark plots comparing baseline vs Nvium over training."""
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:
        metric_key = PRIMARY_METRIC.get(benchmark, "acc,none")
        metric_label = metric_key.replace(",none", "").replace(",", "_")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Baseline
        bl_steps = []
        bl_scores = []
        for r in baseline_results:
            score = r.get("metrics", {}).get(benchmark, {}).get(metric_key)
            if score is not None:
                bl_steps.append(r["step"])
                bl_scores.append(score * 100)

        # Multihead
        mh_steps = []
        mh_scores = []
        for r in Nvium_results:
            score = r.get("metrics", {}).get(benchmark, {}).get(metric_key)
            if score is not None:
                mh_steps.append(r["step"])
                mh_scores.append(score * 100)

        if bl_steps:
            ax.plot(bl_steps, bl_scores, "o-", label="Baseline", color="#2196F3", markersize=5, linewidth=1.5)
        if mh_steps:
            ax.plot(mh_steps, mh_scores, "s-", label="Multihead", color="#F44336", markersize=5, linewidth=1.5)

        # Random baseline reference
        random_baseline = {
            "arc_easy": 25.0, "hellaswag": 25.0, "mmlu": 25.0,
            "openbookqa": 25.0, "piqa": 50.0, "winogrande": 50.0,
        }
        if benchmark in random_baseline:
            ax.axhline(y=random_baseline[benchmark], color="gray", linestyle="--", alpha=0.5, label="Random")

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel(f"{metric_label} (%)", fontsize=12)
        ax.set_title(f"{benchmark.upper()} — {experiment_name}", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

        plot_path = os.path.join(output_dir, f"{benchmark}_{metric_label}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("PLOT", f"Saved {plot_path}")

    # Combined plot with all benchmarks in subplots
    n_benchmarks = len(benchmarks)
    ncols = min(3, n_benchmarks)
    nrows = (n_benchmarks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if n_benchmarks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, benchmark in enumerate(benchmarks):
        ax = axes[i]
        metric_key = PRIMARY_METRIC.get(benchmark, "acc,none")
        metric_label = metric_key.replace(",none", "").replace(",", "_")

        bl_steps, bl_scores = [], []
        for r in baseline_results:
            score = r.get("metrics", {}).get(benchmark, {}).get(metric_key)
            if score is not None:
                bl_steps.append(r["step"])
                bl_scores.append(score * 100)

        mh_steps, mh_scores = [], []
        for r in Nvium_results:
            score = r.get("metrics", {}).get(benchmark, {}).get(metric_key)
            if score is not None:
                mh_steps.append(r["step"])
                mh_scores.append(score * 100)

        if bl_steps:
            ax.plot(bl_steps, bl_scores, "o-", label="Baseline", color="#2196F3", markersize=4, linewidth=1.2)
        if mh_steps:
            ax.plot(mh_steps, mh_scores, "s-", label="Multihead", color="#F44336", markersize=4, linewidth=1.2)

        random_baseline = {
            "arc_easy": 25.0, "hellaswag": 25.0, "mmlu": 25.0,
            "openbookqa": 25.0, "piqa": 50.0, "winogrande": 50.0,
        }
        if benchmark in random_baseline:
            ax.axhline(y=random_baseline[benchmark], color="gray", linestyle="--", alpha=0.5, label="Random")

        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel(f"{metric_label} (%)", fontsize=10)
        ax.set_title(benchmark.upper(), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Benchmark Comparison — {experiment_name}", fontsize=14, y=1.02)
    fig.tight_layout()
    combined_path = os.path.join(output_dir, "all_benchmarks_combined.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("PLOT", f"Saved combined plot: {combined_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def evaluate_single_model(
    model_dir: str,
    benchmarks: List[str],
    batch_size: int,
    limit: Optional[int],
    output_dir: str,
) -> Dict[str, Any]:
    """Evaluate a single model directory (no checkpoint subdirs — the dir itself contains the weights)."""
    exp_cfg = load_exp_config(model_dir)
    model_name = os.path.basename(os.path.normpath(model_dir))

    log("EVAL", f"Evaluating single model: {model_dir}")

    # Monkey-patch AutoModelForCausalLM.from_pretrained
    loader = make_custom_loader(exp_cfg)
    orig = AutoModelForCausalLM.from_pretrained
    AutoModelForCausalLM.from_pretrained = loader

    try:
        eval_kwargs = dict(
            model="hf",
            model_args=f"pretrained={model_dir},trust_remote_code=True",
            tasks=benchmarks,
            batch_size=batch_size,
            log_samples=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if limit is not None:
            eval_kwargs["limit"] = limit

        results = lm_eval.simple_evaluate(**eval_kwargs)
    finally:
        AutoModelForCausalLM.from_pretrained = orig
        for m in loader.model_cache.values():
            del m
        loader.model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Extract metrics
    metrics = {}
    for benchmark in benchmarks:
        task_results = (results.get("results", {}) or {}).get(benchmark, {})
        benchmark_metrics = {}
        keys = BENCHMARK_METRIC_KEYS.get(benchmark, ["acc,none"])
        for k in keys:
            if k in task_results:
                benchmark_metrics[k] = float(task_results[k])
        metrics[benchmark] = benchmark_metrics

    output = {
        "model_dir": model_dir,
        "model_name": model_name,
        "model_type": exp_cfg.get("model_type", "unknown"),
        "benchmarks": benchmarks,
        "metrics": metrics,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, "results.json")
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    log("EVAL", f"Saved results to {output_json}")
    for bm, md in metrics.items():
        parts = [f"{k.split(',')[0]}={v:.4f}" for k, v in md.items()]
        log("EVAL", f"  {bm}: {', '.join(parts)}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints on benchmarks")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to a single model directory (evaluates just this model)")
    parser.add_argument("--baseline_dir", type=str, default=None,
                        help="Path to baseline experiment directory")
    parser.add_argument("--Nvium_dir", type=str, default=None,
                        help="Path to Nvium experiment directory")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["mmlu", "arc_easy", "hellaswag", "piqa", "openbookqa", "winogrande"],
                        help="Benchmarks to evaluate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per benchmark (None = full)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: model_dir/benchmarks/)")
    parser.add_argument("--checkpoint_step", type=int, default=None,
                        help="Only evaluate this checkpoint step (default: all checkpoints)")
    args = parser.parse_args()

    # ── Single model mode ──────────────────────────────────────────────────
    if args.model_dir is not None:
        if args.output_dir is None:
            output_dir = os.path.join(args.model_dir, "benchmarks")
        else:
            output_dir = args.output_dir

        log("MAIN", f"Model:      {args.model_dir}")
        log("MAIN", f"Benchmarks: {args.benchmarks}")
        log("MAIN", f"Limit:      {args.limit}")
        log("MAIN", f"Output:     {output_dir}")

        evaluate_single_model(
            args.model_dir, args.benchmarks, args.batch_size, args.limit, output_dir,
        )
        log("MAIN", "Done!")
        return

    # ── Paired baseline+Nvium mode ─────────────────────────────────────
    if not args.baseline_dir or not args.Nvium_dir:
        parser.error("Provide either --model_dir or both --baseline_dir and --Nvium_dir")

    # Derive experiment name from the Nvium dir
    experiment_name = os.path.basename(os.path.normpath(args.Nvium_dir))

    # Output goes under Nvium_dir/evals/ by default
    if args.output_dir is None:
        output_dir = os.path.join(args.Nvium_dir, "evals")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    json_dir = os.path.join(output_dir, "json_results")
    plot_dir = os.path.join(output_dir, "plots")

    log("MAIN", f"Baseline:  {args.baseline_dir}")
    log("MAIN", f"Multihead: {args.Nvium_dir}")
    log("MAIN", f"Benchmarks: {args.benchmarks}")
    log("MAIN", f"Limit: {args.limit}")
    log("MAIN", f"Output: {output_dir}")

    # Evaluate baseline
    baseline_results = evaluate_experiment(
        args.baseline_dir, args.benchmarks, args.batch_size, args.limit,
        os.path.join(json_dir, "baseline"), label="baseline",
        only_step=args.checkpoint_step,
    )

    # Evaluate Nvium
    Nvium_results = evaluate_experiment(
        args.Nvium_dir, args.benchmarks, args.batch_size, args.limit,
        os.path.join(json_dir, "Nvium"), label="Nvium",
        only_step=args.checkpoint_step,
    )

    # Save combined summary
    summary = {
        "experiment_name": experiment_name,
        "baseline_dir": args.baseline_dir,
        "Nvium_dir": args.Nvium_dir,
        "benchmarks": args.benchmarks,
        "limit": args.limit,
        "baseline": baseline_results,
        "Nvium": Nvium_results,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log("MAIN", f"Saved summary to {summary_path}")

    # Generate plots
    make_plots(baseline_results, Nvium_results, args.benchmarks, plot_dir, experiment_name)

    log("MAIN", "Done!")


if __name__ == "__main__":
    main()
