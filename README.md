# Nvium

<!-- Badges -->
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace Models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/YOUR_HF_ORG)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/YOUR_HF_DATASET)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Nvium: [Paper Title]**
> [Authors] — [Venue, Year]

N-vium is a decoder-only language model pretrained from scratch with the ability to exit computation early at inference time. The name comes from the Latin suffix *-vium*, meaning a meeting of ways.



This is a research repository. The models are trained at relatively small scales for experimentation purposes and are not intended for production use.

This repository contains:
- Full pretraining code for **Nvium** and two baselines (**CALM**, **LayerSkip**)
- Ablation scripts sweeping architecture dimensions (width, depth, number of exits, exit position, loss weighting)
- Speedup measurement scripts on [Spec-Bench](https://github.com/hemingkx/Spec-Bench) benchmark
- Standard LM evaluation via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Supervised fine-tuning (SFT) on [Tulu-3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)

---

## Repository Structure

```
.
├── assets/
│   ├── llama/            # base model config and tokenizer
│   └── dataset/          # dataset files (download separately, see below)
├── config/               # Hydra configs for pretraining (Nvium, CALM, LayerSkip)
├── config_experiments/   # per-experiment config overrides for ablations
├── evals/                # benchmark evaluation code (lm-eval harness)
├── performance/
│   ├── Nvium/            # Spec-Bench speedup measurement for Nvium
│   ├── calm/             # Spec-Bench speedup measurement for CALM
│   └── layerskip/        # Spec-Bench speedup measurement for LayerSkip
├── pretraining/
│   ├── Nvium/            # Nvium model, trainer, and training entry point
│   ├── calm/             # CALM model, trainer, and training entry point
│   └── layerskip/        # LayerSkip model, trainer, and training entry point
├── scripts/
│   ├── Nvium/            # SLURM scripts for Nvium pretraining and ablations
│   ├── calm/             # SLURM scripts for CALM pretraining and benchmarking
│   ├── layerskip/        # SLURM scripts for LayerSkip pretraining and benchmarking
│   ├── evals/            # SLURM script for LM benchmark evaluation
│   └── sft/              # SLURM script for supervised fine-tuning
├── sft/                  # supervised fine-tuning code
├── container_env.sh      # environment setup: paths, W&B, HF token
├── requirements.txt      # Python dependencies
└── requirements-eval.txt # additional dependencies for lm-eval
```

---

## Getting Started

### Hardware

Pretraining experiments were run on NVIDIA GH200 GPUs on a SLURM cluster. The default pretraining configuration uses 4 GPUs. Speedup measurements were run on a single NVIDIA A100 80GB GPU.

### Environment

The codebase runs inside an **NGC PyTorch container** (25.02) via SLURM's `--environment` directive. Set up your environment file (`.toml`) and configure `container_env.sh`:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_ORG/nvium.git
cd nvium

# 2. Edit container_env.sh and fill in:
#    - MODEL_NAME_OR_PATH           (path to base LLaMA config/tokenizer)
#    - DATASET                      (dataset name, default: c4-30B-scaling)
#    - Update --environment path in each sbatch script to your .toml location
#    - WANDB_API_KEY, WANDB_ENTITY  (optional, only needed for W&B logging)
#    - HF_TOKEN                     (optional, only needed for private HF datasets)
```

`container_env.sh` is sourced by every SLURM script and acts as the single configuration point for paths, credentials, and W&B settings.

### W&B Logging

W&B is set to `offline` by default (logs saved locally, no network required). To change the mode, set `WANDB_MODE` before submitting a job:

```bash
WANDB_MODE=online sbatch scripts/Nvium/Nvium_run_width_depth_scaling.sbatch  # stream to wandb.ai
WANDB_MODE=disabled sbatch ...                                                 # turn off entirely
# sync offline runs later:
wandb sync $SCRATCH/wandb/run-*
```

### Pretrained Checkpoints & Dataset

Pretrained checkpoints are available on Hugging Face: [huggingface.co/YOUR_HF_ORG](https://huggingface.co/YOUR_HF_ORG)

Dataset: [YOUR_HF_ORG/YOUR_DATASET](https://huggingface.co/datasets/YOUR_HF_ORG/YOUR_DATASET)

---

## Experiments

All experiments are launched via SLURM sbatch scripts in `scripts/`. Every script reads its configuration from `container_env.sh` and accepts overrides via environment variables at submission time. Results (checkpoints, benchmark YAMLs) are saved under `runs/` inside the repository.

### Pretraining

**Nvium** (`scripts/Nvium/`) trains a multi-exit model from scratch with our method. A matched dense **baseline** (mode `calm_baseline`) trains the same architecture with the standard loss.

**CALM** (`scripts/calm/run_calm.sbatch`) trains the [Confident Adaptive Language Modelling](https://arxiv.org/abs/2207.07061) method.

**LayerSkip** (`scripts/layerskip/run_layerskip.sbatch`) trains the [LayerSkip](https://arxiv.org/abs/2404.16710) method.

### Nvium ablations

All ablation scripts are array jobs — each array task trains one configuration from scratch.

| Script | What it sweeps |
|--------|---------------|
| `scripts/Nvium/Nvium_run_beta_ablation.sbatch` | Loss weight β for the router compute loss (13 values: 0–0.6) |
| `scripts/Nvium/Nvium_run_head_position_ablation.sbatch` | Position of the single early exit head across depth-24 (11 positions) |
| `scripts/Nvium/Nvium_run_number_heads_ablation.sbatch` | Number of exit heads: 2vium through 12vium (6 configs) |
| `scripts/Nvium/Nvium_run_width_depth_scaling.sbatch` | Full width × depth grid: 6 widths × 5 depths = 30 configs |

Each script supports a `MODEL_TYPE=baseline` flag to train the matched dense baseline instead.

### Speedup measurement

Generation speedup is measured on [Spec-Bench](https://github.com/hemingkx/Spec-Bench) (6 categories, 80 prompts each). Results are saved as versioned YAML files inside each checkpoint's `benchmarks/` directory.

| Script | Method | Required inputs |
|--------|--------|----------------|
| `scripts/Nvium/Nvium_run_specbench.sbatch` | Nvium | `FORWARD_PATH` (Nvium ckpt) + `BASELINE_PATH` (dense ckpt) |
| `scripts/calm/run_calm_specbench.sbatch` | CALM | `CALM_CHECKPOINT` |
| `scripts/layerskip/run_layerskip_specbench.sbatch` | LayerSkip | `CHECKPOINT` |

Use `LIMIT=5` for a quick smoke test (5 prompts per category).

### Standard LM Evaluations

`scripts/evals/run_evals.sbatch` runs a checkpoint on standard benchmarks (MMLU, ARC-Easy, HellaSwag, PIQA, OpenBookQA, WinoGrande) via lm-evaluation-harness. Results are saved to `{checkpoint}/benchmarks/results.json`. Pass `MODEL_DIR=/path/to/checkpoint`.

### Supervised Fine-Tuning

`scripts/sft/run_sft.sbatch` fine-tunes a pretrained checkpoint on the Tulu-3 SFT mixture using DDP. Pass `CHECKPOINT=/path/to/checkpoint`. The fine-tuned model is saved to `runs/sft/sft-{run_name}-{checkpoint}/`.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code or models in your research, please cite:

```bibtex
@article{nvium2025,
  title   = {[Paper Title]},
  author  = {[Authors]},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## Contributing

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/YOUR_ORG/nvium/issues). For questions about the research methodology, please refer to the paper. Pull requests that fix bugs or improve documentation are appreciated.
