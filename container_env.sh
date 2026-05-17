#!/usr/bin/env bash
# container_env.sh - source this in sbatch scripts before running experiments

# ─── Paths ───────────────────────────────────────────────────────────────────
export SCRATCH=/capstor/scratch/cscs/$USER

# Persistent Python deps installed with: pip install --no-deps --target $PY_DEPS_DIR
export PY_DEPS_DIR="$SCRATCH/pydeps"
mkdir -p "$PY_DEPS_DIR"
export PYTHONPATH="$PY_DEPS_DIR:${PYTHONPATH:-}"

# ─── TLS certificates ────────────────────────────────────────────────────────
# Set if present (SUSE login node); unset otherwise (NGC container uses its own bundle)
if [ -f /etc/ssl/ca-bundle.pem ]; then
    export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem
    export REQUESTS_CA_BUNDLE=/etc/ssl/ca-bundle.pem
    export PIP_CERT=/etc/ssl/ca-bundle.pem
else
    unset SSL_CERT_FILE
    unset REQUESTS_CA_BUNDLE
    unset PIP_CERT
fi

# ─── Weights & Biases ────────────────────────────────────────────────────────
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT="${WANDB_PROJECT:-TEST}"
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p "$WANDB_DIR"
# WANDB_MODE controls where runs are logged:
#   offline  — save locally to $WANDB_DIR only; sync later with: wandb sync $SCRATCH/wandb/run-*
#   online   — stream directly to wandb.ai (requires WANDB_API_KEY and network access)
#   disabled — turn off wandb entirely (no overhead, useful for evals/benchmarks)
# Override per-job at submission time: WANDB_MODE=online sbatch scripts/calm/run_calm.sbatch
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB__SERVICE_WAIT=300

# ─── Distributed training ────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# ─── Dataset / model defaults ────────────────────────────────────────────────
export DATASET=c4-30B-scaling
export MODEL_NAME_OR_PATH="/users/$USER/project/github/assets/llama"

# ─── Hugging Face ────────────────────────────────────────────────────────────
export HF_TOKEN=YOUR_HF_TOKEN
export TOKENIZERS_PARALLELISM=false
export USE_LOCAL_CACHE=1
