#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%j.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%j.err
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

# ─── Usage ──────────────────────────────────────────────────────────────────
# Evaluate a single model checkpoint on standard benchmarks.
#
#   MODEL_DIR=/path/to/checkpoint sbatch scripts/evals/run_evals.sbatch
#
# Quick test (1 sample per benchmark):
#   LIMIT=1 MODEL_DIR=... sbatch scripts/evals/run_evals.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

# ─── Required checkpoint ────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-}"
if [ -z "$MODEL_DIR" ]; then
    echo "[ERROR] MODEL_DIR is not set. Pass it as: MODEL_DIR=/path/to/checkpoint sbatch ..."
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] Model directory not found: $MODEL_DIR"
    exit 1
fi

# ─── Config ─────────────────────────────────────────────────────────────────
BENCHMARKS="${BENCHMARKS:-mmlu arc_easy hellaswag piqa openbookqa winogrande}"
BATCH_SIZE="${BATCH_SIZE:-16}"

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

MODEL_NAME="$(basename $(dirname $MODEL_DIR))-$(basename $MODEL_DIR)"
echo "=== Benchmark Evaluation ==="
echo "  Model:      $MODEL_NAME"
echo "  Model dir:  $MODEL_DIR"
echo "  Benchmarks: $BENCHMARKS"
echo "  Batch size: $BATCH_SIZE"
echo "  Limit:      ${LIMIT:-full}"
echo ""

python3 evals/eval_benchmarks.py \
    --model_dir "$MODEL_DIR" \
    --benchmarks $BENCHMARKS \
    --batch_size $BATCH_SIZE \
    $LIMIT_ARG

echo "[$(date +%H:%M:%S)] Done: $MODEL_NAME"
