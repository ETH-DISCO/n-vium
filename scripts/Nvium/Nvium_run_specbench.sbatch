#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=specbench-Nvium
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%j.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%j.err
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

# ─── Usage ──────────────────────────────────────────────────────────────────
#   FORWARD_PATH=/path/to/checkpoint BASELINE_PATH=/path/to/checkpoint \
#     sbatch scripts/Nvium/run_Nvium_specdec.sbatch
#
# Quick test (5 prompts per category):
#   LIMIT=5 FORWARD_PATH=... BASELINE_PATH=... sbatch ...
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

# ─── Required checkpoints ───────────────────────────────────────────────────
FORWARD_PATH="${FORWARD_PATH:-}"
BASELINE_PATH="${BASELINE_PATH:-}"

if [ -z "$FORWARD_PATH" ]; then
    echo "[ERROR] FORWARD_PATH is not set."
    exit 1
fi
if [ -z "$BASELINE_PATH" ]; then
    echo "[ERROR] BASELINE_PATH is not set."
    exit 1
fi
if [ ! -d "$FORWARD_PATH" ]; then
    echo "[ERROR] Forward checkpoint not found: $FORWARD_PATH"
    exit 1
fi
if [ ! -d "$BASELINE_PATH" ]; then
    echo "[ERROR] Baseline checkpoint not found: $BASELINE_PATH"
    exit 1
fi

# ─── Benchmark config ───────────────────────────────────────────────────────
TOKENS="${TOKENS:-128}"
RUNS="${RUNS:-3}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-896}"    # = 1024 ctx - 128 gen
HEAD_TEMPS="${HEAD_TEMPS:-1.0}"
ROUTER_TEMPS="${ROUTER_TEMPS:-1.0}"

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

PAIR_NAME=$(basename "$FORWARD_PATH")

echo "=== Spec-Bench Nvium Evaluation ==="
echo "  Pair:     $PAIR_NAME"
echo "  Forward:  $FORWARD_PATH"
echo "  Baseline: $BASELINE_PATH"
echo "  Tokens:   $TOKENS, Runs: $RUNS"
echo "  Limit:    ${LIMIT:-full}"
echo ""

python3 performance/Nvium/measure_specbench_Nvium.py \
    --forward_path "$FORWARD_PATH" \
    --baseline_path "$BASELINE_PATH" \
    --tokens $TOKENS \
    --runs $RUNS \
    --max_input_tokens $MAX_INPUT_TOKENS \
    --head_temps $HEAD_TEMPS \
    --router_temps $ROUTER_TEMPS \
    $LIMIT_ARG

echo "[$(date +%H:%M:%S)] OK: $PAIR_NAME"
