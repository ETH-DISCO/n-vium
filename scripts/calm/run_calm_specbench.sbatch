#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=specbench-calm
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%j.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%j.err
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

# ─── Usage ──────────────────────────────────────────────────────────────────
#   CALM_CHECKPOINT=/path/to/checkpoint \
#     sbatch scripts/calm/run_calm_specbench.sbatch
#
# Optional overrides:
#   THRESHOLDS="0.3 0.5 0.7 0.9"   space-separated list (default: 0.1..1.0)
#   TOKENS=128                       tokens to generate per prompt
#   RUNS=3                           measurement runs per threshold
#   LIMIT=5                          prompts per category (quick test)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

# ─── Required checkpoint ────────────────────────────────────────────────────
CALM_CHECKPOINT="${CALM_CHECKPOINT:-}"

if [ -z "$CALM_CHECKPOINT" ]; then
    echo "[ERROR] CALM_CHECKPOINT is not set."
    exit 1
fi
if [ ! -d "$CALM_CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CALM_CHECKPOINT"
    exit 1
fi

# ─── Benchmark config ───────────────────────────────────────────────────────
THRESHOLDS="${THRESHOLDS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
TOKENS="${TOKENS:-128}"
RUNS="${RUNS:-3}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-896}"    # = 1024 ctx - 128 gen
PROMPTS_PER_CAT="${PROMPTS_PER_CAT:-80}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    PROMPTS_PER_CAT="$LIMIT"
fi

SAMPLE_FLAG=""
if [ "${DO_SAMPLE}" -eq 1 ]; then
    SAMPLE_FLAG="--do_sample"
fi

CKPT_NAME=$(basename "$CALM_CHECKPOINT")

echo "=== Spec-Bench CALM Evaluation ==="
echo "  Checkpoint: $CKPT_NAME"
echo "  Thresholds: $THRESHOLDS"
echo "  Tokens:     $TOKENS, Runs: $RUNS"
echo "  Prompts/cat: $PROMPTS_PER_CAT"
echo ""

python3 performance/calm/measure_specbench_calm.py \
    --checkpoint "$CALM_CHECKPOINT" \
    --cache_dir "$SCRATCH/cache/SpecDec" \
    --tokens $TOKENS \
    --runs $RUNS \
    --max_input_tokens $MAX_INPUT_TOKENS \
    --prompts_per_category $PROMPTS_PER_CAT \
    --thresholds $THRESHOLDS \
    --temperature $TEMPERATURE \
    $SAMPLE_FLAG

echo "[$(date +%H:%M:%S)] Done: $CKPT_NAME"
