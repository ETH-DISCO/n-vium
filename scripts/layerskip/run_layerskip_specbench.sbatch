#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=specbench-layerskip
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%j.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%j.err
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

# ─── Usage ──────────────────────────────────────────────────────────────────
# Runs the full Spec-Bench benchmark on a LayerSkip checkpoint.
# Dense baseline runs once; speculative runs once per (exit_layer, draft_length).
#
#   CHECKPOINT=/path/to/checkpoint sbatch scripts/layerskip/run_layerskip_specdec.sbatch
#
# Quick test (5 prompts per category):
#   PROMPTS_PER_CAT=5 CHECKPOINT=... sbatch scripts/layerskip/run_layerskip_specdec.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

# ─── Checkpoint under test ──────────────────────────────────────────────────
# Pass via environment variable:
#   CHECKPOINT=/path/to/checkpoint sbatch ...
CHECKPOINT="${CHECKPOINT:-}"
if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] CHECKPOINT is not set. Pass it as: CHECKPOINT=/path/to/ckpt sbatch ..."
    exit 1
fi
if [ ! -d "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT"
    exit 1
fi

# ─── Benchmark config ───────────────────────────────────────────────────────
TOKENS="${TOKENS:-128}"
RUNS="${RUNS:-3}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-896}"        # = 1024 ctx - 128 gen
PROMPTS_PER_CAT="${PROMPTS_PER_CAT:-80}"           # full Spec-Bench = 80/cat (6 cats → 480 prompts)
EXIT_LAYERS="${EXIT_LAYERS:-2 4 6}"
DRAFT_LENGTHS="${DRAFT_LENGTHS:-5 8 12}"

echo "=== Spec-Bench LayerSkip Evaluation ==="
echo "  Checkpoint:     $CHECKPOINT"
echo "  Exit layers:    $EXIT_LAYERS"
echo "  Draft lengths:  $DRAFT_LENGTHS"
echo "  Tokens/gen:     $TOKENS"
echo "  Runs/setting:   $RUNS"
echo "  Prompts/cat:    $PROMPTS_PER_CAT"
echo "  Max input:      $MAX_INPUT_TOKENS"
echo ""

python3 performance/layerskip/measure_specbench_layerskip.py \
    --checkpoint "$CHECKPOINT" \
    --tokens $TOKENS \
    --runs $RUNS \
    --prompts_per_category $PROMPTS_PER_CAT \
    --max_input_tokens $MAX_INPUT_TOKENS \
    --exit_layers $EXIT_LAYERS \
    --draft_lengths $DRAFT_LENGTHS

echo "[$(date +%H:%M:%S)] Done"
