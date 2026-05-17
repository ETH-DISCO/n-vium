#!/bin/bash
#SBATCH --job-name=sft
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%j.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%j.err
#SBATCH --environment=/users/alorenc/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

# ─── Usage ──────────────────────────────────────────────────────────────────
# Fine-tune a pretrained checkpoint on Tulu-3 SFT mixture.
#
#   CHECKPOINT=/path/to/checkpoint sbatch scripts/sft/run_sft.sbatch
#
# Quick test:
#   MAX_SAMPLES=100 MAX_STEPS=20 CHECKPOINT=... sbatch scripts/sft/run_sft.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#
# ─── Reproducibility note ────────────────────────────────────────────────────
# Hyperparameters match the original depth/width scaling experiments
# Original global batch size: 16 (per-device) × 1 (GPU) × 2 (grad accum) = 32.
# With 4 GPUs here the global batch is 128 — adjust GRAD_ACCUM if needed.
#
#   LR            = 1e-5
#   EPOCHS        = 2
#   BATCH_SIZE    = 16   (per device)
#   GRAD_ACCUM    = 2
#   MAX_SEQ_LEN   = 1024
#   LOSS_MODE     = full  (mix+compute; multihead only — baseline ignores it)
#   SAVE/EVAL     = every 5% of total training steps
#   LOGGING_STEPS = 10
#   dataset       = allenai/tulu-3-sft-mixture (full, ~892K train examples)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

# ─── Required checkpoint ────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-}"
if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] CHECKPOINT is not set. Pass it as: CHECKPOINT=/path/to/checkpoint sbatch ..."
    exit 1
fi
if [ ! -d "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint directory not found: $CHECKPOINT"
    exit 1
fi

# ─── Training hyperparameters ───────────────────────────────────────────────
SFT_DATASET="${SFT_DATASET:-allenai/tulu-3-sft-mixture}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-}"        # empty = full dataset
MAX_STEPS="${MAX_STEPS:-}"            # empty = train full epochs
SAVE_EVAL_PCT="${SAVE_EVAL_PCT:-5}"   # save/eval every N% of total steps
LOGGING_STEPS="${LOGGING_STEPS:-10}"
LOSS_MODE="${LOSS_MODE:-full}"        # full (mix+compute), mix_only, final_only (multihead only)

RUN_NAME="sft-$(basename $(dirname $CHECKPOINT))-$(basename $CHECKPOINT)"
OUTPUT_DIR="/users/$USER/project/github/runs/sft/${RUN_NAME}"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

# Compute save/eval steps as a percentage of total training steps
TRAIN_SAMPLES=${MAX_SAMPLES:-892375}
EFFECTIVE_BATCH=$(( BATCH_SIZE * GRAD_ACCUM * NUM_GPUS ))
TOTAL_STEPS=$(( (TRAIN_SAMPLES + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH * EPOCHS ))
if [ -n "$MAX_STEPS" ]; then
    TOTAL_STEPS=$MAX_STEPS
fi
SAVE_STEPS=$(( TOTAL_STEPS * SAVE_EVAL_PCT / 100 ))
SAVE_STEPS=$(( SAVE_STEPS > 0 ? SAVE_STEPS : 1 ))
EVAL_STEPS=$SAVE_STEPS

echo "=== Fine-tuning ==="
echo "  Checkpoint:  $CHECKPOINT"
echo "  Output:      $OUTPUT_DIR"
echo "  Dataset:     $SFT_DATASET"
echo "  LR: $LR, Epochs: $EPOCHS, Batch: ${BATCH_SIZE}x${GRAD_ACCUM}, GPUs: $NUM_GPUS"
echo "  Max samples: ${MAX_SAMPLES:-full}, Max steps: ${MAX_STEPS:-auto}"
echo "  Total steps: ~$TOTAL_STEPS, Save/eval every ${SAVE_EVAL_PCT}% = $SAVE_STEPS steps"
echo "  Loss mode:   $LOSS_MODE"
echo ""

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

MAX_STEPS_ARG=""
if [ -n "$MAX_STEPS" ]; then
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
fi

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=$NUM_GPUS \
  sft/sft.py \
  --checkpoint_path "$CHECKPOINT" \
  --dataset_name "$SFT_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --wandb_project "$WANDB_PROJECT" \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --max_seq_length $MAX_SEQ_LEN \
  --save_steps $SAVE_STEPS \
  --eval_steps $EVAL_STEPS \
  --logging_steps $LOGGING_STEPS \
  --sft_loss_mode $LOSS_MODE \
  --gradient_checkpointing \
  $MAX_SAMPLES_ARG \
  $MAX_STEPS_ARG

echo "[$(date +%H:%M:%S)] Done: $RUN_NAME"
