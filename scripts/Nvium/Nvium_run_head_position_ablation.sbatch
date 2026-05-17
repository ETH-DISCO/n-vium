#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=Nvium-head-position-ablation
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%A_%a.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%A_%a.err
#SBATCH --array=0-10
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

set -euo pipefail

# Run mode: set BASELINE=1 to run the vanilla baseline instead of the sweep.
# When BASELINE=1 only array task 0 does work; the rest exit immediately.
BASELINE="${BASELINE:-0}"

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

###############################
#       Weights & Biases      #
###############################
# WANDB_PROJECT defaults to the value set in container_env.sh; override with:
#   WANDB_PROJECT=PROJECT sbatch run_head_position_ablation.sbatch

###############################
#   Fixed architecture        #
###############################
HIDDEN=768
NUM_HEADS=12
HEAD_DIM=64
DEPTH=24
INTER_SIZE=2048
ROUTER_INTER=512

###############################
#   Head position configs     #
###############################
# Early head position (every 2nd layer): 2, 4, 6, ..., 22
# Index:     0   1   2   3   4   5   6   7   8   9   10
POSITIONS=( 2   4   6   8   10  12  14  16  18  20  22 )

###############################
#   Fixed training config     #
###############################
LR=5.0e-4
MIN_LR=5.0e-5
BS=8
GA=8

# Compute loss weight — override via: COMPUTE_LOSS_BETA=0.15 sbatch ...
COMPUTE_LOSS_BETA="${COMPUTE_LOSS_BETA:-0.0}"

# All runs use the same step count (same compute budget = same parameter count
# regardless of head position).
# ~222M params (h=768, 24L, single early head) × 20x Chinchilla:
#   4.44B tokens / (1024 tok/seq × 8 bs × 8 grad_accum × 4 GPUs) = 16,937 steps
MAX_STEPS=16937

###############################
#   Select config for index   #
###############################
i=$SLURM_ARRAY_TASK_ID

if [ "${BASELINE}" = "1" ]; then
    # Only the first array task runs; the rest are no-ops.
    if [ "${i}" -ne 0 ]; then
        echo "BASELINE=1: array task ${i} is a no-op (only task 0 runs)."
        exit 0
    fi

    RUN_NAME="Nvium_baseline_h768_l24"

    echo "=== Head Position Ablation Baseline (no early exit) ==="
    echo "  hidden=${HIDDEN}, depth=${DEPTH}, heads=${NUM_HEADS}, head_dim=${HEAD_DIM}"
    echo "  lr=${LR}, min_lr=${MIN_LR}, batch_size=${BS}, grad_accum=${GA}"
    echo "  dataset=${DATASET}, max_steps=${MAX_STEPS}"
    echo "  run_name=${RUN_NAME}"

    if [ -n "${RESUME_DIR:-}" ]; then
        export HYDRA_RUN_DIR="${RESUME_DIR}"
    else
        export HYDRA_RUN_DIR="/users/$USER/project/github/runs/Nvium-head-position-ablation/baseline/${RUN_NAME}"
    fi

    RESUME_OVERRIDE=""
    if [ -n "${RESUME_CKPT:-}" ]; then
        RESUME_OVERRIDE="+resume_from_checkpoint=${RESUME_CKPT}"
        echo "Resuming from checkpoint: ${RESUME_CKPT}"
    fi

    python3 -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      pretraining/Nvium/Nvium_main.py \
      hydra.run.dir="${HYDRA_RUN_DIR}" \
      model_type=baseline \
      model_overrides.hidden_size=${HIDDEN} \
      model_overrides.num_hidden_layers=${DEPTH} \
      model_overrides.num_attention_heads=${NUM_HEADS} \
      model_overrides.num_key_value_heads=${NUM_HEADS} \
      model_overrides.intermediate_size=${INTER_SIZE} \
      model_overrides.head_dim=${HEAD_DIM} \
      model_name_or_path=${MODEL_NAME_OR_PATH} \
      dataset=${DATASET} \
      training_args.per_device_train_batch_size=${BS} \
      training_args.per_device_eval_batch_size=${BS} \
      training_args.gradient_accumulation_steps=${GA} \
      training_args.learning_rate=${LR} \
      training_args.min_lr=${MIN_LR} \
      training_args.max_steps=${MAX_STEPS} \
      training_args.project_name=${WANDB_PROJECT} \
      training_args.run_name=${RUN_NAME} \
      ${RESUME_OVERRIDE}

else
    POS=${POSITIONS[$i]}
    RUN_NAME="Nvium_h768_l24_pos${POS}_${COMPUTE_LOSS_BETA}"

    echo "=== Head Position Ablation: early exit at layer ${POS}/${DEPTH} ==="
    echo "  hidden=${HIDDEN}, depth=${DEPTH}, heads=${NUM_HEADS}, head_dim=${HEAD_DIM}"
    echo "  lr=${LR}, min_lr=${MIN_LR}, batch_size=${BS}, grad_accum=${GA}"
    echo "  dataset=${DATASET}, max_steps=${MAX_STEPS}"
    echo "  alpha_compute=${COMPUTE_LOSS_BETA}"
    echo "  run_name=${RUN_NAME}"

    if [ -n "${RESUME_DIR:-}" ]; then
        export HYDRA_RUN_DIR="${RESUME_DIR}"
    else
        export HYDRA_RUN_DIR="/users/$USER/project/github/runs/Nvium-head-position-ablation/compute_loss_${COMPUTE_LOSS_BETA}/${RUN_NAME}"
    fi

    RESUME_OVERRIDE=""
    if [ -n "${RESUME_CKPT:-}" ]; then
        RESUME_OVERRIDE="+resume_from_checkpoint=${RESUME_CKPT}"
        echo "Resuming from checkpoint: ${RESUME_CKPT}"
    fi

    python3 -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      pretraining/Nvium/Nvium_main.py \
      hydra.run.dir="${HYDRA_RUN_DIR}" \
      model_type=Nvium \
      early_layer_idx=[${POS}] \
      router.hidden_size=${HIDDEN} \
      router.intermediate_size=${ROUTER_INTER} \
      early_adapter.dim=${HIDDEN} \
      final_adapter.dim=${HIDDEN} \
      model_overrides.hidden_size=${HIDDEN} \
      model_overrides.num_hidden_layers=${DEPTH} \
      model_overrides.num_attention_heads=${NUM_HEADS} \
      model_overrides.num_key_value_heads=${NUM_HEADS} \
      model_overrides.intermediate_size=${INTER_SIZE} \
      model_overrides.head_dim=${HEAD_DIM} \
      model_name_or_path=${MODEL_NAME_OR_PATH} \
      dataset=${DATASET} \
      training_args.per_device_train_batch_size=${BS} \
      training_args.per_device_eval_batch_size=${BS} \
      training_args.gradient_accumulation_steps=${GA} \
      training_args.learning_rate=${LR} \
      training_args.min_lr=${MIN_LR} \
      training_args.max_steps=${MAX_STEPS} \
      training_args.project_name=${WANDB_PROJECT} \
      training_args.run_name=${RUN_NAME} \
      loss.warmup.router_compute_loss=false \
      loss.main.router_compute_loss=true \
      loss.alphas_warmup.router_compute_loss=0.0 \
      loss.alphas_main.router_compute_loss=${COMPUTE_LOSS_BETA} \
      ${RESUME_OVERRIDE}

fi
