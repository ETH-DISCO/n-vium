#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=Nvium-number-heads-ablation
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%A_%a.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%A_%a.err
#SBATCH --array=0-5
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

###############################
#       Weights & Biases      #
###############################
# WANDB_PROJECT defaults to the value set in container_env.sh; override with:
#   WANDB_PROJECT=PROJECT sbatch run_number_heads_ablation.sbatch

###############################
#   Fixed architecture        #
###############################
HIDDEN=768
NUM_HEADS=12
HEAD_DIM=64
DEPTH=24
INTER_SIZE=2048
ROUTER_HIDDEN=768
ROUTER_INTER=512
ADAPTER_DIM=768

###############################
#   Ablation configurations   #
###############################
# Naming: N-vium = model with N total heads (N-1 early + 1 final)
# Index:              0     1     2     3     4     5
TOTAL_HEADS=(         2     3     4     6     8     12)

# Early exits = total_heads - 1, spread evenly (step = 24 / total_heads)
# 2  heads: step=12 -> [12]
# 3  heads: step=8  -> [8,16]
# 4  heads: step=6  -> [6,12,18]
# 6  heads: step=4  -> [4,8,12,16,20]
# 8  heads: step=3  -> [3,6,9,12,15,18,21]
# 12 heads: step=2  -> [2,4,6,8,10,12,14,16,18,20,22]
EARLY_EXITS=(
    "[12]"
    "[8,16]"
    "[6,12,18]"
    "[4,8,12,16,20]"
    "[3,6,9,12,15,18,21]"
    "[2,4,6,8,10,12,14,16,18,20,22]"
)

# Router balance/warmup target = 1 / total_heads (equal load across all exits)
# Index:              0      1      2      3      4      5
BALANCE_TARGETS=(   0.500  0.333  0.250  0.167  0.125  0.083)

# Baseline intermediate_size param-matched (slightly exceeding) the Nvium
# total parameter count for each ablation. Each Nvium exit adds ~1.97M params
# (router 0.79M + early adapter 1.18M + norm + router head); the final adapter
# adds ~1.18M. Each +1 of intermediate_size adds ~55,296 params to the 24L
# backbone. Values below are the smallest multiple of 8 whose backbone delta
# vs INTER_SIZE=2048 meets/exceeds the Nvium extras.
# Index:              0     1     2     3     4     5
BASELINE_INTER=(    2112  2144  2184  2248  2320  2464)

###############################
#   Fixed training config     #
###############################
LR=6.0e-4
MIN_LR=6.0e-5
BS=16
GA=4

# Compute loss weight — override via: COMPUTE_LOSS_BETA=0.15 sbatch ...
COMPUTE_LOSS_BETA="${COMPUTE_LOSS_BETA:-0.1}"

# Nvium step count (same compute budget across head counts).
# Derived from 20x Chinchilla for the 24L h=768 Nvium model (~226M params):
#   4.523B tokens / (1024 tok/seq * 16 bs * 4 grad_accum * 4 GPUs) = 17,254 steps
NVIUM_MAX_STEPS=17254
# Baseline step count is slightly higher to match the baseline's smaller
# per-step token budget at the param-matched intermediate sizes.
BASELINE_MAX_STEPS=17287

###############################
#   Mode: baseline or Nvium   #
#   BASELINE=1 sbatch --array=0-5 ...  → param-matched baselines per index
#   sbatch --array=0-5 ...             → Nvium (default)
###############################
BASELINE="${BASELINE:-}"

###############################
#   Select config for index   #
###############################
i=$SLURM_ARRAY_TASK_ID

# Per-index batch size override (reduce BS, increase GA to avoid OOM;
# effective batch size unchanged). Index 5 (12 heads, 11 early exits) OOMs at BS=16.
if [ "$i" -eq 5 ]; then
    BS=8
    GA=8
fi

if [ -n "$BASELINE" ]; then
    #############################################
    # BASELINE MODE — param-matched per index
    #############################################
    NH=${TOTAL_HEADS[$i]}
    BASE_INTER=${BASELINE_INTER[$i]}
    RUN_NAME="baseline_matched_${NH}heads_d24_h768_inter${BASE_INTER}"

    echo "=== BASELINE (param-matched to ${NH}-head Nvium) ==="
    echo "  hidden=${HIDDEN}, depth=${DEPTH}, heads=${NUM_HEADS}, head_dim=${HEAD_DIM}"
    echo "  intermediate_size=${BASE_INTER}"
    echo "  lr=${LR}, min_lr=${MIN_LR}, batch_size=${BS}, grad_accum=${GA}"
    echo "  dataset=${DATASET}, max_steps=${BASELINE_MAX_STEPS}"
    echo "  run_name=${RUN_NAME}"

    if [ -n "${RESUME_DIR:-}" ]; then
        export HYDRA_RUN_DIR="${RESUME_DIR}"
    else
        export HYDRA_RUN_DIR="/users/$USER/project/github/runs/Nvium-number-heads-ablation/baseline_matched/${RUN_NAME}"
    fi

    RESUME_OVERRIDE=""
    if [ -n "${RESUME_CKPT:-}" ]; then
        RESUME_OVERRIDE="resume_from_checkpoint=${RESUME_CKPT}"
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
      model_overrides.intermediate_size=${BASE_INTER} \
      model_overrides.head_dim=${HEAD_DIM} \
      model_name_or_path=${MODEL_NAME_OR_PATH} \
      dataset=${DATASET} \
      training_args.per_device_train_batch_size=${BS} \
      training_args.per_device_eval_batch_size=${BS} \
      training_args.gradient_accumulation_steps=${GA} \
      training_args.learning_rate=${LR} \
      training_args.min_lr=${MIN_LR} \
      training_args.lr_scheduler_type=cosine_with_min_lr \
      training_args.max_steps=${BASELINE_MAX_STEPS} \
      training_args.project_name=${WANDB_PROJECT} \
      training_args.run_name=${RUN_NAME} \
      ${RESUME_OVERRIDE}

else
    #############################################
    # Nvium MODE (default)
    # Naming: ${NH}vium = model with NH total heads
    #############################################
    NH=${TOTAL_HEADS[$i]}
    EXITS=${EARLY_EXITS[$i]}
    BALANCE_TARGET=${BALANCE_TARGETS[$i]}
    RUN_NAME="${NH}vium_h768_l24_${COMPUTE_LOSS_BETA}"

    echo "=== ${NH}vium: ${NH} total heads ($(( NH - 1 )) early + 1 final) ==="
    echo "  early_exits=${EXITS}"
    echo "  balance_target=${BALANCE_TARGET}"
    echo "  hidden=${HIDDEN}, depth=${DEPTH}, heads=${NUM_HEADS}, head_dim=${HEAD_DIM}"
    echo "  lr=${LR}, min_lr=${MIN_LR}, batch_size=${BS}, grad_accum=${GA}"
    echo "  dataset=${DATASET}, max_steps=${NVIUM_MAX_STEPS}"
    echo "  compute_loss_beta=${COMPUTE_LOSS_BETA}"
    echo "  run_name=${RUN_NAME}"

    if [ -n "${RESUME_DIR:-}" ]; then
        export HYDRA_RUN_DIR="${RESUME_DIR}"
    else
        export HYDRA_RUN_DIR="/users/$USER/project/github/runs/Nvium-number-heads-ablation/compute_loss_${COMPUTE_LOSS_BETA}/${RUN_NAME}"
    fi

    RESUME_OVERRIDE=""
    if [ -n "${RESUME_CKPT:-}" ]; then
        RESUME_OVERRIDE="resume_from_checkpoint=${RESUME_CKPT}"
        echo "Resuming from checkpoint: ${RESUME_CKPT}"
    fi

    python3 -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      pretraining/Nvium/Nvium_main.py \
      hydra.run.dir="${HYDRA_RUN_DIR}" \
      model_type=Nvium \
      early_layer_idx=${EXITS} \
      router.hidden_size=${ROUTER_HIDDEN} \
      router.intermediate_size=${ROUTER_INTER} \
      router.router_balance_target=${BALANCE_TARGET} \
      router.router_warmup_target=${BALANCE_TARGET} \
      early_adapter.dim=${ADAPTER_DIM} \
      final_adapter.dim=${ADAPTER_DIM} \
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
      training_args.lr_scheduler_type=cosine_with_min_lr \
      training_args.max_steps=${NVIUM_MAX_STEPS} \
      training_args.project_name=${WANDB_PROJECT} \
      training_args.run_name=${RUN_NAME} \
      loss.warmup.router_compute_loss=false \
      loss.main.router_compute_loss=true \
      loss.alphas_warmup.router_compute_loss=0.0 \
      loss.alphas_main.router_compute_loss=${COMPUTE_LOSS_BETA} \
      ${RESUME_OVERRIDE}
fi
