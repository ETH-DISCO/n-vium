#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=Nvium-beta-ablation
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%A_%a.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%A_%a.err
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path
#SBATCH --array=0-12

# ─── Usage ──────────────────────────────────────────────────────────────────
#   Sweep beta (router_compute_loss weight) over 13 values; each array task
#   trains one setting from scratch.
#
#   sbatch scripts/Nvium/Nvium_run_beta_ablation.sbatch
#
#   Run the dense baseline only (array task 0):
#   MODEL_TYPE=baseline sbatch --array=0 scripts/Nvium/Nvium_run_beta_ablation.sbatch
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

MODEL_TYPE="${MODEL_TYPE:-Nvium}"

BETA_VALUES=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6)
BETA=${BETA_VALUES[$SLURM_ARRAY_TASK_ID]}

if [ "$MODEL_TYPE" = "baseline" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -gt 0 ]; then
        echo "Baseline mode: skipping array task ${SLURM_ARRAY_TASK_ID}"
        exit 0
    fi
    RUN_NAME="baseline_h768_l24"
    RUN_DIR="/users/$USER/project/github/runs/Nvium-beta-ablation/baseline"
    OVERRIDES=(
        "model_type=baseline"
        "resume_from_checkpoint=null"
        "model_name_or_path=${MODEL_NAME_OR_PATH}"
        "dataset=${DATASET}"
        "training_args.run_name=${RUN_NAME}"
        "training_args.project_name=${WANDB_PROJECT}"
        "hydra.run.dir=${RUN_DIR}"
        # Architecture — param-matched baseline (intermediate_size bumped to match Nvium param count)
        "model_overrides.hidden_size=768"
        "model_overrides.num_hidden_layers=24"
        "model_overrides.num_attention_heads=12"
        "model_overrides.num_key_value_heads=12"
        "model_overrides.intermediate_size=2184"
        "model_overrides.head_dim=64"
        # Training
        "training_args.per_device_train_batch_size=16"
        "training_args.per_device_eval_batch_size=16"
        "training_args.gradient_accumulation_steps=4"
        "training_args.learning_rate=0.0006"
        "training_args.min_lr=6.0e-05"
        "training_args.warmup_ratio=0.01"
        "training_args.weight_decay=0.1"
        "training_args.max_grad_norm=1.0"
        "training_args.max_steps=17287"
    )
else
    RUN_NAME="Nvium_h768_l24_beta${BETA}"
    RUN_DIR="/users/$USER/project/github/runs/Nvium-beta-ablation/${RUN_NAME}"
    OVERRIDES=(
        "model_type=Nvium"
        "model_name_or_path=${MODEL_NAME_OR_PATH}"
        "dataset=${DATASET}"
        "training_args.run_name=${RUN_NAME}"
        "training_args.project_name=${WANDB_PROJECT}"
        "hydra.run.dir=${RUN_DIR}"
        # Architecture
        "model_overrides.hidden_size=768"
        "model_overrides.num_hidden_layers=24"
        "model_overrides.num_attention_heads=12"
        "model_overrides.num_key_value_heads=12"
        "model_overrides.intermediate_size=2048"
        "model_overrides.head_dim=64"
        # Early exits
        "early_layer_idx=[6,12,18]"
        # Router
        "router.intermediate_size=512"
        # Training
        "training_args.per_device_train_batch_size=16"
        "training_args.per_device_eval_batch_size=16"
        "training_args.gradient_accumulation_steps=4"
        "training_args.learning_rate=0.0006"
        "training_args.min_lr=6.0e-05"
        "training_args.warmup_ratio=0.01"
        "training_args.weight_decay=0.1"
        "training_args.max_grad_norm=1.0"
        "training_args.max_steps=17254"
    )
    if [ "${BETA}" = "0" ]; then
        OVERRIDES+=("loss.main.router_compute_loss=false" "loss.alphas_main.router_compute_loss=0.0")
    else
        OVERRIDES+=("loss.alphas_main.router_compute_loss=${BETA}")
    fi
fi

echo "=== Beta Ablation [${MODEL_TYPE}]: beta=${BETA} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "  run_name: $RUN_NAME"
echo "  run_dir:  $RUN_DIR"
echo ""

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  pretraining/Nvium/Nvium_main.py \
  "${OVERRIDES[@]}"

echo "[$(date +%H:%M:%S)] Done: $RUN_NAME"
