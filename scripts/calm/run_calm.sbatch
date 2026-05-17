#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=calm-h768-l24
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%j.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%j.err
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

# ─── Usage ──────────────────────────────────────────────────────────────────
# Two modes, both train from scratch:
#
#   MODE=calm (default)
#     CALM pretraining with early exits, linear-weighted CE.
#     sbatch scripts/calm/run_calm.sbatch
#
#   MODE=calm_baseline
#     Matched dense baseline: same arch/LR/steps/data, final CE only.
#     MODE=calm_baseline sbatch scripts/calm/run_calm.sbatch
#
# Override run name:
#   RUN_NAME=my_run sbatch scripts/calm/run_calm.sbatch
#
# Resume from checkpoint:
#   RESUME_CHECKPOINT=/path/to/checkpoint sbatch scripts/calm/run_calm.sbatch
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh


MODE="${MODE:-calm}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [ "${MODE}" = "calm" ]; then
    RUN_NAME="${RUN_NAME:-calm_h768_l24}"
    HYDRA_RUN_DIR="/users/$USER/project/github/runs/calm/${RUN_NAME}"

    RESUME_ARG=""
    if [ -n "${RESUME_CHECKPOINT}" ]; then
        RESUME_ARG="+resume_from_checkpoint=${RESUME_CHECKPOINT}"
    fi

    echo "=== CALM Training (from scratch) ==="
    echo "  Run name: $RUN_NAME"
    echo "  Run dir:  $HYDRA_RUN_DIR"
    echo ""

    python3 -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      pretraining/calm/calm_main.py \
      --config-name calm_config \
      hydra.run.dir="${HYDRA_RUN_DIR}" \
      baseline_checkpoint=null \
      model_name_or_path=${MODEL_NAME_OR_PATH} \
      dataset=${DATASET} \
      training_args.run_name="${RUN_NAME}" \
      training_args.project_name="${WANDB_PROJECT}" \
      ${RESUME_ARG}

elif [ "${MODE}" = "calm_baseline" ]; then
    RUN_NAME="${RUN_NAME:-baseline_h768_l24}"
    HYDRA_RUN_DIR="/users/$USER/project/github/runs/calm/${RUN_NAME}"

    echo "=== CALM Baseline Training (from scratch) ==="
    echo "  Run name: $RUN_NAME"
    echo "  Run dir:  $HYDRA_RUN_DIR"
    echo ""

    python3 -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      pretraining/Nvium/Nvium_main.py \
      hydra.run.dir="${HYDRA_RUN_DIR}" \
      model_type=baseline \
      resume_from_checkpoint=${RESUME_CHECKPOINT:-null} \
      model_overrides.hidden_size=768 \
      model_overrides.num_hidden_layers=24 \
      model_overrides.num_attention_heads=12 \
      model_overrides.num_key_value_heads=12 \
      model_overrides.intermediate_size=2184 \
      model_overrides.head_dim=64 \
      training_args.learning_rate=5.0e-4 \
      training_args.min_lr=5.0e-5 \
      training_args.max_steps=17251 \
      training_args.per_device_train_batch_size=16 \
      training_args.per_device_eval_batch_size=16 \
      training_args.gradient_accumulation_steps=4 \
      training_args.project_name="${WANDB_PROJECT}" \
      training_args.run_name="${RUN_NAME}"

else
    echo "[ERROR] Unknown MODE=${MODE}. Use 'calm' or 'calm_baseline'."
    exit 1
fi

echo "[$(date +%H:%M:%S)] Done: $RUN_NAME"
