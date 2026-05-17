#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=layerskip-h768-l24
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/project/github/runs/logs/%x-%j.out
#SBATCH --error=/users/%u/project/github/runs/logs/%x-%j.err
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

set -euo pipefail

mkdir -p /users/$USER/project/github/runs/logs

cd /users/$USER/project/github
source /users/$USER/project/github/container_env.sh

###############################
#   Weights & Biases
###############################
# WANDB_PROJECT defaults to the value set in container_env.sh; override with:
#   WANDB_PROJECT=PROJECT sbatch run_layerskip.sbatch

###############################
#   LayerSkip hyperparameters
#
#   Architecture matches baseline_h768_l24:
#     intermediate_size = 2184
#     max_steps         = 17284
#     lr                = 4e-4,  min_lr = 4e-5
#     batch             = 16 per-device × 4 grad-accum × 4 GPUs = 256 global
#
#   LayerSkip recipe (paper values for from-scratch pretraining):
#     p_max             = 0.1
#     dropout_schedule  = exponential   (S(t) ramps 0→1 over training)
#     escale            = 0.2
#     curriculum_type   = gradual
###############################

P_MAX="${P_MAX:-0.3}"
ESCALE="${ESCALE:-1.0}"
CURRICULUM="${CURRICULUM:-gradual}"
DROPOUT_SCHEDULE="${DROPOUT_SCHEDULE:-exponential}"
MAX_STEPS=17284
GA="${GA:-4}"

RUN_NAME="layerskip_h768_l24_pmax${P_MAX}_escale${ESCALE}"

echo "=== LayerSkip Pretraining: h768, L24 ==="
echo "  intermediate_size=2184  (matches baseline_h768_l24)"
echo "  max_steps=${MAX_STEPS}  (matches baseline)"
echo "  lr=4e-4, min_lr=4e-5"
echo "  p_max=${P_MAX}, escale=${ESCALE}"
echo "  curriculum=${CURRICULUM}, dropout_schedule=${DROPOUT_SCHEDULE}"
echo "  run_name=${RUN_NAME}"

export HYDRA_RUN_DIR="/users/$USER/project/github/runs/layerskip/${RUN_NAME}"

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  pretraining/layerskip/layerskip_main.py \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  model_name_or_path=${MODEL_NAME_OR_PATH} \
  dataset=${DATASET} \
  training_args.max_steps=${MAX_STEPS} \
  training_args.gradient_accumulation_steps=${GA} \
  training_args.project_name=${WANDB_PROJECT} \
  training_args.run_name=${RUN_NAME} \
  layerskip.p_max=${P_MAX} \
  layerskip.escale=${ESCALE} \
  layerskip.curriculum_type=${CURRICULUM} \
  layerskip.dropout_schedule=${DROPOUT_SCHEDULE} \
  layerskip.total_training_steps=${MAX_STEPS}
