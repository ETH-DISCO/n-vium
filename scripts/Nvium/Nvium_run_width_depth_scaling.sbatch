#!/bin/bash
#SBATCH -A go60
#SBATCH -p normal
#SBATCH --job-name=Nvium-depth-width-scaling
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/users/%u/store/outputs/logs/%x-%A_%a.out
#SBATCH --error=/users/%u/store/outputs/logs/%x-%A_%a.err
#SBATCH --array=0-29
#SBATCH --environment=/users/$USER/project/ngc-pytorch-myenv-25.02.toml  # set to your .toml path

set -euo pipefail

mkdir -p /users/$USER/store/outputs/logs

cd /users/$USER/project/Nvium
source /users/$USER/project/Nvium/container_env.sh

# Diagnostics: confirm key env vars are set
echo "DEBUG: USER=${USER}, SCRATCH=${SCRATCH}, PY_DEPS_DIR=${PY_DEPS_DIR}"

export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=/users/$USER/store/outputs/logs/elastic_error_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json

###############################
#       Weights & Biases      #
###############################
# WANDB_PROJECT defaults to the value set in container_env.sh; override with:
#   WANDB_PROJECT=PROJECT sbatch run_width_depth_scaling.sbatch

###############################
#   Model type toggle         #
#   Usage:                    #
#     MODEL_TYPE=Nvium    sbatch run_width_depth_scaling.sbatch
#     MODEL_TYPE=baseline sbatch run_width_depth_scaling.sbatch
###############################
MODEL_TYPE="${MODEL_TYPE:-Nvium}"

###############################
#   Compute loss weight       #
#   Usage:                    #
#     COMPUTE_LOSS_BETA=0.15 sbatch run_width_depth_scaling.sbatch
###############################
COMPUTE_LOSS_BETA="${COMPUTE_LOSS_BETA:-0.0}"

###############################
#   Adaptive penalty toggle   #
#   Usage:                    #
#     ADAPTIVE=1 sbatch --array=29 run_width_depth_scaling.sbatch
#   Only applies to h1536/l48 (array index 29). Requires the baseline
#   CE records to have been saved first by running MODEL_TYPE=baseline.
###############################
ADAPTIVE="${ADAPTIVE:-0}"

###############################
#   Grid definition           #
#   30 runs: 6 widths x 5 depths
#   index i = width_idx * 5 + depth_idx
###############################

# Width axis (hidden_size)
WIDTHS=(          128    256    512    768    1024   1536)
N_HEADS=(           4      8      8     12      16     24)
HEAD_DIMS=(        32     32     64     64      64     64)
# Multihead model MLP intermediate: nearest mult of 8 to 8/3 * h
INTER_SIZES=(     344    688   1368   2048    2736   4096)
# Router: hidden = hidden_size, intermediate: nearest even to 0.66 * hidden_size
ROUTER_HIDDENS=(  128    256    512    768    1024   1536)
ROUTER_INTERS=(    88    176    344    512     688   1016)
# Adapter dim = hidden_size (1:1 ratio, constant across widths)
ADAPTER_DIMS=(    128    256    512    768    1024   1536)
# Learning rate and min_lr (from Pythia/EleutherAI scaling table)
LRS=(           1.0e-3 1.0e-3 1.0e-3 4.0e-4 3.0e-4 3.0e-4)
MIN_LRS=(       1.0e-4 1.0e-4 1.0e-4 4.0e-5 3.0e-5 3.0e-5)

# Depth axis (num_hidden_layers)
DEPTHS=(6 12 24 36 48)

# Quadrivium early exit layer indices per depth.
EARLY_LAYERS=(
    "[1,2,4]"      # depth=6  (exception: not div by 4)
    "[3,6,9]"      # depth=12
    "[6,12,18]"    # depth=24
    "[9,18,27]"    # depth=36
    "[12,24,36]"   # depth=48
)

# Batch size / grad accum: effective batch = per_device * grad_accum * 4 GPUs (sequences × 1024 tokens)
#   hidden=128:   64 seqs =  64K tokens
#   hidden=256:  128 seqs = 128K tokens
#   hidden=512:  256 seqs = 256K tokens
#   hidden=768:  256 seqs = 256K tokens
#   hidden=1024: 512 seqs = 512K tokens
#   hidden=1536: 512 seqs = 512K tokens
# Rows = widths (0=128..5=1536), cols = depths (0=6..4=48), row-major.
BATCH_SIZES=(
    16 16 16 16 16   # hidden=128
    16 16 16 16 16   # hidden=256
    16 16 16 16 16   # hidden=512
    16 16 16 16  8   # hidden=768
    16 16 16  8  8   # hidden=1024
     8  8  8  8  8   # hidden=1536
)
GRAD_ACCUMS=(
     1  1  1  1  1   # hidden=128  (16*1*4=64 seqs)
     2  2  2  2  2   # hidden=256  (16*2*4=128 seqs)
     4  4  4  4  4   # hidden=512  (16*4*4=256 seqs)
     4  4  4  4  8   # hidden=768  (16*4*4=256 seqs; last: 8*8*4=256)
     8  8  8 16 16   # hidden=1024 (16*8*4=512; 8*16*4=512)
    16 16 16 16 16   # hidden=1536 (8*16*4=512 seqs)
)
GRAD_CKPTS=(
    false false false false false   # hidden=128
    false false false false false   # hidden=256
    false false false false false   # hidden=512
    false false false false false   # hidden=768
    false false false false false   # hidden=1024
    false false false false false   # hidden=1536
)



MH_MAX_STEPS=(
      2924   3288   4012   4736   5460   # hidden=128  (x4: 64K tok/step)
      3346   4070   5518   6966   8412   # hidden=256  (x2: 128K tok/step)
      4184   5625   8509  11393  14277   # hidden=512  (unchanged: 256K tok/step)
      7531  10771  17251  23731  30211   # hidden=768  (unchanged: 256K tok/step)
      5866   8750  14517  20285  26052   # hidden=1024 (/2: 512K tok/step)
     11308  17788  30748  43708  56668   # hidden=1536 (/2: 512K tok/step)
)

# Baseline intermediate sizes (mult of 8, varies by both width AND depth)
# Chosen as smallest mult-of-8 such that baseline_params >= multihead_params
BASELINE_INTER_SIZES=(
       432    392    368    360    360   # hidden=128
       864    776    736    720    712   # hidden=256
      1712   1544   1456   1432   1416   # hidden=512
      2568   2312   2184   2136   2120   # hidden=768
      3424   3080   2912   2856   2824   # hidden=1024
      5120   4608   4352   4272   4224   # hidden=1536
)

BASELINE_MAX_STEPS=(
      2924   3292   4016   4740   5488   # hidden=128  (x4: 64K tok/step)
      3348   4072   5530   6978   8426   # hidden=256  (x2: 128K tok/step)
      4184   5632   8515  11422  14305   # hidden=512  (unchanged: 256K tok/step)
      7539  10787  17284  23747  30278   # hidden=768  (unchanged: 256K tok/step)
      5868   8752  14530  20309  26065   # hidden=1024 (/2: 512K tok/step)
     11310  17790  30750  43744  56670   # hidden=1536 (/2: 512K tok/step)
)

###############################
#   Decode config for index   #
###############################
i=$SLURM_ARRAY_TASK_ID
width_idx=$(( i / 5 ))
depth_idx=$(( i % 5 ))
flat_idx=$(( width_idx * 5 + depth_idx ))

HIDDEN=${WIDTHS[$width_idx]}
DEPTH=${DEPTHS[$depth_idx]}
EARLY_LAYER=${EARLY_LAYERS[$depth_idx]}

NUM_HEADS=${N_HEADS[$width_idx]}
HEAD_DIM=${HEAD_DIMS[$width_idx]}
ROUTER_HIDDEN=${ROUTER_HIDDENS[$width_idx]}
ROUTER_INTER=${ROUTER_INTERS[$width_idx]}
ADAPTER_DIM=${ADAPTER_DIMS[$width_idx]}
LR=${LRS[$width_idx]}
MLR=${MIN_LRS[$width_idx]}

BS=${BATCH_SIZES[$flat_idx]}
GA=${GRAD_ACCUMS[$flat_idx]}
GRAD_CKPT=${GRAD_CKPTS[$flat_idx]}

# -----------------------------------------------------------------------
# Hardcoded overrides for h1536/l48 (the ~1.49B run, array index 29).
# Default h=1536 entry uses NUM_HEADS=24, HEAD_DIM=64, LR=3e-4.
# This cell uses 12 heads × 128 dim and LR=2e-4 to match the reference
# multihead run. The WSD scheduler also matches the reference (Nvium
# default is cosine_with_min_lr; reference used warmup_stable_decay).
# Set ADAPTIVE=1 to enable the adaptive penalty controller.
# -----------------------------------------------------------------------
ADAPTIVE_OVERRIDES_1536=""
if [ "$width_idx" -eq 5 ] && [ "$depth_idx" -eq 4 ]; then
    NUM_HEADS=12
    HEAD_DIM=128
    LR=2.0e-4
    MLR=2.0e-5
    if [ "${ADAPTIVE}" = "1" ]; then
        BASELINE_CE_PATH="$SCRATCH/outputs/Nvium-depth-width-scaling/baseline/baseline_h1536_l48/checkpoints/baseline_ce_records_full.json"
        ADAPTIVE_OVERRIDES_1536="adaptive_penalty.enabled=true \
  adaptive_penalty.k_up=8.0 \
  adaptive_penalty.k_down=4.0 \
  adaptive_penalty.max_step=100.0 \
  adaptive_penalty.max_weight=10.0 \
  loss.main.router_compute_loss=true \
  adaptive_penalty.baseline_ce_path=${BASELINE_CE_PATH}"
    fi
fi

if [ "$MODEL_TYPE" = "baseline" ]; then
    INTER=${BASELINE_INTER_SIZES[$flat_idx]}
    MAX_STEP=${BASELINE_MAX_STEPS[$flat_idx]}
    RUN_NAME="baseline_h${HIDDEN}_l${DEPTH}"
else
    INTER=${INTER_SIZES[$width_idx]}
    MAX_STEP=${MH_MAX_STEPS[$flat_idx]}
    RUN_NAME="Nvium_h${HIDDEN}_l${DEPTH}"
    if [ "${COMPUTE_LOSS_BETA}" != "0.0" ]; then
        RUN_NAME="${RUN_NAME}_${COMPUTE_LOSS_BETA}"
    fi
fi

echo "=== Width-Depth Scaling Nvium [${MODEL_TYPE}]: hidden=${HIDDEN}, layers=${DEPTH} ==="
echo "  n_heads=${NUM_HEADS}, head_dim=${HEAD_DIM}, intermediate=${INTER}"
if [ "$MODEL_TYPE" = "Nvium" ]; then
    echo "  early_layer_idx=${EARLY_LAYER} (3 early exits + 1 final)"
    echo "  router: hidden=${ROUTER_HIDDEN}, inter=${ROUTER_INTER}"
    echo "  adapter_dim=${ADAPTER_DIM}"
    echo "  compute_loss_beta=${COMPUTE_LOSS_BETA}"
fi
echo "  lr=${LR}, min_lr=${MLR}"
echo "  batch=${BS}, grad_accum=${GA}, grad_ckpt=${GRAD_CKPT}"
echo "  max_steps=${MAX_STEP}  (Chinchilla 20x)"
echo "  run_name=${RUN_NAME}"

if [ "$MODEL_TYPE" = "baseline" ]; then
    export HYDRA_RUN_DIR="$SCRATCH/outputs/Nvium-depth-width-scaling/baseline/${RUN_NAME}"
else
    export HYDRA_RUN_DIR="$SCRATCH/outputs/Nvium-depth-width-scaling/compute_loss_${COMPUTE_LOSS_BETA}/${RUN_NAME}"
fi

# Build model-type-specific overrides
if [ "$MODEL_TYPE" = "baseline" ]; then
    MODEL_OVERRIDES="model_type=baseline \
  resume_from_checkpoint=null \
  training_args.save_baseline_ce=true"
else
    # For h1536/l48 the reference run used warmup_stable_decay; all other
    # cells use the Nvium config default (cosine_with_min_lr).
    if [ "$width_idx" -eq 5 ] && [ "$depth_idx" -eq 4 ]; then
        SCHED_OVERRIDES="training_args.lr_scheduler_type=warmup_stable_decay training_args.lr_decay_ratio=0.1"
    else
        SCHED_OVERRIDES=""
    fi
    MODEL_OVERRIDES="model_type=Nvium \
  resume_from_checkpoint=${RESUME_CHECKPOINT:-null} \
  early_layer_idx=${EARLY_LAYER} \
  router.hidden_size=${ROUTER_HIDDEN} \
  router.intermediate_size=${ROUTER_INTER} \
  early_adapter.dim=${ADAPTER_DIM} \
  final_adapter.dim=${ADAPTER_DIM} \
  loss.warmup.router_compute_loss=false \
  loss.main.router_compute_loss=$([ "${COMPUTE_LOSS_BETA}" = "0.0" ] && echo false || echo true) \
  loss.alphas_warmup.router_compute_loss=${COMPUTE_LOSS_BETA} \
  loss.alphas_main.router_compute_loss=${COMPUTE_LOSS_BETA} \
  ${SCHED_OVERRIDES} \
  ${ADAPTIVE_OVERRIDES_1536}"
fi

echo "--- Import diagnostic ---"
python3 -c "
import sys, os, traceback
sys.stderr = sys.stdout
sys.path.insert(0, 'pretraining/Nvium')
os.environ.setdefault('RANK', '0')
try:
    import Nvium_main
    print('Import OK')
except Exception:
    traceback.print_exc()
    sys.exit(1)
" && echo "--- Import OK, launching distributed ---" || { echo "--- Import FAILED ---"; exit 1; }

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  pretraining/Nvium/Nvium_main.py \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  ${MODEL_OVERRIDES} \
  model_overrides.hidden_size=${HIDDEN} \
  model_overrides.num_hidden_layers=${DEPTH} \
  model_overrides.num_attention_heads=${NUM_HEADS} \
  model_overrides.num_key_value_heads=${NUM_HEADS} \
  model_overrides.intermediate_size=${INTER} \
  model_overrides.head_dim=${HEAD_DIM} \
  model_name_or_path=${MODEL_NAME_OR_PATH} \
  dataset=${DATASET} \
  cache_locally=true \
  training_args.learning_rate=${LR} \
  training_args.min_lr=${MLR} \
  training_args.per_device_train_batch_size=${BS} \
  training_args.per_device_eval_batch_size=${BS} \
  training_args.gradient_accumulation_steps=${GA} \
  training_args.gradient_checkpointing=${GRAD_CKPT} \
  training_args.max_steps=${MAX_STEP} \
  training_args.project_name=${WANDB_PROJECT} \
  training_args.run_name=${RUN_NAME}