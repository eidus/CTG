#!/bin/bash
# Run one model's two 512-scene evaluations back to back on a single GPU.
#
#   ./run_pair_512.sh <gpu> <run>
#   ./run_pair_512.sh 2 run6
#
# guided goes first because it is the longer of the two (~8.8h vs ~6.9h at
# num_scenes_per_batch=1), so if the allocation is cut short the shorter job is the one left
# unfinished. Both read eval_configs/full_512.json, so every run in the comparison sees the
# same 512 scenes in the same order.
#
# One srun holds the GPU for both jobs: re-queueing between them risks losing the allocation
# to another user partway through.
set -u

ROOT=/scratch/cilab/ujkim/Baselines/CTG
GPU=$1
RUN=$2

CKPT_DIR=$ROOT/scene_diffuser_trained_models/scene_diff_${RUN}/run0
if [ ! -f "$CKPT_DIR/checkpoints/iter100000.ckpt" ]; then
  echo "no checkpoint at $CKPT_DIR/checkpoints/iter100000.ckpt" >&2
  exit 1
fi

mkdir -p "$ROOT/eval_logs"

srun --gres=gpu:1 --cpus-per-task=8 --ntasks=1 -J "ctg_${RUN}_pair" \
  singularity exec --cleanenv --nv \
  --bind $ROOT/CTG:/workspace/CTG \
  --bind $ROOT/ctg_trajdata:/workspace/ctg_trajdata \
  $ROOT/ctg_v2.sif bash -lc "
    set -u
    export CUDA_VISIBLE_DEVICES=$GPU
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

    for pair in 'guided heuristic' 'unguided none'; do
      set -- \$pair
      MODE=\$1
      SRC=\$2
      NAME=${RUN}_\${MODE}_512
      echo \"=== \$(date '+%F %T')  start \$NAME on GPU $GPU\"
      conda run --no-capture-output -n ctg \
      python /workspace/CTG/scripts/scene_editor.py \
        --config_file $ROOT/eval_configs/full_512.json \
        --results_root_dir $ROOT/eval_results \
        --env trajdata \
        --registered_name trajdata_nuplan_scene_diff_dt05 \
        --eval_class SceneDiffuser \
        --num_scenes_per_batch 1 \
        --policy_ckpt_dir $CKPT_DIR \
        --policy_ckpt_key iter100000.ckpt \
        --editing_source \$SRC \
        --prefix \${NAME}_ > $ROOT/eval_logs/\${NAME}.log 2>&1
      echo \"=== \$(date '+%F %T')  done  \$NAME exit=\$?\"
    done
  "
