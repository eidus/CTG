#!/bin/bash
# Launch the six 512-scene nuPlan evaluations: three seeds x {unguided, guided}.
#
# All six read the same eval_configs/full_512.json, so the scene set is identical across
# seeds and guidance settings and the runs stay directly comparable. The scene list itself
# comes from scripts/select_eval_scenes.py (one scene per log, seed 0); eval_results/
# eval_scenes_512.csv records exactly which scenes those are.
#
# Usage:
#   ./launch_eval_512.sh "6 7"        # two GPUs, jobs queued round-robin onto them
#   ./launch_eval_512.sh "2 3 4 5 6 7" 4   # six GPUs, num_scenes_per_batch=4
#
# Arg 1: space-separated GPU ids to use (default "6 7")
# Arg 2: num_scenes_per_batch (default 1 -- raise only if the bench showed it fits)
set -u

ROOT=/scratch/cilab/ujkim/Baselines/CTG
GPUS=(${1:-6 7})
BATCH=${2:-1}
LOGDIR=$ROOT/eval_logs
mkdir -p "$LOGDIR"

# run6=seed 1, run8=seed 0, run9=seed 2. run7 is the cosine-LR experiment, not a seed.
RUNS=(run6 run8 run9)
# Longest jobs first so the trailing queue on each GPU is short.
MODES=(guided unguided)

i=0
for mode in "${MODES[@]}"; do
  for run in "${RUNS[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    i=$((i + 1))

    if [ "$mode" = "guided" ]; then
      SRC=heuristic
    else
      SRC=none
    fi
    NAME=${run}_${mode}_512
    echo "launch $NAME -> GPU $gpu (batch $BATCH)"

    srun --gres=gpu:1 --cpus-per-task=8 --ntasks=1 -J "ctg_${run}_${mode}" \
      singularity exec --cleanenv --nv \
      --bind $ROOT/CTG:/workspace/CTG \
      --bind $ROOT/ctg_trajdata:/workspace/ctg_trajdata \
      $ROOT/ctg_v2.sif bash -lc "
        CUDA_VISIBLE_DEVICES=$gpu PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        conda run --no-capture-output -n ctg \
        python /workspace/CTG/scripts/scene_editor.py \
          --config_file $ROOT/eval_configs/full_512.json \
          --results_root_dir $ROOT/eval_results \
          --env trajdata \
          --registered_name trajdata_nuplan_scene_diff_dt05 \
          --eval_class SceneDiffuser \
          --num_scenes_per_batch $BATCH \
          --policy_ckpt_dir $ROOT/scene_diffuser_trained_models/scene_diff_${run}/run0 \
          --policy_ckpt_key iter100000.ckpt \
          --editing_source $SRC \
          --prefix ${NAME}_
      " > "$LOGDIR/${NAME}.log" 2>&1 &
  done
done

echo
echo "launched ${i} jobs across ${#GPUS[@]} GPU(s); logs in $LOGDIR"
echo "watch:   tail -f $LOGDIR/run6_guided_512.log"
echo "queue:   squeue -u \$USER -n ctg_run6_guided"
wait
