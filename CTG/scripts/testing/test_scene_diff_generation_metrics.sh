#!/usr/bin/env bash

# ========== ADCG-Diffusion parity note ==========
# This is CTG's counterpart to ADCG-Diffusion/JARCD's
# scripts/testing/test_jarcd_generation_metrics.sh: run the training
# script (train.py) in a single validation-only pass against a trained
# checkpoint, with generation-quality metrics turned on, so that
# "generate + evaluate the generation" happens together the same way
# JARCD's `py_func=validate` + `model.downstream_task=jarcd_generation_eval`
# does. Unlike ADCG this runs the SAME registered experiment/config
# (trajdata_nuplan_scene_diff, SceneDiffuser jointly generating every
# agent in the scene) that is used for normal training -- no separate
# algorithm or config, just an eval-mode switch overlaid at run time --
# so the resulting numbers are directly usable as a baseline to compare
# against JARCD's own generation-quality numbers.
#
# This is NOT the same thing as scripts/scene_editor.py: that script runs
# closed-loop, guided, multi-step rollout simulation (CTG's counterpart to
# ADCG's section-8.3 `test_jarcd_closed_loop.sh`). This script instead
# measures open-loop, one-shot generation quality straight out of the
# model's own validation_step -- CTG's counterpart to ADCG's section-8.1
# `test_jarcd_generation_metrics.sh`.
#
# See tbsim/utils/generation_quality_metrics.py for what's actually
# computed (physical validity / map compliance / realism JSD) and
# tbsim/algos/algos.py's SceneDiffuserTrafficModel._compute_generation_quality_metrics
# for where it's wired into the validation loop.

# ========== Logging & Experiment Settings ==========
OUTPUT_DIR="YOUR_OUTPUT_DIR"           # Where results/logs will be saved (root_dir)
EXPERIMENT_NAME="YOUR_EXPERIMENT_NAME" # Overrides the registered config's default name

# ========== Model Checkpoint ==========
CHECKPOINT_PATH="YOUR_CHECKPOINT_PATH.ckpt"   # Trained trajdata_nuplan_scene_diff checkpoint

# ========== Dataset ==========
DATASET_PATH="YOUR_DATASET_ROOT_PATH"  # trajdata dataset root (see ctg_trajdata/DATASETS.md)

# ========== Weights & Biases (optional -- only used if the registered
#             config has train.logging.log_wandb enabled) ==========
WANDB_PROJECT="YOUR_WANDB_PROJECT"

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

python -W ignore scripts/train.py \
    --config_name trajdata_nuplan_scene_diff \
    --name "$EXPERIMENT_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_path "$DATASET_PATH" \
    --wandb_project_name "$WANDB_PROJECT" \
    --ckpt_path "$CHECKPOINT_PATH" \
    --validate_only \
    --eval_generation_quality \
    "$@"
    # ^ forward any extra overrides (e.g. --debug for a quick smoke run
    #   on a couple of validation batches before a full baseline pass)
