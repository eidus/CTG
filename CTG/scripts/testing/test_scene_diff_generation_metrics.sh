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
# NOTE: trajdata_nuplan_scene_diff reads its data location from
# `NuplanTrajdataSceneTrainConfig.trajdata_data_dirs`/`trajdata_source_valid`
# in tbsim/configs/trajdata_nuplan_scene_config.py, NOT from a CLI flag --
# train.py's --dataset_path only feeds the (unused-by-trajdata) legacy
# `cfg.train.dataset_path` field. Edit that config file's
# `trajdata_data_dirs` dict to point "nuplan_mini" (or whichever
# trajdata_source_valid dataset you're using) at your real nuPlan data
# root before running this script.

# ========== Weights & Biases ==========
# NOTE: this registered config has train.logging.log_wandb=True by
# default, and train.py asserts WANDB_APIKEY is set whenever wandb
# logging is on (see `export WANDB_APIKEY=<key>` below). Skip this
# entirely for a first smoke test by adding --debug (see below) --
# debug mode suppresses all logging regardless of log_wandb/log_tb.
WANDB_PROJECT="YOUR_WANDB_PROJECT"
# export WANDB_APIKEY="YOUR_WANDB_API_KEY"

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

python -W ignore scripts/train.py \
    --config_name trajdata_nuplan_scene_diff \
    --name "$EXPERIMENT_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project_name "$WANDB_PROJECT" \
    --ckpt_path "$CHECKPOINT_PATH" \
    --validate_only \
    --eval_generation_quality \
    "$@"
    # ^ forward any extra overrides (e.g. --debug for a quick smoke run
    #   on a couple of validation batches before a full baseline pass --
    #   also skips the WANDB_APIKEY requirement above)
