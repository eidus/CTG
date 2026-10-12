# CTG++ on nuPlan — state of the work

Reproducing CTG++ (scene-level diffusion traffic simulation) on nuPlan, as a baseline.
Last updated 2026-10-12.

The model is an **unconditional prior**. Constraints belong in inference-time guidance, never
in the training loss — `loss_weights.collision_loss = 0.0` and `offroad_loss = 0.0` in
`algo_config.py` are correct, not oversights. The `# 0.05` / `# 0.1` comments beside them are
CTG-v1 leftovers. Do not turn them on to improve collision numbers.

---

## Where things are

```
/scratch/cilab/ujkim/Baselines/CTG/
├── CTG/                              code (git repo root is the parent)
├── ctg_trajdata/                     vendored trajdata
├── ctg_v2.sif                        singularity image, conda env "ctg"
├── scene_diffuser_trained_models/    checkpoints (gitignored, ~1GB per run)
├── eval_configs/                     --config_file inputs
├── eval_results/                     evaluation outputs
└── bench/                            scratch, not worth keeping
```

Data: `/scratch/cilab/ujkim/data/nuPlan/{root,val,test}`, each with a `trainval/` holding the
`.db` logs. trajdata hardcodes the `trainval` subfolder for every non-mini nuplan env and
resolves maps from `<path>/../maps`. Cache: `/scratch/cilab/ujkim/.unified_data_cache`.

## Models

| name | seed | notes |
|---|---|---|
| `scene_diff_run6` | 1 | the reference run |
| `scene_diff_run7` | 1 | cosine LR experiment — **worse**, not a seed variant |
| `scene_diff_run8` | 0 | seed replicate |
| `scene_diff_run9` | 2 | seed replicate |

The three seeds are **run6 / run8 / run9**. run7 changed the LR schedule, not the seed: on 100
test scenes it was slightly worse on off-road rate (0.076 vs 0.066, paired Wilcoxon p=0.005)
and critical failures (0.320 vs 0.292, p=0.015). The cosine schedule stays available behind
`optim_params.policy.learning_rate.scheduler`, default `"none"`.

Checkpoint to evaluate: `scene_diff_<run>/run0/checkpoints/iter100000.ckpt`. Pass the key as
`iter100000.ckpt` with the extension — `iter100000` alone matches two files and trips an assert.

## The 512-scene evaluation (complete)

Six runs, three seeds × {guided, unguided}, all on the **same 512 scenes**:

| run | `_512_` | `_tail_` | total |
|---|---|---|---|
| run6 guided / unguided | 487 | 25 | 512 |
| run8 guided / unguided | 487 | 25 | 512 |
| run9 guided | 487 | 25 | 512 |
| run9 unguided | 512 | — | 512 |

**Five of the six are split across two directories and still need merging**: the first 487 are
in `<run>_<mode>_512_scene_edit_eval/`, the last 25 in `<run>_<mode>_tail_scene_edit_eval/`.
run9 unguided alone ran all 512 in one go. `stats.json` is `{metric: [per-scene value]}`, so
merging is list concatenation; `eval_results/eval_scenes_512.csv` gives the canonical order in
its `position` column, with a `batch` column marking `first487` vs `tail25`.

The scene set is 512 scenes across **375 distinct logs** (las_vegas 349, singapore 68,
pittsburgh 58, boston 37).

## The index trap — read before touching scene selection

`eval_scenes` indexes `UnifiedDataset._scene_index`, **not** the cache's `scenes_list.dill`:

```
scenes_list.dill   19281 scenes, cache-build order
_scene_index       18209 scenes, sorted, minus scenes too short for history+future
```

Selecting from the wrong list is silent for any index inside both ranges. That is what
happened here: five 512-scene evaluations ran to completion on *different scenes than
intended*, and only died 487 scenes in, when an index finally exceeded 18209, with
`IndexError: list index out of range` from `dataset.get_scene`. The 487 that ran were a valid
sample (350 logs), just not the intended one-per-log draw.

Workflow, in order:

```bash
# 1. dump the order (few minutes, no GPU needed)
python scripts/dump_scene_index.py --out eval_results/scene_index_order.json

# 2. select against it
python scripts/select_eval_scenes.py \
    --scene-index eval_results/scene_index_order.json \
    --out-dir eval_results
```

`select_eval_scenes.py` defaults to one scene per log. A log is one continuous drive cut into
~20s pieces, so consecutive scenes share a route and a traffic situation — `np.arange(512)`
draws from just 36 of the 1349 test logs, which is why that is never the right default. It
also drops scenes under 360 raw frames: a 30-step rollout at dt=0.5 needs 3s history + 15s
future, and `_scene_index` only guarantees the 8s the dataset asked for.

Verify a selection before launching anything long:

```python
sel   = json.load(open("eval_results/eval_scenes_512.json"))
order = json.load(open("eval_results/scene_index_order.json"))
assert max(sel) < len(order)
```

## Running an evaluation

`--config_file` is merged with `cfg.update(**external)`, which recurses into nested dicts, so a
JSON holding only `{"trajdata": {"eval_scenes": [...], "num_scenes_to_evaluate": N}}` overrides
just those fields. It is applied **before** the env block is flattened to the top level, so the
`trajdata` nesting is required. This is how six jobs share one scene list without any of them
editing a config file — no repeat of the `base.py` seed-edit race.

```bash
srun --gres=gpu:1 --cpus-per-task=8 --ntasks=1 -J ctg_r6g \
singularity exec --cleanenv --nv \
--bind /scratch/cilab/ujkim/Baselines/CTG/CTG:/workspace/CTG \
--bind /scratch/cilab/ujkim/Baselines/CTG/ctg_trajdata:/workspace/ctg_trajdata \
/scratch/cilab/ujkim/Baselines/CTG/ctg_v2.sif bash -lc '
CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
conda run --no-capture-output -n ctg \
python /workspace/CTG/scripts/scene_editor.py \
--config_file /scratch/cilab/ujkim/Baselines/CTG/eval_configs/full_512.json \
--results_root_dir /scratch/cilab/ujkim/Baselines/CTG/eval_results \
--env trajdata --registered_name trajdata_nuplan_scene_diff_dt05 \
--eval_class SceneDiffuser --num_scenes_per_batch 1 \
--policy_ckpt_dir /scratch/cilab/ujkim/Baselines/CTG/scene_diffuser_trained_models/scene_diff_run6/run0 \
--policy_ckpt_key iter100000.ckpt \
--editing_source heuristic --prefix run6_guided_512_
'
```

- `--editing_source heuristic` = guided, `none` = unguided, `--eval_class GroundTruth` = replay baseline
- `--prefix` decides the output directory; **changing it is the only thing stopping a rerun from
  overwriting previous results**
- `--results_root_dir` must be absolute or output lands relative to the launch directory

Cost per scene, measured by solving across a 4-scene and a 100-scene run (startup is 202s):
guided **61.2s**, unguided **48.4s**, GroundTruth 18.8s. 512 scenes is ~8.8h guided, ~6.9h
unguided. Roughly a third of each scene is fixed overhead (env reset, map rasterisation,
metrics, hdf5 write) rather than model compute.

`num_scenes_per_batch` must stay **1**. Raising it is not an OOM problem — A100 40GB has room —
but `MapCollisionLoss.forward` (`tbsim/utils/guidance_loss.py`, around line 858) masks `x`,
`drivable_map`, `extent` and `raster_from_agent` by `agt_mask` and then reads `curr_speed`
unmasked. With one scene per batch `agt_mask` covers every agent so the sizes happen to agree;
with more, `moving.view((B,1,1))` raises a shape error. Masking `curr_speed` alongside the rest
is a one-line fix and is a no-op at batch 1 (guidance applies to all agents), but it was not
applied — batch > 1 has not been exercised beyond that point, so expect more of the same.

## Output format

`data.hdf5` holds one group per scene, keyed `<scene name>_<start frame>`, every array shaped
`(agents, timesteps, ...)`. Agent count varies per scene; `freeze_agents=True` fixes the set at
the start of the rollout.

| dataset | meaning |
|---|---|
| `centroid`, `yaw` | what actually happened — use these to replay or visualise |
| `action_traj_positions/yaws` | the 10-frame (5s) plan issued at each step |
| `action_sample_positions/yaws` | **all 5 candidate samples**, including unchosen ones |
| `extent`, `track_id`, `world_from_agent`, `map_names`, `attn_weights` | |

Plan is not execution: `n_step_action=5` means only the first 5 steps of each plan run before
replanning. `action_sample_*` is most of the 167MB per run, and is the raw material for a
diversity metric — `num_sim_per_scene=1` means diversity was never computed, but the samples
are on disk.

`stats.json` is the summary derived from the hdf5; the hdf5 is the source of truth.

## Metrics: two that cannot be used

- **Every `*_disk_*` metric.** Ground truth scores 0.045 off-road under the disk
  approximation against 0.003 by bounding box — it flags real human driving as off-road. The
  disk collision breakdown is also internally inconsistent: per-type rates are all 0 while
  `coll_any` is non-zero.
- **`all_diversity`.** `OccupancyDiversity.get_episode_metrics()` returns `{}` unless
  `num_sim_per_scene > 1`, which is 1. The metric never appears in results.

Usable: collision rate (total and by type), off-road rate, critical failure rate, comfort
(speed / jerk / lateral and longitudinal acceleration), coverage, ADE/FDE, and `guide_*`
compliance metrics when guidance is on. Comfort metrics have no good direction — closer to
ground truth is better, lower is not.

## Earlier results (100 scenes, superseded by the 512-scene runs)

Guidance works. Off-road rate 0.066 → 0.002 against a ground-truth 0.003, collision 0.217 →
0.142 against 0.046, all p < 0.0001 by paired Wilcoxon. The cost is speed: 2.55 → 1.64 m/s
where ground truth is 2.10, so part of the gain is just moving less. Side collisions were 89%
of all collisions.

Those 100 scenes were `np.arange(100)` — 8 logs, 2 of 4 cities. Do not quote them as a nuPlan
result; the 512-scene runs replace them.

## Open questions

- Is the public nuPlan test split the right target? The 2023 Challenge leaderboard uses a
  private set, and CTG++'s own numbers are nuScenes, so neither comparison is direct.
- Diversity is unmeasured and cheap to add (`num_sim_per_scene` > 1, evaluation only).
- GPT guidance (`GPTLoss`) is untouched. Only the hand-written heuristics have been run.
- Adversarial guidance is untouched. `CollisionLoss`, `FrontCollisionLoss` and
  `CollideLeftSideLoss` in `guidance_loss.py` already implement "agent `target_ind` should
  collide with agent `ref_ind`" — the attack side of CTG++, with agent indices hardcoded by
  hand. Relevant if the goal is generating an attacker plus reactive agents: the joint
  scene-level denoising means guidance on one agent moves the others, but there is no victim
  planner in the loop, no search over which agent attacks, and no plausibility constraint
  beyond the prior.

## Things that bit us

- `trajdata_source_valid` used to alias `trajdata_source_train`: every `val/*` metric was a
  training-set metric.
- `root/trainval` was a symlink to `train_singapore`, so training silently used 2,396 of
  13,179 logs. trajdata matches zero `.db` files for a missing city **without raising**.
  Always check the scene count against the `.db` count on disk.
- `train_vegas_4/2021.06.21.22.56.42_veh-38_01258_02103.db` is corrupt in the source zip —
  excluded, 13,179 logs remain.
- W&B x-axis is the log-call count, not the step: `_step` ≈ `global_step / log_every_n_steps`.
  A "plateau at 2k" on the chart is 20k real steps.
- `get_exp_dir` prompts `overwrite? (y/n)` when a run name already exists and **`y` deletes the
  whole directory**, checkpoints included. `--remove_exp_dir` does it with no prompt.
- To rebuild the scene index for newly added logs, delete `<cache>/<env>/scenes_list.dill`
  only. `trajdata_rebuild_cache = True` also forces map rebuilds and re-bakes valid caches.
- The nuPlan and `--editing_source none` paths were never run upstream. Several call sites
  branched only on `nusc`/`l5`, and `guidance_config = None` reached `len()` in three places.
  Before a long run on an untried path, grep it for `nusc`/`l5` branching and unguarded `len()`.
- A `/tmp` path is node-local: singularity cannot bind it from the compute node. Scripts and
  logs that a job must reach belong under `/scratch`.

## Next step

Merge the `_512_` + `_tail_` halves for the five split runs, then build the six-way comparison:
three seeds × guided/unguided, with a GroundTruth baseline on the same 512 scenes. Seed spread
across run6/8/9 is the new thing these runs make available — previous numbers were single-seed.
