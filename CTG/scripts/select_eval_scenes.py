"""
Pick a fixed, reproducible set of nuPlan test scenes to evaluate on.

WHICH INDEX SPACE -- read this before changing anything here.

`eval_scenes` in the evaluation config indexes `UnifiedDataset._scene_index`, NOT the
`scenes_list.dill` in the trajdata cache. The two differ in both order and length:

    scenes_list.dill   19281 scenes, cache-build order
    _scene_index       18209 scenes, sorted, minus every scene that cannot supply the
                       requested history (3s) + future (5s)

An earlier version of this script selected from scenes_list.dill. Indices below 18209 were
still valid, so five 512-scene evaluations ran to completion on *different scenes than
intended* with no error at all; the runs only died once an index exceeded 18209, 487 scenes
in, with an opaque `IndexError: list index out of range` from `dataset.get_scene`. Selecting
from the wrong list fails silently -- take the `--scene-index` argument seriously.

Produce that file first:

    python scripts/dump_scene_index.py --out eval_results/scene_index_order.json

Default strategy ("per_log"): sample LOGS without replacement, then one scene from each. A
log is one continuous drive chopped into ~20s pieces, so consecutive scenes share a route,
a time of day and a traffic situation -- `np.arange(512)` draws 512 scenes from just 36 of
the 1349 test logs. One scene per log gives 512 independent drives instead.

Scenes shorter than --min-length frames are dropped: a 30-step rollout at dt=0.5 needs 3s of
history + 15s of future = 360 frames at the raw 20Hz, and `_scene_index` only guarantees the
8s the dataset itself asked for. Lengths are joined from scenes_list.dill by scene *name*,
never by position.

Writes to --out-dir:
  eval_scenes_<N>.json       indices, for scene_editor's --config_file
  eval_scenes_<N>.csv        index, name, log (.db), city, length: what was actually used
  eval_scenes_<N>.meta.json  arguments and summary stats, so the choice is auditable

Usage:
  python scripts/select_eval_scenes.py \
      --scene-index eval_results/scene_index_order.json \
      --out-dir     eval_results
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import dill


def log_of(name):
    # trajdata names a scene "<log>=<hash>"; the log is the .db file it came from.
    return name.split("=")[0]


def load_scene_index(path):
    with open(path) as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("%s is not a non-empty list of scene names" % path)
    if names[0].endswith(".dill"):
        raise ValueError(
            "%s holds cache filenames, not scene names -- regenerate it with "
            "scripts/dump_scene_index.py" % path
        )
    return names


def load_scene_meta(cache_dir, env):
    """name -> (length, location), joined by name because positions do not correspond."""
    path = os.path.join(cache_dir, env, "scenes_list.dill")
    with open(path, "rb") as f:
        scenes = dill.load(f)
    return {s.name: (s.length, s.location) for s in scenes}


def select(names, meta, num, strategy, min_length, seed, exclude):
    eligible = [
        i
        for i, n in enumerate(names)
        if i not in exclude and meta.get(n, (0, None))[0] >= min_length
    ]
    if len(eligible) < num:
        raise ValueError(
            "only %d scenes are eligible (>= %d frames, not excluded), need %d"
            % (len(eligible), min_length, num)
        )
    rng = random.Random(seed)

    if strategy == "random":
        return sorted(rng.sample(eligible, num))

    by_log = defaultdict(list)
    for i in eligible:
        by_log[log_of(names[i])].append(i)
    logs = sorted(by_log)
    if len(logs) < num:
        raise ValueError(
            "only %d logs have an eligible scene, cannot take %d distinct logs"
            % (len(logs), num)
        )
    return sorted(rng.choice(by_log[l]) for l in rng.sample(logs, num))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--scene-index",
        required=True,
        help="JSON list of scene names in _scene_index order (scripts/dump_scene_index.py)",
    )
    p.add_argument("--cache-dir", default="/scratch/cilab/ujkim/.unified_data_cache")
    p.add_argument("--env", default="nuplan_test")
    p.add_argument("--num", type=int, default=512)
    p.add_argument("--strategy", choices=["per_log", "random"], default="per_log")
    p.add_argument(
        "--min-length",
        type=int,
        default=360,
        help="minimum scene length in raw 20Hz frames (360 = 3s history + 15s rollout)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--exclude",
        default=None,
        help="JSON list of indices to keep out of the draw (e.g. an already-evaluated set)",
    )
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    names = load_scene_index(args.scene_index)
    meta = load_scene_meta(args.cache_dir, args.env)
    exclude = set(json.load(open(args.exclude))) if args.exclude else set()

    sel = select(names, meta, args.num, args.strategy, args.min_length, args.seed, exclude)

    # Every index must address the dataset that will actually be built, or the run dies partway.
    assert max(sel) < len(names), "selected an index past the end of _scene_index"
    assert len(set(sel)) == args.num, "duplicate indices selected"

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "eval_scenes_%d" % args.num)

    with open(stem + ".json", "w") as f:
        json.dump(sel, f)

    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "scene_index", "scene_name", "log_db", "location", "length_frames"])
        for pos, i in enumerate(sel):
            n = names[i]
            ln, city = meta.get(n, (-1, "?"))
            w.writerow([pos, i, n, log_of(n) + ".db", city, ln])

    selected_names = [names[i] for i in sel]
    out_meta = {
        "scene_index_file": os.path.abspath(args.scene_index),
        "scene_index_len": len(names),
        "env": args.env,
        "num": args.num,
        "strategy": args.strategy,
        "min_length": args.min_length,
        "seed": args.seed,
        "excluded": len(exclude),
        "selected_unique_logs": len({log_of(n) for n in selected_names}),
        "cities": dict(Counter(meta.get(n, (0, "?"))[1] for n in selected_names)),
    }
    with open(stem + ".meta.json", "w") as f:
        json.dump(out_meta, f, indent=2)

    print("wrote %s.{json,csv,meta.json}" % stem)
    for k, v in out_meta.items():
        print("  %-22s %s" % (k, v))


if __name__ == "__main__":
    main()
