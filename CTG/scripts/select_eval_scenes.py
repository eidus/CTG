"""
Pick a fixed, reproducible set of nuPlan test scenes to evaluate on.

Why this exists: `eval_scenes = np.arange(N)` looks like a random sample but is not. trajdata's
scene list is grouped by log, and each log is one continuous drive chopped into ~20s pieces, so
the first 512 indices come from only 36 of the 1349 test logs -- the same few routes measured 14
times each. The earlier 100-scene runs covered 8 logs and 2 of the 4 cities.

Default strategy ("per_log"): sample LOGS without replacement, then one scene from each. That
gives 512 independent drives instead of 36. Scenes shorter than --min-length frames are dropped
first: a 30-step rollout at dt=0.5 needs 3s of history + 15s of future = 360 frames at the raw
20Hz, and ~13% of test scenes are end-of-log stubs below that.

Writes three files to --out-dir:
  eval_scenes_<N>.json  -- just the indices, for scene_editor's --config_file
  eval_scenes_<N>.csv   -- index, scene name, log (.db file), city, length: what was actually used
  eval_scenes_<N>.meta.json -- the arguments and summary stats, so the choice is auditable

Usage:
  python scripts/select_eval_scenes.py --out-dir /scratch/cilab/ujkim/Baselines/CTG/eval_results
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import dill


def load_scenes(cache_dir, env):
    path = os.path.join(cache_dir, env, "scenes_list.dill")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "No cached scene list at %s -- run an eval once to build the cache first." % path
        )
    with open(path, "rb") as f:
        return dill.load(f)


def log_of(scene):
    # trajdata names a scene "<log>=<hash>"; the log is the .db file it came from.
    return scene.name.split("=")[0]


def select(scenes, num, strategy, min_length, seed):
    eligible = [i for i, s in enumerate(scenes) if s.length >= min_length]
    if len(eligible) < num:
        raise ValueError(
            "Only %d scenes are >= %d frames, need %d" % (len(eligible), min_length, num)
        )
    rng = random.Random(seed)

    if strategy == "random":
        return sorted(rng.sample(eligible, num))

    by_log = defaultdict(list)
    for i in eligible:
        by_log[log_of(scenes[i])].append(i)
    logs = sorted(by_log)
    if len(logs) < num:
        raise ValueError(
            "Only %d logs have an eligible scene, cannot take %d distinct logs" % (len(logs), num)
        )
    return sorted(rng.choice(by_log[l]) for l in rng.sample(logs, num))


def main():
    p = argparse.ArgumentParser()
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
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    scenes = load_scenes(args.cache_dir, args.env)
    sel = select(scenes, args.num, args.strategy, args.min_length, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "eval_scenes_%d" % args.num)

    with open(stem + ".json", "w") as f:
        json.dump(sel, f)

    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_index", "scene_name", "log_db", "location", "length_frames"])
        for i in sel:
            s = scenes[i]
            w.writerow([i, s.name, log_of(s) + ".db", s.location, s.length])

    cities = Counter(scenes[i].location for i in sel)
    logs = {log_of(scenes[i]) for i in sel}
    meta = {
        "env": args.env,
        "num": args.num,
        "strategy": args.strategy,
        "min_length": args.min_length,
        "seed": args.seed,
        "total_scenes_in_split": len(scenes),
        "total_logs_in_split": len({log_of(s) for s in scenes}),
        "selected_unique_logs": len(logs),
        "cities": dict(cities),
    }
    with open(stem + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("wrote %s.{json,csv,meta.json}" % stem)
    for k, v in meta.items():
        print("  %-24s %s" % (k, v))


if __name__ == "__main__":
    main()
