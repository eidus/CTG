"""
Dump the evaluation dataset's `_scene_index` -- the list `eval_scenes` actually indexes.

scene_editor's `--config_file` takes `trajdata.eval_scenes` as positions in
`UnifiedDataset._scene_index`, which is NOT the cache's `scenes_list.dill`: it is sorted
differently and shorter (18209 vs 19281 for nuplan_test), because UnifiedDataset drops every
scene that cannot supply the requested history and future. Indexing the wrong list fails
silently for any index inside both ranges, and only raises `IndexError` from
`dataset.get_scene` once an index runs past the end.

So: dump the order once, then select scenes against it (scripts/select_eval_scenes.py).

Building the dataset takes a few minutes -- it walks the whole scene index -- but needs no GPU.

Usage (inside the container, /scratch is visible to the compute nodes; a path under /tmp is
node-local and will not be there):
  python scripts/dump_scene_index.py --out /scratch/.../eval_results/scene_index_order.json
"""

import argparse
import contextlib
import io
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registered-name", default="trajdata_nuplan_scene_diff_dt05")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # The config prints a great deal at import and build time; keep stdout for the result.
    with contextlib.redirect_stdout(io.StringIO()):
        from tbsim.configs.registry import get_registered_experiment_config
        from tbsim.configs.scene_edit_config import SceneEditingConfig
        from tbsim.evaluation.env_builders import EnvUnifiedBuilder

        exp_cfg = get_registered_experiment_config(args.registered_name)
        cfg = SceneEditingConfig(registered_name=args.registered_name)
        # scene_editor copies the env-specific block to the top level before building the env;
        # mirror that so the dataset here matches the one evaluation will construct.
        for k in cfg[cfg.env]:
            cfg[k] = cfg[cfg.env][k]
        cfg.lock()

        env = EnvUnifiedBuilder(eval_config=cfg, exp_config=exp_cfg, device="cpu").get_env()

    # Cache layout is <cache>/<env>/<scene name>/scene_metadata_dt<dt>.dill, so the scene name
    # is the parent directory -- the last component is the same filename for every scene.
    names = [str(path).split(os.sep)[-2] for path in env.dataset._scene_index]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(names, f)

    print("len(_scene_index) =", len(names))
    print("first:", names[0])
    print("last :", names[-1])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
