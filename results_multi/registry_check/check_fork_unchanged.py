"""Is the fork-only pipeline unchanged? (multi-object phase 1.2: run before and after objects.json is introduced)

    GELSIGHT_SENSOR=mini MUJOCO_GL=egl python results_multi/registry_check/check_fork_unchanged.py TAG regen
    GELSIGHT_SENSOR=mini MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 python results_multi/registry_check/check_fork_unchanged.py TAG search
        -> results_multi/registry_check/fork_unchanged_TAG_PART.json

The rendering device matters for bit-exact images: the dataset shards were rendered (and verified) on the default EGL
device (the NVIDIA GPU), the stored search episodes on device 2 (the Intel iGPU). On the other device the physics and
labels still agree exactly, but the rendered depth / images differ in the last bits.

(a) regeneration: stored dataset_mini presses (test2-0000 and train2-0000, 20 each) are simulated again from their
    seeds by collect_dataset.Collector and compared field by field (collect_dataset.compare: every HDF5 field and every
    contact row, exact equality);
(b) the final model's search: the first 20 stored clean episodes of results_centre/mini_robust (seed 0) are run again
    with centre_v3.SearcherV3.episode_v2 and compared with the stored traces, every field except the timing.
A digest of all regenerated / re-run outputs is stored, so TAG=before and TAG=after can be compared directly.
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
assert os.environ.get("GELSIGHT_SENSOR") == "mini", "run with GELSIGHT_SENSOR=mini"
import numpy as np  # noqa: E402

import collect_dataset as cd  # noqa: E402

tag, part = sys.argv[1], sys.argv[2]
out = dict(tag=tag, part=part, egl_device=os.environ.get("MUJOCO_EGL_DEVICE_ID", "default"), regen={}, search={})
h = hashlib.sha256()
t0 = time.time()
for shard in (("dataset_mini/test2-0000.h5", "dataset_mini/train2-0000.h5") if part == "regen" else ()):
    res, col = cd.regenerate(shard, list(range(20)))
    out["regen"][shard] = dict(n=len(res), identical=sum(r["identical"] for r in res),
                               differing=[r for r in res if not r["identical"]])
    import h5py
    with h5py.File(shard) as f:
        ref = f["reference_image"][()]
        for r in res:
            s = col.sample(r["seed"], ref)
            for k in sorted(cd.FIELDS):
                h.update(k.encode())
                h.update(np.ascontiguousarray(np.asarray(s[k], dtype=cd.FIELDS[k][1])).tobytes())
    print(f"regen {shard}: {out['regen'][shard]['identical']}/20 identical ({time.time() - t0:.0f} s)", flush=True)

import centre_v3 as v3  # noqa: E402

E = json.load(open("results_centre/mini_robust/search/episodes_v2_100k_s0_clean.json"))[:20] if part == "search" else []
S = None if not E else v3.SearcherV3({}, "net", "results_centre/mini_robust/model_100k_seed0.pt")
same = 0
for e in E:
    f = S.episode_v2(e["index"])
    keys = [k for k in e if k not in ("seconds", "first_part")]
    eq = all(json.dumps(f[k], sort_keys=True) == json.dumps(e[k], sort_keys=True) for k in keys)
    same += eq
    h.update(json.dumps({k: f[k] for k in keys}, sort_keys=True).encode())
    if not eq:
        out["search"].setdefault("differing", []).append(dict(index=e["index"], fields=[k for k in keys if json.dumps(
            f[k], sort_keys=True) != json.dumps(e[k], sort_keys=True)]))
out["search"].update(n=len(E), identical_to_stored=same)
out["digest"] = h.hexdigest()
out["seconds"] = time.time() - t0
print(f"search: {same}/{len(E)} episodes identical to the stored ones; digest {out['digest'][:16]} ({out['seconds']:.0f} s)")
json.dump(out, open(os.path.join(HERE, f"fork_unchanged_{tag}_{part}.json"), "w"), indent=1)
