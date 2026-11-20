"""
Freeze the simulation state: results_centre/FROZEN_SIM/ with copies of the code, the final model(s) and the
dataset metadata, and a manifest of SHA-256 hashes (copied files, every model file under results_centre/, the
dataset shards in place), plus the software environment. Nothing outside FROZEN_SIM/ is changed.

    python freeze_sim.py --final results_centre/mini_robust/model_100k_seed0.pt [--final ...] [--shards dataset_mini]
    python freeze_sim.py --verify            # re-hash and compare with the manifest
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_centre", "FROZEN_SIM")
CODE = ["*.py", "*.xml", "*.md", "requirements.txt", "MODELS.sha256", "ForkMilimeter.stl", "meshes/fork_stl/*",
        "results_centre/**/*.sh", "results_centre/**/*.py", "results_centre/*.md", "prezentacija/*.md",
        "prezentacija/*.sh"]
META = ["dataset/poses/stable_poses.json", "dataset/metadata.json", "dataset_v3/metadata.json", "dataset_mini/metadata.json",
        "dataset_v3/abs_templates.npz", "results_moving/fork_detector.joblib"]


def sha(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, HERE)


def files(patterns):
    out = []
    for p in patterns:
        out += [f for f in glob.glob(os.path.join(HERE, p), recursive=True) if os.path.isfile(f)]
    return sorted(set(f for f in out if "FROZEN_SIM" not in f and "__pycache__" not in f))


def freeze(finals, shard_dirs):
    if os.path.exists(OUT):
        raise SystemExit(f"{rel(OUT)} exists; a freeze is never overwritten (move it away first)")
    t0 = time.time()
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), copied={}, models_in_place={}, shards_in_place={}, final=[])
    for group, pats in (("code", CODE), ("dataset_meta", META)):
        for f in files(pats):
            dst = os.path.join(OUT, group, rel(f))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(f, dst)
            man["copied"][rel(dst)] = dict(source=rel(f), sha256=sha(dst), bytes=os.path.getsize(dst))
    for fp in finals:
        fp = os.path.join(HERE, fp)
        for f in (fp, fp.replace(".pt", ".json")):
            dst = os.path.join(OUT, "models", rel(f))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(f, dst)
            man["copied"][rel(dst)] = dict(source=rel(f), sha256=sha(dst), bytes=os.path.getsize(dst))
        man["final"].append(rel(fp))
    for f in sorted(glob.glob(os.path.join(HERE, "results_centre", "**", "*.pt"), recursive=True)
                    + glob.glob(os.path.join(HERE, "results_centre", "model*.pt"))):
        if "FROZEN_SIM" not in f:
            man["models_in_place"][rel(f)] = sha(f)
    for d in shard_dirs:
        for f in sorted(glob.glob(os.path.join(HERE, d, "*.h5"))):
            man["shards_in_place"][rel(f)] = sha(f)
    env = dict(python=sys.version, platform=platform.platform())
    for mod in ("mujoco", "torch", "numpy", "cv2", "scipy", "trimesh", "h5py"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:                                   # noqa: BLE001
            env[mod] = f"not importable: {e}"
    for cmd, key in ((["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], "gpu"),
                     ([sys.executable, "-m", "pip", "freeze"], "pip_freeze")):
        try:
            env[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
        except Exception as e:                                   # noqa: BLE001
            env[key] = f"failed: {e}"
    man["environment"] = env
    man["seconds"] = time.time() - t0
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
    with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
        for k, v in man["copied"].items():
            f.write(f"{v['sha256']}  {k}\n")
        for k, v in list(man["models_in_place"].items()) + list(man["shards_in_place"].items()):
            f.write(f"{v}  ../../{k}\n")
    print(f"froze {len(man['copied'])} files, hashed {len(man['models_in_place'])} models and "
          f"{len(man['shards_in_place'])} shards in place, {man['seconds']:.0f} s -> {rel(OUT)}")
    return man


def verify():
    man = json.load(open(os.path.join(OUT, "MANIFEST.json")))
    bad = [k for k, v in man["copied"].items() if sha(os.path.join(HERE, k)) != v["sha256"]]
    bad += [k for k, v in man["models_in_place"].items() if not os.path.exists(os.path.join(HERE, k)) or sha(os.path.join(HERE, k)) != v]
    bad += [k for k, v in man["shards_in_place"].items() if not os.path.exists(os.path.join(HERE, k)) or sha(os.path.join(HERE, k)) != v]
    n = len(man["copied"]) + len(man["models_in_place"]) + len(man["shards_in_place"])
    print(f"verified {n} hashes: {'all match' if not bad else f'{len(bad)} differ: {bad[:10]}'}")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--shards", nargs="*", default=["dataset_mini"])
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        verify()
    else:
        freeze(a.final, a.shards)


if __name__ == "__main__":
    main()
