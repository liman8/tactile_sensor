"""
Multi-object tactile dataset (multi-object run, phase 2): dataset_multi/<object>/<split>-NNNN.h5, Mini sensor.

    python collect_multi.py collect [--objects fork_stl fork2_stl knife_stl spoon_stl] [--train 30000 --val 2000 --test 3000]
    python collect_multi.py regen dataset_multi/knife_stl/test-0000.h5 0 1 2      # re-simulate from the seeds, compare
    python collect_multi.py gallery                                              # label check per object
    python collect_multi.py cache                                                # -> dataset_multi/cache_multi.npz

Every press is made by collect_dataset.Collector after collect_dataset.use_object(name, scene="gelsight_multi.xml"),
with the dataset_mini settings: Mini sensor, the object welded to the table in one of its NATURAL stable poses
(dataset_multi/poses/<name>/), 70 % on the object / 20 % on the table / 10 % hovering, force 0.2-10 N
(log-uniform), sensor tilt up to 3 deg, object within +-30 mm of the table origin at any yaw.

What a shard keeps (only what the network uses, plus the labels and the seeds):
  X        (n, 72, 93) float16   reconstructed depth at the model resolution: the press's depth_recon (fork_detector
                                 photometric model, as collect_dataset) stored as float16, then INTER_AREA to 72 x 93
                                 = exactly what centre_v3.prepare() makes of a dataset_mini shard
  bank     (n, 5, 72, 93) f16    train shards only: the 5 structured-noise variants the training draws from
                                 (centre_v2.structured_noise + random_variant, rng = [BANK_SEED, press seed, k],
                                 photometric reconstruction, downsample), what centre_v3.build_bank3 makes for dataset_mini
  labels   every scalar / vector field of collect_dataset.FIELDS except the images and the free-run placeholders,
           + object_id (registry order), part (the object's part carrying most of the gel-object contact force, -1
           without object contact), face (0 face down / 1 face up / -1 where the object or the pose has no face)
  gel_contacts  the contact rows, as collect_dataset
  per shard: reference_image (its seed in attrs), marker_rest, metadata_json
The RGB image, the full-resolution depth maps and the marker field are not kept: `regen` rebuilds a press from its seed.

Seeds: press = 100,000,000 * (object id + 1) + 10,000,000 * split index + i (splits train, val, test);
reference image of shard k = 3,000,000,000 + 100,000 * object id + 1000 * split index + k. None of them collide
with dataset/, dataset_v3/ or dataset_mini/ (< 1e8 and 1e9 + small).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("GELSIGHT_SENSOR", "mini")          # train_centre's input size (72 x 93) and GelSightSim

import cv2
import numpy as np

import objects

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("MULTI_DATA", os.path.join(HERE, "dataset_multi"))     # MULTI_DATA: smoke tests elsewhere
SCENE = "gelsight_multi.xml"
SPLITS = ["train", "val", "test"]
SIZES = dict(train=30000, val=2000, test=3000)
SHARD = 1000
BANK_SEED = 5050
K_VARIANTS = 5
H_IN, W_IN = 72, 93
SKIP = {"img", "depth_gel", "depth_fork", "depth_recon", "marker_shift", "free_qpos", "free_qvel", "free_fork_pos_w",
        "free_fork_quat_w", "free_shift_mm", "free_origin_shift_mm", "free_rot_deg", "free_settled",
        "free_force_touch", "free_force_residual"}


def press_seed(oid, split, i):
    return 100_000_000 * (oid + 1) + 10_000_000 * SPLITS.index(split) + i


def ref_seed(oid, split, shard):
    return 3_000_000_000 + 100_000 * oid + 1000 * SPLITS.index(split) + shard


def face_label(name, pose_id):
    o = objects.get(name)
    if not o["face_defined"]:
        return -1
    if name == "fork_stl":                      # dataset_mini convention: pose 0 = face down, 1 = face up
        return pose_id if pose_id in (0, 1) else -1
    p = o["poses"]
    return 0 if pose_id == p["face_down_pose"] else (1 if pose_id == p["face_up_pose"] else -1)


def dominant_part(name, rows, Rf, pf):
    """The object's part (along its long axis) carrying most of the gel-object normal force; -1 without contact."""
    starts = objects.part_starts_mm(name)
    k = [r for r in rows if r[7] == 0]
    if not k:
        return -1
    x = np.array([(Rf.T @ (r[1] - pf))[0] * 1000 for r in k])
    w = np.bincount(objects.part_from_x(starts, x), weights=np.maximum([r[3] for r in k], 1e-12),
                    minlength=len(starts) + 1)
    return int(np.argmax(w))


def poses_path(name):
    """The object's stable poses (registry: fork_stl -> dataset_mini/poses/, the new objects -> dataset_multi/poses/)."""
    return os.path.join(HERE, objects.get(name)["poses"]["mini"])


def _setup(name):
    import collect_dataset as cd
    cd.set_sensor("mini")
    cd.use_object(name, scene=SCENE)
    return cd


def config(cd):
    return cd.Config(frac_on_fork=0.7, frac_table=0.2, frac_hover=0.1, propped_frac=0.0, free_run=False,
                     part_weights={p: 1.0 for p in cd.PARTS}, fork_xy_range_m=0.03, tilt_max_deg=3.0,
                     force_min_n=0.2, force_max_n=10.0, sensor="mini")


def make_press(cd, col, name, seed, ref, with_bank, rest):
    """One press: the stored fields (dict) from collect_dataset.Collector.sample."""
    import centre_v2 as v2
    s = col.sample(seed, ref)
    rec16 = np.asarray(s["depth_recon"], np.float16)
    X = cv2.resize(rec16.astype(np.float32), (W_IN, H_IN), interpolation=cv2.INTER_AREA).astype(np.float16)
    out = {k: s[k] for k in cd.FIELDS if k not in SKIP}
    Rf, pf = cd.quat2mat(s["fork_quat_w"]), np.asarray(s["fork_pos_w"])
    out.update(X=X, object_id=objects.object_id(name), part=dominant_part(name, s["gel_contacts"], Rf, pf),
               face=face_label(name, int(s["pose_id"])))
    if with_bank:
        B = np.zeros((K_VARIANTS, H_IN, W_IN), np.float16)
        mk = rest + s["marker_shift"]
        for k in range(K_VARIANTS):
            rng = np.random.default_rng([BANK_SEED, int(seed), k])
            I2, R2 = v2.structured_noise(s["img"], ref, rng, mk, **v2.random_variant(rng))
            B[k] = cv2.resize(col.det.pm.height_map(I2, R2, v2.MM_PER_PX_CAM)[0].astype(np.float32), (W_IN, H_IN),
                              interpolation=cv2.INTER_AREA).astype(np.float16)
        out["bank"] = B
    out["gel_contacts"] = s["gel_contacts"]
    return out


def write_shard(args):
    name, split, shard, idx, path = args
    import h5py
    cd = _setup(name)
    oid = objects.object_id(name)
    pj = poses_path(name)
    poses_json, poses_npz = json.load(open(pj)), np.load(pj.replace(".json", ".npz"))
    cfg = config(cd)
    col = cd.Collector(poses_json, poses_npz, cfg)
    rs = ref_seed(oid, split, shard)
    col.sim.rng = np.random.default_rng(rs)
    ref = col.sim.reference_image()
    rest = col.sim._markers.astype(np.float32)
    with_bank = split == "train"
    n, t0, rows, cnt = len(idx), time.time(), [], np.zeros(len(idx), np.int64)
    tmp = path + ".part"
    with h5py.File(tmp, "w") as f:
        meta = cd.metadata(col, poses_json, cfg)
        meta.update(multi=dict(object=name, object_id=oid, registry=objects.get(name), scene=SCENE, split=split,
                               stored=__doc__.split("What a shard keeps")[1].split("Seeds:")[0].strip(),
                               bank_seed=BANK_SEED, k_variants=K_VARIANTS, input_hw=[H_IN, W_IN]))
        f.create_dataset("metadata_json", data=json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        f.attrs.update(object=name, object_id=oid, split=split, shard=shard, n=n, ref_seed=rs,
                       first_seed=press_seed(oid, split, idx[0]))
        f["reference_image"] = ref
        f["marker_rest"] = rest
        ds = {}
        for j, i in enumerate(idx):
            p = make_press(cd, col, name, press_seed(oid, split, i), ref, with_bank, rest)
            if not ds:
                for k, v in p.items():
                    if k == "gel_contacts":
                        continue
                    v = np.asarray(v)
                    big = k in ("X", "bank")
                    ds[k] = f.create_dataset(k, (n,) + v.shape, dtype=v.dtype if v.dtype != np.float64 or big else np.float64,
                                             chunks=((1,) + v.shape) if big else None,
                                             compression="gzip" if big else None, compression_opts=4 if big else None,
                                             shuffle=big)
            for k, d in ds.items():
                d[j] = p[k]
            rows += p["gel_contacts"]
            cnt[j] = len(p["gel_contacts"])
            if (j + 1) % 250 == 0:
                print(f"  {name} {split}-{shard:04d}: {j + 1}/{n} ({(time.time() - t0) / (j + 1):.3f} s/press)", flush=True)
        gp = f.create_group("gel_contacts")
        gp["count"] = cnt
        gp["start"] = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        for c, (nm, shp, dt) in enumerate(cd.GEL_C):
            arr = np.array([r[c] for r in rows], dtype=dt).reshape((len(rows),) + shp)
            gp.create_dataset(nm, data=arr, compression="gzip", compression_opts=4, shuffle=True)
        f.attrs["seconds"] = time.time() - t0
    os.replace(tmp, path)
    return dict(path=path, n=n, seconds=time.time() - t0, bytes=os.path.getsize(path), split=split, object=name)


def collect(names, sizes, workers=24):
    from multiprocessing import get_context
    tasks = []
    for name in names:
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        for split in SPLITS:
            for shard in range(int(np.ceil(sizes[split] / SHARD))):
                idx = list(range(shard * SHARD, min(sizes[split], (shard + 1) * SHARD)))
                path = os.path.join(OUT, name, f"{split}-{shard:04d}.h5")
                if os.path.exists(path):
                    continue
                tasks.append((name, split, shard, idx, path))
    tasks.sort(key=lambda t: (t[1] != "test", t[1] != "val", t[2], t[0]))     # test and val first, objects interleaved
    print(f"{len(tasks)} shards, {sum(len(t[3]) for t in tasks)} presses, {workers} workers", flush=True)
    t0, res = time.time(), []
    with get_context("spawn").Pool(min(workers, max(len(tasks), 1))) as pool:
        for r in pool.imap_unordered(write_shard, tasks):
            res.append(r)
            print(f"wrote {os.path.relpath(r['path'], HERE)}: {r['n']} presses, {r['seconds']:.0f} s, {r['bytes'] / 1e6:.0f} MB "
                  f"({len(res)}/{len(tasks)}, {time.time() - t0:.0f} s)", flush=True)
    if res:
        by = {}
        for r in res:
            by.setdefault(r["split"] == "train", []).append(r)
        rep = dict(wall_s=time.time() - t0, workers=workers, shards=len(res), presses=sum(r["n"] for r in res),
                   mb_per_10k_presses={("train (with bank)" if k else "val / test"): sum(r["bytes"] for r in v) /
                                       sum(r["n"] for r in v) * 1e4 / 1e6 for k, v in by.items()},
                   seconds_per_press_per_worker=float(np.mean([r["seconds"] / r["n"] for r in res])))
        json.dump(rep, open(os.path.join(OUT, f"throughput_{time.strftime('%Y%m%d_%H%M')}.json"), "w"), indent=1)
        print(json.dumps(rep, indent=1))


def regen(path, indices):
    import h5py
    with h5py.File(path) as f:
        name, split, shard = f.attrs["object"], f.attrs["split"], int(f.attrs["shard"])
        cd = _setup(name)
        pj = poses_path(name)
        col = cd.Collector(json.load(open(pj)), np.load(pj.replace(".json", ".npz")), config(cd))
        ref, rest = f["reference_image"][()], f["marker_rest"][()]
        out = []
        for i in indices:
            p = make_press(cd, col, name, int(f["seed"][i]), ref, split == "train", rest)
            bad = [k for k in f if k not in ("metadata_json", "reference_image", "marker_rest", "gel_contacts")
                   and not np.array_equal(np.asarray(f[k][i]), np.asarray(p[k], dtype=f[k].dtype), equal_nan=f[k].dtype.kind == "f")]
            a, c = int(f["gel_contacts"]["start"][i]), int(f["gel_contacts"]["count"][i])
            for ci, (nm, shp, dt) in enumerate(cd.GEL_C):
                b = np.array([r[ci] for r in p["gel_contacts"]], dtype=dt).reshape((len(p["gel_contacts"]),) + shp)
                if not np.array_equal(f["gel_contacts"][nm][a:a + c], b):
                    bad.append("gel_contacts/" + nm)
            out.append(dict(index=int(i), seed=int(f["seed"][i]), identical=not bad, differing=bad))
    return out


def shards(names=None, split=None):
    names = names or objects.names()
    return sorted(p for n in names for p in glob.glob(os.path.join(OUT, n, f"{split or '*'}-[0-9][0-9][0-9][0-9].h5")))


CACHE_KEYS = ["seed", "object_id", "part", "face", "sample_type", "pose_id", "pose_category", "contact_fork",
              "contact_table", "com_s", "bbox_s", "fork_pos_s", "fork_quat_s", "force_cmd", "target_part", "n_tines",
              "tilt_deg", "peak_indent_mm", "fork_visible", "contact_visible", "flawed"]


def build_cache():
    """dataset_multi/cache_multi.npz (X + labels of every shard, train / val / test) and dataset_multi/cache_multi_bank.npy
    (the train rows' bank, memory-mapped by the training)."""
    import h5py
    cols = {k: [] for k in CACHE_KEYS + ["X", "split", "index", "shard"]}
    train_rows = []
    for p in shards():
        with h5py.File(p) as f:
            n = int(f.attrs["n"])
            cols["X"].append(f["X"][()])
            cols["split"].append(np.full(n, SPLITS.index(f.attrs["split"])))
            cols["index"].append(np.arange(n))
            cols["shard"].append(np.full(n, os.path.relpath(p, OUT)))
            for k in CACHE_KEYS:
                cols[k].append(f[k][()])
            if f.attrs["split"] == "train":
                train_rows.append((p, n))
    C = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(os.path.join(OUT, "cache_multi.npz"), **C)
    tr = np.nonzero(C["split"] == 0)[0]
    bank = np.lib.format.open_memmap(os.path.join(OUT, "cache_multi_bank.npy"), mode="w+", dtype=np.float16,
                                     shape=(len(tr), K_VARIANTS, H_IN, W_IN))
    pos = 0
    for p, n in train_rows:
        with h5py.File(p) as f:
            bank[pos:pos + n] = f["bank"][()]
        pos += n
    assert pos == len(tr)
    bank.flush()
    json.dump(dict(train_rows=tr.tolist()[:5] + ["..."], n_train=int(len(tr)), note="bank row r = the r-th train row of "
                   "cache_multi.npz (shards in sorted order)"), open(os.path.join(OUT, "cache_multi_bank.json"), "w"))
    print(f"cache: {len(C['seed'])} presses ({len(tr)} train), bank {bank.shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("collect")
    a.add_argument("--objects", nargs="+", default=None)
    a.add_argument("--train", type=int, default=SIZES["train"])
    a.add_argument("--val", type=int, default=SIZES["val"])
    a.add_argument("--test", type=int, default=SIZES["test"])
    a.add_argument("--workers", type=int, default=24)
    a = sub.add_parser("regen")
    a.add_argument("shard")
    a.add_argument("index", type=int, nargs="+")
    sub.add_parser("cache")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.objects or objects.names(), dict(train=args.train, val=args.val, test=args.test), args.workers)
    elif args.cmd == "regen":
        print(json.dumps(regen(args.shard, args.index), indent=1))
    elif args.cmd == "cache":
        build_cache()


if __name__ == "__main__":
    main()
