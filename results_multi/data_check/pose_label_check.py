"""Pose addendum C1 / C2: the pose labels and their conventions, checked on stored presses before training.

    python results_multi/data_check/pose_label_check.py
      -> pose_label_gallery_<object>.png (30 presses each), pose_label_check.json

(1) Gallery: the CAD outline placed at the LABELLED pose only - the centre of mass (com_w), the yaw (the label, in the
    sensor frame, turned into the table frame through the stored sensor pose) and the stable-pose id (its rest
    orientation from the object's stable poses) - is projected into the sensor image and drawn on the stored depth X.
    It must coincide with the depth signal. The footprint is also compared with the one from the raw body pose
    (fork_pos_s / fork_quat_s): IoU and the largest vertex distance.
(2) Yaw under augmentation: on presses with an elongated contact, the contact's axis from the image (structure tensor,
    mod 180 deg) is compared with the yaw label (mod 180 deg), before and after centre_v2.augment_robust (rotation, zoom,
    shifts, mirror for symmetric objects) with the label transformed by centre_multi.augment_yaw. The median
    difference must stay small and the same.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

import objects  # noqa: E402
import train_centre as tc  # noqa: E402
import centre_multi as cm  # noqa: E402

S, H, W, MPP = 4, 72, 93, 0.2
OUT = os.path.dirname(os.path.abspath(__file__))


def rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def labelled_pose_world(name, rest, com_w, yaw_s, pose_id, sensor_quat_w, com_body):
    """Object body orientation and position in the world from the pose label (centre, yaw, stable pose)."""
    rp, rq = rest[pose_id]
    R_rest = tc.quat2mat(np.asarray(rq, float) / np.linalg.norm(rq))
    Rs = tc.quat2mat(sensor_quat_w)
    u_w = Rs @ np.array([np.cos(yaw_s), np.sin(yaw_s), 0.0])                 # the label's +x direction, world
    psi_w = np.arctan2(u_w[1], u_w[0])
    psi_rest = np.arctan2(R_rest[1, 0], R_rest[0, 0])
    Rw = rz(psi_w - psi_rest) @ R_rest
    z_body = rp[2]                                                          # resting height of the body origin
    pos = np.array([com_w[0], com_w[1], 0.0]) - (Rw @ com_body) * [1, 1, 0]
    pos[2] = z_body
    return Rw, pos


def footprint(mesh_mm, Rw, pos, sensor_pos_w, sensor_quat_w):
    Rs = tc.quat2mat(sensor_quat_w)
    V = (mesh_mm.vertices / 1000.0) @ Rw.T + pos                            # world
    Vs = (V - sensor_pos_w) @ Rs                                            # sensor frame
    P = Vs[:, :2] * 1000.0 / MPP
    pts = np.stack([(P[:, 0] + (W - 1) / 2) * S + S / 2, (P[:, 1] + (H - 1) / 2) * S + S / 2], 1)
    m = np.zeros((H * S, W * S), np.uint8)
    for f in mesh_mm.faces:
        cv2.fillPoly(m, [np.round(pts[f] * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    return m, Vs


def gallery(C, name, n=30):
    import h5py
    mesh = trimesh.load(os.path.join(objects.mesh_dir(name), f"{name}_visual.stl"), process=True)
    P = json.load(open(os.path.join(ROOT, objects.get(name)["poses"]["mini"])))["poses"]
    rest = {p["id"]: (p["rest_pos_m"], p["quat_wxyz"]) for p in P}
    com_body = np.array(objects.get(name)["com_mm"]) / 1000.0
    sel = cm.rows(C, [name], "test")
    on = sel[(C["cls"][sel] == 2) & (C["sample_type"][sel] == 0)]
    pick = np.random.default_rng(5).choice(on, n, replace=False)
    yaw = np.arctan2(*cm.yaw_vectors(C, pick).T[::-1])
    tiles, ious, dists = [], [], []
    for g, ys in zip(pick, yaw):
        shard, i = C["shard"][g], int(C["index"][g])
        with h5py.File(os.path.join(cm.DATA, shard)) as f:
            sp, sq, cw, fp, fq = (f["sensor_pos_w"][i], f["sensor_quat_w"][i], f["com_w"][i], f["fork_pos_w"][i],
                                  f["fork_quat_w"][i])
        pid = int(C["pose_id"][g])
        Rw, pos = labelled_pose_world(name, rest, cw, ys, pid, sq, com_body)
        m_lab, V_lab = footprint(mesh, Rw, pos, sp, sq)
        m_raw, V_raw = footprint(mesh, tc.quat2mat(fq), fp, sp, sq)
        ious.append(float(((m_lab > 0) & (m_raw > 0)).sum() / max(((m_lab > 0) | (m_raw > 0)).sum(), 1)))
        dists.append(float(np.linalg.norm(V_lab - V_raw, axis=1).max() * 1000))
        v = np.clip(C["X"][g].astype(np.float32) / 1.5, 0, 1)
        img = cv2.resize(cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), (W * S, H * S),
                         interpolation=cv2.INTER_NEAREST)
        cnt, _ = cv2.findContours(m_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, (80, 255, 80), 1)
        c = C["com_s"][g][:2] * 1000.0 / MPP
        p = np.array([(c[0] + (W - 1) / 2) * S + S / 2, (c[1] + (H - 1) / 2) * S + S / 2])
        ctr = np.clip(p, 6, [W * S - 7, H * S - 7])
        d = np.array([np.cos(ys), np.sin(ys)])
        cv2.arrowedLine(img, tuple(np.round(ctr).astype(int)), tuple(np.round(ctr + 34 * d).astype(int)), (60, 60, 255), 2,
                        tipLength=0.3)
        bar = np.zeros((22, W * S, 3), np.uint8)
        pn = objects.part_names(name)[C["part"][g]] if C["part"][g] >= 0 else "-"
        cv2.putText(bar, f"#{i} {pn} pose {pid} yaw {np.degrees(ys) % 360:.0f} deg IoU {ious[-1]:.3f}", (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([bar, img], 0))
    rows_ = [np.concatenate(tiles[r:r + 5], 1) for r in range(0, len(tiles), 5)]
    head = np.zeros((28, rows_[0].shape[1], 3), np.uint8)
    cv2.putText(head, f"{name}: stored X | green = CAD outline at the LABELLED pose (centre + yaw + stable pose) | red arrow = "
                "labelled yaw (+x, towards the head) from the centre of mass (clamped to the pad)", (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, f"pose_label_gallery_{name}.png"), np.concatenate([head] + rows_, 0))
    return dict(n=n, footprint_iou_min=float(min(ious)), footprint_iou_mean=float(np.mean(ious)),
                max_vertex_dist_mm=float(max(dists)))


def yaw_augment_check(C, n=3000):
    import torch
    import centre_v2 as v2
    import search_viz as sv
    out = {}
    sel = cm.rows(C, objects.names(), "test")
    on = sel[(C["cls"][sel] == 2)]
    on = np.random.default_rng(3).choice(on, n, replace=False)
    lab = cm.labels(C, on, objects.names())
    X = torch.tensor(C["X"][on].astype(np.float32))[:, None]

    def axis_diff(Xn, u):
        d = []
        for x, uu in zip(Xn, u):
            g = sv.press_geometry(x)
            if g is None or not np.isfinite(g["axis"]) or g["coherence"] < 0.6:
                d.append(np.nan)
                continue
            a = np.degrees(g["axis"] - np.arctan2(uu[1], uu[0])) % 180
            d.append(min(a, 180 - a))
        return np.array(d)
    d0 = axis_diff(X[:, 0].numpy(), lab["yaw_u"])
    gen = torch.Generator().manual_seed(1)
    xa, _, draws = v2.augment_robust(X, torch.tensor(lab["vec"], dtype=torch.float32), gen,
                                     allow_mirror=torch.tensor(lab["mirror_ok"]))
    ua = cm.augment_yaw(torch.tensor(lab["yaw_u"], dtype=torch.float32), draws).numpy()
    d1 = axis_diff(xa[:, 0].numpy(), ua)
    ok = np.isfinite(d0) & np.isfinite(d1)
    wrong = cm.augment_yaw(torch.tensor(lab["yaw_u"], dtype=torch.float32), dict(draws, mir=torch.ones_like(draws["mir"]))).numpy()
    mir = draws["mir"].numpy() < 0
    d2 = axis_diff(xa[:, 0].numpy()[mir], wrong[mir])
    out.update(n_coherent=int(ok.sum()), median_axis_vs_label_deg_before=float(np.nanmedian(d0[ok])),
               median_axis_vs_label_deg_after=float(np.nanmedian(d1[ok])),
               p90_before=float(np.nanpercentile(d0[ok], 90)), p90_after=float(np.nanpercentile(d1[ok], 90)),
               mirrored=int(mir.sum()), mirrored_only_symmetric=bool(np.all(lab["mirror_ok"][mir])),
               control_mirror_ignored_median_deg=float(np.nanmedian(d2)) if len(d2) else None)
    return out


def main():
    C = cm.load_cache()
    rep = {name: gallery(C, name) for name in objects.names()}
    rep["yaw_augmentation"] = yaw_augment_check(C)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(OUT, "pose_label_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
