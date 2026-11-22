"""Phase 2 label check: per object, stored test presses (dataset_multi) with the object's outline drawn from the LABELS
(fork_pos_s / fork_quat_s = the object's pose in the sensor frame, and its visual mesh) onto the stored network input
X (reconstructed depth, 72 x 93 px at 0.2 mm), and the centre of mass from com_s (red cross, or a red arrow at the
border pointing to it when it is outside the pad). If the labels are right, the depth signal lies inside the outline.
    python results_multi/data_check/label_gallery.py   -> results_multi/data_check/label_gallery_<object>.png
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

import objects  # noqa: E402
import train_centre as tc  # noqa: E402

S, H, W, MPP = 4, 72, 93, 0.2          # upscale, input size, mm per input pixel
OUT = os.path.dirname(os.path.abspath(__file__))


def footprint_px(mesh_mm, pos_s, quat_s):
    """Object footprint in the sensor x-y plane, as a filled mask on the upscaled input grid."""
    R = tc.quat2mat(quat_s)
    V = mesh_mm.vertices / 1000.0 @ R.T + pos_s                     # sensor frame, m
    P = V[:, :2] * 1000.0 / MPP                                     # input pixels from the centre
    cx, cy = (W - 1) / 2, (H - 1) / 2
    pts = np.stack([(P[:, 0] + cx) * S + S / 2, (P[:, 1] + cy) * S + S / 2], 1)
    mask = np.zeros((H * S, W * S), np.uint8)
    for f in mesh_mm.faces:
        cv2.fillPoly(mask, [np.round(pts[f] * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    return mask


def tile(x, mask, com_s, title):
    v = np.clip(x.astype(np.float32) / 1.5, 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (80, 255, 80), 1)
    c = np.array(com_s[:2]) * 1000.0 / MPP
    p = np.array([(c[0] + (W - 1) / 2) * S + S / 2, (c[1] + (H - 1) / 2) * S + S / 2])
    if 0 <= p[0] < W * S and 0 <= p[1] < H * S:
        q = tuple(np.round(p).astype(int))
        cv2.drawMarker(img, q, (60, 60, 255), cv2.MARKER_CROSS, 16, 2)
    else:
        ctr = np.array([W * S / 2, H * S / 2])
        d = (p - ctr) / np.linalg.norm(p - ctr)
        t = min((W * S / 2 - 8) / max(abs(d[0]), 1e-9), (H * S / 2 - 8) / max(abs(d[1]), 1e-9))
        a = ctr + d * t
        cv2.arrowedLine(img, tuple(np.round(a - d * 40).astype(int)), tuple(np.round(a).astype(int)), (60, 60, 255), 2,
                        tipLength=0.35)
    bar = np.zeros((22, W * S, 3), np.uint8)
    cv2.putText(bar, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def main():
    for name in objects.names():
        mesh = trimesh.load(os.path.join(objects.mesh_dir(name), f"{name}_visual.stl"), process=True)
        parts = objects.part_names(name)
        with h5py.File(os.path.join(ROOT, "dataset_multi", name, "test-0000.h5")) as f:
            st, cf, part, face, pose = (f["sample_type"][()], f["contact_fork"][()], f["part"][()], f["face"][()],
                                        f["pose_id"][()])
            picks = []
            for k in range(len(parts)):                       # 3 object presses per part, then table / hover
                ok = np.nonzero((st == 0) & cf & (part == k))[0]
                picks += list(ok[:3])
            picks += list(np.nonzero(st == 1)[0][:2]) + list(np.nonzero(st == 2)[0][:1])
            tiles = []
            for i in picks[:16]:
                m = footprint_px(mesh, f["fork_pos_s"][i], f["fork_quat_s"][i])
                lab = (["object", "table", "hover"][st[i]] + (f" {parts[part[i]]}" if part[i] >= 0 else "")
                       + f" pose {pose[i]}" + (f" face {'down' if face[i] == 0 else 'up'}" if face[i] >= 0 else "")
                       + f" {float(f['force_cmd'][i]):.1f}N")
                tiles.append(tile(f["X"][i], m, f["com_s"][i], f"#{i} {lab}"))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows_ = [np.concatenate(tiles[r:r + 4], 1) for r in range(0, len(tiles), 4)]
        head = np.zeros((28, rows_[0].shape[1], 3), np.uint8)
        cv2.putText(head, f"{name}: stored X (reconstructed depth, 0-1.5 mm) | green = object outline from the labels | "
                    "red = centre of mass (arrow: outside the pad)", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"label_gallery_{name}.png"), np.concatenate([head] + rows_, 0))
        print("wrote", f"label_gallery_{name}.png", len(picks), "presses")


if __name__ == "__main__":
    main()
