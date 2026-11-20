"""
record_real.py - grab GelSight Mini frames (+ reference frames) with the robot pose of every press.

    *** UNTESTED WITHOUT HARDWARE. Written against the GelSight Mini's standard video interface
    *** (a UVC camera, read with OpenCV), not run against a real sensor or robot.

    python record_real.py --out sessions/gauge_01                     # interactive: r = reference, Enter = press, q = quit
    python record_real.py --out sessions/noise_01 --noise-frames 600 --interval 1.0   # no-contact frames over time
    python record_real.py --list                                      # list video devices and their names

Assumptions (check them on the real device):
  * The Mini shows up as a UVC camera whose name contains "GelSight Mini" (Linux: /sys/class/video4linux/
    videoN/name). Use --device N to pick another index.
  * Stream geometry: the script asks for --width x --height (default 3280 x 2464, MJPG) and uses whatever
    the camera delivers. Like gsrobotics' resize_crop_mini, it crops 1/7 of the height / width on each
    side and resizes to 320 x 240, the size the simulator and the networks use. Check that the crop removes
    the black border of YOUR unit, and that the resulting image has the same handedness as the simulator
    (image column = sensor +x, image row = sensor +y): if not, flip with --flip-x / --flip-y.
  * Auto exposure / white balance must be off or settled: a changing exposure looks like illumination
    drift. The script discards --warmup frames and can lock exposure via CAP_PROP_* if the driver allows.

Robot pose: --robot manual (type it, or press Enter to skip), --robot file:PATH (a JSON {"tcp": [x, y, z,
rx, ry, rz], "force_n": F} kept up to date by your robot program), or --robot ur:HOST (ur_rtde's
RTDEReceiveInterface.getActualTCPPose(), untested). Units: metres and the robot's rotation vector.

Output (one directory per session):
  session.json          device, stream size, crop, settings, start time
  refs/ref_NNNN.png     reference (no-contact) frames, each the mean of --ref-avg frames, 320 x 240 RGB
  presses/press_NNNN.png  pressed frames, 320 x 240 RGB
  presses.jsonl         one line per press: frame, reference used, time, robot TCP pose, force, notes
  noise/frame_NNNN.png + noise.jsonl   with --noise-frames (no contact): frames over time for measure_noise.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

OUT_W, OUT_H = 320, 240
BANNER = "record_real.py: UNTESTED WITHOUT HARDWARE - check the assumptions in the docstring before trusting the frames"


def list_devices():
    out = []
    for p in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        n = int(p.split("video")[-1].split("/")[0])
        out.append((n, open(p).read().strip()))
    return out


def find_mini():
    for n, name in list_devices():
        if "gelsight" in name.lower() and "mini" in name.lower():
            return n
    return None


class MiniCamera:
    """GelSight Mini through OpenCV (UVC). crop + resize like gsrobotics' resize_crop_mini."""

    def __init__(self, device=None, width=3280, height=2464, warmup=30, flip_x=False, flip_y=False):
        if device is None:
            device = find_mini()
            if device is None:
                raise SystemExit("no video device named 'GelSight Mini' found; use --device N (see --list)")
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device, backend)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open video device {device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.device, self.flip_x, self.flip_y = device, flip_x, flip_y
        for _ in range(warmup):                       # let exposure / white balance settle
            self.cap.read()
        ok, f = self.cap.read()
        if not ok:
            raise SystemExit("camera returned no frame")
        self.raw_shape = f.shape
        self.info = dict(device=device, raw_width=int(f.shape[1]), raw_height=int(f.shape[0]),
                         fourcc="MJPG", requested=[width, height],
                         auto_exposure=self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                         exposure=self.cap.get(cv2.CAP_PROP_EXPOSURE))

    def lock_exposure(self):
        """Try to freeze exposure / white balance at their current values (driver dependent)."""
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)       # V4L2: 1 = manual
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    @staticmethod
    def crop_resize(f):
        h, w = f.shape[:2]
        by, bx = int(h / 7), int(w / 7)
        f = f[by:h - by, bx:w - bx]
        return cv2.resize(f, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

    def grab(self, n=1):
        """Mean of n frames, 320 x 240, RGB uint8."""
        acc = None
        for _ in range(n):
            ok, f = self.cap.read()
            if not ok:
                raise RuntimeError("frame grab failed")
            g = self.crop_resize(f).astype(np.float32)
            acc = g if acc is None else acc + g
        img = np.clip(np.round(acc / n), 0, 255).astype(np.uint8)[..., ::-1]     # BGR -> RGB
        if self.flip_x:
            img = img[:, ::-1]
        if self.flip_y:
            img = img[::-1]
        return np.ascontiguousarray(img)

    def close(self):
        self.cap.release()


class PoseSource:
    def get(self):
        return None


class ManualPose(PoseSource):
    def get(self):
        s = input("  robot TCP pose x y z rx ry rz [m, rad] (Enter = skip): ").strip()
        return dict(tcp=[float(v) for v in s.split()]) if s else None


class FilePose(PoseSource):
    def __init__(self, path):
        self.path = path

    def get(self):
        with open(self.path) as f:
            return json.load(f)


class URPose(PoseSource):
    """Universal Robots via ur_rtde (pip install ur_rtde). UNTESTED."""

    def __init__(self, host):
        import rtde_receive
        self.r = rtde_receive.RTDEReceiveInterface(host)

    def get(self):
        d = dict(tcp=list(self.r.getActualTCPPose()))
        try:
            d["tcp_force"] = list(self.r.getActualTCPForce())
        except Exception:
            pass
        return d


def make_pose_source(spec):
    if spec == "manual":
        return ManualPose()
    if spec.startswith("file:"):
        return FilePose(spec[5:])
    if spec.startswith("ur:"):
        return URPose(spec[3:])
    if spec == "none":
        return PoseSource()
    raise SystemExit(f"unknown --robot {spec}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="session directory")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--width", type=int, default=3280)
    ap.add_argument("--height", type=int, default=2464)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--ref-avg", type=int, default=10, help="frames averaged into a reference")
    ap.add_argument("--press-avg", type=int, default=1, help="frames averaged into a press frame")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--lock-exposure", action="store_true")
    ap.add_argument("--robot", default="manual", help="manual | none | file:PATH | ur:HOST")
    ap.add_argument("--noise-frames", type=int, default=0, help="record this many no-contact frames and exit")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between noise frames")
    args = ap.parse_args()
    print(BANNER, file=sys.stderr)
    if args.list:
        for n, name in list_devices():
            print(f"/dev/video{n}: {name}")
        return
    if not args.out:
        raise SystemExit("--out is required")
    os.makedirs(os.path.join(args.out, "refs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "presses"), exist_ok=True)
    cam = MiniCamera(args.device, args.width, args.height, args.warmup, args.flip_x, args.flip_y)
    if args.lock_exposure:
        cam.lock_exposure()
    session = dict(camera=cam.info, output_size=[OUT_W, OUT_H], crop="1/7 of each side, then INTER_AREA resize",
                   flip_x=args.flip_x, flip_y=args.flip_y, ref_avg=args.ref_avg, press_avg=args.press_avg,
                   robot=args.robot, started=time.strftime("%Y-%m-%dT%H:%M:%S"), untested_without_hardware=True)
    with open(os.path.join(args.out, "session.json"), "w") as f:
        json.dump(session, f, indent=1)
    if args.noise_frames:
        os.makedirs(os.path.join(args.out, "noise"), exist_ok=True)
        with open(os.path.join(args.out, "noise.jsonl"), "a") as log:
            t0 = time.time()
            for k in range(args.noise_frames):
                img = cam.grab(1)
                fn = f"noise/frame_{k:05d}.png"
                cv2.imwrite(os.path.join(args.out, fn), img[..., ::-1])
                log.write(json.dumps(dict(frame=fn, t=time.time() - t0)) + "\n")
                log.flush()
                time.sleep(max(0.0, args.interval - (time.time() - t0 - k * args.interval)))
        cam.close()
        return
    poses = make_pose_source(args.robot)
    ref_file, n_ref, n_press = None, 0, 0
    with open(os.path.join(args.out, "presses.jsonl"), "a") as log:
        while True:
            cmd = input("[r] reference  [Enter] press  [q] quit > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "r":
                img = cam.grab(args.ref_avg)
                ref_file = f"refs/ref_{n_ref:04d}.png"
                cv2.imwrite(os.path.join(args.out, ref_file), img[..., ::-1])
                n_ref += 1
                print(f"  saved {ref_file}")
                continue
            if ref_file is None:
                print("  take a reference first (no contact, r)")
                continue
            pose = poses.get()
            img = cam.grab(args.press_avg)
            fn = f"presses/press_{n_press:04d}.png"
            cv2.imwrite(os.path.join(args.out, fn), img[..., ::-1])
            note = input("  note (Enter = none): ").strip()
            log.write(json.dumps(dict(frame=fn, reference=ref_file, t=time.time(), robot=pose, note=note)) + "\n")
            log.flush()
            n_press += 1
            print(f"  saved {fn}")
    cam.close()


if __name__ == "__main__":
    main()
