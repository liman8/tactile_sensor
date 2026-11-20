"""A gif for every episode of videos/mini_robust_s0/ (search_video.py writes one only for the first), made from the
mp4 with the same settings: every second frame, half size, 2000 / fps ms per frame. Decoded from the mp4, so with
its compression. Run from the project root: python results_centre/videos/mini_robust_s0_gif/make_gifs.py"""
import glob
import os

import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "mini_robust_s0")
for mp4 in sorted(glob.glob(os.path.join(SRC, "episode_*.mp4"))):
    out = os.path.join(HERE, os.path.basename(mp4).replace(".mp4", ".gif"))
    if os.path.exists(out):
        print("exists, skipped:", out)
        continue
    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    h, w = frames[0].shape[:2]
    ims = [Image.fromarray(cv2.resize(f, (w // 2, h // 2), interpolation=cv2.INTER_AREA)[..., ::-1]) for f in frames[::2]]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(2000 / fps), loop=0, optimize=True)
    print(f"{os.path.basename(out)}: {len(frames)} frames -> {len(ims)}, {os.path.getsize(out) / 1e6:.1f} MB")
