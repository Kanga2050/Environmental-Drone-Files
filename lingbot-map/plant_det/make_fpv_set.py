"""Copy a YOLO dataset with FPV-style degradation (plant_det/fpv.py) applied to
a fraction of the training images (the rest hardlinked clean) and to all val."""
import argparse
import os
import shutil
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

from plant_det.fpv import degrade


def job(t):
    src, dst, do = t
    if os.path.exists(dst):
        return
    if not do:
        os.link(src, dst)
        return
    rng = np.random.default_rng(zlib.crc32(dst.encode()))
    im = cv2.imread(src)
    if im is not None:
        cv2.imwrite(dst, degrade(im, rng), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--frac", type=float, default=0.65)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    jobs = []
    for split, frac in (("train", a.frac), ("val", 1.0)):
        si, di = os.path.join(a.src, "images", split), os.path.join(a.dst, "images", split)
        os.makedirs(di, exist_ok=True)
        sl, dl = os.path.join(a.src, "labels", split), os.path.join(a.dst, "labels", split)
        if not os.path.exists(dl):
            shutil.copytree(sl, dl)
        for f in sorted(os.listdir(si)):
            do = (zlib.crc32(f.encode()) % 1000) / 1000 < frac
            jobs.append((os.path.join(si, f), os.path.join(di, f), do))
    with Pool(a.workers) as p:
        for i, _ in enumerate(p.imap_unordered(job, jobs, chunksize=64)):
            if i % 10000 == 0:
                print(i, len(jobs), flush=True)
    with open(os.path.join(a.dst, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.dst)}\ntrain: images/train\nval: images/val\n"
                f"names:\n  0: plant\n  1: tree\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
