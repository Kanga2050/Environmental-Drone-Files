
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map")
from detect_plants import stack_crops, upscale_crop

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_final"

def chroma_noise(rgb):

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = []
    for c in (1, 2):
        ch = lab[:, :, c]
        out.append(float(np.mean(np.abs(ch - cv2.GaussianBlur(ch, (0, 0), 1.6)))))
    return float(np.mean(out))

def detail(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())

plants = json.load(open(os.path.join(ART, "plants.json"), encoding="utf-8"))["plants"]
crops_dir = os.path.join(ART, "crops")

by_inst = defaultdict(list)
for p in plants:
    for fid in p["frames"]:
        for f in sorted(glob.glob(os.path.join(crops_dir, f"{fid:06d}_*.jpg"))):
            by_inst[p["id"]].append(f)

rows = []
for pid, files in sorted(by_inst.items()):
    if len(files) < 2:
        continue
    imgs = []
    for f in files[:6]:
        im = cv2.imread(f)
        if im is not None:
            imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if len(imgs) < 2:
        continue

    imgs.sort(key=lambda a: -a.shape[0] * a.shape[1])
    ref = imgs[0]
    comp = stack_crops(imgs)
    if comp is None:
        continue
    rows.append((pid, len(imgs), chroma_noise(ref), chroma_noise(comp),
                 detail(ref), detail(comp), ref.shape[1], ref.shape[0]))

if not rows:
    sys.exit("no instance had two readable crops")

a = np.array([r[2:6] for r in rows], dtype=np.float64)
n = np.array([r[1] for r in rows])
w = np.array([r[6] for r in rows])
h = np.array([r[7] for r in rows])

print(f"{len(rows)} instances, {n.mean():.1f} crops stacked on average")
print(f"crop size: median {int(np.median(w))}x{int(np.median(h))} px, "
      f"smallest {w.min()}x{h.min()}")
print()
print(f"{'':22s} {'single':>9s} {'stacked':>9s} {'change':>9s}")
print(f"{'chroma noise (a/b)':22s} {a[:,0].mean():9.2f} {a[:,1].mean():9.2f} "
      f"{100*(a[:,1].mean()/a[:,0].mean()-1):+8.1f}%")
print(f"{'detail (laplacian var)':22s} {a[:,2].mean():9.1f} {a[:,3].mean():9.1f} "
      f"{100*(a[:,3].mean()/a[:,2].mean()-1):+8.1f}%")
print()
better = int((a[:, 1] < a[:, 0]).sum())
kept = int((a[:, 3] > 0.7 * a[:, 2]).sum())
print(f"noise reduced on {better}/{len(rows)} instances")
print(f"detail kept above 70% on {kept}/{len(rows)} instances")

u = np.array([[detail(cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB),
                                 (224, 224), interpolation=cv2.INTER_LINEAR)),
               detail(cv2.resize(upscale_crop(
                   cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), 224),
                   (224, 224), interpolation=cv2.INTER_AREA))]
              for files in by_inst.values() for f in files[:1]])
print()
print(f"resize to 224: bilinear direct {u[:,0].mean():7.1f} detail, "
      f"lanczos-up then area {u[:,1].mean():7.1f} "
      f"({100*(u[:,1].mean()/u[:,0].mean()-1):+.1f}%)")
