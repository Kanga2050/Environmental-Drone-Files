
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
import detect_plants as dp

OUT = os.path.dirname(os.path.abspath(__file__))

def old_quality(c):
    if c.size == 0:
        return 0.0
    g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    veg = dp.vegetation_fraction(c)
    det = min(float(cv2.Laplacian(g, cv2.CV_64F).var()) / 300.0, 1.0)
    size = min(float(c.shape[0] * c.shape[1]) / (400.0 * 400.0), 1.0)
    sky = float((g > 205).mean())
    return float(max(veg, .02) * max(det, .05) * max(size, .05) * (1 - sky))

ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", default=r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_final")
ap.add_argument("--views", type=int, default=5)
ap.add_argument("--plantnet", default=None)
ap.add_argument("--arms", default="ABCD")
ap.add_argument("--models_dir", default=r"C:\Users\shaur\AppData\Local\lingbot-map\models")
args = ap.parse_args()

art = args.artifacts
meta = json.load(open(os.path.join(art, "meta.json"), encoding="utf-8"))
model_hw = tuple(meta["model_hw"])
plants = json.load(open(os.path.join(art, "plants.json"), encoding="utf-8"))["plants"]
pts = np.load(os.path.join(art, "plant_points.npz"))

if args.plantnet:
    clf = dp.PlantNetClassifier(args.plantnet)
    tag = "plantnet"
    print("Pl@ntNet: crops below are uploaded to an external service.\n")
else:
    clf = dp.BioClipClassifier(os.path.join(args.models_dir, "bioclip"), "cuda",
                               dp.DEFAULT_SPECIES)
    tag = "bioclip"

by_frame = defaultdict(list)
for f in glob.glob(os.path.join(art, "crops", "*.jpg")):
    by_frame[int(os.path.basename(f).split("_")[0])].append(f)

calls = {"n": 0}

def top1(views, fused):

    if not views:
        return None, 0.0
    if fused:
        calls["n"] += 1
        r = clf.classify_many(views, topk=3)
        return (r[0][0], r[0][2]) if r else (None, 0.0)
    votes, best = Counter(), 0.0
    for v in views:
        calls["n"] += 1
        r = clf(v, topk=3)
        if r:
            votes[r[0][0]] += 1
            best = max(best, r[0][2])
    return (votes.most_common(1)[0][0] if votes else None), best

ARMS = {
    "A": ("detector crops, old rank, vote", "det", old_quality, False),
    "B": ("detector crops, old rank, fused", "det", old_quality, True),
    "C": ("detector crops, sharp rank, fused", "det", dp.crop_quality, True),
    "D": ("3D crops, sharp rank, fused", "3d", dp.crop_quality, True),
}

results = {k: {"names": [], "consistent": 0, "n": 0, "score": []}
           for k in args.arms}

for p in plants:
    pooled = pts.get(f"pts_{p['id']}")
    pooled = np.asarray(pooled, np.float32) if pooled is not None else None
    det_crops, td_crops = [], []
    for fid in p["frames"]:
        for f in sorted(by_frame.get(fid, [])):
            im = cv2.imread(f)
            if im is not None:
                det_crops.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if pooled is not None and len(pooled):
            c = dp.crop_from_instance(art, fid, pooled, model_hw)
            if c is not None:
                td_crops.append(c)

    for key in args.arms:
        label, src, rank, fused = ARMS[key]
        pool = det_crops if src == "det" else td_crops
        if len(pool) < 2:
            continue
        pool = sorted(pool, key=lambda a: -rank(a))
        sel = pool[:args.views]
        if src == "det":
            sel = [dp.foliage_subcrop(a) for a in sel] if key in ("A", "B") else sel
        name, score = top1(sel, fused)
        if name is None:
            continue

        h1, h2 = pool[0::2][:args.views], pool[1::2][:args.views]
        n1, _ = top1(h1, fused)
        n2, _ = top1(h2, fused)
        r = results[key]
        r["names"].append(name)
        r["score"].append(score)
        r["n"] += 1
        r["consistent"] += int(bool(n1) and n1 == n2)

print(f"{len(plants)} instances, classifier {tag}, {calls['n']} calls\n")
print(f"{'arm':4s} {'what':36s} {'consistent':>11s} {'distinct':>9s} {'mean score':>11s}")
for key in args.arms:
    r = results[key]
    if not r["n"]:
        continue
    print(f"{key:4s} {ARMS[key][0]:36s} "
          f"{r['consistent']:>5d}/{r['n']:<5d} "
          f"{len(set(r['names'])):>9d} {np.mean(r['score']):>11.3f}")

json.dump({k: {kk: (vv if not isinstance(vv, list) else vv)
               for kk, vv in v.items()} for k, v in results.items()},
          open(os.path.join(OUT, f"arms_{tag}.json"), "w"), indent=1)
