"""Compare plant assignments on the real map without ground truth.

For each plant, using its own 2D detections:
  box_precision  share of its points that project inside its detection box
                 (in views where the point is in the image); low = leakage
  seed_recall    share of its lifted seed points with an assigned point nearby
  ground_leak    share of its points lying on the ground plane
Old = radius rule around lifted points (view_map3d's default before),
new = plant_seg.npz.

    python -m plant_seg.eval_yard --artifacts ../artifacts_yard_avg
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from . import prompt as P
from .yard import Yard


def plant_scores(y, idx, views, seeds, radius):
    if len(idx) == 0:
        return {"n": 0, "box_precision": 0.0, "seed_recall": 0.0, "ground_leak": 0.0}
    xyz = y.xyz[idx]
    prec = []
    for v in views:
        u, vv, z = P.project(xyz, v["E"], v["K"])
        h, w = v["depth"].shape
        on = (z > 0) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        if on.sum() < 10:
            continue
        x0, y0, x1, y1 = v["box"]
        inb = on & (u >= x0) & (u <= x1) & (vv >= y0) & (vv <= y1)
        prec.append(inb.sum() / on.sum())
    d, _ = cKDTree(xyz).query(seeds, k=1, workers=-1)
    return {"n": int(len(idx)),
            "box_precision": float(np.mean(prec)) if prec else float("nan"),
            "seed_recall": float((d <= radius).mean()),
            "ground_leak": float((xyz[:, 2] < 2 * y.s.voxel).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--seg", default="plant_seg.npz")
    args = ap.parse_args()
    from view_map3d import MapScene
    s = MapScene(args.artifacts, use_seg=False)
    y = Yard(s)
    views, seeds = y.views()
    seg = np.load(os.path.join(args.artifacts, args.seg))
    new_tags = seg["tags"][s.order]
    radius = 2.5 * s.voxel
    rows = []
    for i, p in enumerate(s.plants):
        pid = p["id"]
        if pid not in seeds or not views.get(pid):
            continue
        old = plant_scores(y, s.member.get(pid, np.zeros(0, np.int64)), views[pid],
                           seeds[pid], radius)
        new = plant_scores(y, np.flatnonzero(new_tags == i), views[pid], seeds[pid], radius)
        rows.append({"id": pid, "category": p["category"], "views": len(views[pid]),
                     "old": old, "new": new})
        print(f"#{pid:2d} {p['category']:<20} n {old['n']:>7,} -> {new['n']:>7,}   "
              f"box prec {old['box_precision']:.2f} -> {new['box_precision']:.2f}   "
              f"seed recall {old['seed_recall']:.2f} -> {new['seed_recall']:.2f}   "
              f"ground {old['ground_leak']:.2f} -> {new['ground_leak']:.2f}")
    agg = {}
    for k in ("box_precision", "seed_recall", "ground_leak"):
        for which in ("old", "new"):
            vals = [r[which][k] for r in rows if np.isfinite(r[which][k])]
            agg[f"{which}_{k}"] = float(np.mean(vals))
    print(json.dumps(agg, indent=1))
    with open(os.path.join(args.artifacts, args.seg.replace(".npz", "_eval.json")), "w") as f:
        json.dump({"summary": agg, "plants": rows}, f, indent=1)


if __name__ == "__main__":
    main()
