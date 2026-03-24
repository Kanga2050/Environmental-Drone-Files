
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
import detect_plants as dp

OUT = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", default=r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_final")
ap.add_argument("--key", required=True)
ap.add_argument("--views", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

art = args.artifacts
plants = json.load(open(os.path.join(art, "plants.json"), encoding="utf-8"))["plants"]
clf = dp.PlantNetClassifier(args.key)
print("Every crop below is uploaded to Pl@ntNet.\n")

by_frame = defaultdict(list)
for f in glob.glob(os.path.join(art, "crops", "*.jpg")):
    by_frame[int(os.path.basename(f).split("_")[0])].append(f)

rows = []
todo = plants[:args.limit] if args.limit else plants
for p in todo:
    pool = []
    for fid in p["frames"]:
        for f in sorted(by_frame.get(fid, [])):
            im = cv2.imread(f)
            if im is not None:
                pool.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if len(pool) < 2:
        continue
    pool.sort(key=lambda a: -dp.crop_quality(a))
    sel = pool[:args.views]

    single = clf(dp.foliage_subcrop(sel[0]), topk=3)
    fused = clf.classify_many(sel, topk=3)
    h1 = clf.classify_many(pool[0::2][:args.views], topk=1)
    h2 = clf.classify_many(pool[1::2][:args.views], topk=1)

    row = {
        "id": p["id"], "views": len(pool),
        "single": single[0][0] if single else None,
        "single_score": single[0][2] if single else 0.0,
        "fused": fused[0][0] if fused else None,
        "fused_common": fused[0][1] if fused else "",
        "fused_score": fused[0][2] if fused else 0.0,
        "consistent": bool(h1 and h2 and h1[0][0] == h2[0][0]),
        "half_names": [h1[0][0] if h1 else None, h2[0][0] if h2 else None],
    }
    rows.append(row)
    print(f"#{row['id']:<3d} {row['views']:>3d} views | "
          f"single {str(row['single'])[:24]:<24s} {row['single_score']:.3f} | "
          f"fused {str(row['fused'])[:24]:<24s} {row['fused_score']:.3f}"
          f"{'  [consistent]' if row['consistent'] else ''}")

if not rows:
    sys.exit("nothing classified")

s = np.array([r["single_score"] for r in rows])
f = np.array([r["fused_score"] for r in rows])
print(f"\n{len(rows)} instances")
print(f"{'':30s} {'single':>9s} {'fused':>9s}")
print(f"{'top-1 score, mean':30s} {s.mean():9.3f} {f.mean():9.3f}")
print(f"{'top-1 score, median':30s} {np.median(s):9.3f} {np.median(f):9.3f}")
for thr in (0.20, 0.30, 0.50):
    print(f"{'scoring >= ' + format(thr, '.2f'):30s} "
          f"{int((s >= thr).sum()):9d} {int((f >= thr).sum()):9d}")
print(f"{'split-half consistent (fused)':30s} {'':9s} "
      f"{sum(r['consistent'] for r in rows):>4d}/{len(rows)}")

gate = [r for r in rows if r["fused_score"] >= 0.20 and r["consistent"]
        and r["views"] >= 5]
print(f"\nnamed under score>=0.20 AND split-half consistent AND >=5 views: "
      f"{len(gate)} of {len(rows)}")
for r in sorted(gate, key=lambda r: -r["fused_score"]):
    print(f"   #{r['id']:<3d} {r['fused']} ({r['fused_common']})  "
          f"{r['fused_score']:.3f}, {r['views']} views")

json.dump(rows, open(os.path.join(OUT, "plantnet_eval.json"), "w"), indent=1)
print("\nwrote plantnet_eval.json")
