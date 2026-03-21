
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_final"
OUT = os.path.dirname(os.path.abspath(__file__))

def sharpness(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    contrast = g.std() ** 2 + 1e-6
    return lap, lap / contrast

plants = json.load(open(os.path.join(ART, "plants.json"), encoding="utf-8"))["plants"]
crops_dir = os.path.join(ART, "crops")

by_inst = defaultdict(list)
for p in plants:
    for fid in p["frames"]:
        for f in sorted(glob.glob(os.path.join(crops_dir, f"{fid:06d}_*.jpg"))):
            by_inst[p["id"]].append(f)

gain, rows = [], []
best_tiles, used_tiles = [], []
for p in plants:
    files = by_inst.get(p["id"], [])
    if len(files) < 2:
        continue
    used_path = os.path.join(crops_dir, p.get("crop", ""))
    scores = {}
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        scores[f] = sharpness(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    if used_path not in scores:
        continue
    best = max(scores, key=lambda f: scores[f][1])
    u = scores[used_path][1]
    b = scores[best][1]
    rows.append((p["id"], len(files), u, b, best != used_path))
    gain.append(b / max(u, 1e-6))
    if len(best_tiles) < 8 and best != used_path and b > 1.4 * u:
        def tile(path, label, colour):
            im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            h, w = im.shape[:2]
            s = 230.0 / max(h, w)
            im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
            out = np.zeros((230, 230, 3), np.uint8)
            out[:im.shape[0], :im.shape[1]] = im
            cv2.putText(out, label, (4, 16), 0, 0.42, colour, 1)
            return out
        used_tiles.append(tile(used_path, f"#{p['id']} used {u:.2f}", (90, 160, 255)))
        best_tiles.append(tile(best, f"#{p['id']} best {b:.2f}", (120, 255, 140)))

g = np.array(gain)
print(f"{len(rows)} instances with more than one view")
print(f"a sharper view exists for {int(sum(r[4] for r in rows))} of them")
print(f"sharpness ratio best/used: median {np.median(g):.2f}x  "
      f"mean {g.mean():.2f}x  p90 {np.percentile(g, 90):.2f}x")
print(f"instances where the best view is >1.5x sharper: {int((g > 1.5).sum())}")
print(f"instances where the best view is >2x sharper:   {int((g > 2.0).sum())}")
print()
print("worst offenders (used vs best, normalised sharpness):")
for pid, n, u, b, diff in sorted(rows, key=lambda r: -(r[3] / max(r[2], 1e-6)))[:8]:
    print(f"  #{pid:<3d} {n:>2d} views   used {u:6.2f}   best {b:6.2f}   "
          f"{b / max(u, 1e-6):5.2f}x")

if best_tiles:
    cv2.imwrite(os.path.join(OUT, "sharpness_pairs.jpg"),
                cv2.cvtColor(np.vstack([np.hstack(used_tiles),
                                        np.hstack(best_tiles)]), cv2.COLOR_RGB2BGR))
    print("\nwrote sharpness_pairs.jpg (top row: currently used, bottom: sharpest)")
