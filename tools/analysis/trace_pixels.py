
import json
import os
import sys

import numpy as np

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
VOXEL = 0.005
DOWNSAMPLE = 4
CONF_THR = 1.5

meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))
H, W = meta["model_hw"]
print(f"model grid {H}x{W} = {H * W:,} pixels per frame")
print(f"downsample {DOWNSAMPLE} -> {len(range(0, H, DOWNSAMPLE))}x"
      f"{len(range(0, W, DOWNSAMPLE))} = "
      f"{len(range(0, H, DOWNSAMPLE)) * len(range(0, W, DOWNSAMPLE)):,} "
      f"candidate points per frame\n")

def keys(P):
    q = np.floor(P / VOXEL).astype(np.int64) + (1 << 20)
    return ((q[:, 0] & 0x1FFFFF) | ((q[:, 1] & 0x1FFFFF) << 21)
            | ((q[:, 2] & 0x1FFFFF) << 42))

rows = []
for fid in meta["saved_frame_ids"][20:80:6]:
    f = os.path.join(ART, "frames", f"{fid:06d}.npz")
    if not os.path.exists(f):
        continue
    d = np.load(f)
    depth = d["depth"].astype(np.float32)
    conf = d["conf"].astype(np.float32)
    world = unproject_depth_map_to_point_map(
        depth[None], d["extrinsic"][None], d["intrinsic"][None])[0]

    w = world[::DOWNSAMPLE, ::DOWNSAMPLE].reshape(-1, 3)
    c = conf[::DOWNSAMPLE, ::DOWNSAMPLE].reshape(-1)
    keep = np.isfinite(w).all(1) & (c > CONF_THR)
    w = w[keep]
    if not len(w):
        continue
    k = keys(w)
    uniq = len(np.unique(k))
    rows.append((fid, keep.sum(), uniq, len(w) / max(uniq, 1), c[keep].mean()))

print(f"{'frame':>7s} {'pixels kept':>12s} {'distinct voxels':>16s} "
      f"{'pixels/voxel':>13s} {'mean conf':>10s}")
for fid, n, u, r, mc in rows:
    print(f"{fid:>7d} {n:>12,} {u:>16,} {r:>13.2f} {mc:>10.2f}")
a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
print(f"{'mean':>7s} {a[:,0].mean():>12,.0f} {a[:,1].mean():>16,.0f} "
      f"{a[:,2].mean():>13.2f} {a[:,3].mean():>10.2f}")

cloud = np.load(os.path.join(ART, "cloud.npz"))
w = cloud["obs_weight"]
print(f"\nobs_weight in cloud.npz: min {w.min():.2f}  median {np.median(w):.2f}  "
      f"p90 {np.percentile(w, 90):.1f}  max {w.max():.0f}")
print(f"values below 1.0: {100 * (w < 1).mean():.1f}%   "
      f"below the confidence threshold {CONF_THR}: {100 * (w < CONF_THR).mean():.1f}%")
mean_conf = a[:, 3].mean()
print(f"\nmean per-pixel confidence is {mean_conf:.2f}, and a voxel's weight is the")
print(f"SUM of the confidences that hit it, so an implied contribution count is")
print(f"weight / {mean_conf:.2f}:")
est = w / mean_conf
print(f"  implied contributions: median {np.median(est):.1f}  "
      f"p90 {np.percentile(est, 90):.1f}  max {est.max():.0f}")
for lo, hi, label in ((0, 1.5, "under one full observation"),
                      (1.5, 2.5, "about 1-2"), (2.5, 5, "about 2-5"),
                      (5, 1e9, "5 or more")):
    m = (est >= lo) & (est < hi)
    print(f"  {label:<28s} {100 * m.mean():5.1f}% of voxels")
