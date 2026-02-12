
import json
import os

import numpy as np

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))
H, W = meta["model_hw"]

F = []
for fid in meta["saved_frame_ids"]:
    p = os.path.join(ART, "frames", f"{fid:06d}.npz")
    if os.path.exists(p):
        d = np.load(p)
        E = d["extrinsic"].astype(np.float64)
        F.append((fid, E, d["intrinsic"].astype(np.float64), p))

pos = np.array([-E[:, :3].T @ E[:, 3] for _, E, _, _ in F])
ids = np.array([f[0] for f in F])
fwd = np.array([E[2, :3] for _, E, _, _ in F])
print(f"{len(F)} poses, path length "
      f"{np.linalg.norm(np.diff(pos, axis=0), axis=1).sum():.2f} u")
print(f"extent {np.ptp(pos, axis=0).round(3)} u")

D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
T = np.abs(ids[:, None] - ids[None])
scale = np.percentile(D, 90)
cand = np.argwhere((D < 0.10 * scale) & (T > 1500))
print(f"\nrevisit pairs (<{0.10 * scale:.3f} u apart, >1500 frames apart): "
      f"{len(cand) // 2}")
seen, rows = set(), []
for i, j in cand:
    if i >= j or (i, j) in seen:
        continue
    seen.add((i, j))
    ang = np.degrees(np.arccos(np.clip(fwd[i] @ fwd[j], -1, 1)))
    rows.append((ids[i], ids[j], D[i, j], ang))
rows.sort(key=lambda r: r[2])
print(f"{'frame A':>8s} {'frame B':>8s} {'gap (u)':>9s} {'view angle':>11s}")
for a, b, d_, ang in rows[:10]:
    print(f"{a:>8d} {b:>8d} {d_:>9.4f} {ang:>10.1f}deg")

fx = np.array([K[0, 0] for _, _, K, _ in F])
half = len(fx) // 2
print(f"\nfocal length: first half mean {fx[:half].mean():.1f}, "
      f"second half mean {fx[half:].mean():.1f} "
      f"({100 * (fx[half:].mean() / fx[:half].mean() - 1):+.1f}%)")
t = np.arange(len(fx))
sl = np.polyfit(t, fx, 1)[0]
print(f"trend {sl * len(fx):+.1f} px across the flight "
      f"({100 * sl * len(fx) / fx.mean():+.1f}% of mean)")

step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
print(f"\nper-keyframe step: median {np.median(step):.4f} u, "
      f"first half {np.median(step[:half]):.4f}, "
      f"second half {np.median(step[half:]):.4f} "
      f"({100 * (np.median(step[half:]) / np.median(step[:half]) - 1):+.1f}%)")
print(f"correlation fx vs step: r = {np.corrcoef(fx[:-1], step)[0, 1]:+.3f}")
