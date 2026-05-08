
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

BASE = r"C:\Users\shaur\OneDrive\Desktop\Test"
A = os.path.join(BASE, "ab_average")
L = os.path.join(BASE, "ab_latest")

a = np.load(os.path.join(A, "cloud.npz"))
l = np.load(os.path.join(L, "cloud.npz"))
Pa, Ca = a["points"], a["colors"].astype(np.float32)
Pl, Cl = l["points"], l["colors"].astype(np.float32)
print(f"average: {len(Pa):,} points   latest: {len(Pl):,} points")
if len(Pa) != len(Pl):
    sys.exit("different point counts -- not a paired comparison")

d = np.linalg.norm(Pa - Pl, axis=1)
print(f"same-index positions agree to {np.percentile(d, 99) * 1000:.2f} mm "
      f"at p99 (voxel is 5.0 mm), so the rows are the same voxels")

rng = np.random.default_rng(0)
sample = rng.choice(len(Pa), 150_000, replace=False)

tree = cKDTree(Pa)
_, nn = tree.query(Pa[sample], k=9, workers=-1)
nn = nn[:, 1:]

def roughness(C):
    diff = C[sample][:, None, :] - C[nn]
    return np.abs(diff).mean(axis=(1, 2))

ra, rl = roughness(Ca), roughness(Cl)
print()
print(f"{'':34s} {'latest':>9s} {'average':>9s} {'change':>9s}")
print(f"{'local colour roughness (0-255)':34s} {rl.mean():9.2f} {ra.mean():9.2f} "
      f"{100 * (ra.mean() / rl.mean() - 1):+8.1f}%")
print(f"{'  median':34s} {np.median(rl):9.2f} {np.median(ra):9.2f} "
      f"{100 * (np.median(ra) / np.median(rl) - 1):+8.1f}%")
print(f"points where averaging is smoother: {int((ra < rl).sum()):,} of "
      f"{len(sample):,} ({100 * (ra < rl).mean():.1f}%)")

def to_lab_ab(C):
    import cv2
    return cv2.cvtColor((C.reshape(-1, 1, 3) / 255).astype(np.float32),
                        cv2.COLOR_RGB2LAB).reshape(-1, 3)[:, 1:]

ab_a, ab_l = to_lab_ab(Ca), to_lab_ab(Cl)

def rough_ab(X):
    return np.abs(X[sample][:, None, :] - X[nn]).mean(axis=(1, 2))

ca, cl = rough_ab(ab_a), rough_ab(ab_l)
print(f"{'local chroma roughness (a*b*)':34s} {cl.mean():9.3f} {ca.mean():9.3f} "
      f"{100 * (ca.mean() / cl.mean() - 1):+8.1f}%")

La = 0.299 * Ca[:, 0] + 0.587 * Ca[:, 1] + 0.114 * Ca[:, 2]
Ll = 0.299 * Cl[:, 0] + 0.587 * Cl[:, 1] + 0.114 * Cl[:, 2]
rla = np.abs(La[sample][:, None] - La[nn]).mean(1)
rll = np.abs(Ll[sample][:, None] - Ll[nn]).mean(1)
print(f"{'local luma roughness':34s} {rll.mean():9.2f} {rla.mean():9.2f} "
      f"{100 * (rla.mean() / rll.mean() - 1):+8.1f}%")

print()
delta = np.abs(Ca - Cl).mean(1)
print(f"per-voxel colour change, mean {delta.mean():.1f} / 255, "
      f"p90 {np.percentile(delta, 90):.1f}")
if "obs_weight" in a:
    w = a["obs_weight"]
    print(f"observations per voxel: median {np.median(w):.0f}, "
          f"p90 {np.percentile(w, 90):.0f}, max {w.max():.0f}")

    for lo, hi in ((0, 2), (2, 5), (5, 15), (15, 1e9)):
        m = (w[sample] >= lo) & (w[sample] < hi)
        if m.sum() > 500:
            print(f"  {lo:>3.0f}-{hi if hi < 1e9 else '+':>4} obs "
                  f"({m.sum():>6,} pts): roughness "
                  f"{rl[m].mean():6.2f} -> {ra[m].mean():6.2f} "
                  f"({100 * (ra[m].mean() / rl[m].mean() - 1):+.1f}%)")
