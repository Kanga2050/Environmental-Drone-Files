import argparse, glob, os
import numpy as np
from scipy.spatial import cKDTree

def read_ply(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            s = f.readline().decode("ascii", "ignore").strip()
            if s.startswith("element vertex"): n = int(s.split()[-1])
            elif s.startswith("property"): props.append(s.split()[-1])
            elif s == "end_header": break
        dt = np.dtype([(q, "<f4") if q in ("x","y","z","nx","ny","nz") else (q,"u1") for q in props])
        d = np.frombuffer(f.read(n * dt.itemsize), dt)
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32)

def sub(P, k, rng):
    return P if len(P) <= k else P[rng.choice(len(P), k, replace=False)]

def inbox(P, Q, pad):
    lo, hi = Q.min(0) - pad, Q.max(0) + pad
    return P[np.all((P >= lo) & (P <= hi), 1)]

def coincide(P, Q, thr, mult=10.0):
    if len(P) < 100 or len(Q) < 100: return None, None, 0
    d, _ = cKDTree(Q).query(P, workers=-1)
    near = d < thr * mult
    if near.sum() < 100: return None, None, int(near.sum())
    return float((d[near] < thr).mean()), float(np.median(d[near])), int(near.sum())

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", required=True)
ap.add_argument("--repair", default=None)
ap.add_argument("--merged", default=None)
ap.add_argument("--seams", default="11-12,15-16,16-17,22-23")
ap.add_argument("--all", action="store_true")
ap.add_argument("--pct", type=float, default=1.0)
ap.add_argument("--sample", type=int, default=150000)
ap.add_argument("--scale", type=float, default=0.0)
a = ap.parse_args()
rng = np.random.default_rng(0)

scale = a.scale
if scale <= 0:
    S = [sub(read_ply(f), 20000, rng) for f in sorted(glob.glob(os.path.join(a.blocks, "block_*.ply")))]
    S = np.concatenate([q for q in S if len(q)])
    scale = float(np.percentile(np.linalg.norm(S - np.median(S, 0), axis=1), 90))
    print(f"scale from {len(S):,} sampled block points (raw GLOMAP units)")
thr = a.pct / 100.0 * scale
print(f"scene p90 radius {scale:.3f}, threshold {a.pct}% = {thr:.4f}")

nb = len(glob.glob(os.path.join(a.blocks, "block_*.ply")))
seams = [(i, i + 1) for i in range(nb - 1)] if a.all else \
        [tuple(int(v) for v in s.split("-")) for s in a.seams.split(",")]

cache = {}
def blk(i):
    if i not in cache:
        p = os.path.join(a.blocks, f"block_{i:04d}.ply")
        cache[i] = sub(read_ply(p), a.sample, rng) if os.path.exists(p) else np.empty((0, 3), np.float32)
    return cache[i]

print(f"{'seam':>7} {'A<->B':>9} {'medAB':>8} {'nOverlap':>9} {'R->A':>9} {'R->B':>9} {'Rpts':>9}  verdict")
for x, y in seams:
    A, B = blk(x), blk(y)
    fab, mab, nab = coincide(A, B, thr)
    ra = rb = None; nr = 0
    if a.repair:
        rp = os.path.join(a.repair, f"repair_{x:02d}_{y:02d}.ply")
        if os.path.exists(rp):
            R = read_ply(rp); nr = len(R); Rs = sub(R, a.sample, rng)
            ra, _, na = coincide(Rs, A, thr)
            rb, _, nbb = coincide(Rs, B, thr)
    f = lambda v: "     n/a" if v is None else f"{v*100:7.1f}%"
    m = "     n/a" if mab is None else f"{mab:8.4f}"
    if ra is not None and rb is not None:
        verdict = "bridged" if min(ra, rb) > 0.5 else ("partial" if min(ra, rb) > 0.15 else "NOT bridged")
    else:
        verdict = "aligned" if fab is not None and fab > 0.5 else "broken"
    print(f"{x:3d}-{y:<3d} {f(fab)} {m} {nab:9,} {f(ra)} {f(rb)} {nr:9,}  {verdict}")
