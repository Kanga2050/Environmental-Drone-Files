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
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)

def icp(A, B, tb, iters=20, rad=0.5):
    X = A.copy(); R = np.eye(3)
    for _ in range(iters):
        d, j = tb.query(X, workers=-1)
        m = d < rad
        if m.sum() < 200: break
        P, Q = X[m], B[j[m]]
        pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
        r = Vt.T @ U.T
        if np.linalg.det(r) < 0: Vt[-1] *= -1; r = Vt.T @ U.T
        X = (X - pc) @ r.T + qc; R = r @ R
    return X, np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", required=True)
ap.add_argument("--gap", type=int, default=3)
ap.add_argument("--sample", type=int, default=40000)
ap.add_argument("--pct", type=float, default=1.0)
ap.add_argument("--mult", type=float, default=10.0)
ap.add_argument("--min_near", type=int, default=2000)
ap.add_argument("--top", type=int, default=18)
a = ap.parse_args()
rng = np.random.default_rng(0)

fs = sorted(glob.glob(os.path.join(a.blocks, "block_*.ply")))
Bs, ids = [], []
for f in fs:
    P = read_ply(f)
    if len(P) < 1000: continue
    Bs.append(P if len(P) <= a.sample else P[rng.choice(len(P), a.sample, replace=False)])
    ids.append(int(os.path.basename(f)[6:10]))
S = np.concatenate(Bs)
scale = float(np.percentile(np.linalg.norm(S - np.median(S, 0), axis=1), 90))
thr = a.pct / 100.0 * scale
print(f"{len(Bs)} blocks, scene p90 radius {scale:.3f}, threshold {thr:.4f}")
trees = [cKDTree(P) for P in Bs]

rows = []
for x in range(len(Bs)):
    for y in range(x + 1, len(Bs)):
        if ids[y] - ids[x] < a.gap: continue
        d, _ = trees[y].query(Bs[x], workers=-1)
        near = d < thr * a.mult
        nn = int(near.sum())
        if nn < a.min_near: continue
        c = float((d[near] < thr).mean())
        rows.append((nn * (1 - c), ids[x], ids[y], nn, c, float(np.median(d[near]))))
rows.sort(reverse=True)
print(f"{len(rows)} co-located distant pairs (gap>={a.gap}, >={a.min_near} co-located pts)")
print(f"{'pair':>9} {'colocated':>10} {'coinc':>7} {'median':>8} {'afterICP':>9} {'rot':>6}  verdict")
for _, x, y, nn, c, md in rows[:a.top]:
    ix, iy = ids.index(x), ids.index(y)
    X, ang = icp(Bs[ix], Bs[iy], trees[iy])
    d2, _ = trees[iy].query(X, workers=-1)
    n2 = d2 < thr * a.mult
    c2 = float((d2[n2] < thr).mean()) if n2.sum() >= 100 else float("nan")
    v = "DUPLICATED, rigid offset" if c2 - c > 0.15 else ("coincident" if c > 0.6 else "no rigid fix")
    print(f"{x:3d}-{y:<5d} {nn:10,} {c*100:6.1f}% {md:8.4f} {c2*100:8.1f}% {ang:6.2f}  {v}")
