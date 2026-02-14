
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
OUT = os.path.dirname(os.path.abspath(__file__))
CONF, SUB, RADIUS = 1.5, 8000, 0.12
MAX_GAP_U, MIN_DT, MAX_ANG = 0.16, 1500, 25

meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))
frames = {fid: p for fid in meta["saved_frame_ids"]
          if os.path.exists(p := os.path.join(ART, "frames", f"{fid:06d}.npz"))}
_cache = {}

def cloud(fid, seed=0):
    if (fid, seed) in _cache:
        return _cache[fid, seed]
    d = np.load(frames[fid])
    depth = d["depth"].astype(np.float32)
    depth = depth[..., 0] if depth.ndim == 3 else depth
    conf, E, K = d["conf"].astype(np.float32), d["extrinsic"], d["intrinsic"]
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:2, 0:W:2]
    dd, cc = depth[ys, xs], conf[ys, xs]
    ok = (dd > 0) & (cc >= CONF) & np.isfinite(dd)
    xs, ys, dd = xs[ok], ys[ok], dd[ok]
    cam = np.stack([(xs - K[0, 2]) * dd / K[0, 0],
                    (ys - K[1, 2]) * dd / K[1, 1], dd], 1)
    w = ((cam - E[:, 3]) @ E[:, :3]).astype(np.float64)
    if len(w) > SUB:
        w = w[np.random.default_rng(seed).choice(len(w), SUB, replace=False)]
    _cache[fid, seed] = w
    if len(_cache) > 120:
        _cache.pop(next(iter(_cache)))
    return w

def rigid(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return R, mq - R @ mp

def icp(src, dst, iters=50):
    tree = cKDTree(dst)
    cur, R_t, t_t = src.copy(), np.eye(3), np.zeros(3)
    for _ in range(iters):
        dist, j = tree.query(cur, distance_upper_bound=RADIUS, workers=-1)
        keep = np.isfinite(dist)
        if keep.sum() < 300:
            break
        R, t = rigid(cur[keep], dst[j[keep]])
        cur = cur @ R.T + t
        R_t, t_t = R @ R_t, R @ t_t + t
    dist, _ = tree.query(cur, distance_upper_bound=RADIUS, workers=-1)
    ok = np.isfinite(dist)
    rms = float(np.sqrt((dist[ok] ** 2).mean())) if ok.any() else np.inf
    return R_t, t_t, rms, float(ok.mean())

def rms_of(src, dst):
    dist, _ = cKDTree(dst).query(src, distance_upper_bound=RADIUS, workers=-1)
    ok = np.isfinite(dist)
    return (float(np.sqrt((dist[ok] ** 2).mean())) if ok.any() else np.inf,
            float(ok.mean()))

ids = np.array(sorted(frames))
E = {f: np.load(frames[f])["extrinsic"].astype(np.float64) for f in ids}
pos = np.array([-E[f][:, :3].T @ E[f][:, 3] for f in ids])
fwd = np.array([E[f][2, :3] for f in ids])
D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
T = np.abs(ids[:, None] - ids[None])
ANG = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))
cand = [(int(ids[i]), int(ids[j]), float(D[i, j]), float(ANG[i, j]))
        for i, j in np.argwhere((D < MAX_GAP_U) & (T > MIN_DT) & (ANG < MAX_ANG))
        if i < j]
print(f"{len(cand)} candidate revisit pairs\n")

edges, rej = [], {}
for k, (a, b, gap, ang) in enumerate(sorted(cand, key=lambda r: r[3])):
    A, B = cloud(a), cloud(b)
    if len(A) < 1000 or len(B) < 1000:
        rej["too few points"] = rej.get("too few points", 0) + 1
        continue
    rms0, ovl0 = rms_of(B, A)
    if ovl0 < 0.30:
        rej["little overlap"] = rej.get("little overlap", 0) + 1
        continue
    R, t, rms, ovl = icp(B, A)
    rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if rms > 0.85 * rms0:
        rej["no improvement"] = rej.get("no improvement", 0) + 1
        continue
    if rot > ang + 10:
        rej["implausible rotation"] = rej.get("implausible rotation", 0) + 1
        continue

    R2, t2, _, _ = icp(A, B)
    comp = R2 @ R
    sym_rot = np.degrees(np.arccos(np.clip((np.trace(comp) - 1) / 2, -1, 1)))
    sym_t = np.linalg.norm(R2 @ t + t2)
    if sym_rot > 3.0 or sym_t > 0.03:
        rej["asymmetric"] = rej.get("asymmetric", 0) + 1
        continue

    R3, t3, _, _ = icp(cloud(b, 1), cloud(a, 1))
    if (np.linalg.norm(t3 - t) > 0.03 or
            np.degrees(np.arccos(np.clip((np.trace(R3 @ R.T) - 1) / 2, -1, 1))) > 3):
        rej["unstable"] = rej.get("unstable", 0) + 1
        continue
    edges.append({"a": a, "b": b, "gap_u": gap, "view_deg": ang,
                  "R": R.tolist(), "t": t.tolist(), "rms0": rms0, "rms": rms,
                  "overlap": ovl, "shift_u": float(np.linalg.norm(t)),
                  "rot_deg": float(rot)})
    print(f"  KEEP {a:>5d}<->{b:<5d} gap {gap:.3f} view {ang:4.1f}deg  "
          f"rms {rms0:.4f}->{rms:.4f}  shift {np.linalg.norm(t):.3f}u "
          f"rot {rot:4.1f}deg")

print(f"\n{len(edges)} constraints survived of {len(cand)} candidates")
for r, n in sorted(rej.items(), key=lambda kv: -kv[1]):
    print(f"   rejected, {r:<22s} {n}")
if edges:
    sh = np.array([e["shift_u"] for e in edges])
    ro = np.array([e["rot_deg"] for e in edges])
    print(f"\n  drift at a revisit: shift median {np.median(sh):.3f} u "
          f"(range {sh.min():.3f}-{sh.max():.3f})")
    print(f"                      rot   median {np.median(ro):.2f} deg")
    print(f"  residual {np.median([e['rms0'] for e in edges]):.4f} -> "
          f"{np.median([e['rms'] for e in edges]):.4f} u")
json.dump(edges, open(os.path.join(OUT, "loop_edges.json"), "w"), indent=1)
print(f"\nwrote loop_edges.json ({len(edges)} edges)")
