
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
CONF, SUB, RADIUS = 1.5, 9000, 0.12
meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))
rng = np.random.default_rng(0)
frames = {fid: p for fid in meta["saved_frame_ids"]
          if os.path.exists(p := os.path.join(ART, "frames", f"{fid:06d}.npz"))}

def cloud(fid):
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
    return w[rng.choice(len(w), SUB, replace=False)] if len(w) > SUB else w

def fit(P, Q, allow_scale):
    mp, mq = P.mean(0), Q.mean(0)
    A, B = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(A.T @ B / len(P))
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1, 1, d]) @ Vt).T
    s = 1.0
    if allow_scale:
        var = (A ** 2).sum() / len(P)
        if var > 0:
            s = float(np.clip((S * [1, 1, d]).sum() / var, 0.9, 1.111))
    return s, R, mq - s * R @ mp

def icp(src, dst, allow_scale, iters=60):
    tree = cKDTree(dst)
    cur = src.copy()
    s_t, R_t, t_t = 1.0, np.eye(3), np.zeros(3)
    for _ in range(iters):
        dist, j = tree.query(cur, distance_upper_bound=RADIUS, workers=-1)
        keep = np.isfinite(dist)
        if keep.sum() < 200:
            break
        s, R, t = fit(cur[keep], dst[j[keep]], allow_scale)
        cur = s * cur @ R.T + t
        s_t, R_t, t_t = s * s_t, R @ R_t, s * R @ t_t + t
    dist, _ = tree.query(cur, distance_upper_bound=RADIUS, workers=-1)
    ok = np.isfinite(dist)
    rms = float(np.sqrt((dist[ok] ** 2).mean())) if ok.sum() else np.nan
    return s_t, R_t, t_t, rms, float(ok.mean())

ids = np.array(sorted(frames))
E = {f: np.load(frames[f])["extrinsic"].astype(np.float64) for f in ids}
pos = np.array([-E[f][:, :3].T @ E[f][:, 3] for f in ids])
fwd = np.array([E[f][2, :3] for f in ids])
D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
T = np.abs(ids[:, None] - ids[None])
ANG = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))

pairs = sorted(((int(ids[i]), int(ids[j]), float(D[i, j]), float(ANG[i, j]),
                 int(T[i, j]))
                for i, j in np.argwhere((D < 0.16) & (T > 1500) & (ANG < 15))
                if i < j), key=lambda r: r[3])[:24]

print(f"{len(pairs)} candidate revisit pairs\n")
print(f"{'A':>6s} {'B':>6s} {'gap':>6s} {'ang':>5s} {'rms0':>7s} "
      f"{'rigid':>7s} {'simil':>7s} {'scale':>7s} {'rot':>5s} {'ovl':>5s}  note")
good = []
for a, b, d_, an, dt in pairs:
    A, B = cloud(a), cloud(b)
    if len(A) < 800 or len(B) < 800:
        continue
    dist0, _ = cKDTree(A).query(B, distance_upper_bound=RADIUS, workers=-1)
    m0 = np.isfinite(dist0)
    if m0.mean() < 0.25:
        continue
    rms0 = float(np.sqrt((dist0[m0] ** 2).mean()))
    _, Rr, tr, rms_r, ovl_r = icp(B, A, False)
    s, Rs, ts, rms_s, ovl_s = icp(B, A, True)
    rot = np.degrees(np.arccos(np.clip((np.trace(Rs) - 1) / 2, -1, 1)))
    bad = rot > an + 12
    note = "ICP failed" if bad else ""
    print(f"{a:>6d} {b:>6d} {d_:>6.3f} {an:>5.1f} {rms0:>7.4f} {rms_r:>7.4f} "
          f"{rms_s:>7.4f} {s:>7.4f} {rot:>5.1f} {ovl_s:>5.2f}  {note}")
    if not bad:
        good.append((rms0, rms_r, rms_s, s, np.linalg.norm(ts), rot, dt))

if good:
    g = np.array(good)
    print(f"\n{len(g)} pairs converged sanely (rotation consistent with the cameras)")
    print(f"  residual, no alignment   median {np.median(g[:, 0]):.4f} u")
    print(f"  residual, rigid fit      median {np.median(g[:, 1]):.4f} u")
    print(f"  residual, + free scale   median {np.median(g[:, 2]):.4f} u")
    gain = 100 * (1 - np.median(g[:, 2]) / np.median(g[:, 1]))
    print(f"  allowing scale buys      {gain:+.1f}% on the residual")
    print(f"  scale factor             median {np.median(g[:, 3]):.4f}  "
          f"range {g[:, 3].min():.4f}-{g[:, 3].max():.4f}")
    print(f"  translation needed       median {np.median(g[:, 4]):.4f} u")
    print(f"  rotation needed          median {np.median(g[:, 5]):.2f} deg")
    print(f"\n  -> between revisits the map is misplaced by "
          f"{np.median(g[:, 4]):.3f} u and mis-sized by "
          f"{100 * abs(np.median(g[:, 3]) - 1):.1f}%")
