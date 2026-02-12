
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
CONF, SUB, ITERS = 1.5, 9000, 40
meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))
rng = np.random.default_rng(0)

frames = {}
for fid in meta["saved_frame_ids"]:
    p = os.path.join(ART, "frames", f"{fid:06d}.npz")
    if os.path.exists(p):
        frames[fid] = p

def cloud(fid):
    d = np.load(frames[fid])
    depth = d["depth"].astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    conf, E, K = d["conf"].astype(np.float32), d["extrinsic"], d["intrinsic"]
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:2, 0:W:2]
    dd, cc = depth[ys, xs], conf[ys, xs]
    ok = (dd > 0) & (cc >= CONF) & np.isfinite(dd)
    xs, ys, dd = xs[ok], ys[ok], dd[ok]
    cam = np.stack([(xs - K[0, 2]) * dd / K[0, 0],
                    (ys - K[1, 2]) * dd / K[1, 1], dd], 1)
    world = (cam - E[:, 3]) @ E[:, :3]
    if len(world) > SUB:
        world = world[rng.choice(len(world), SUB, replace=False)]
    return world.astype(np.float64)

def umeyama(P, Q):

    mp, mq = P.mean(0), Q.mean(0)
    A, B = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(A.T @ B / len(P))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = (U @ D @ Vt).T
    var = (A ** 2).sum() / len(P)
    s = float((S * np.array([1, 1, d])).sum() / var) if var > 0 else 1.0
    return s, R, mq - s * R @ mp

def icp(src, dst, iters=ITERS):
    tree = cKDTree(dst)
    s_tot, R_tot, t_tot = 1.0, np.eye(3), np.zeros(3)
    cur = src.copy()
    thr = None
    for k in range(iters):
        dist, j = tree.query(cur, workers=-1)
        if thr is None:
            thr = max(np.percentile(dist, 80), 1e-3)
        keep = dist < thr
        if keep.sum() < 100:
            break
        s, R, t = umeyama(cur[keep], dst[j[keep]])
        cur = s * cur @ R.T + t
        s_tot, R_tot, t_tot = s * s_tot, R @ R_tot, s * R @ t_tot + t
        thr = max(thr * 0.9, 2.5 * np.median(dist[keep]))
    dist, _ = tree.query(cur, workers=-1)
    inl = dist < np.percentile(dist, 80)
    return s_tot, R_tot, t_tot, float(np.sqrt((dist[inl] ** 2).mean()))

ids = np.array(sorted(frames))
E = {f: np.load(frames[f])["extrinsic"].astype(np.float64) for f in ids}
pos = np.array([-E[f][:, :3].T @ E[f][:, 3] for f in ids])
fwd = np.array([E[f][2, :3] for f in ids])
D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
T = np.abs(ids[:, None] - ids[None])
ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))

pairs = []
for i, j in np.argwhere((D < 0.16) & (T > 1500) & (ang < 20)):
    if i < j:
        pairs.append((int(ids[i]), int(ids[j]), float(D[i, j]),
                      float(ang[i, j]), int(T[i, j])))
pairs.sort(key=lambda r: r[3])
pairs = pairs[:28]
print(f"{len(pairs)} revisit pairs (same place, <20 deg apart, >1500 frames)\n")
print(f"{'A':>6s} {'B':>6s} {'gap':>7s} {'ang':>6s} {'rms0':>8s} {'rms1':>8s} "
      f"{'scale':>7s} {'transl':>8s} {'rot':>6s}")

rows = []
for a, b, d_, an, dt in pairs:
    try:
        A, B = cloud(a), cloud(b)
    except Exception:
        continue
    if len(A) < 500 or len(B) < 500:
        continue
    rms0 = float(np.sqrt((cKDTree(A).query(B, workers=-1)[0] ** 2).mean()))
    s, R, t, rms1 = icp(B, A)
    rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    rows.append((a, b, d_, an, rms0, rms1, s, np.linalg.norm(t), rot, dt))
    print(f"{a:>6d} {b:>6d} {d_:>7.3f} {an:>5.1f} {rms0:>8.4f} {rms1:>8.4f} "
          f"{s:>7.4f} {np.linalg.norm(t):>8.4f} {rot:>5.1f}")

if rows:
    r = np.array([[x[4], x[5], x[6], x[7], x[8], x[9]] for x in rows])
    print(f"\n{len(rows)} pairs aligned")
    print(f"  residual before ICP : median {np.median(r[:, 0]):.4f} u")
    print(f"  residual after  ICP : median {np.median(r[:, 1]):.4f} u "
          f"({100 * (1 - np.median(r[:, 1]) / np.median(r[:, 0])):.0f}% removed)")
    print(f"  scale factor        : median {np.median(r[:, 2]):.4f}  "
          f"range {r[:, 2].min():.4f}-{r[:, 2].max():.4f}")
    print(f"  translation         : median {np.median(r[:, 3]):.4f} u")
    print(f"  rotation            : median {np.median(r[:, 4]):.2f} deg")
    dev = np.abs(r[:, 2] - 1)
    print(f"\n  |scale-1| median {dev.median() if hasattr(dev,'median') else np.median(dev):.4f}"
          f"  -> {100 * np.median(dev):.1f}% size error between revisits")
    print(f"  translation vs frame gap: r = {np.corrcoef(r[:, 5], r[:, 3])[0, 1]:+.3f}")
    print(f"  scale dev  vs frame gap: r = {np.corrcoef(r[:, 5], dev)[0, 1]:+.3f}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "icp_rows.npy"), r)
