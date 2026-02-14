
import json
import os

import numpy as np
from scipy.spatial import cKDTree

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
OUT = os.path.dirname(os.path.abspath(__file__))
WIN, RADIUS, MIN_PTS = 200, 0.10, 4000

cloud = np.load(os.path.join(ART, "cloud.npz"))
P = cloud["points"].astype(np.float64)
FF = cloud["first_frame"].astype(np.int32)
print(f"{len(P):,} points, first_frame {FF.min()}..{FF.max()}")
meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))

frames = {fid: p for fid in meta["saved_frame_ids"]
          if os.path.exists(p := os.path.join(ART, "frames", f"{fid:06d}.npz"))}
ids = np.array(sorted(frames))
E = {f: np.load(frames[f])["extrinsic"].astype(np.float64) for f in ids}
pos = np.array([-E[f][:, :3].T @ E[f][:, 3] for f in ids])
fwd = np.array([E[f][2, :3] for f in ids])
D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
T = np.abs(ids[:, None] - ids[None])
ANG = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))

def chunk(t):
    m = (FF >= t - WIN) & (FF <= t + WIN)
    return P[m]

def rigid(A, B):
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return R, mb - R @ ma

def icp(src, dst, iters=50):
    tree = cKDTree(dst)
    cur, R_t, t_t = src.copy(), np.eye(3), np.zeros(3)
    for _ in range(iters):
        d, j = tree.query(cur, distance_upper_bound=RADIUS, workers=-1)
        k = np.isfinite(d)
        if k.sum() < 500:
            break
        R, t = rigid(cur[k], dst[j[k]])
        cur = cur @ R.T + t
        R_t, t_t = R @ R_t, R @ t_t + t
    d, _ = tree.query(cur, distance_upper_bound=RADIUS, workers=-1)
    k = np.isfinite(d)
    return R_t, t_t, (float(np.sqrt((d[k] ** 2).mean())) if k.any() else np.inf), float(k.mean())

cand = sorted(((int(ids[i]), int(ids[j]), float(D[i, j]), float(ANG[i, j]))
               for i, j in np.argwhere((D < 0.16) & (T > 1500) & (ANG < 25))
               if i < j), key=lambda r: r[3])

seen, pairs = set(), []
for a, b, g, an in cand:
    key = (a // 400, b // 400)
    if key in seen:
        continue
    seen.add(key)
    pairs.append((a, b, g, an))
print(f"{len(cand)} candidates -> {len(pairs)} distinct revisits\n")

edges, rej = [], {}
for a, b, gap, ang in pairs:
    A, B = chunk(a), chunk(b)
    if len(A) < MIN_PTS or len(B) < MIN_PTS:
        rej["thin chunk"] = rej.get("thin chunk", 0) + 1
        continue
    d0, _ = cKDTree(A).query(B, distance_upper_bound=RADIUS, workers=-1)
    ov0 = float(np.isfinite(d0).mean())
    if ov0 < 0.20:
        rej["little overlap"] = rej.get("little overlap", 0) + 1
        continue
    rms0 = float(np.sqrt((d0[np.isfinite(d0)] ** 2).mean()))
    R, t, rms, ovl = icp(B, A)
    rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    R2, t2, _, _ = icp(A, B)
    sym = np.degrees(np.arccos(np.clip((np.trace(R2 @ R) - 1) / 2, -1, 1)))
    symt = np.linalg.norm(R2 @ t + t2)
    ok = (rms < 0.9 * rms0) and rot < ang + 10 and sym < 3.0 and symt < 0.03
    tag = "KEEP" if ok else "drop"
    if not ok:
        why = ("no improvement" if rms >= 0.9 * rms0 else
               "implausible rotation" if rot >= ang + 10 else "asymmetric")
        rej[why] = rej.get(why, 0) + 1
    else:
        edges.append({"a": a, "b": b, "R": R.tolist(), "t": t.tolist(),
                      "shift_u": float(np.linalg.norm(t)), "rot_deg": float(rot),
                      "rms0": rms0, "rms": rms, "overlap": ovl,
                      "n_a": int(len(A)), "n_b": int(len(B))})
    print(f"  {tag} {a:>5d}<->{b:<5d} pts {len(A):>6,}/{len(B):<6,} ovl {ov0:.2f} "
          f"rms {rms0:.4f}->{rms:.4f} shift {np.linalg.norm(t):.3f}u "
          f"rot {rot:4.1f} sym {sym:4.1f}")

print(f"\n{len(edges)} constraints survived of {len(pairs)}")
for r, n in sorted(rej.items(), key=lambda kv: -kv[1]):
    print(f"   rejected, {r:<22s} {n}")
if edges:
    sh = np.array([e["shift_u"] for e in edges])
    print(f"\n  drift at a revisit: shift median {np.median(sh):.3f} u "
          f"(range {sh.min():.3f}-{sh.max():.3f})")
    print(f"  residual {np.median([e['rms0'] for e in edges]):.4f} -> "
          f"{np.median([e['rms'] for e in edges]):.4f} u")
json.dump(edges, open(os.path.join(OUT, "loop_edges.json"), "w"), indent=1)
print(f"\nwrote loop_edges.json ({len(edges)} edges)")
