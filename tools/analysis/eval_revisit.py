import json
import os
import sys

import numpy as np

GT = json.load(open(r"C:\Users\shaur\AppData\Local\sfm-work\revisit_gt.json"))


def poses(art):
    meta = json.load(open(os.path.join(art, "meta.json"), encoding="utf-8"))
    src = meta.get("source_frame_ids") or []
    out = {}
    for fid in meta.get("saved_frame_ids", []):
        p = os.path.join(art, "frames", f"{fid:06d}.npz")
        if not os.path.exists(p):
            continue
        E = np.load(p)["extrinsic"].astype(np.float64)
        v = src[fid] if fid < len(src) else fid
        out[int(v)] = -E[:, :3].T @ E[:, 3]
    return out


def score(art):
    P = poses(art)
    if not P:
        return None
    ids = np.array(sorted(P))
    pos = np.array([P[i] for i in ids])
    scale = np.percentile(np.linalg.norm(pos - np.median(pos, 0), axis=1), 90)

    def near(f):
        j = np.argmin(np.abs(ids - f))
        return (ids[j], P[ids[j]]) if abs(ids[j] - f) <= 60 else (None, None)

    d, used = [], []
    for g in GT:
        fa, pa = near(g["a"])
        fb, pb = near(g["b"])
        if pa is None or pb is None:
            continue
        d.append(np.linalg.norm(pa - pb) / max(scale, 1e-9))
        used.append((g["a"], g["b"]))
    if not d:
        return None
    d = np.array(d)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() / max(scale, 1e-9)
    return {"n": len(d), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            "mean": float(d.mean()), "under_10pct": float((d < 0.10).mean()),
            "frames": len(ids), "path": float(step)}


print(f"{'artifacts':<28s} {'rev':>4s} {'median':>8s} {'p90':>8s} {'<10%':>6s} {'frames':>7s}")
for art in sys.argv[1:]:
    s = score(art)
    if s is None:
        print(f"{os.path.basename(art):<28s}  no usable poses")
        continue
    print(f"{os.path.basename(art):<28s} {s['n']:>4d} {s['median']:>8.3f} "
          f"{s['p90']:>8.3f} {100 * s['under_10pct']:>5.0f}% {s['frames']:>7d}")
