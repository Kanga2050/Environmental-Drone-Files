
import json
import os
import sys

import cv2
import numpy as np

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_avg"
meta = json.load(open(os.path.join(ART, "meta.json"), encoding="utf-8"))
H, W = meta["model_hw"]
cloud = np.load(os.path.join(ART, "cloud.npz"))
P, C = cloud["points"], cloud["colors"].astype(np.float32)
print(f"model_hw {W}x{H}, cloud {len(P):,} points")

RANGE = 40

def probe(fid):
    f = np.load(os.path.join(ART, "frames", f"{fid:06d}.npz"))
    jpg = os.path.join(ART, "frames", f"{fid:06d}.jpg")
    img = cv2.cvtColor(cv2.imread(jpg), cv2.COLOR_BGR2RGB).astype(np.float32)
    ih, iw = img.shape[:2]
    E, K = f["extrinsic"], f["intrinsic"]

    Xc = P @ E[:, :3].T + E[:, 3]
    z = Xc[:, 2]
    keep = z > 0.05
    Xc, col = Xc[keep], C[keep]
    z = Xc[:, 2]
    u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / z + K[1, 2]

    su, sv = iw / W, ih / H
    pu, pv = u * su, v * sv
    m = (pu >= RANGE) & (pu < iw - RANGE) & (pv >= RANGE) & (pv < ih - RANGE)
    pu, pv, col, zz = pu[m], pv[m], col[m], z[m]
    if len(pu) < 2000:
        return None

    o = np.argsort(zz)[:60000]
    pu, pv, col = pu[o], pv[o], col[o]

    best, surf = None, np.zeros((2 * RANGE + 1, 2 * RANGE + 1), np.float32)
    for i, dv in enumerate(range(-RANGE, RANGE + 1)):
        for j, du in enumerate(range(-RANGE, RANGE + 1)):
            s = img[np.clip((pv + dv).astype(np.int32), 0, ih - 1),
                    np.clip((pu + du).astype(np.int32), 0, iw - 1)]
            e = float(np.abs(s - col).mean())
            surf[i, j] = e
            if best is None or e < best[0]:
                best = (e, du, dv)
    i0 = np.unravel_index(np.argmin(surf), surf.shape)
    return dict(n=len(pu), err=best[0], du=best[1], dv=best[2],
                err0=float(surf[RANGE, RANGE]), iw=iw, ih=ih,
                su=su, sv=sv)

ids = [i for i in meta["saved_frame_ids"][::18]][:14]
print(f"\n{'frame':>6s} {'pts':>7s} {'err@(0,0)':>10s} {'best err':>9s} "
      f"{'du':>5s} {'dv':>5s}")
rows = []
for fid in ids:
    r = probe(fid)
    if not r:
        continue
    rows.append(r)
    print(f"{fid:>6d} {r['n']:>7,} {r['err0']:>10.2f} {r['err']:>9.2f} "
          f"{r['du']:>5d} {r['dv']:>5d}")

if rows:
    du = np.array([r["du"] for r in rows], float)
    dv = np.array([r["dv"] for r in rows], float)
    print(f"\nphoto is {rows[0]['iw']}x{rows[0]['ih']}, "
          f"scale to model {rows[0]['su']:.4f} x {rows[0]['sv']:.4f}")
    print(f"best shift: du median {np.median(du):+.1f} px "
          f"(mean {du.mean():+.1f}, sd {du.std():.1f})")
    print(f"            dv median {np.median(dv):+.1f} px "
          f"(mean {dv.mean():+.1f}, sd {dv.std():.1f})")
    print(f"as a fraction of width/height: "
          f"{np.median(du) / rows[0]['iw'] * 100:+.2f}% / "
          f"{np.median(dv) / rows[0]['ih'] * 100:+.2f}%")
    imp = np.array([r["err0"] - r["err"] for r in rows])
    print(f"colour error improvement from shifting: mean {imp.mean():.2f}/255 "
          f"({100 * (imp / np.array([r['err0'] for r in rows])).mean():.1f}%)")
