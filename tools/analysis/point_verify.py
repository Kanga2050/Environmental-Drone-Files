
import colorsys
import json
import os
import sys

import cv2
import numpy as np

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)

ART = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_full"
OUT = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[2] if len(sys.argv) > 2 else "point_verify"

cloud = np.load(os.path.join(ART, "cloud.npz"))
P = cloud["points"].astype(np.float32)
C = cloud["colors"].astype(np.uint8)
plants = json.load(open(os.path.join(ART, "plants.json")))["plants"]
meta = json.load(open(os.path.join(ART, "meta.json")))
voxel = float(meta.get("voxel_size", 0.005))

inst_file = os.path.join(ART, "plant_points.npz")
inst = np.load(inst_file) if os.path.exists(inst_file) else None
palette = [tuple(int(c * 255) for c in
                 colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 1.0))
           for i in range(len(plants))]

from scipy.spatial import cKDTree
RAD = voxel * float(os.environ.get("HL_RADIUS", "2.5"))
tree = cKDTree(P)
hl = C.copy()
marked = np.zeros(len(P), bool)
for p, col in zip(plants, palette):
    key = f"pts_{p['id']}"
    if inst is None or key not in inst or not len(inst[key]):
        continue
    ip = inst[key].astype(np.float32)
    hits = [h for h in tree.query_ball_point(ip, r=RAD, workers=-1) if h]
    if not hits:
        continue
    idx = np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]))
    hl[idx] = col
    marked[idx] = True
print(f"{len(plants)} plants | highlighted {marked.sum():,} of {len(P):,} "
      f"cloud points ({marked.mean():.1%})")

def splat(fid, cols, W=518, H=420, rad=1):
    d = np.load(os.path.join(ART, "frames", f"{fid:06d}.npz"), allow_pickle=True)
    E, K = d["extrinsic"], d["intrinsic"]
    R, t = E[:, :3], E[:, 3]
    Q = (P @ R.T) + t
    z = Q[:, 2]
    uv = Q @ K.T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    u = uv[ok, 0].astype(np.int32)
    v = uv[ok, 1].astype(np.int32)
    zz = z[ok]
    cc = cols[ok]
    order = np.argsort(-zz)
    u, v, cc = u[order], v[order], cc[order]
    img = np.zeros((H, W, 3), np.uint8)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            uu = np.clip(u + dx, 0, W - 1)
            vv = np.clip(v + dy, 0, H - 1)
            img[vv, uu] = cc
    return img

good = []
for fid in meta["saved_frame_ids"]:
    f = os.path.join(ART, "frames", f"{fid:06d}.npz")
    if not os.path.exists(f):
        continue
    c = np.load(f, allow_pickle=True)["conf"].astype(np.float32)
    if (c > 1.5).mean() > 0.4:
        good.append(fid)
if not good:
    good = meta["saved_frame_ids"][:8]
picks = [good[int(len(good) * f)] for f in (0.08, 0.3, 0.55, 0.8)]

rows = []
for fid in picks:
    photo = cv2.resize(cv2.imread(os.path.join(ART, "frames", f"{fid:06d}.jpg")), (518, 420))
    rows.append(np.hstack([
        photo,
        cv2.cvtColor(splat(fid, C), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(splat(fid, hl), cv2.COLOR_RGB2BGR)]))
    print("rendered", fid)
cv2.imwrite(os.path.join(OUT, f"{TAG}.jpg"), np.vstack(rows))
print("wrote", f"{TAG}.jpg")
