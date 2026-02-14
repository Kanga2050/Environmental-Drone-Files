import sqlite3
import struct
import sys

import numpy as np

W = r"C:\Users\shaur\AppData\Local\sfm-work"
M = 2147483647


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def read_images(path):
    out = {}
    with open(path, "rb") as f:
        for _ in range(rd(f, "<Q")[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = rd(f, "<idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            f.read(24 * rd(f, "<Q")[0])
            out[iid] = (np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]),
                        name.decode())
    return out


def q2R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ang(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))


imgs = read_images(rf"{W}\sparse_glomap\0\images.bin")
print(f"{len(imgs)} registered poses")
R = {i: q2R(q) for i, (q, t, n) in imgs.items()}
C = {i: -q2R(q).T @ t for i, (q, t, n) in imgs.items()}
num = {i: int(n[:6]) for i, (q, t, n) in imgs.items()}

c = sqlite3.connect(rf"{W}\db313.db")
rows = c.execute(
    "select pair_id, rows, config, qvec from two_view_geometries where rows>0").fetchall()
print(f"{len(rows):,} verified pairs in db")

seq, loop = [], []
for pid, nin, cfg, qb in rows:
    i1, i2 = pid // M, pid % M
    if i1 not in R or i2 not in R or qb is None:
        continue
    q = np.frombuffer(qb, np.float64)
    if q.size != 4 or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        continue
    err = ang(q2R(q), R[i2] @ R[i1].T)
    gap = abs(num[i1] - num[i2])
    dist = float(np.linalg.norm(C[i1] - C[i2]))
    (loop if gap > 600 else seq).append((err, nin, gap, dist))

for name, v in (("sequential", seq), ("long-range (loop)", loop)):
    if not v:
        print(f"{name}: none")
        continue
    e = np.array([x[0] for x in v])
    d = np.array([x[3] for x in v])
    print(f"\n{name}: {len(v):,} pairs")
    print(f"  rotation error vs two-view: median {np.median(e):6.2f} deg, "
          f"p90 {np.percentile(e, 90):6.2f}, >20deg: {100 * (e > 20).mean():5.1f}%")
    print(f"  camera separation in model: median {np.median(d):.3f} u, "
          f"p90 {np.percentile(d, 90):.3f}")

if loop:
    e = np.array([x[0] for x in loop])
    d = np.array([x[3] for x in loop])
    n = np.array([x[1] for x in loop])
    strong = n >= 150
    print(f"\nstrong loop pairs (>=150 inliers): {strong.sum():,}")
    if strong.any():
        print(f"  rotation error median {np.median(e[strong]):.2f} deg, "
              f">20deg: {100 * (e[strong] > 20).mean():.1f}%")
        print(f"  camera separation median {np.median(d[strong]):.3f} u")
