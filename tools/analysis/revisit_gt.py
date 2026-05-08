import json
import sqlite3
import struct

import numpy as np

W = r"C:\Users\shaur\AppData\Local\sfm-work"
M = 2147483647


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


names = {}
with open(rf"{W}\sparse_glomap\0\images.bin", "rb") as f:
    for _ in range(rd(f, "<Q")[0]):
        iid, *_r, _cid = rd(f, "<idddddddi")
        n = b""
        while (c := f.read(1)) != b"\x00":
            n += c
        f.read(24 * rd(f, "<Q")[0])
        names[iid] = int(n.decode()[:6])

c = sqlite3.connect(rf"{W}\db313.db")
pairs = []
for pid, nin in c.execute(
        "select pair_id, rows from two_view_geometries where rows>=100"):
    i1, i2 = pid // M, pid % M
    if i1 in names and i2 in names:
        a, b = sorted((names[i1], names[i2]))
        if b - a > 600:
            pairs.append((a, b, nin))

pairs.sort(key=lambda r: -r[2])
seen, out = set(), []
for a, b, n in pairs:
    k = (a // 200, b // 200)
    if k in seen:
        continue
    seen.add(k)
    out.append({"a": a, "b": b, "inliers": n})
json.dump(out, open(rf"{W}\revisit_gt.json", "w"), indent=1)
gaps = np.array([o["b"] - o["a"] for o in out])
print(f"{len(pairs):,} verified long-range pairs -> {len(out)} distinct revisits")
print(f"frame gap: median {np.median(gaps):.0f}, min {gaps.min()}, max {gaps.max()}")
print(f"inliers: median {np.median([o['inliers'] for o in out]):.0f}")
print(f"revisits with both frames < 3600: "
      f"{sum(1 for o in out if o['b'] < 3600)}")
