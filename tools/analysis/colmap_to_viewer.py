
import argparse
import json
import os
import struct

import numpy as np

def read_next(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))

def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(read_next(f, "<Q")[0]):
            cid, model, w, h = read_next(f, "<iiQQ")
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                       7: 5, 8: 4, 9: 5, 10: 12}.get(model, 4)
            params = read_next(f, "<" + "d" * nparams)
            cams[cid] = {"model": model, "w": w, "h": h,
                         "params": np.array(params)}
    return cams

def read_images(path):
    imgs = {}
    with open(path, "rb") as f:
        for _ in range(read_next(f, "<Q")[0]):
            iid, *q, tx, ty, tz, cid = read_next(f, "<idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            npts = read_next(f, "<Q")[0]
            f.read(24 * npts)
            imgs[iid] = {"q": np.array(q), "t": np.array([tx, ty, tz]),
                         "cam": cid, "name": name.decode(), "n2d": npts}
    return imgs

def read_points(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        for _ in range(read_next(f, "<Q")[0]):
            _, x, y, z, r, g, b, _ = read_next(f, "<QdddBBBd")
            f.read(8 * read_next(f, "<Q")[0])
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, np.float32), np.array(rgb, np.uint8)

def read_ply(path):
    with open(path, "rb") as f:
        hdr, n, props = b"", 0, []
        while True:
            line = f.readline()
            hdr += line
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s.startswith("property"):
                props.append(s.split()[-1])
            elif s == "end_header":
                break
        fmt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if "nx" in props:
            fmt += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        fmt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        d = np.frombuffer(f.read(np.dtype(fmt).itemsize * n), dtype=fmt, count=n)
    return (np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32),
            np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.uint8))

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="sparse model dir (bin files)")
ap.add_argument("--ply", default=None, help="dense fused.ply, if there is one")
ap.add_argument("--out", required=True)
ap.add_argument("--voxel", type=float, default=0.005)
ap.add_argument("--max_points", type=int, default=6_000_000)
args = ap.parse_args()

cams = read_cameras(os.path.join(args.model, "cameras.bin"))
imgs = read_images(os.path.join(args.model, "images.bin"))
print(f"{len(imgs)} registered images, {len(cams)} camera(s)")
for cid, c in cams.items():
    print(f"  camera {cid}: model {c['model']} {c['w']}x{c['h']} "
          f"params {np.round(c['params'], 2)}")

if args.ply and os.path.exists(args.ply):
    P, C = read_ply(args.ply)
    print(f"dense cloud: {len(P):,} points")
else:
    P, C = read_points(os.path.join(args.model, "points3D.bin"))
    print(f"sparse cloud: {len(P):,} points")

keep = np.isfinite(P).all(1)
P, C = P[keep], C[keep]
centre = np.median(P, axis=0)
P -= centre
r = np.percentile(np.linalg.norm(P, axis=1), 95)
scale = 1.0 / max(r, 1e-9)
P *= scale
print(f"normalised: centre {np.round(centre, 3)}, p95 radius {r:.3f} -> 1.0")

rng = np.random.default_rng(0)
if len(P) > args.max_points:
    sel = rng.choice(len(P), args.max_points, replace=False)
    P, C = P[sel], C[sel]
    print(f"subsampled to {len(P):,}")

os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
np.savez_compressed(os.path.join(args.out, "cloud.npz"), points=P, colors=C)

order = sorted(imgs, key=lambda i: imgs[i]["name"])
saved = []
for iid in order:
    im = imgs[iid]
    fid = int(os.path.splitext(im["name"])[0])
    c = cams[im["cam"]]
    R = qvec2rotmat(im["q"])
    t = im["t"] * scale - R @ centre * scale
    E = np.concatenate([R, t[:, None]], 1).astype(np.float32)
    p = c["params"]
    fx, fy, cx, cy = ((p[0], p[1], p[2], p[3]) if c["model"] in (4, 5, 6, 10)
                      else (p[0], p[0], p[1], p[2]))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    np.savez(os.path.join(args.out, "frames", f"{fid:06d}.npz"),
             extrinsic=E, intrinsic=K)
    saved.append(fid)

json.dump({"model_hw": [int(cams[imgs[order[0]]["cam"]]["h"]),
                        int(cams[imgs[order[0]]["cam"]]["w"])],
           "num_frames": 8711, "fps": 29.97, "voxel_size": args.voxel,
           "conf_threshold": 0.0, "saved_frame_ids": saved,
           "source": "COLMAP/GLOMAP reconstruction"},
          open(os.path.join(args.out, "meta.json"), "w"), indent=1)
json.dump({"plants": [], "classifier": "none", "scene_extent": 2.0,
           "cluster_dist": 0.1}, open(os.path.join(args.out, "plants.json"), "w"))
print(f"wrote {args.out}: {len(P):,} points, {len(saved)} poses")
