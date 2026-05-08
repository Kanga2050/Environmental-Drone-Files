
import argparse
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

COLMAP = r"C:\Users\shaur\AppData\Local\sfm-tools\colmap\bin\colmap.exe"

def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

def read_images(path):

    out = {}
    with open(path, "rb") as f:
        for _ in range(rd(f, "<Q")[0]):
            iid, *_q, _tx, _ty, _tz, _cid = rd(f, "<idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n2d = rd(f, "<Q")[0]
            buf = f.read(24 * n2d)
            ids = np.frombuffer(buf, dtype=np.dtype([("x", "<f8"), ("y", "<f8"),
                                                     ("p", "<i8")]))["p"]
            out[iid] = (name.decode(), ids[ids >= 0])
    return out

def free_gb(path):
    t, u, f = shutil.disk_usage(path)
    return f / 1e9

def run(args, log):
    with open(log, "ab") as fh:
        p = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--sparse", required=True)
ap.add_argument("--work", required=True, help="dense workspace (will be large)")
ap.add_argument("--out", required=True, help="final merged ply")
ap.add_argument("--block", type=int, default=250)
ap.add_argument("--overlap", type=int, default=40)
ap.add_argument("--num_src", type=int, default=12)
ap.add_argument("--min_free_gb", type=float, default=4.0)
ap.add_argument("--voxel", type=float, default=0.0)
args = ap.parse_args()

log = os.path.join(os.path.dirname(args.out), "mvs.log")
os.makedirs(os.path.dirname(args.out), exist_ok=True)

if not os.path.exists(os.path.join(args.work, "stereo", "patch-match.cfg")):
    print("undistorting (once, for the whole sequence)...", flush=True)
    rc = run([COLMAP, "image_undistorter", "--image_path", args.images,
              "--input_path", args.sparse, "--output_path", args.work,
              "--output_type", "COLMAP", "--max_image_size", "-1"], log)
    if rc:
        sys.exit(f"image_undistorter failed ({rc}); see {log}")
print(f"undistorted workspace ready, {free_gb(args.work):.1f} GB free")

imgs = read_images(os.path.join(args.work, "sparse", "images.bin"))
names = [imgs[i][0] for i in sorted(imgs, key=lambda i: imgs[i][0])]
pts = {imgs[i][0]: set(imgs[i][1].tolist()) for i in imgs}
print(f"{len(names)} registered images in the dense model")

by_pt = defaultdict(list)
for n in names:
    for p in pts[n]:
        by_pt[p].append(n)
cov = defaultdict(lambda: defaultdict(int))
for p, ns in by_pt.items():
    if len(ns) > 60:
        continue
    for a in ns:
        for b in ns:
            if a != b:
                cov[a][b] += 1
print("co-visibility built")

stereo = os.path.join(args.work, "stereo")
dm, nm, cg = (os.path.join(stereo, d) for d in
              ("depth_maps", "normal_maps", "consistency_graphs"))
step = max(1, args.block - args.overlap)
blocks = [names[i:i + args.block] for i in range(0, len(names), step)]
blocks = [b for b in blocks if len(b) >= 3]
print(f"{len(blocks)} blocks of <={args.block} (step {step}, overlap {args.overlap})\n")

plys, t0 = [], time.time()
for bi, block in enumerate(blocks):
    chunk_ply = os.path.join(os.path.dirname(args.out), f"block_{bi:04d}.ply")
    if os.path.exists(chunk_ply):
        plys.append(chunk_ply)
        continue
    if free_gb(args.work) < args.min_free_gb:
        print(f"STOPPING: only {free_gb(args.work):.1f} GB free")
        break
    inblock = set(block)
    with open(os.path.join(stereo, "patch-match.cfg"), "w") as f:
        for n in block:
            src = sorted((c for c in cov[n] if c in inblock),
                         key=lambda c: -cov[n][c])[:args.num_src]
            f.write(n + "\n" + (", ".join(src) if src else "__auto__, 8") + "\n")
    with open(os.path.join(stereo, "fusion.cfg"), "w") as f:
        f.write("\n".join(block) + "\n")

    rc = run([COLMAP, "patch_match_stereo", "--workspace_path", args.work,
              "--workspace_format", "COLMAP",
              "--PatchMatchStereo.max_image_size", "-1",
              "--PatchMatchStereo.geom_consistency", "true"], log)
    if rc:
        print(f"block {bi}: patch_match_stereo failed ({rc}), skipping")
    else:
        rc = run([COLMAP, "stereo_fusion", "--workspace_path", args.work,
                  "--workspace_format", "COLMAP", "--input_type", "geometric",
                  "--output_path", chunk_ply], log)
        if rc == 0 and os.path.exists(chunk_ply):
            plys.append(chunk_ply)

    for d in (dm, nm, cg):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass
    done = bi + 1
    el = time.time() - t0
    sz = os.path.getsize(chunk_ply) / 1e6 if os.path.exists(chunk_ply) else 0
    print(f"block {done}/{len(blocks)}  {sz:7.0f} MB  "
          f"{el / 60:6.1f} min elapsed  eta {(el / done) * (len(blocks) - done) / 3600:5.1f} h  "
          f"free {free_gb(args.work):4.1f} GB", flush=True)

print(f"\n{len(plys)} block clouds; merging")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_to_viewer import read_ply

P, C = [], []
for p in plys:
    a, b = read_ply(p)
    P.append(a)
    C.append(b)
P, C = np.concatenate(P), np.concatenate(C)
print(f"merged {len(P):,} points")
if args.voxel > 0:
    key = np.floor(P / args.voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    P, C = P[idx], C[idx]
    print(f"voxel {args.voxel}: {len(P):,} points after deduplicating seams")
np.savez_compressed(args.out.replace(".ply", ".npz"), points=P, colors=C)
print(f"wrote {args.out.replace('.ply', '.npz')}")
