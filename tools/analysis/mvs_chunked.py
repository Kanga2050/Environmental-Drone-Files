
import argparse
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict

TARGET_GAPS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)

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
ap.add_argument("--cache_size", type=float, default=8.0)
ap.add_argument("--max_image_size", type=int, default=-1)
ap.add_argument("--depth_min", type=float, default=-1.0)
ap.add_argument("--depth_max", type=float, default=-1.0)
ap.add_argument("--min_shared", type=int, default=30)
ap.add_argument("--fusion_cache", type=float, default=4.0)
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
    if len(ns) > 400:
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
    bpos = {n: k for k, n in enumerate(block)}
    with open(os.path.join(stereo, "patch-match.cfg"), "w") as f:
        for n in block:
            i = bpos[n]
            out = []
            for t in TARGET_GAPS:
                if len(out) >= args.num_src:
                    break
                cand = [c for c in block if c not in out and c != n
                        and cov[n].get(c, 0) >= args.min_shared]
                if not cand:
                    continue
                out.append(min(cand, key=lambda c: abs(abs(bpos[c] - i) - t)))
            for t in TARGET_GAPS:
                if len(out) >= args.num_src: break
                cand = [c for c in block if c not in out and c != n]
                if not cand: break
                out.append(min(cand, key=lambda c: abs(abs(bpos[c] - i) - t)))
            f.write(n + "\n" + ", ".join(out) + "\n")
    with open(os.path.join(stereo, "fusion.cfg"), "w") as f:
        f.write("\n".join(block) + "\n")

    rc = run([COLMAP, "patch_match_stereo", "--workspace_path", args.work,
              "--workspace_format", "COLMAP",
              "--PatchMatchStereo.max_image_size", str(args.max_image_size),
              "--PatchMatchStereo.cache_size", str(args.cache_size),
              "--PatchMatchStereo.depth_min", str(args.depth_min),
              "--PatchMatchStereo.depth_max", str(args.depth_max),
              "--PatchMatchStereo.geom_consistency", "true"], log)
    ngeo = len([f for f in os.listdir(dm) if f.endswith("geometric.bin")]) if os.path.isdir(dm) else 0
    if rc:
        print(f"block {bi}: patch_match_stereo failed ({rc}), "
              f"{ngeo}/{len(block)} geometric maps present; fusing anyway", flush=True)
    if ngeo >= 3:
        with open(os.path.join(stereo, "fusion.cfg"), "w") as f:
            f.write("\n".join(n for n in block
                              if os.path.exists(os.path.join(dm, n + ".geometric.bin"))) + "\n")
        rc = run([COLMAP, "stereo_fusion", "--workspace_path", args.work,
                  "--workspace_format", "COLMAP", "--input_type", "geometric",
                  "--StereoFusion.use_cache", "1",
                  "--StereoFusion.cache_size", str(args.fusion_cache),
                  "--output_path", chunk_ply], log)
        if rc == 0 and os.path.exists(chunk_ply):
            plys.append(chunk_ply)
    else:
        print(f"block {bi}: only {ngeo} geometric maps, nothing to fuse", flush=True)

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
def read_ply(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            s = f.readline().decode("ascii", "ignore").strip()
            if s.startswith("element vertex"): n = int(s.split()[-1])
            elif s.startswith("property"): props.append(s.split()[-1])
            elif s == "end_header": break
        dt = np.dtype([(q, "<f4") if q in ("x","y","z","nx","ny","nz") else (q,"u1") for q in props])
        d = np.frombuffer(f.read(n * dt.itemsize), dt)
    return (np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32),
            np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.uint8))

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
