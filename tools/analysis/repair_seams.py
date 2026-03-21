import argparse, json, os, struct, subprocess, sys, time
from collections import defaultdict
import numpy as np

COLMAP = r"C:\Users\shaur\AppData\Local\sfm-tools\colmap\bin\colmap.exe"
TARGET_GAPS = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)

def rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

def read_images(path):
    out = {}
    with open(path, "rb") as f:
        for _ in range(rd(f, "<Q")[0]):
            iid, *_q, _tx, _ty, _tz, _cid = rd(f, "<idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00": name += c
            n2d = rd(f, "<Q")[0]
            ids = np.frombuffer(f.read(24 * n2d),
                                dtype=np.dtype([("x","<f8"),("y","<f8"),("p","<i8")]))["p"]
            out[iid] = (name.decode(), ids[ids >= 0])
    return out

def run(a, log):
    with open(log, "ab") as fh:
        return subprocess.run(a, stdout=fh, stderr=subprocess.STDOUT).returncode

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seams", required=True, help="comma list a-b")
ap.add_argument("--step", type=int, default=210)
ap.add_argument("--block", type=int, default=250)
ap.add_argument("--win", type=int, default=250)
ap.add_argument("--num_src", type=int, default=12)
ap.add_argument("--min_shared", type=int, default=30)
ap.add_argument("--cache_size", type=float, default=6.0)
ap.add_argument("--fusion_cache", type=float, default=4.0)
ap.add_argument("--depth_min", type=float, default=0.3)
ap.add_argument("--depth_max", type=float, default=20.0)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
log = os.path.join(a.out, "repair.log")
imgs = read_images(os.path.join(a.work, "sparse", "images.bin"))
names = [imgs[i][0] for i in sorted(imgs, key=lambda i: imgs[i][0])]
pts = {imgs[i][0]: set(imgs[i][1].tolist()) for i in imgs}
by_pt = defaultdict(list)
for n in names:
    for p in pts[n]: by_pt[p].append(n)
cov = defaultdict(lambda: defaultdict(int))
for p, ns in by_pt.items():
    if len(ns) > 400: continue
    for x in ns:
        for y in ns:
            if x != y: cov[x][y] += 1
print(f"{len(names)} images, co-visibility built", flush=True)

stereo = os.path.join(a.work, "stereo")
dm, nm, cg = (os.path.join(stereo, d) for d in ("depth_maps","normal_maps","consistency_graphs"))
t0 = time.time()
for s in a.seams.split(","):
    x, y = (int(v) for v in s.split("-"))
    centre = (y * a.step + x * a.step + a.block) // 2
    lo = max(0, centre - a.win // 2); hi = min(len(names), lo + a.win)
    blk = names[lo:hi]
    ply = os.path.join(a.out, f"repair_{x:02d}_{y:02d}.ply")
    if os.path.exists(ply):
        print(f"seam {x}-{y}: exists, skip", flush=True); continue
    for d in (dm, nm, cg):
        if os.path.isdir(d):
            for fn in os.listdir(d):
                try: os.remove(os.path.join(d, fn))
                except OSError: pass
    bpos = {n: k for k, n in enumerate(blk)}
    with open(os.path.join(stereo, "patch-match.cfg"), "w") as f:
        for n in blk:
            i = bpos[n]; out = []
            for t in TARGET_GAPS:
                if len(out) >= a.num_src: break
                cand = [c for c in blk if c not in out and c != n and cov[n].get(c, 0) >= a.min_shared]
                if not cand: continue
                out.append(min(cand, key=lambda c: abs(abs(bpos[c] - i) - t)))
            for t in TARGET_GAPS:
                if len(out) >= a.num_src: break
                cand = [c for c in blk if c not in out and c != n]
                if not cand: break
                out.append(min(cand, key=lambda c: abs(abs(bpos[c] - i) - t)))
            f.write(n + "\n" + ", ".join(out) + "\n")
    with open(os.path.join(stereo, "fusion.cfg"), "w") as f:
        f.write("\n".join(blk) + "\n")
    print(f"seam {x}-{y}: {blk[0]}-{blk[-1]} ({len(blk)} frames)", flush=True)
    rc = run([COLMAP, "patch_match_stereo", "--workspace_path", a.work, "--workspace_format", "COLMAP",
              "--PatchMatchStereo.max_image_size", "-1",
              "--PatchMatchStereo.cache_size", str(a.cache_size),
              "--PatchMatchStereo.depth_min", str(a.depth_min),
              "--PatchMatchStereo.depth_max", str(a.depth_max),
              "--PatchMatchStereo.geom_consistency", "true"], log)
    ngeo = len([f for f in os.listdir(dm) if f.endswith("geometric.bin")]) if os.path.isdir(dm) else 0
    if rc: print(f"  patch_match failed ({rc}), {ngeo}/{len(blk)} maps; fusing anyway", flush=True)
    if ngeo >= 3:
        with open(os.path.join(stereo, "fusion.cfg"), "w") as f:
            f.write("\n".join(n for n in blk
                              if os.path.exists(os.path.join(dm, n + ".geometric.bin"))) + "\n")
        run([COLMAP, "stereo_fusion", "--workspace_path", a.work, "--workspace_format", "COLMAP",
             "--input_type", "geometric", "--StereoFusion.use_cache", "1",
             "--StereoFusion.cache_size", str(a.fusion_cache), "--output_path", ply], log)
    sz = os.path.getsize(ply) / 1e6 if os.path.exists(ply) else 0
    print(f"  -> {sz:.0f} MB  ({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
print("repair done", flush=True)
