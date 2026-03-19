import argparse, glob, os
import numpy as np

def read_ply(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            s = f.readline().decode("ascii", "ignore").strip()
            if s.startswith("element vertex"): n = int(s.split()[-1])
            elif s.startswith("property"): props.append(s.split()[-1])
            elif s == "end_header": break
        dt = np.dtype([(p, "<f4") if p in ("x","y","z","nx","ny","nz") else (p,"u1") for p in props])
        d = np.frombuffer(f.read(n * dt.itemsize), dt)
    return (np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32),
            np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.uint8))

def write_ply(path, P, C):
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    dt = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")])
    a = np.empty(len(P), dt)
    a["x"], a["y"], a["z"] = P[:,0], P[:,1], P[:,2]
    a["red"], a["green"], a["blue"] = C[:,0], C[:,1], C[:,2]
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(a.tobytes())

ap = argparse.ArgumentParser()
ap.add_argument("--blocks", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--voxel", type=float, default=0.004)
a = ap.parse_args()

fs = []
for tok in a.blocks.split(","):
    fs += sorted(glob.glob(os.path.join(tok, "block_*.ply") if os.path.isdir(tok) else tok))
Ps, Cs = [], []
for f in fs:
    p, c = read_ply(f)
    Ps.append(p); Cs.append(c)
    print(f"{os.path.basename(f)}: {len(p):,}")
P, C = np.concatenate(Ps), np.concatenate(Cs)
print(f"merged {len(P):,} from {len(fs)} blocks")
if a.voxel > 0:
    _, idx = np.unique(np.floor(P / a.voxel).astype(np.int64), axis=0, return_index=True)
    P, C = P[idx], C[idx]
    print(f"voxel {a.voxel}: {len(P):,} after seam dedup")
write_ply(a.out, P, C)
print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.0f} MB)")
