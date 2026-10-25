import json, os, sys
import numpy as np

ART = sys.argv[1]  # map folder
OUT = sys.argv[2]  # output folder
os.makedirs(OUT, exist_ok=True)
P = np.load(os.path.join(ART, "poses.npy"))
meta = json.load(open(os.path.join(ART, "meta.json")))
c = np.load(os.path.join(ART, "cloud.npz"))
pts = c["points"]
ff = c["first_frame"]
lf = c["last_frame"]
rng = np.random.default_rng(0)
sub = pts[rng.choice(len(pts), 400000, replace=False)]

C = P[:, :3, 3]
V = P[:, :3, 2]
up = -V.mean(0)
up /= np.linalg.norm(up)
best = None
for _ in range(400):
    a, b, d = sub[rng.choice(len(sub), 3, replace=False)]
    n = np.cross(b - a, d - a)
    if np.linalg.norm(n) < 1e-9:
        continue
    n /= np.linalg.norm(n)
    if abs(n @ up) < 0.9:
        continue
    if n @ up < 0:
        n = -n
    off = -n @ a
    inl = np.abs(sub @ n + off) < 0.02
    if best is None or inl.sum() > best[2]:
        best = (n, off, inl.sum())
n, off, _ = best
inl = np.abs(sub @ n + off) < 0.05
A = np.c_[sub[inl], np.ones(inl.sum())]
sol = np.linalg.lstsq(A, np.zeros(inl.sum()) - 0, rcond=None)
h_pts = sub @ n + off
ground = np.percentile(h_pts, 5)
off -= ground
height = C @ n + off
e1 = np.cross(n, [0, 0, 1.0])
if np.linalg.norm(e1) < 1e-3:
    e1 = np.cross(n, [1.0, 0, 0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(n, e1)
t = -(C @ n + off) / np.minimum(V @ n, -1e-3)
F = C + V * t[:, None]
xy = np.c_[F @ e1, F @ e2]
cxy = np.c_[C @ e1, C @ e2]
tilt = np.degrees(np.arccos(np.clip(-(V @ n), -1, 1)))

fps = meta.get("fps", 30.0)
N = len(P)
step = 15
idx = np.arange(0, N, step)
cell = np.median(height) * 0.5
gx = np.floor(xy[idx] / cell).astype(int)
seen = {}
passes = np.zeros(len(idx), int)
revisit = np.zeros(len(idx), bool)
gap = int(10 * fps / step)
for k, g in enumerate(map(tuple, gx)):
    hist = seen.setdefault(g, [])
    if not hist or k - hist[-1] > gap:
        if hist:
            revisit[k] = True
        passes[k] = sum(1 for i in range(len(hist)) if i == 0 or hist[i] - hist[i - 1] > gap) + 1
    else:
        passes[k] = passes[k - 1] if k else 1
    hist.append(k)
new = np.bincount(ff, minlength=N)[:N].astype(float)
last = np.bincount(lf, minlength=N)[:N].astype(float)
win = int(fps)
newr = np.convolve(new, np.ones(win) / win, "same")
cnf = []
for fid in meta["saved_frame_ids"]:
    d = np.load(os.path.join(ART, "frames", f"{fid:06d}.npz"))
    cf = d["conf"].astype(np.float32)
    cnf.append((fid, float(np.median(cf)), float((cf > 1.5).mean())))
cnf = np.array(cnf)
cells = {}
for k, g in enumerate(map(tuple, gx)):
    cells.setdefault(g, set()).add(k)
cell_passes = {}
for g, ks in cells.items():
    ks = sorted(ks)
    p = 1 + sum(1 for a, b in zip(ks, ks[1:]) if b - a > gap)
    cell_passes[g] = p
np.savez(os.path.join(OUT, "track.npz"), t=np.arange(N) / fps, height=height, xy=xy, cxy=cxy, tilt=tilt,
         idx=idx, passes=passes, revisit=revisit, newr=newr, cnf=cnf, n=n, off=off, e1=e1, e2=e2, cell=cell,
         cell_keys=np.array(list(cell_passes.keys())), cell_passes=np.array(list(cell_passes.values())))
print(f"frames {N}, {N / fps / 60:.1f} min")
print(f"height above ground (GCT units): median {np.median(height):.3f}, 5-95% {np.percentile(height, 5):.3f}-{np.percentile(height, 95):.3f}")
print(f"camera tilt from straight down: median {np.median(tilt):.0f} deg")
span = np.ptp(xy, 0)
print(f"footprint track spans {span[0]:.2f} x {span[1]:.2f} units; cell {cell:.3f}")
pc = np.array(list(cell_passes.values()))
print(f"cells {len(pc)}; passes per cell: 1:{(pc == 1).sum()} 2:{(pc == 2).sum()} 3:{(pc == 3).sum()} 4+:{(pc >= 4).sum()}")
starts = np.flatnonzero(revisit)
print(f"revisit events {len(starts)}; first few at s: {(idx[starts][:15] / fps).round(0).tolist()}")
print(f"median conf by minute: {[round(float(np.median(cnf[(cnf[:, 0] >= a * 60 * fps) & (cnf[:, 0] < (a + 1) * 60 * fps), 1])), 2) for a in range(int(N / fps / 60) + 1)]}")
