import json, os, sys
import cv2
import numpy as np

ART = sys.argv[1]  # map folder
VID = sys.argv[2]  # source video
OUT = sys.argv[3]  # flight_track.py output folder
T = np.load(os.path.join(OUT, "track.npz"))
meta = json.load(open(os.path.join(ART, "meta.json")))
plants = json.load(open(os.path.join(ART, "plants.json")))["plants"]
n, off, e1, e2 = T["n"], float(T["off"]), T["e1"], T["e2"]
fps = float(meta.get("fps", 30.0))
src_ids = meta.get("frame_source_ids") or list(range(len(T["height"])))

W, H = 1920, 1080
MAP = 1040
c = np.load(os.path.join(ART, "cloud.npz"))
rng = np.random.default_rng(1)
sel = rng.choice(len(c["points"]), min(len(c["points"]), 12_000_000), replace=False)
p = c["points"][sel]
col = c["colors"][sel]
u, v, h = p @ e1, p @ e2, p @ n + off
def smooth(a, w=31):
    k = np.ones(w) / w
    return np.c_[[np.convolve(np.pad(a[:, i], (w // 2, w // 2), mode="edge"), k, "valid") for i in range(a.shape[1])]].T


xy = smooth(T["xy"])
cxy = smooth(T["cxy"])
lo = np.percentile(np.r_[u, xy[:, 0]], 0.5), np.percentile(np.r_[v, xy[:, 1]], 0.5)
hi = np.percentile(np.r_[u, xy[:, 0]], 99.5), np.percentile(np.r_[v, xy[:, 1]], 99.5)
span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.05
cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
s = MAP / span


def to_px(a, b):
    return ((a - cx) * s + MAP / 2 + 20).astype(np.int32), (MAP / 2 - (b - cy) * s + 20).astype(np.int32)


px, py = to_px(u, v)
ok = (px >= 20) & (px < MAP + 20) & (py >= 20) & (py < MAP + 20)
order = np.argsort(h[ok])
bg = np.full((H, W, 3), 18, np.uint8)
bg[py[ok][order], px[ok][order]] = col[ok][order][:, ::-1]
bg[20:MAP + 20, 20:MAP + 20] = cv2.dilate(bg[20:MAP + 20, 20:MAP + 20], np.ones((2, 2), np.uint8))
dim = (bg[20:MAP + 20, 20:MAP + 20] * 0.55).astype(np.uint8)
bg[20:MAP + 20, 20:MAP + 20] = dim

cell = float(T["cell"])
PASS_COL = [(0, 0, 0), (209, 158, 69), (0, 205, 255), (52, 104, 235), (180, 70, 220)]

height = T["height"]
newr = T["newr"]
cnf = T["cnf"]
tsec = T["t"]
idx = T["idx"]
N = len(height)
gap = int(10 * fps)
speed = 5
frames_out = int(N / speed) if len(sys.argv) < 2 else int(sys.argv[1])
writer = cv2.VideoWriter(os.path.join(OUT, "flight.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
cap = cv2.VideoCapture(VID)
cells_hist = {}
passes_now = {}
gx_all = np.floor(xy / cell).astype(int)
plant_px = []
for pl in plants:
    cc = np.array(pl["centroid"])
    a, b = to_px(np.array([cc @ e1]), np.array([cc @ e2]))
    form = pl["hierarchy"]["level"]
    name = pl["hierarchy"][form]["name"] if form in pl["hierarchy"] else "plant"
    plant_px.append((int(a[0]), int(b[0]), min(pl["frames"]) if pl.get("frames") else 0, name))

panel_x = MAP + 60
cam_w, cam_h = W - panel_x - 30, int((W - panel_x - 30) * 9 / 16)
plot_y0 = 30 + cam_h + 45


def plot(img, y0, hgt, series, color, label, k, fmt):
    x0, x1 = panel_x, W - 30
    cv2.rectangle(img, (x0, y0), (x1, y0 + hgt), (60, 60, 60), 1)
    xs = np.linspace(x0, x1, len(series)).astype(int)
    lo_, hi_ = np.percentile(series, 1), np.percentile(series, 99)
    ys = (y0 + hgt - (np.clip(series, lo_, hi_) - lo_) / max(hi_ - lo_, 1e-9) * (hgt - 6) - 3).astype(int)
    cv2.polylines(img, [np.c_[xs, ys]], False, (90, 90, 90), 1, cv2.LINE_AA)
    i = k if len(series) == N else int(np.searchsorted(series_t, k))
    cv2.polylines(img, [np.c_[xs, ys][: max(2, i + 1)]], False, color, 2, cv2.LINE_AA)
    xm = int(xs[min(i, len(xs) - 1)])
    cv2.line(img, (xm, y0), (xm, y0 + hgt), (230, 230, 230), 1)
    cv2.putText(img, label, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, fmt, (x1 - 150, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


series_t = cnf[:, 0]
revisits = 0
for j in range(frames_out):
    k = min(N - 1, j * speed)
    for kk in range(max(0, k - speed + 1), k + 1):
        g = tuple(gx_all[kk])
        hist = cells_hist.setdefault(g, [])
        if not hist or kk - hist[-1] > gap:
            passes_now[g] = passes_now.get(g, 0) + 1
            if passes_now[g] > 1:
                revisits += 1
        hist.append(kk)
    img = bg.copy()
    over = img[20:MAP + 20, 20:MAP + 20].copy()
    for g, pc in passes_now.items():
        a0, b0 = to_px(np.array([g[0] * cell]), np.array([g[1] * cell + cell]))
        a1, b1 = to_px(np.array([g[0] * cell + cell]), np.array([g[1] * cell]))
        cv2.rectangle(over, (int(a0[0]) - 20, int(b0[0]) - 20), (int(a1[0]) - 20, int(b1[0]) - 20),
                      PASS_COL[min(pc, 4)], -1)
    img[20:MAP + 20, 20:MAP + 20] = cv2.addWeighted(img[20:MAP + 20, 20:MAP + 20], 0.62, over, 0.38, 0)
    tr = np.c_[to_px(cxy[: k + 1:3, 0], cxy[: k + 1:3, 1])]
    if len(tr) > 1:
        cv2.polylines(img, [tr], False, (255, 255, 255), 1, cv2.LINE_AA)
    a, b = to_px(np.array([xy[k, 0]]), np.array([xy[k, 1]]))
    ca, cb = to_px(np.array([cxy[k, 0]]), np.array([cxy[k, 1]]))
    cv2.line(img, (int(ca[0]), int(cb[0])), (int(a[0]), int(b[0])), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (int(a[0]), int(b[0])), int(cell * s * 0.5), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (int(ca[0]), int(cb[0])), 7, (255, 255, 255), -1, cv2.LINE_AA)
    for (x, y, f0, name) in plant_px:
        if f0 <= k:
            cv2.circle(img, (x, y), 5, (80, 230, 80), 2, cv2.LINE_AA)
    sid = int(src_ids[k]) if k < len(src_ids) else k
    cap.set(cv2.CAP_PROP_POS_FRAMES, sid)
    okf, fr = cap.read()
    if okf:
        img[30:30 + cam_h, panel_x:panel_x + cam_w] = cv2.resize(fr, (cam_w, cam_h))
    cv2.putText(img, "drone camera (recorded video)", (panel_x, 30 + cam_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    ph = 105
    plot(img, plot_y0, ph, height, (255, 200, 80), "height above ground (GCT units)", k, f"{height[k]:.2f}")
    plot(img, plot_y0 + ph + 15, ph, newr, (120, 230, 120), "new map points per frame", k, f"{newr[k]:,.0f}")
    ci = min(len(cnf) - 1, int(np.searchsorted(series_t, k)))
    plot(img, plot_y0 + 2 * (ph + 15), ph, cnf[:, 1], (80, 160, 255), "median depth confidence (saved frames)", k,
         f"{cnf[ci, 1]:.2f}")
    ly = H - 70
    cv2.putText(img, f"t {tsec[k]:5.1f} s   x{speed}   revisits {revisits}   cells {len(passes_now)}",
                (panel_x, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    lx = panel_x
    for pc in range(1, 5):
        cv2.rectangle(img, (lx, ly + 18), (lx + 22, ly + 36), PASS_COL[pc], -1)
        cv2.putText(img, f"{pc}{'+' if pc == 4 else ''} pass{'es' if pc > 1 else ''}", (lx + 28, ly + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        lx += 140
    cv2.circle(img, (lx + 10, ly + 27), 6, (80, 230, 80), 2, cv2.LINE_AA)
    cv2.putText(img, "plant", (lx + 22, ly + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "Manta V3 - flight path from GCT poses", (30, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    writer.write(img)
    if j == frames_out - 1:
        cv2.imwrite(os.path.join(OUT, "flight_last.png"), img)
writer.release()
print("frames", frames_out)
