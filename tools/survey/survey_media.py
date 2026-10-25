import argparse, json, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--mount", choices=["down", "forward"], required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sample", type=int, default=30_000_000)
ap.add_argument("--hero", type=int, default=-1, help="saved frame id for the hero view (-1 = auto)")
ap.add_argument("--back", type=float, default=0.25)
ap.add_argument("--lift", type=float, default=0.08)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ART = a.map
meta = json.load(open(os.path.join(ART, "meta.json")))
plants = json.load(open(os.path.join(ART, "plants.json")))["plants"]
P = np.load(os.path.join(ART, "poses.npy"))
fps = float(meta.get("fps", 30))
src_ids = meta.get("frame_source_ids") or list(range(len(P)))

c = np.load(os.path.join(ART, "cloud.npz"), mmap_mode="r")
npts = len(c["points"])
rng = np.random.default_rng(3)
sel = np.sort(rng.choice(npts, min(npts, a.sample), replace=False))
pts = np.asarray(c["points"])[sel].astype(np.float32)
col = np.asarray(c["colors"])[sel]
segp = os.path.join(ART, "plant_seg.npz")
tag = np.load(segp)["tags"][sel].astype(np.int32) if os.path.exists(segp) else np.full(len(sel), -1, np.int32)
del c

up = -P[:, :3, 2].mean(0) if a.mount == "down" else -P[:, :3, 1].mean(0)
up /= np.linalg.norm(up)
h = pts @ up
ground = np.percentile(h, 3)
e1 = np.cross(up, [0, 0, 1.0])
if np.linalg.norm(e1) < 1e-3:
    e1 = np.cross(up, [1.0, 0, 0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(up, e1)
cam = P[:, :3, 3]
pal = np.array([cv2.cvtColor(np.uint8([[[int(179 * ((i * 0.618) % 1)), 210, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
                for i in range(max(1, len(plants)))], np.uint8)
idx_of = {p["id"]: k for k, p in enumerate(plants)}


def label(p):
    hh = p["hierarchy"]
    lv = hh["level"]
    if lv in hh:
        return hh[lv]["name"], hh[lv]["conf"], lv
    return "plant", p.get("category_confidence", 0.0), "plant"


def health_txt(p):
    hp = p.get("health")
    return f" · {hp['label']}" if hp else ""


bgr = col[:, ::-1].copy()
tagged = tag >= 0
tint = bgr.copy()
tint[tagged] = (0.35 * bgr[tagged] + 0.65 * pal[np.clip([idx_of.get(t, 0) for t in tag[tagged]], 0, len(pal) - 1)]).astype(np.uint8)

# top-down plant map with path
S = 1800
u, v = pts @ e1, pts @ e2
cu_, cv_ = np.r_[u, cam @ e1], np.r_[v, cam @ e2]
lo = np.percentile(cu_, 2), np.percentile(cv_, 2)
hi = np.percentile(cu_, 98), np.percentile(cv_, 98)
span = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.04
mu, mv = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
sc = S / span
tx = lambda x, y: (((x - mu) * sc + S / 2).astype(int), (S / 2 - (y - mv) * sc).astype(int))
px, py = tx(u, v)
ok = (px >= 0) & (px < S) & (py >= 0) & (py < S)
o = np.argsort(h[ok])
img = np.full((S, S, 3), 16, np.uint8)
img[py[ok][o], px[ok][o]] = (tint[ok][o] * 0.85).astype(np.uint8)
img = cv2.dilate(img, np.ones((2, 2), np.uint8))
from scipy.ndimage import uniform_filter1d
cs_ = uniform_filter1d(cam, 31, axis=0, mode="nearest")[::5]
cx, cy = tx(cs_ @ e1, cs_ @ e2)
cv2.polylines(img, [np.c_[cx, cy]], False, (255, 255, 255), 1, cv2.LINE_AA)
for p in plants:
    ccn = np.array(p["centroid"])
    x, y = tx(np.array([ccn @ e1]), np.array([ccn @ e2]))
    x, y = int(x[0]), int(y[0])
    name, conf, lv = label(p)
    deep = lv in ("family", "genus", "species")
    cv2.circle(img, (x, y), 9, (255, 255, 255), 2, cv2.LINE_AA)
    t = f"{p['id']}" + (f" {name}" if deep else "")
    for th, cc in ((3, (0, 0, 0)), (1, (120, 255, 255) if deep else (255, 255, 255))):
        cv2.putText(img, t, (x + 11, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55 if deep else 0.45, cc, th, cv2.LINE_AA)
cv2.imwrite(os.path.join(a.out, f"Manta V3 - {a.name} Plant Map.png"), img)

hw = tuple(meta["model_hw"])


def render(c2w, K, W, H, colors, labels=False, fov_scale=1.0):
    w2c = np.linalg.inv(c2w)
    cm = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = cm[:, 2]
    m = z > 1e-3
    uv = cm[m] @ K.T
    x = uv[:, 0] / uv[:, 2]
    y = uv[:, 1] / uv[:, 2]
    k = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, zz, cc = x[k].astype(int), y[k].astype(int), z[m][k], colors[m][k]
    o = np.argsort(-zz)
    out = np.full((H, W, 3), 14, np.uint8)
    near = zz[o] < np.percentile(zz, 30)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        out[np.clip(y[o] + dy, 0, H - 1), np.clip(x[o] + dx, 0, W - 1)] = cc[o]
    on = o[near]
    for dx, dy in ((2, 0), (0, 2), (2, 1), (1, 2), (2, 2)):
        out[np.clip(y[on] + dy, 0, H - 1), np.clip(x[on] + dx, 0, W - 1)] = cc[on]
    if labels:
        items = []
        for p in plants:
            q = w2c[:3, :3] @ np.array(p["centroid"]) + w2c[:3, 3]
            if q[2] <= 0.05:
                continue
            s = K @ q
            X, Y = s[0] / s[2], s[1] / s[2]
            if 10 <= X < W - 10 and 10 <= Y < H - 10:
                items.append((q[2], int(X), int(Y), p))
        items.sort(key=lambda t: t[0])
        placed = []
        for d, X, Y, p in items[:14]:
            name, conf, lv = label(p)
            form = p["hierarchy"].get("form", {}).get("name", p["category"])
            txt = f"#{p['id']} {form}" + (f" {name}" if lv in ("family", "genus", "species") else f" {conf:.2f}") + health_txt(p)
            fs = 0.62
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            x0, y0 = X - tw // 2 - 8, Y - 36
            while any(x0 < r[2] and x0 + tw + 16 > r[0] and y0 < r[3] and y0 + th + 12 > r[1] for r in placed) and y0 > 0:
                y0 -= th + 16
            placed.append((x0 - 4, y0 - 4, x0 + tw + 20, y0 + th + 16))
            ov = out.copy()
            cv2.rectangle(ov, (x0, y0), (x0 + tw + 16, y0 + th + 12), (30, 30, 30), -1)
            out = cv2.addWeighted(ov, 0.8, out, 0.2, 0)
            cv2.circle(out, (x0 + 8, y0 + th // 2 + 6), 4, tuple(int(v) for v in pal[idx_of[p["id"]] % len(pal)]), -1)
            cv2.putText(out, txt, (x0 + 16, y0 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.line(out, (X, y0 + th + 12), (X, Y), (200, 200, 200), 1, cv2.LINE_AA)
    return out


saved = meta["saved_frame_ids"]
OW, OH = 640, int(640 * hw[0] / hw[1])
vis = []
for fid in saved:
    d = np.load(os.path.join(ART, "frames", f"{fid:06d}.npz"))
    vis.append((sum(1 for p in plants if fid in p.get("frames", [])), fid))
vis.sort(reverse=True)
picked = []
gap = max(300, len(P) // 8)
for cnt, fid in vis:
    if all(abs(fid - q) > gap for q in picked):
        picked.append(fid)
    if len(picked) == 4:
        break
picked.sort()


def frame_pose(fid):
    d = np.load(os.path.join(ART, "frames", f"{fid:06d}.npz"))
    E = np.eye(4)
    E[:3] = d["extrinsic"]
    K = d["intrinsic"].astype(np.float64).copy()
    return np.linalg.inv(E), K


rows = []
for fid in picked:
    c2w, K = frame_pose(fid)
    K2 = K.copy()
    K2[:2] *= OW / hw[1]
    cimg = cv2.resize(cv2.imread(os.path.join(ART, "frames", f"{fid:06d}.jpg")), (OW, OH))
    grey = (bgr.astype(np.float32).mean(1, keepdims=True).repeat(3, 1) * 0.45).astype(np.uint8)
    grey[tagged] = tint[tagged]
    rows.append(np.hstack([cimg, np.full((OH, 8, 3), 255, np.uint8), render(c2w, K2, OW, OH, bgr),
                           np.full((OH, 8, 3), 255, np.uint8), render(c2w, K2, OW, OH, grey)]))
panel = np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in rows])[:-8]
cv2.imwrite(os.path.join(a.out, f"Manta V3 - {a.name} Camera, Point Cloud and Plant Points.png"), panel)

hero_fid = a.hero if a.hero >= 0 else picked[0]
c2w, K = frame_pose(hero_fid)
fwd = c2w[:3, 2]
pose = c2w.copy()
pose[:3, 3] = c2w[:3, 3] - fwd * a.back + up * a.lift
W, H = 1920, 1080
Kh = np.array([[W * 0.55, 0, W / 2], [0, W * 0.55, H / 2], [0, 0, 1]])
hero = render(pose, Kh, W, H, tint, labels=True)
cv2.imwrite(os.path.join(a.out, f"Manta V3 - {a.name} 3D View.png"), hero)

cap = cv2.VideoCapture(a.video)
cams = []
for k, fid in enumerate(picked):
    sid = int(src_ids[fid]) if fid < len(src_ids) else fid
    cap.set(cv2.CAP_PROP_POS_FRAMES, sid)
    okf, fr = cap.read()
    if okf:
        p_ = os.path.join(a.out, f"Manta V3 - {a.name} Camera {k + 1}.jpg")
        cv2.imwrite(p_, cv2.resize(fr, (1600, 900)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cams.append(p_)

rank = {"species": 4, "genus": 3, "family": 2, "form": 1, "plant": 0}
best = sorted(plants, key=lambda p: (-rank.get(p["hierarchy"]["level"], 0), -p.get("views", 0)))
tiles, crops = [], []
for p in best[:24]:
    im = cv2.imread(os.path.join(ART, "crops", p["crop"])) if p.get("crop") else None
    if im is None:
        continue
    if len(crops) < 12:
        name, conf, lv = label(p)
        form = p["hierarchy"].get("form", {}).get("name", p["category"])
        cap_ = f"#{p['id']} {form}" + (f" · {name}" if lv in ("family", "genus", "species") else "") + health_txt(p)
        pth = os.path.join(a.out, f"Manta V3 - {a.name} Plant {len(crops) + 1:02d}.jpg")
        cv2.imwrite(pth, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        crops.append((pth, cap_))
    hh_, ww_ = im.shape[:2]
    s_ = 240 / max(hh_, ww_)
    im = cv2.resize(im, (max(1, int(ww_ * s_)), max(1, int(hh_ * s_))))
    t = np.full((300, 260, 3), 255, np.uint8)
    t[10:10 + im.shape[0], 10 + (240 - im.shape[1]) // 2:10 + (240 - im.shape[1]) // 2 + im.shape[1]] = im
    name, conf, lv = label(p)
    form = p["hierarchy"].get("form", {}).get("name", p["category"])
    cv2.putText(t, f"#{p['id']}  {form}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    sub = f"{lv}: {name} {conf:.0%}" if lv in ("family", "genus", "species") else f"{p['views']} views"
    if p.get("health"):
        sub += f" | {p['health']['label']}"
    cv2.putText(t, sub, (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 90, 160), 1, cv2.LINE_AA)
    tiles.append(t)
while tiles and len(tiles) % 6:
    tiles.append(np.full((300, 260, 3), 255, np.uint8))
if tiles:
    grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(os.path.join(a.out, f"Manta V3 - {a.name} Plant Identifications.png"), grid)

forms = {}
for p in plants:
    f_ = p["hierarchy"].get("form", {}).get("name", p["category"])
    forms[f_] = forms.get(f_, 0) + 1
deep = [(p["id"], *label(p)) for p in plants if p["hierarchy"]["level"] in ("family", "genus", "species")]
health = {}
for p in plants:
    if p.get("health"):
        health[p["health"]["label"]] = health.get(p["health"]["label"], 0) + 1
plant_pts = int(tagged.sum() / len(tagged) * npts)
veg = (bgr[:, 1].astype(int) * 2 - bgr[:, 0] - bgr[:, 2]) > 20
summary = dict(name=a.name, points=int(npts), frames=len(P), minutes=len(P) / fps / 60, plants=len(plants), forms=forms,
               deep=deep, health=health, plant_point_share=float(tagged.mean()), green_point_share=float(veg.mean()),
               picked=picked, hero=hero_fid, cams=cams, crops=crops,
               mean_plant_views=float(np.mean([p["views"] for p in plants])) if plants else 0)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if k not in ("crops", "cams")}, default=str))
