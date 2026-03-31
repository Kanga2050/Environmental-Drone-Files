
import colorsys, json, os, sys, time
import cv2, numpy as np, viser
from playwright.sync_api import sync_playwright
from scipy.spatial import cKDTree

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_final"
OUT = os.path.dirname(os.path.abspath(__file__))
PORT, W, H = 8093, 1200, 800

cloud = np.load(os.path.join(ART, "cloud.npz"))
P = cloud["points"].astype(np.float32)
C = cloud["colors"].astype(np.uint8)
plants = json.load(open(os.path.join(ART, "plants.json")))["plants"]
meta = json.load(open(os.path.join(ART, "meta.json")))
voxel = float(meta.get("voxel_size", 0.005))
inst = np.load(os.path.join(ART, "plant_points.npz"))
pal = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, .85, 1.0))
       for i in range(len(plants))]

tree = cKDTree(P)
hl = C.copy()
n_marked = 0
for p, col in zip(plants, pal):
    k = f"pts_{p['id']}"
    if k not in inst or not len(inst[k]):
        continue
    hits = [h for h in tree.query_ball_point(inst[k].astype(np.float32),
                                             r=voxel * 2.5, workers=-1) if h]
    if not hits:
        continue
    idx = np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]))
    hl[idx] = col
    n_marked += len(idx)
print(f"highlighted {n_marked:,} points across {len(plants)} plants")

server = viser.ViserServer(host="0.0.0.0", port=PORT)
plain = server.scene.add_point_cloud("/plain", points=P, colors=C,
                                     point_size=0.0035, point_shape="rounded")
mark = server.scene.add_point_cloud("/marked", points=P, colors=hl,
                                    point_size=0.0035, point_shape="rounded")

lo, hi = P.min(0), P.max(0)
ctr = (lo + hi) / 2
rad = float(np.linalg.norm(hi - lo)) / 2
samp = P[np.random.default_rng(0).choice(len(P), 60000, replace=False)]
axis = np.linalg.svd(samp - samp.mean(0), full_matrices=False)[2][0]
axis /= np.linalg.norm(axis)
side = np.cross(axis, np.array([0.0, 1.0, 0.0]))
side /= max(np.linalg.norm(side), 1e-9)
VIEWS = {
    "along": ctr - axis * 1.05 * rad - np.array([0.0, 0.30 * rad, 0.0]),
    "across": ctr + side * 0.95 * rad - np.array([0.0, 0.40 * rad, 0.0]),
    "top": ctr - np.array([0.0, 1.3 * rad, 0.0]),
}
UPS = {"top": tuple(float(v) for v in axis)}

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                 "--enable-unsafe-swiftshader"])
    page = b.new_page(viewport={"width": W, "height": H})
    page.goto(f"http://localhost:{PORT}", wait_until="load")
    for _ in range(90):
        if server.get_clients():
            break
        time.sleep(0.5)
    cl = list(server.get_clients().values())[0]
    for vn, eye in VIEWS.items():
        for ln, a, m in [("plain", True, False), ("marked", False, True)]:
            plain.visible, mark.visible = a, m
            cl.camera.up_direction = UPS.get(vn, (0.0, -1.0, 0.0))
            cl.camera.position = tuple(float(v) for v in eye)
            cl.camera.look_at = tuple(float(v) for v in ctr)
            time.sleep(5.0)
            img = np.asarray(cl.get_render(height=H, width=W, transport_format="png"))[:, :, :3]
            cv2.imwrite(os.path.join(OUT, f"fin_{vn}_{ln}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("captured", vn, ln, flush=True)
    b.close()

for vn in VIEWS:
    ims = [cv2.imread(os.path.join(OUT, f"fin_{vn}_{k}.png")) for k in ("plain", "marked")]
    cv2.imwrite(os.path.join(OUT, f"fin_cmp_{vn}.png"), np.hstack(ims))
print("done")
