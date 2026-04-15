
import os
import sys
import time

import cv2
import numpy as np
import trimesh
import viser
from playwright.sync_api import sync_playwright

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
from build_surface import (close_mask, diffuse_glb, keep_contiguous, mesh_edges,
                           srgb_to_linear)
import colorsys
import json

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_full"
OUT = os.path.dirname(os.path.abspath(__file__))
PORT = 8094
W, H = 1150, 780

cloud = np.load(os.path.join(ART, "cloud.npz"))
pts, cols = cloud["points"].astype(np.float32), cloud["colors"].astype(np.uint8)
surf = np.load(os.path.join(ART, "surface.npz"))
V, F, C = surf["vertices"], surf["faces"], surf["colors"]
plants = json.load(open(os.path.join(ART, "plants.json")))["plants"]

src, dst = mesh_edges(F, len(V))
palette = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, .85, 1.0))
           for i in range(len(plants))]

hl = C.copy()
for p, col in zip(plants, palette):
    k = p["id"]
    idx, d = surf[f"plant_{k}_idx"], surf[f"plant_{k}_block"]
    diag = float(surf[f"plant_{k}_diag"])
    sel = idx[np.isfinite(d) & (d <= 0.16 * diag)]
    if not len(sel):
        continue
    m = np.zeros(len(V), bool)
    m[sel] = True
    m = close_mask(m, src, dst, 2)
    m = keep_contiguous(m, V, src, dst, 0.05)
    hl[m] = col
    print(f"#{k} {p['category']:<10} diag {diag:.2f} -> {int(m.sum()):,} verts")

def mesh_of(colors):
    rgba = np.concatenate([srgb_to_linear(colors),
                           np.full((len(colors), 1), 255, np.uint8)], axis=1)
    return trimesh.Trimesh(vertices=V, faces=F, vertex_colors=rgba, process=False)

server = viser.ViserServer(host="0.0.0.0", port=PORT)
pc = server.scene.add_point_cloud("/cloud", points=pts, colors=cols,
                                  point_size=0.004, point_shape="rounded")
server.scene.add_light_ambient("/ambient", color=(255, 255, 255), intensity=1.8)
plain = server.scene.add_glb("/plain", glb_data=diffuse_glb(mesh_of(C)))
marked = server.scene.add_glb("/marked", glb_data=diffuse_glb(mesh_of(hl)))

lo, hi = pts.min(0), pts.max(0)
ctr = (lo + hi) / 2
rad = float(np.linalg.norm(hi - lo)) / 2
UP = (0.0, -1.0, 0.0)

samp = pts[np.random.default_rng(0).choice(len(pts), min(40000, len(pts)), replace=False)]
axis = np.linalg.svd(samp - samp.mean(0), full_matrices=False)[2][0]
axis = axis / np.linalg.norm(axis)
side = np.cross(axis, np.array([0.0, 1.0, 0.0]))
side = side / max(np.linalg.norm(side), 1e-9)
VIEWS = {
    "along": ctr - axis * 1.25 * rad - np.array([0.0, 0.35 * rad, 0.0]),
    "across": ctr + side * 1.15 * rad - np.array([0.0, 0.45 * rad, 0.0]),
    "top": ctr - np.array([0.0, 1.5 * rad, 0.0]),
}
UPS = {"top": tuple(float(v) for v in axis)}

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                      "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": W, "height": H})
    page.goto(f"http://localhost:{PORT}", wait_until="load")
    for _ in range(60):
        if server.get_clients():
            break
        time.sleep(0.5)
    client = list(server.get_clients().values())[0]

    for vname, eye in VIEWS.items():
        for lname, a, b, c in [("points", True, False, False),
                               ("surface", False, True, False),
                               ("marked", False, False, True)]:
            pc.visible, plain.visible, marked.visible = a, b, c
            client.camera.up_direction = UPS.get(vname, UP)
            client.camera.position = tuple(float(v) for v in eye)
            client.camera.look_at = tuple(float(v) for v in ctr)
            time.sleep(3.0)
            img = np.asarray(client.get_render(height=H, width=W,
                                               transport_format="png"))[:, :, :3]
            cv2.imwrite(os.path.join(OUT, f"yard_{vname}_{lname}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("captured", vname, lname)
    browser.close()

for v in VIEWS:
    ims = [cv2.imread(os.path.join(OUT, f"yard_{v}_{k}.png"))
           for k in ("points", "surface", "marked")]
    cv2.imwrite(os.path.join(OUT, f"yard_cmp_{v}.png"), np.hstack(ims))
print("wrote yard_cmp_wide.png / yard_cmp_high.png")
