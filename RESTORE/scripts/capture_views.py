"""Capture what viser actually renders, rather than what I think it renders.

My offline rasteriser draws vertex colours straight to pixels. The browser does
not: a mesh goes through a lit PBR material, a point cloud does not. The two can
carry identical colours and still look nothing alike on screen, which is exactly
the kind of difference an offline check cannot see.

Serves the same scene view_map.py does on a spare port, drives a headless
Chromium to it, and asks viser for a real render from that client.
"""
import os
import sys
import time

import cv2
import numpy as np
import trimesh
import viser
from playwright.sync_api import sync_playwright

sys.path.insert(0, r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map")
from build_surface import diffuse_glb, srgb_to_linear

ART = r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts"
OUT = os.path.dirname(os.path.abspath(__file__))
PORT = 8099
W, H = 1100, 760

cloud = np.load(os.path.join(ART, "cloud.npz"))
pts, cols = cloud["points"].astype(np.float32), cloud["colors"].astype(np.uint8)

surf = np.load(os.path.join(ART, "surface.npz"))
V, F, C = surf["vertices"], surf["faces"], surf["colors"]
rgba = np.concatenate([srgb_to_linear(C), np.full((len(C), 1), 255, np.uint8)], axis=1)
mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=rgba, process=False)

server = viser.ViserServer(host="0.0.0.0", port=PORT)
pc = server.scene.add_point_cloud("/cloud", points=pts, colors=cols,
                                  point_size=0.004, point_shape="rounded")
mh = server.scene.add_glb("/surface", glb_data=diffuse_glb(mesh))
server.scene.add_light_ambient("/ambient", color=(255, 255, 255), intensity=1.8)

# The reconstruction's world frame is the first camera's: +z points into the
# scene and -y is up, so a viewpoint must sit at lower z than its subject to be
# looking at the front of it.
CENTER = np.array([0.10, -0.12, 0.71])
VIEWS = {
    "front": CENTER + np.array([0.10, -0.12, -0.45]),
    "side": CENTER + np.array([0.42, -0.16, -0.28]),
}
UP = (0.0, -1.0, 0.0)
LAYERS = [("points", True, False), ("surface", False, True)]

shots = {}
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                      "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": W, "height": H})
    page.goto(f"http://localhost:{PORT}", wait_until="load")

    for _ in range(60):
        if server.get_clients():
            break
        time.sleep(0.5)
    clients = server.get_clients()
    if not clients:
        print("no client connected")
        browser.close()
        sys.exit(1)
    client = list(clients.values())[0]
    print(f"client connected: {list(clients)}")

    for vname, eye in VIEWS.items():
        for lname, show_pc, show_mesh in LAYERS:
            key = f"{vname}_{lname}"
            pc.visible = show_pc
            mh.visible = show_mesh
            client.camera.up_direction = UP
            client.camera.position = tuple(float(v) for v in eye)
            client.camera.look_at = tuple(float(v) for v in CENTER)
            time.sleep(3.0)
            img = np.asarray(
                client.get_render(height=H, width=W, transport_format="png"))[:, :, :3]
            shots[key] = img
            cv2.imwrite(os.path.join(OUT, f"shot_{key}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  captured {key}")
    browser.close()

# Compare only where each actually drew something -- the black background would
# otherwise dominate, and the two layers cover different fractions of the frame.
print()
for key, im in shots.items():
    f = im.reshape(-1, 3).astype(np.float32)
    lit = f.sum(axis=1) > 30
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    print(f"{key:16s} luma {f[lit].mean():6.1f}  sat {hsv[lit, 1].mean():6.1f}  "
          f"coverage {lit.mean():5.1%}")

print()
for v in VIEWS:
    a = shots[f"{v}_points"].reshape(-1, 3).astype(np.float32)
    b = shots[f"{v}_surface"].reshape(-1, 3).astype(np.float32)
    # Only pixels both layers drew, so the comparison is like for like.
    both = (a.sum(axis=1) > 30) & (b.sum(axis=1) > 30)
    print(f"{v:8s} on {both.mean():5.1%} shared pixels: "
          f"points {a[both].mean():6.1f} vs surface {b[both].mean():6.1f} "
          f"({100 * (b[both].mean() / max(a[both].mean(), 1e-6) - 1):+.1f}%)")

for v in VIEWS:
    pair = np.hstack([shots[f"{v}_points"], shots[f"{v}_surface"]])
    cv2.imwrite(os.path.join(OUT, f"compare_{v}.png"),
                cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
print("wrote compare_front.png / compare_side.png")
