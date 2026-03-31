
import numpy as np
from playwright.sync_api import sync_playwright

def mat(a):
    return np.array(a, float).reshape(4, 4).T

def project(P, world, proj, rect):
    view = np.linalg.inv(mat(world))
    X = np.c_[P, np.ones(len(P))] @ view.T
    C = X @ mat(proj).T
    w = C[:, 3]
    ndc = C[:, :3] / w[:, None]
    x = rect["x"] + (ndc[:, 0] * 0.5 + 0.5) * rect["w"]
    y = rect["y"] + (0.5 - ndc[:, 1] * 0.5) * rect["h"]
    return np.c_[x, y], w

for w_, h_, tag in ((1500, 900, "wide"), (1000, 900, "windowed")):
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=swiftshader",
                                     "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": w_, "height": h_})
        pg.goto("http://127.0.0.1:8090/", wait_until="networkidle")
        pg.wait_for_timeout(5000)
        pg.locator("[data-tab=frames]").click()
        pg.wait_for_timeout(500)
        pg.locator("#frameSlider").evaluate(
            "(el)=>{el.value=120;el.dispatchEvent(new Event('input'))}")
        pg.wait_for_timeout(1200)
        pg.locator("#frameEnter").click()
        pg.locator("#frameEnter").evaluate("(el) => el.blur()")
        pg.wait_for_timeout(2500)
        a = pg.evaluate("window.__dbg()")

        st = pg.locator("#stage").bounding_box()
        cx, cy = st["x"] + st["width"] / 2, st["y"] + st["height"] / 2
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        pg.mouse.move(cx + 1, cy)
        pg.wait_for_timeout(800)
        pg.mouse.up()
        c = pg.evaluate("window.__dbg()")
        b.close()

    W = mat(a["world"])
    pos, R = W[:3, 3], W[:3, :3]
    g = np.linspace(-0.32, 0.32, 9)
    dirs = [np.array([np.tan(u), np.tan(v), -1.0]) for u in g for v in g]
    P = np.array([pos + R @ (d / np.linalg.norm(d)) * 3.0 for d in dirs])

    sa, wa = project(P, a["world"], a["proj"], a["rect"])
    sc, wc = project(P, c["world"], c["proj"], c["rect"])
    m = (wa > 0) & (wc > 0)
    d = np.linalg.norm(sa[m] - sc[m], axis=1)
    print(f"--- {tag} {w_}x{h_} ---")
    print(f"  in : fov {a['fov']:.2f}  rect {[round(v) for v in a['rect'].values()]}")
    print(f"  out: fov {c['fov']:.2f}  rect {[round(v) for v in c['rect'].values()]}")
    print(f"  screen displacement over {m.sum()} points: "
          f"median {np.median(d):.1f} px, max {d.max():.1f} px")

    ca = np.array([a["rect"]["x"] + a["rect"]["w"] / 2,
                   a["rect"]["y"] + a["rect"]["h"] / 2])
    cc = np.array([c["rect"]["x"] + c["rect"]["w"] / 2,
                   c["rect"]["y"] + c["rect"]["h"] / 2])
    ra = np.linalg.norm(sa[m] - ca, axis=1)
    rc = np.linalg.norm(sc[m] - cc, axis=1)
    k = np.median(rc / np.maximum(ra, 1e-6))
    print(f"  radial scale factor about centre: {k:.4f} "
          f"({'no zoom' if abs(k - 1) < 0.02 else 'ZOOM'})")
