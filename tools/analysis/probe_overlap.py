
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = (r"C:\Users\shaur\AppData\Local\Temp\probe"
       r"\c--Users-shaur-OneDrive-Desktop-Test"
       r"\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad")

def edges(a):
    gx = np.abs(np.diff(a, axis=1))[:-1, :]
    gy = np.abs(np.diff(a, axis=0))[:, :-1]
    return gx + gy

def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=swiftshader",
                                 "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1500, "height": 900})
    pg.goto("http://127.0.0.1:8090/", wait_until="networkidle")
    pg.wait_for_timeout(5000)
    pg.locator("[data-tab=frames]").click()
    pg.wait_for_timeout(500)
    pg.locator("#frameSlider").evaluate(
        "(el) => { el.value = 120; el.dispatchEvent(new Event('input')); }")
    pg.wait_for_timeout(1200)

    pg.locator("#showOwn").check()
    pg.locator("#showOwn").evaluate("(el) => el.blur()")
    pg.wait_for_timeout(400)
    pg.locator("#frameEnter").click()
    pg.locator("#frameEnter").evaluate("(el) => el.blur()")
    pg.wait_for_timeout(3000)

    st = pg.locator("#stage").bounding_box()
    ar = 518 / 420
    vw, vh = st["width"], st["width"] / ar
    if vh > st["height"]:
        vh, vw = st["height"], st["height"] * ar
    clip = {"x": st["x"] + (st["width"] - vw) / 2,
            "y": st["y"] + (st["height"] - vh) / 2, "width": vw, "height": vh}

    pg.locator("#opacitySlider").evaluate(
        "(el)=>{el.value=100;el.dispatchEvent(new Event('input'))}")
    pg.wait_for_timeout(900)
    pg.screenshot(path=f"{OUT}\\ov_photo.png", clip=clip)

    pg.locator("#opacitySlider").evaluate(
        "(el)=>{el.value=0;el.dispatchEvent(new Event('input'))}")
    pg.wait_for_timeout(900)
    pg.screenshot(path=f"{OUT}\\ov_points.png", clip=clip)

    pg.locator("#opacitySlider").evaluate(
        "(el)=>{el.value=55;el.dispatchEvent(new Event('input'))}")
    pg.wait_for_timeout(900)
    pg.screenshot(path=f"{OUT}\\ov_blend.png", clip=clip)
    b.close()

ph = np.asarray(Image.open(f"{OUT}\\ov_photo.png").convert("L"), np.float32)
pt = np.asarray(Image.open(f"{OUT}\\ov_points.png").convert("L"), np.float32)
ep, et = edges(ph), edges(pt)
print(f"letterbox {ph.shape[1]}x{ph.shape[0]}")
print(f"edge correlation photo vs points at (0,0): {corr(ep, et):+.4f}")

best = (None, 0, 0)
R = 24
for dv in range(-R, R + 1, 2):
    for du in range(-R, R + 1, 2):
        a = np.roll(np.roll(et, dv, axis=0), du, axis=1)
        c = corr(ep, a)
        if best[0] is None or c > best[0]:
            best = (c, du, dv)
print(f"best edge correlation {best[0]:+.4f} at shift du={best[1]}, dv={best[2]}")
print(f"as a fraction of the view: {best[1] / ph.shape[1] * 100:+.2f}% x, "
      f"{best[2] / ph.shape[0] * 100:+.2f}% y")
