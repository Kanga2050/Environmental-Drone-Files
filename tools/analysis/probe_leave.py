
import numpy as np
from playwright.sync_api import sync_playwright

OUT = (r"C:\Users\shaur\AppData\Local\Temp\probe"
       r"\c--Users-shaur-OneDrive-Desktop-Test"
       r"\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad")

def shot(pg, path, clip):
    pg.screenshot(path=path, clip=clip)
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)

def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0

for w, h, tag in ((1500, 900, "wide"), (1000, 900, "windowed")):
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=swiftshader",
                                     "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.goto("http://127.0.0.1:8090/", wait_until="networkidle")
        pg.wait_for_timeout(5000)
        pg.locator("[data-tab=frames]").click()
        pg.wait_for_timeout(600)

        pg.locator("#frameSlider").evaluate(
            "(el) => { el.value = 120; el.dispatchEvent(new Event('input')); }")
        pg.wait_for_timeout(1500)

        pg.keyboard.press("g")
        pg.wait_for_timeout(2500)
        pg.keyboard.press("o")
        pg.wait_for_timeout(1200)
        stage = pg.locator("#stage").bounding_box()

        side = int(min(stage["width"], stage["height"]) * 0.5)
        clip = {"x": stage["x"] + (stage["width"] - side) / 2,
                "y": stage["y"] + (stage["height"] - side) / 2,
                "width": side, "height": side}
        a = shot(pg, f"{OUT}\\leave_{tag}_in.png", clip)

        pg.keyboard.press("g")
        pg.wait_for_timeout(2500)
        c = shot(pg, f"{OUT}\\leave_{tag}_out.png", clip)

        print(f"{tag} {w}x{h}: correlation in-frame vs out-of-frame = "
              f"{corr(a, c):+.3f}")
        b.close()
