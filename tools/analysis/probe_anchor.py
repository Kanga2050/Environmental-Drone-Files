
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

OUT = (r"C:\Users\shaur\AppData\Local\Temp\probe"
       r"\c--Users-shaur-OneDrive-Desktop-Test"
       r"\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad")

def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
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
        pg.wait_for_timeout(500)
        pg.locator("#frameSlider").evaluate(
            "(el)=>{el.value=120;el.dispatchEvent(new Event('input'))}")
        pg.wait_for_timeout(1200)
        pg.locator("#frameEnter").click()
        pg.locator("#frameEnter").evaluate("(el) => el.blur()")
        pg.wait_for_timeout(2500)
        pg.wait_for_timeout(800)

        pg.evaluate("""() => {
          for (const id of ['overlaySvg','labels','hud','tools','panel'])
            { const e = document.getElementById(id); if (e) e.style.display='none'; }
        }""")
        pg.wait_for_timeout(400)

        st = pg.locator("#stage").bounding_box()
        ar = 518 / 420
        vw, vh = st["width"], st["width"] / ar
        if vh > st["height"]:
            vh, vw = st["height"], st["height"] * ar

        k = 0.7
        clip = {"x": st["x"] + (st["width"] - vw * k) / 2,
                "y": st["y"] + (st["height"] - vh * k) / 2,
                "width": vw * k, "height": vh * k}
        pg.screenshot(path=f"{OUT}\\an_{tag}_in.png", clip=clip)

        cx, cy = st["x"] + st["width"] / 2, st["y"] + st["height"] / 2
        pg.mouse.move(cx, cy)
        pg.mouse.down()
        pg.mouse.move(cx + 1, cy)
        pg.wait_for_timeout(900)
        pg.mouse.up()
        pg.wait_for_timeout(600)
        pg.screenshot(path=f"{OUT}\\an_{tag}_out.png", clip=clip)
        b.close()

    a = np.asarray(Image.open(f"{OUT}\\an_{tag}_in.png").convert("L"), np.float32)
    c = np.asarray(Image.open(f"{OUT}\\an_{tag}_out.png").convert("L"), np.float32)
    print(f"{tag} {w}x{h}: same-rect correlation {corr(a, c):+.4f}, "
          f"mean abs diff {np.abs(a - c).mean():.2f}")
