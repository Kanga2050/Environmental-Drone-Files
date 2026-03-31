
import numpy as np
from playwright.sync_api import sync_playwright

OUT = (r"C:\Users\shaur\AppData\Local\Temp\probe"
       r"\c--Users-shaur-OneDrive-Desktop-Test"
       r"\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad")

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=swiftshader",
                                 "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1400, "height": 900})
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
    pg.screenshot(path=f"{OUT}\\turn_0.png")

    st = pg.locator("#stage").bounding_box()
    cx, cy = st["x"] + st["width"] / 2, st["y"] + st["height"] / 2
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    for k in range(1, 9):
        pg.mouse.move(cx + k * 22, cy)
        pg.wait_for_timeout(120)
    pg.mouse.up()
    pg.wait_for_timeout(1200)
    c = pg.evaluate("window.__dbg()")
    pg.screenshot(path=f"{OUT}\\turn_1.png")
    b.close()

move = float(np.linalg.norm(np.array(a["pos"]) - np.array(c["pos"])))
dyaw = np.degrees(c["yaw"] - a["yaw"])
print(f"turned {dyaw:+.1f} deg in yaw")
print(f"camera moved {move:.6f} u   (must be 0: the pinned photo is only")
print(f"                              correct under pure rotation)")
print(f"pitch {np.degrees(a['pitch']):.2f} -> {np.degrees(c['pitch']):.2f}, "
      f"roll {np.degrees(a['roll']):.2f} -> {np.degrees(c['roll']):.2f}")
print(f"anchored photo present after turn: {c.get('lock') is False}")
