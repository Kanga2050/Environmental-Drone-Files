
import json
import math

import numpy as np
from playwright.sync_api import sync_playwright

def fwd_of(world):
    m = np.array(world, float).reshape(4, 4).T
    return -m[:3, 2] / np.linalg.norm(m[:3, 2])

def up_of(world):
    m = np.array(world, float).reshape(4, 4).T
    return m[:3, 1] / np.linalg.norm(m[:3, 1])

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=swiftshader",
                                 "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1000, "height": 900})
    pg.goto("http://127.0.0.1:8090/", wait_until="networkidle")
    pg.wait_for_timeout(5000)
    pg.locator("[data-tab=frames]").click()
    pg.wait_for_timeout(500)
    pg.locator("#frameSlider").evaluate(
        "(el) => { el.value = 120; el.dispatchEvent(new Event('input')); }")
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
    pg.wait_for_timeout(600)
    c = pg.evaluate("window.__dbg()")
    pg.mouse.up()
    b.close()

fa, fc = fwd_of(a["world"]), fwd_of(c["world"])
ua, uc = up_of(a["world"]), up_of(c["world"])
turn = math.degrees(math.acos(float(np.clip(fa @ fc, -1, 1))))
rollc = math.degrees(math.acos(float(np.clip(ua @ uc, -1, 1))))
move = float(np.linalg.norm(np.array(a["pos"]) - np.array(c["pos"])))

print(f"{'':22s} {'in frame':>12s} {'after 1px drag':>16s}")
print(f"{'frame-locked':22s} {str(a['lock']):>12s} {str(c['lock']):>16s}")
print(f"{'fov (deg)':22s} {a['fov']:12.2f} {c['fov']:16.2f}")
print(f"{'yaw (deg)':22s} {math.degrees(a['yaw']):12.3f} "
      f"{math.degrees(c['yaw']):16.3f}")
print(f"{'pitch (deg)':22s} {math.degrees(a['pitch']):12.3f} "
      f"{math.degrees(c['pitch']):16.3f}")
print(f"{'roll (deg)':22s} {math.degrees(a['roll']):12.3f} "
      f"{math.degrees(c['roll']):16.3f}")
print()
print(f"view direction turned by {turn:.3f} deg   (a 1px drag should be 0.183)")
print(f"up vector rotated by     {rollc:.3f} deg")
print(f"camera moved             {move:.5f} u")
