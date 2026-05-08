
import numpy as np
from playwright.sync_api import sync_playwright

OUT = (r"C:\Users\shaur\AppData\Local\Temp\probe"
       r"\c--Users-shaur-OneDrive-Desktop-Test"
       r"\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad")

def stand(pg, idx=120):
    pg.locator("[data-tab=frames]").click()
    pg.wait_for_timeout(400)
    pg.locator("#frameSlider").evaluate(
        f"(el)=>{{el.value={idx};el.dispatchEvent(new Event('input'))}}")
    pg.wait_for_timeout(1000)
    pg.locator("#frameEnter").click()
    pg.locator("#frameEnter").evaluate("(el) => el.blur()")
    pg.wait_for_timeout(2200)

def drag(pg, st, dx):
    cx, cy = st["x"] + st["width"] / 2, st["y"] + st["height"] / 2
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    pg.mouse.move(cx + dx, cy)
    pg.wait_for_timeout(500)
    pg.mouse.up()
    pg.wait_for_timeout(500)

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=swiftshader",
                                 "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    pg.goto("http://127.0.0.1:8090/", wait_until="networkidle")
    pg.wait_for_timeout(4000)

    pg.reload(wait_until="networkidle")
    pg.wait_for_timeout(5000)
    st = pg.locator("#stage").bounding_box()

    stand(pg)
    print(f"standing in frame        lock={pg.evaluate('window.__dbg().lock')}")
    drag(pg, st, 40)
    d = pg.evaluate("window.__dbg()")
    p_rot = np.array(d["pos"])
    print(f"after a 40px drag        anchored={d['anchored']:>5}  "
          f"(photo should survive)")

    pg.keyboard.down("KeyW")
    pg.wait_for_timeout(900)
    pg.keyboard.up("KeyW")
    pg.wait_for_timeout(600)
    d2 = pg.evaluate("window.__dbg()")
    moved = float(np.linalg.norm(np.array(d2["pos"]) - p_rot))
    print(f"after holding W          anchored={d2['anchored']:>5}  "
          f"(photo should be gone), camera moved {moved:.3f} u")

    stand(pg)
    drag(pg, st, 12)
    a3 = pg.evaluate("window.__dbg().anchored")
    pg.mouse.move(st["x"] + st["width"] / 2, st["y"] + st["height"] / 2)
    pg.mouse.wheel(0, -200)
    pg.wait_for_timeout(700)
    a4 = pg.evaluate("window.__dbg().anchored")
    print(f"wheel                    anchored {a3} -> {a4}  (should end -1)")

    pg.locator("[data-tab=plants]").click()
    pg.wait_for_timeout(900)
    card = pg.locator("#plantList .card").first
    pid = card.evaluate("(el) => el.dataset.i")
    card.locator(".head").click()
    pg.wait_for_timeout(2600)
    d3 = pg.evaluate("window.__dbg()")
    label = pg.locator("#frameVal").inner_text()
    print()
    print(f"clicked plant card {pid}      lock={d3['lock']}  now showing {label}")
    pg.screenshot(path=f"{OUT}\\plant_stand.png")
    b.close()
