
import sys

from playwright.sync_api import sync_playwright

OUT = r"C:\Users\shaur\AppData\Local\Temp\probe\c--Users-shaur-OneDrive-Desktop-Test\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad"

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1500, "height": 950})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append("console:" + m.text)
          if m.type == "error" else None)
    pg.goto("http://127.0.0.1:8090/", wait_until="networkidle")
    pg.wait_for_timeout(6000)

    pg.keyboard.press("Escape")
    rows = pg.locator("#plantList .card")
    n = rows.count()
    print("plant rows found:", n)
    if n:
        rows.nth(0).click()
        pg.wait_for_timeout(600)
        pg.locator("[data-tab=sel]").click()
        pg.wait_for_timeout(1500)

    bars = pg.locator(".dist .bar").count()
    labs = pg.locator(".distlab").all_inner_texts()
    print("distribution bars rendered:", bars)
    print("section labels:", labs)
    print("warn banners:", pg.locator(".warn").count())

    pg.screenshot(path=OUT + r"\dist_panel.png")

    p = pg.locator("#tab-sel")
    if p.count() and p.is_visible():
        p.screenshot(path=OUT + r"\dist_panel_only.png")
    print("errors:", errs[:6] if errs else "none")
    b.close()
