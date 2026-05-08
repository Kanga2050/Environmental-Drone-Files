
import sys

from playwright.sync_api import sync_playwright

PROBE = """() => {
  const stage = document.getElementById('stage');
  const cv = document.getElementById('gl');
  const ph = document.getElementById('framePhoto');
  const s = stage.getBoundingClientRect(), p = ph.getBoundingClientRect();
  return {
    dpr: window.devicePixelRatio,
    stage: [s.width, s.height],
    canvas_client: [cv.clientWidth, cv.clientHeight],
    canvas_attr: [cv.width, cv.height],
    photo: [p.x - s.x, p.y - s.y, p.width, p.height],
    photo_natural: [ph.naturalWidth, ph.naturalHeight],
  };
}"""

for scale in (1.0, 1.5):
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=swiftshader",
                                     "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1500, "height": 900},
                        device_scale_factor=scale)
        pg.goto("http://127.0.0.1:8090/", wait_until="networkidle")
        pg.wait_for_timeout(5000)
        pg.locator("[data-tab=frames]").click()
        pg.wait_for_timeout(800)
        pg.keyboard.press("g")
        pg.wait_for_timeout(2500)

        r = pg.evaluate(PROBE)
        print(f"--- device_scale_factor {scale} ---")
        print(f"  devicePixelRatio      {r['dpr']}")
        print(f"  stage (CSS px)        {r['stage']}")
        print(f"  canvas clientW/H      {r['canvas_client']}")
        print(f"  canvas width/height   {r['canvas_attr']}")
        print(f"  photo x,y,w,h in stage{[round(v, 1) for v in r['photo']]}")
        sw, sh = r["stage"]
        px, py, pw_, ph_ = r["photo"]
        over = (px < -1 or py < -1 or px + pw_ > sw + 1 or py + ph_ > sh + 1)
        print(f"  photo fits inside stage: {not over}"
              + ("" if not over else "   <-- overflows"))
        pg.screenshot(path=(r"C:\Users\shaur\AppData\Local\Temp\probe"
                            r"\c--Users-shaur-OneDrive-Desktop-Test"
                            r"\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad"
                            rf"\stand_dpr{str(scale).replace('.', '_')}.png"))
        b.close()
