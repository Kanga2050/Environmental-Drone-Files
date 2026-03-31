
import asyncio
import os
import sys

from playwright.async_api import async_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090/"
TAG = sys.argv[2] if len(sys.argv) > 2 else "v3d"

async def main():
    errors, logs = [], []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=[
            "--use-gl=angle", "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
        ])
        page = await browser.new_page(viewport={"width": 1500, "height": 900})
        page.on("console", lambda m: (logs.append(f"{m.type}: {m.text}"),
                                      errors.append(m.text) if m.type == "error" else None))
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        await page.goto(URL, wait_until="networkidle", timeout=120_000)

        await page.wait_for_function(
            "() => document.getElementById('loading').classList.contains('hidden')",
            timeout=180_000)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_01_start.png"))

        await page.mouse.move(700, 450)
        await page.keyboard.down("w")
        await page.wait_for_timeout(1200)
        await page.keyboard.up("w")
        await page.mouse.down()
        await page.mouse.move(920, 430, steps=12)
        await page.mouse.up()
        await page.wait_for_timeout(600)
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_02_fly.png"))

        cards = await page.query_selector_all("#plantList .card .head")
        print(f"plant cards: {len(cards)}")
        if cards:
            await cards[0].click()
            await page.wait_for_timeout(1800)
            await page.screenshot(path=os.path.join(OUT, f"{TAG}_03_plant.png"))

        await page.click('#tabs button[data-tab="frames"]')
        await page.evaluate("""() => {
            const s = document.getElementById('frameSlider');
            s.value = 120; s.dispatchEvent(new Event('input'));
        }""")
        await page.wait_for_timeout(800)
        await page.click("#frameEnter")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_04_frame.png"))

        await page.evaluate("""() => {
            const s = document.getElementById('opacitySlider');
            s.value = 0; s.dispatchEvent(new Event('input'));
        }""")
        await page.wait_for_timeout(1200)
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_05_frame_points.png"))

        await page.evaluate("""() => {
            const s = document.getElementById('opacitySlider');
            s.value = 100; s.dispatchEvent(new Event('input'));
        }""")
        await page.wait_for_timeout(800)
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_06_frame_photo.png"))

        await page.evaluate("""() => {
            const s = document.getElementById('opacitySlider');
            s.value = 0; s.dispatchEvent(new Event('input'));
            document.getElementById('showOwn').click();
        }""")
        await page.wait_for_timeout(3000)
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_07_own.png"))
        await page.evaluate("() => document.getElementById('showOwn').click()")

        await page.keyboard.down("Control")
        await page.mouse.move(400, 300)
        await page.mouse.down()
        await page.mouse.move(760, 640, steps=12)
        await page.mouse.up()
        await page.keyboard.up("Control")
        await page.wait_for_timeout(3500)
        in_frame_sel = await page.inner_text("#selBody")
        print("--- selection made while in frame view ---")
        print(in_frame_sel[:260].replace(chr(10), " | "))
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_09_selframe.png"))

        for mode, name in ((1, "height"), (2, "observations")):
            await page.evaluate("""(m) => {
                const s = document.getElementById('colorMode');
                s.value = String(m); s.dispatchEvent(new Event('change'));
            }""", mode)
            await page.wait_for_timeout(1500)
            await page.screenshot(path=os.path.join(OUT, f"{TAG}_10_color_{name}.png"))
        await page.evaluate("""() => {
            const s = document.getElementById('colorMode');
            s.value = '0'; s.dispatchEvent(new Event('change'));
        }""")

        await page.keyboard.press("Escape")
        await page.wait_for_timeout(400)
        await page.keyboard.down("Control")
        await page.mouse.move(520, 330)
        await page.mouse.down()
        await page.mouse.move(820, 620, steps=14)
        await page.mouse.up()
        await page.keyboard.up("Control")
        await page.wait_for_timeout(3500)
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_08_select.png"))

        sel = await page.inner_text("#selBody")
        print("--- selection panel ---")
        print(sel[:900])

        await browser.close()

    print("\n--- console errors ---")
    for e in errors:
        print("  ", e)
    if not errors:
        print("   none")
    return 1 if errors else 0

sys.exit(asyncio.run(main()))
