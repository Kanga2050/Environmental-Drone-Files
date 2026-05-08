
import asyncio, os, sys
from playwright.async_api import async_playwright
OUT = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091/"

async def main():
    errs = []
    async with async_playwright() as pw:
        b = await pw.chromium.launch(args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
        p = await b.new_page(viewport={"width":1500,"height":900})
        p.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
        p.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        await p.goto(URL, wait_until="networkidle", timeout=120000)
        await p.wait_for_function("() => document.getElementById('loading').classList.contains('hidden')", timeout=180000)
        await p.wait_for_timeout(2500)

        print("buildUp enabled:", await p.evaluate("() => !document.getElementById('buildUp').disabled"))
        await p.click('#tabs button[data-tab="view"]')
        await p.evaluate("() => { const s=document.getElementById('colorMode'); s.value='2'; s.dispatchEvent(new Event('change')); }")
        await p.wait_for_timeout(2000)
        await p.screenshot(path=os.path.join(OUT,"prov_01_observations.png"))
        print("hud:", await p.inner_text("#hudInfo"))

        await p.evaluate("() => { const s=document.getElementById('colorMode'); s.value='0'; s.dispatchEvent(new Event('change')); }")
        await p.click('#tabs button[data-tab="frames"]')
        await p.evaluate("() => { const s=document.getElementById('frameSlider'); s.value=20; s.dispatchEvent(new Event('input')); }")
        await p.wait_for_timeout(600)
        await p.evaluate("() => document.getElementById('showCreated').click()")
        await p.wait_for_timeout(2500)
        await p.screenshot(path=os.path.join(OUT,"prov_02_created.png"))
        print("created hud:", await p.inner_text("#hudInfo"))
        await p.evaluate("() => document.getElementById('showCreated').click()")

        await p.evaluate("() => document.getElementById('buildUp').click()")
        counts = []
        for fi in (5, 30, 60, 84):
            await p.evaluate("(i) => { const s=document.getElementById('frameSlider'); s.value=String(i); s.dispatchEvent(new Event('input')); }", fi)
            await p.wait_for_timeout(900)
            n = await p.evaluate("""() => {
                const v = window.viewer, S = v.S;
                const lim = v.cloud.material.uniforms.uBuildTo.value;
                let c = 0;
                for (let i = 0; i < S.N; i++) if (S.firstFrame[i] <= lim) c++;
                return [lim, c];
            }""")
            counts.append(n)
            print(f"  slider {fi:>3d} -> frame {n[0]:>5d}: {n[1]:,} points revealed")
        await p.screenshot(path=os.path.join(OUT,"prov_03_buildup.png"))
        rising = all(counts[i][1] <= counts[i+1][1] for i in range(len(counts)-1))
        print("revealed count rises monotonically with the frame:", rising)
        await b.close()
    print("console errors:", errs or "none")
asyncio.run(main())
