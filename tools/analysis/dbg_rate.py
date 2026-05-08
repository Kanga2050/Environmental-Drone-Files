import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as pw:
        b = await pw.chromium.launch(args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
        p = await b.new_page(viewport={"width":1500,"height":900})
        await p.goto("http://127.0.0.1:8090/", wait_until="networkidle", timeout=120000)
        await p.wait_for_function("() => document.getElementById('loading').classList.contains('hidden')", timeout=180000)
        await p.wait_for_timeout(2000)
        for dens in (100, 5):
            await p.evaluate("(d) => { const s=document.getElementById('denSlider'); s.value=String(d); s.dispatchEvent(new Event('input')); }", dens)
            await p.wait_for_timeout(800)
            r = await p.evaluate("""() => new Promise(res => {
                let n = 0; const t0 = performance.now();
                const step = () => { n++; if (performance.now()-t0 < 2000) requestAnimationFrame(step); else res(n/((performance.now()-t0)/1000)); };
                requestAnimationFrame(step);
            })""")
            print(f"density {dens:3d}%  ~{r:.1f} rAF/s (software renderer)")
        await b.close()
asyncio.run(main())
