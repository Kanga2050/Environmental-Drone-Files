import asyncio, json
from playwright.async_api import async_playwright
async def main():
    msgs=[]
    async with async_playwright() as pw:
        b=await pw.chromium.launch(args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
        p=await b.new_page(viewport={"width":1500,"height":900})
        p.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))
        p.on("pageerror", lambda e: msgs.append(f"[pageerror] {e}"))
        await p.goto("http://127.0.0.1:8090/", wait_until="networkidle", timeout=120000)
        await p.wait_for_function("() => document.getElementById('loading').classList.contains('hidden')", timeout=180000)
        await p.wait_for_timeout(3000)

        r = await p.evaluate("""() => {
            const v = window.viewer;
            const a = v.selectRect(300, 250, 800, 700, false);
            const b = v.selectRect(300, 250, 800, 700, true);
            return {all: a.length, front: b.length,
                    canvas:[document.getElementById('gl').clientWidth,
                            document.getElementById('gl').clientHeight]};
        }""")
        print("direct selectRect:", r)

        await p.keyboard.down("Control")
        await p.mouse.move(400, 300)
        await p.mouse.down()
        await p.mouse.move(700, 600, steps=10)
        rub = await p.evaluate("() => { const r=document.getElementById('rubber'); return [r.style.display, r.style.width, r.style.height]; }")
        print("rubber during drag:", rub)
        await p.mouse.up()
        await p.keyboard.up("Control")
        await p.wait_for_timeout(4000)
        txt = await p.inner_text("#selBody")
        print("--- panel ---"); print(txt[:700])
        await b.close()
    for m in msgs: print(m[:300])
asyncio.run(main())
