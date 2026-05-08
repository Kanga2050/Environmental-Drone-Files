import asyncio, json
from playwright.async_api import async_playwright
async def main():
    msgs=[]
    async with async_playwright() as pw:
        b=await pw.chromium.launch(args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"])
        p=await b.new_page(viewport={"width":1200,"height":800})
        p.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))
        p.on("pageerror", lambda e: msgs.append(f"[pageerror] {e}"))
        await p.goto("http://127.0.0.1:8090/", wait_until="networkidle", timeout=120000)
        await p.wait_for_function("() => document.getElementById('loading').classList.contains('hidden')", timeout=180000)
        await p.wait_for_timeout(3000)
        info = await p.evaluate("""() => {
            const v = window.viewer; const S = v.S;
            const P = S.positions;
            let mn=[1e9,1e9,1e9], mx=[-1e9,-1e9,-1e9], nan=0;
            for (let i=0;i<300000;i++){
              for(let k=0;k<3;k++){const x=P[i*3+k];
                if(!isFinite(x)){nan++;continue;}
                if(x<mn[k])mn[k]=x; if(x>mx[k])mx[k]=x;}
            }
            const c = v.cloud;
            return {
              n: S.N, nan, mn, mx,
              drawRange: c.geometry.drawRange,
              visible: c.visible,
              camPos: v.cam.pos.toArray(),
              camYaw: v.cam.yaw, camPitch: v.cam.pitch,
              matWorld: v.camera.matrixWorld.elements,
              proj: v.camera.projectionMatrix.elements,
              uPx: c.material.uniforms.uPx.value,
              uSize: c.material.uniforms.uSize.value,
              colors0: [S.colors[0],S.colors[1],S.colors[2]],
              tags0: [S.tags[0],S.tags[1],S.tags[2]],
              renderInfo: JSON.parse(JSON.stringify(v.renderer.info.render)),
              memory: JSON.parse(JSON.stringify(v.renderer.info.memory)),
              progs: v.renderer.info.programs.length,
              glErr: v.renderer.getContext().getError(),
            };
        }""")
        print(json.dumps({k:info[k] for k in ("n","renderInfo","memory","progs","glErr","camPos","camPitch","uPx")}, indent=1))
        print("--- console ---")
        for m in msgs: print(m[:400])
        await b.close()
asyncio.run(main())
