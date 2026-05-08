
import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090/"
TAG = sys.argv[2] if len(sys.argv) > 2 else "cam"

async def main():
    errors = []
    async with async_playwright() as pw:
        b = await pw.chromium.launch(args=[
            "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        p = await b.new_page(viewport={"width": 1500, "height": 900})
        p.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        p.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        await p.goto(URL, wait_until="networkidle", timeout=120_000)
        await p.wait_for_function(
            "() => document.getElementById('loading').classList.contains('hidden')",
            timeout=180_000)
        await p.wait_for_timeout(2500)

        geo = await p.evaluate("""() => {
            const v = window.viewer, T = v.THREE;
            const f = v.S.scene.frames[40];
            const E = f.extrinsic;
            const r2 = new T.Vector3(E[8], E[9], E[10]);   // camera forward
            const r0 = new T.Vector3(E[0], E[1], E[2]);
            const r1 = new T.Vector3(E[4], E[5], E[6]);
            const pos = new T.Vector3(
              -(r0.x*E[3] + r1.x*E[7] + r2.x*E[11]),
              -(r0.y*E[3] + r1.y*E[7] + r2.y*E[11]),
              -(r0.z*E[3] + r1.z*E[7] + r2.z*E[11]));
            // Average the four frustum corners drawn in the scene for this frame.
            const seg = v.world.children
              .flatMap(o => o.children || [])
              .find(o => o.name === 'frusta');
            const arr = seg.geometry.attributes.position.array;
            const base = 40 * 16 * 3;      // 16 points per frame
            let cx = 0, cy = 0, cz = 0, n = 0;
            for (let k = 1; k < 16; k += 2) {
              cx += arr[base + k*3]; cy += arr[base + k*3+1]; cz += arr[base + k*3+2];
              n++;
            }
            const centre = new T.Vector3(cx/n, cy/n, cz/n).sub(pos).normalize();
            const cross = new T.Vector3().crossVectors(r0, r1);
            return {
              forward_dot: centre.dot(r2),
              right_handed: cross.dot(r2),
              apex_matches_position: seg.geometry.attributes.position.array[base] - pos.x,
            };
        }""")
        print("frustum points along camera forward (want ~+1):",
              round(geo["forward_dot"], 4))
        print("r0 x r1 . r2 (want ~+1, right-handed):",
              round(geo["right_handed"], 4))

        await p.click('#tabs button[data-tab="frames"]')
        await p.evaluate("""() => {
            const s = document.getElementById('frameSlider');
            s.value = 40; s.dispatchEvent(new Event('input'));
        }""")
        await p.wait_for_timeout(600)
        await p.evaluate("() => window.viewer.S && document.getElementById('cbTraj').checked")
        await p.screenshot(path=os.path.join(OUT, f"{TAG}_01_path_axes.png"))

        await p.evaluate("() => document.getElementById('showPhotos').click()")
        await p.wait_for_timeout(4000)
        await p.screenshot(path=os.path.join(OUT, f"{TAG}_02_photos.png"))
        n_photos = await p.evaluate(
            "() => window.viewer.world.children.flatMap(o=>o.children||[])"
            ".find(o=>o.name==='photos').children.length")
        print("photo quads placed:", n_photos)
        await p.evaluate("() => document.getElementById('showPhotos').click()")

        await p.click("#framePlay")
        await p.wait_for_timeout(2500)
        idx1 = await p.evaluate("() => window.viewer.S.frameIdx")
        await p.wait_for_timeout(2000)
        idx2 = await p.evaluate("() => window.viewer.S.frameIdx")
        await p.screenshot(path=os.path.join(OUT, f"{TAG}_03_playing.png"))
        await p.click("#framePlay")
        print(f"playback advanced {idx1} -> {idx2}")

        buildup_enabled = await p.evaluate(
            "() => !document.getElementById('buildUp').disabled")
        print("build-up available (needs provenance):", buildup_enabled)

        await b.close()
    print("\nconsole errors:", errors or "none")
    return 1 if errors else 0

sys.exit(asyncio.run(main()))
