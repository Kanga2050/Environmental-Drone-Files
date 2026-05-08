import io
p = r"C:\Users\shaur\AppData\Local\Temp\probe\c--Users-shaur-OneDrive-Desktop-Test\8aeaf542-92a0-4c97-b340-f6e129ba16fe\scratchpad\shoot3d.py"
s = io.open(p, encoding="utf-8").read()
old = '''        # Rectangle selection with Ctrl held.
        await page.keyboard.press("Escape")'''
new = '''        # Rectangle selection WHILE standing in a frame. This is the case that
        # exposes a viewport-mapping bug: the render is letterboxed there, so a
        # selection computed against the whole canvas addresses the wrong pixels
        # and comes back empty over a cloud that is plainly on screen.
        await page.keyboard.down("Control")
        await page.mouse.move(400, 300)
        await page.mouse.down()
        await page.mouse.move(760, 640, steps=12)
        await page.mouse.up()
        await page.keyboard.up("Control")
        await page.wait_for_timeout(3500)
        in_frame_sel = await page.inner_text("#selBody")
        print("--- selection made while in frame view ---")
        print(in_frame_sel[:260].replace("\n", " | "))
        await page.screenshot(path=os.path.join(OUT, f"{TAG}_09_selframe.png"))

        # Colour-by modes.
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

        # Rectangle selection with Ctrl held.
        await page.keyboard.press("Escape")'''
assert old in s
io.open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("shoot script extended")
