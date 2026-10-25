"""Maker-portfolio collage from survey_media.py output.

    python collage.py MEDIA_DIR NAME "Title" "Subtitle" OUT.jpg [--panel_row 0]
"""
import argparse, json, os
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("media")
ap.add_argument("name")
ap.add_argument("title")
ap.add_argument("subtitle")
ap.add_argument("out")
ap.add_argument("--panel_row", type=int, default=0)
ap.add_argument("--hero_caption", default="The map in 3D, viewed along the trail, with each plant's label")
a = ap.parse_args()

W, M, G = 2400, 48, 14
BG, INK, SUB = (18, 20, 23), (240, 242, 244), (165, 172, 180)
F = lambda s, b=False: ImageFont.truetype("arialbd.ttf" if b else "arial.ttf", s)
pre = os.path.join(a.media, f"Manta V3 - {a.name}")
S = json.load(open(os.path.join(a.media, "summary.json")))
blocks = []


def fit(im, w):
    return im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)


def row(ims, caps, width, h0):
    """Justify images into one row of the given width at a common height."""
    rs = [im.width / im.height for im in ims]
    h = round((width - G * (len(ims) - 1)) / sum(rs))
    out = Image.new("RGB", (width, h + (40 if caps else 0)), BG)
    d = ImageDraw.Draw(out)
    x = 0
    for k, im in enumerate(ims):
        w = round(h * rs[k]) if k < len(ims) - 1 else width - x
        out.paste(im.resize((w, h), Image.LANCZOS), (x, 0))
        if caps:
            d.text((x + 2, h + 8), caps[k], fill=SUB, font=F(22))
        x += w + G
    return out


hero = fit(Image.open(pre + " 3D View.png").convert("RGB"), W - 2 * M)
blocks.append((hero, a.hero_caption))

panel = Image.open(pre + " Camera, Point Cloud and Plant Points.png").convert("RGB")
rh = (panel.height + 8) // 4
strip = panel.crop((0, a.panel_row * rh, panel.width, a.panel_row * rh + rh - 8))
blocks.append((fit(strip, W - 2 * M), "The same moment three ways: the drone camera, the 3D map from that camera, and the points assigned to each plant"))

cams = [Image.open(c).convert("RGB") for c in S["cams"]]
blocks.append((row(cams, ["Drone camera"] * len(cams), W - 2 * M, 0), None))
crops = [(Image.open(p).convert("RGB"), c) for p, c in S["crops"]]
half = (len(crops) + 1) // 2
for part in (crops[:half], crops[half:]):
    if part:
        blocks.append((row([c[0] for c in part], [c[1] for c in part], W - 2 * M, 0), None))

H = M + 70 + 50 + 40
for im, cap in blocks:
    H += im.height + (46 if cap else 0) + 26
H += M - 26
out = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(out)
d.text((M, M), a.title, fill=INK, font=F(58, True))
d.text((M, M + 78), a.subtitle, fill=SUB, font=F(30))
y = M + 160
for im, cap in blocks:
    out.paste(im, (M, y))
    y += im.height
    if cap:
        d.text((M + 2, y + 10), cap, fill=SUB, font=F(24))
        y += 46
    y += 26
out.save(a.out, quality=90)
print(a.out, out.size)
