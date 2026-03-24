
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
import detect_plants as dp

OUT = os.path.dirname(os.path.abspath(__file__))

def old_crop_quality(crop_rgb):

    if crop_rgb.size == 0:
        return 0.0
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    veg = dp.vegetation_fraction(crop_rgb)
    detail = min(float(cv2.Laplacian(g, cv2.CV_64F).var()) / 300.0, 1.0)
    size = min(float(crop_rgb.shape[0] * crop_rgb.shape[1]) / (400.0 * 400.0), 1.0)
    sky = float((g > 205).mean())
    return float(max(veg, 0.02) * max(detail, 0.05) * max(size, 0.05) * (1.0 - sky))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=r"C:\Users\shaur\OneDrive\Desktop\Test\artifacts_yard_final")
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--plantnet", default=None, help="API key; uploads crops")
    ap.add_argument("--models_dir", default=r"C:\Users\shaur\AppData\Local\lingbot-map\models")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    art = args.artifacts
    meta = json.load(open(os.path.join(art, "meta.json"), encoding="utf-8"))
    model_hw = tuple(meta["model_hw"])
    plants = json.load(open(os.path.join(art, "plants.json"), encoding="utf-8"))["plants"]
    pts = np.load(os.path.join(art, "plant_points.npz"))
    crops_dir = os.path.join(art, "crops")

    if args.plantnet:
        clf = dp.PlantNetClassifier(args.plantnet)
        name = "plantnet"
        print("Pl@ntNet: every crop below is uploaded to an external service.")
    else:
        clf = dp.BioClipClassifier(os.path.join(args.models_dir, "bioclip"),
                                   "cuda", dp.DEFAULT_SPECIES)
        name = "bioclip"
    print(f"classifier: {name}\n")

    by_frame = defaultdict(list)
    for f in glob.glob(os.path.join(crops_dir, "*.jpg")):
        by_frame[int(os.path.basename(f).split("_")[0])].append(f)

    rows = []
    tiles_old, tiles_new = [], []
    todo = plants[:args.limit] if args.limit else plants
    for p in todo:
        pooled = pts.get(f"pts_{p['id']}")
        if pooled is None or not len(pooled):
            continue
        pooled = np.asarray(pooled, np.float32)

        old_views, new_views = [], []
        for fid in p["frames"]:
            for f in sorted(by_frame.get(fid, [])):
                im = cv2.imread(f)
                if im is not None:
                    old_views.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            c = dp.crop_from_instance(art, fid, pooled, model_hw)
            if c is not None:
                new_views.append(c)
        if not old_views or not new_views:
            continue

        old_views.sort(key=lambda a: -old_crop_quality(a))
        new_views.sort(key=lambda a: -dp.crop_quality(a))
        old_sel = [dp.foliage_subcrop(a) for a in old_views[:args.views]]
        new_sel = new_views[:args.views]

        votes = Counter()
        best_raw = 0.0
        for v in old_sel:
            res = clf(v, topk=3)
            if res:
                votes[res[0][0]] += 1
                best_raw = max(best_raw, res[0][2])
        old_name = votes.most_common(1)[0][0] if votes else "-"
        old_share = (votes.most_common(1)[0][1] / sum(votes.values())) if votes else 0.0

        fused = clf.classify_many(new_sel, topk=3)
        new_name = fused[0][0] if fused else "-"
        new_raw = fused[0][2] if fused else 0.0

        half = max(1, len(new_views) // 2)
        a = clf.classify_many(new_views[:half][:args.views], topk=1)
        b = clf.classify_many(new_views[half:][:args.views], topk=1)
        consistent = bool(a and b and a[0][0] == b[0][0])

        rows.append({"id": p["id"], "old": old_name, "old_raw": best_raw,
                     "old_share": old_share, "new": new_name, "new_raw": new_raw,
                     "consistent": consistent,
                     "old_sharp": dp.crop_sharpness(old_sel[0]),
                     "new_sharp": dp.crop_sharpness(new_sel[0]),
                     "old_px": old_sel[0].shape[:2], "new_px": new_sel[0].shape[:2]})
        print(f"#{p['id']:<3d} old {old_name[:26]:<26s} {best_raw:.3f}  ->  "
              f"new {new_name[:26]:<26s} {new_raw:.3f}"
              f"{'  [consistent]' if consistent else ''}")

        if len(tiles_old) < 9:
            def tile(im, label, colour):
                h, w = im.shape[:2]
                s = 220.0 / max(h, w)
                im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
                o = np.zeros((220, 220, 3), np.uint8)
                o[:im.shape[0], :im.shape[1]] = im
                cv2.putText(o, label, (4, 15), 0, 0.4, colour, 1)
                return o
            tiles_old.append(tile(old_sel[0], f"#{p['id']} old", (120, 170, 255)))
            tiles_new.append(tile(new_sel[0], f"#{p['id']} new", (130, 255, 150)))

    if not rows:
        sys.exit("nothing to compare")

    o_raw = np.array([r["old_raw"] for r in rows])
    n_raw = np.array([r["new_raw"] for r in rows])
    o_sh = np.array([r["old_sharp"] for r in rows])
    n_sh = np.array([r["new_sharp"] for r in rows])
    o_px = np.array([r["old_px"][0] * r["old_px"][1] for r in rows], float)
    n_px = np.array([r["new_px"][0] * r["new_px"][1] for r in rows], float)

    print(f"\n{len(rows)} instances")
    print(f"{'':26s} {'old':>10s} {'new':>10s}")
    print(f"{'top-1 raw score, mean':26s} {o_raw.mean():10.3f} {n_raw.mean():10.3f}")
    print(f"{'top-1 raw score, median':26s} {np.median(o_raw):10.3f} {np.median(n_raw):10.3f}")
    print(f"{'scoring above 0.20':26s} {int((o_raw>=0.2).sum()):10d} {int((n_raw>=0.2).sum()):10d}")
    print(f"{'scoring above 0.30':26s} {int((o_raw>=0.3).sum()):10d} {int((n_raw>=0.3).sum()):10d}")
    print(f"{'chosen crop sharpness':26s} {o_sh.mean():10.3f} {n_sh.mean():10.3f}")
    print(f"{'chosen crop megapixels':26s} {o_px.mean()/1e6:10.3f} {n_px.mean()/1e6:10.3f}")
    print(f"{'distinct species named':26s} "
          f"{len({r['old'] for r in rows}):10d} {len({r['new'] for r in rows}):10d}")
    print(f"\nsplit-half self-consistent (new arm): "
          f"{sum(r['consistent'] for r in rows)}/{len(rows)}")
    agree = sum(r["old"] == r["new"] for r in rows)
    print(f"arms agree on the name: {agree}/{len(rows)}")

    if tiles_old:
        cv2.imwrite(os.path.join(OUT, f"species_crops_{name}.jpg"),
                    cv2.cvtColor(np.vstack([np.hstack(tiles_old),
                                            np.hstack(tiles_new)]), cv2.COLOR_RGB2BGR))
        print(f"wrote species_crops_{name}.jpg (top: old choice, bottom: new)")

    json.dump(rows, open(os.path.join(OUT, f"species_eval_{name}.json"), "w"), indent=1)

main()
