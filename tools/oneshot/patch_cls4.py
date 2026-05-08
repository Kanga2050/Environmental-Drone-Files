
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    ap.add_argument("--fuse_views", action=argparse.BooleanOptionalAction, default=True,''',
    '''    ap.add_argument("--crop_from_3d", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Re-cut each classifier view around where the fused "
                         "instance's own 3D points land in that frame, instead of "
                         "using the per-frame detection box. The box is whatever "
                         "the detector drew, which here is routinely a plant plus "
                         "the fence behind it plus a band of sky")
    ap.add_argument("--fuse_views", action=argparse.BooleanOptionalAction, default=True,'''))

subs.append((
    '''            scored = []
            for m in inst["members"]:
                img = cv2.imread(os.path.join(crops_dir, m["crop"]))
                if img is None:
                    continue
                rgb_crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                q = crop_quality(rgb_crop) if args.rank_crops else m["score"]
                scored.append((q, m, rgb_crop))''',
    '''            pooled = np.concatenate([m["points"] for m in inst["members"]]) \\
                if args.crop_from_3d and inst["members"] else None
            if pooled is not None and len(pooled) > 4000:
                pooled = pooled[np.random.default_rng(0).choice(
                    len(pooled), 4000, replace=False)]
            scored = []
            for m in inst["members"]:
                rgb_crop = None
                if pooled is not None:
                    rgb_crop = crop_from_instance(art, m["frame"], pooled, model_hw)
                if rgb_crop is None:
                    img = cv2.imread(os.path.join(crops_dir, m["crop"]))
                    if img is None:
                        continue
                    rgb_crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                q = crop_quality(rgb_crop) if args.rank_crops else m["score"]
                scored.append((q, m, rgb_crop))'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\\n" + old[:260])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("crop_from_3d wired")
