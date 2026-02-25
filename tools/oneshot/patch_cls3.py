
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
subs = []

OLD_Q = '''def crop_quality(crop_rgb):
    """How much a crop is worth showing a species classifier.

    The highest-*detection*-scoring view is not the most identifiable one. A
    detector is happy with a whole tree seen from forty metres against bright
    sky; a classifier needs leaf texture. On this footage the best-scoring crops
    were backlit canopy silhouettes and shots that were more fence than plant,
    which is why an open-world classifier scored every instance under 0.03.

    Four factors, multiplied so that failing any one is disqualifying:

      vegetation  share of the crop that is actually plant rather than fence
      detail      Laplacian variance, which collapses on motion-blurred frames
      size        more pixels on the subject, with diminishing returns
      backlight   penalises crops that are mostly blown-out sky, where foliage
                  is a silhouette carrying no colour or texture
    """
    if crop_rgb.size == 0:
        return 0.0
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    veg = vegetation_fraction(crop_rgb)
    detail = min(float(cv2.Laplacian(g, cv2.CV_64F).var()) / 300.0, 1.0)
    size = min(float(crop_rgb.shape[0] * crop_rgb.shape[1]) / (400.0 * 400.0), 1.0)
    sky = float((g > 205).mean())
    return float(max(veg, 0.02) * max(detail, 0.05) * max(size, 0.05) * (1.0 - sky))'''

NEW_Q = '''def crop_sharpness(crop_rgb):
    """Scale-invariant, contrast-normalised sharpness. Higher is less blurred.

    Both normalisations matter and the previous version had neither. Laplacian
    variance on the raw crop grows with the crop's size and with how busy it is,
    so a large high-contrast smear outscored a small sharp leaf; resizing to a
    fixed grid removes the first effect and dividing by the crop's own variance
    removes the second, leaving something that tracks focus and motion blur.
    """
    if crop_rgb.size == 0:
        return 0.0
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var() / (g.var() + 1e-6))

def crop_quality(crop_rgb):
    """How much a crop is worth showing a species classifier.

    The highest-*detection*-scoring view is not the most identifiable one: a
    detector is happy with a whole tree seen from forty metres against bright
    sky, where a classifier needs leaf texture.

    Three factors, multiplied so that failing any one is disqualifying:

      sharpness   the one that actually decides this. A hand-flown drone
                  smears most frames, and for 17 of 42 instances here a view
                  existed that was more than twice as sharp as the one being
                  used -- detail that was recorded and then thrown away.
      size        more pixels on the subject, with diminishing returns
      backlight   penalises crops that are mostly blown-out sky, where foliage
                  is a silhouette carrying no colour or texture

    A vegetation-index term used to lead this product. It was measured against
    every crop in this scene and does not discriminate: the best colour-only
    index separates plant pixels from frame pixels at AUC 0.57, so including it
    mostly injected noise into the ranking and let blurred frames win.
    """
    if crop_rgb.size == 0:
        return 0.0
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    sharp = crop_sharpness(crop_rgb)
    # Saturating at 1.0 was the other half of the old bug: every crop above a
    # low bar scored the same, so blur stopped mattering entirely. This ramp
    # keeps separating crops well past the point where they look acceptable.
    sharp = sharp / (sharp + 0.25)
    size = min(float(crop_rgb.shape[0] * crop_rgb.shape[1]) / (400.0 * 400.0), 1.0)
    sky = float((g > 205).mean())
    return float(max(sharp, 0.01) * max(size, 0.05) * (1.0 - sky))'''

subs.append((OLD_Q, NEW_Q))

subs.append((
    '''def vote_label(members, key):''',
    '''def crop_from_instance(art_dir, fid, points_3d, model_hw, pad=0.10,
                       lo_pct=3.0, hi_pct=97.0, min_px=64):
    """Cut a crop around where this instance's own 3D points land in a frame.

    The detection box is 2D and per-frame: it is whatever the detector drew,
    which on this footage is routinely a tree plus the fence behind it plus a
    band of sky. The fused instance knows better -- it has the actual points
    belonging to that plant, pooled over every view -- so projecting those back
    into a frame gives a box around the plant rather than around the detector's
    opinion of the plant.

    Percentiles rather than the extremes, because a handful of stray points
    behind a leaf would otherwise re-inflate the box to the whole frame.
    """
    try:
        depth, conf, extr, intr, rgb = load_frame(art_dir, fid)
    except (FileNotFoundError, KeyError):
        return None
    if rgb is None or not len(points_3d):
        return None
    R, t = extr[:, :3], extr[:, 3]
    q = points_3d @ R.T + t
    z = q[:, 2]
    ok = z > 0.05
    if ok.sum() < 8:
        return None
    q = q[ok]
    u = q[:, 0] * intr[0, 0] / q[:, 2] + intr[0, 2]
    v = q[:, 1] * intr[1, 1] / q[:, 2] + intr[1, 2]
    H, W = model_hw
    inb = (u > -W) & (u < 2 * W) & (v > -H) & (v < 2 * H)
    if inb.sum() < 8:
        return None
    u, v = u[inb], v[inb]
    x0, x1 = np.percentile(u, [lo_pct, hi_pct])
    y0, y1 = np.percentile(v, [lo_pct, hi_pct])
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    x0 -= w * pad; x1 += w * pad
    y0 -= h * pad; y1 += h * pad

    scale, y_off = source_to_model_transform(rgb.shape[:2], model_hw)
    sx0 = int(max(0, x0 / scale))
    sx1 = int(min(rgb.shape[1], x1 / scale))
    sy0 = int(max(0, (y0 + y_off) / scale))
    sy1 = int(min(rgb.shape[0], (y1 + y_off) / scale))
    if sx1 - sx0 < min_px or sy1 - sy0 < min_px:
        return None
    return rgb[sy0:sy1, sx0:sx1]

def vote_label(members, key):'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\\n" + old[:240])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("crop_quality rewritten, crop_from_instance added")
