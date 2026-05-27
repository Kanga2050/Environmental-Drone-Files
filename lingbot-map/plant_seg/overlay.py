"""Draw old vs new plant point assignments over a real photo of each plant.

    python -m plant_seg.overlay --artifacts ../artifacts_yard_avg --out overlays
"""
import argparse
import os

import cv2
import numpy as np

from detect_plants import source_to_model_transform
from . import prompt as P
from .yard import Yard


def draw(y, frame, idx, color, box=None):
    img = cv2.imread(frame["jpg"])
    h, w = img.shape[:2]
    scale, y_off = source_to_model_transform((h, w), y.model_hw)
    E = y.aligned_extrinsic(frame["extrinsic"])
    u, v, z = P.project(y.xyz[idx], E, frame["intrinsic"])
    x, yy = u / scale, (v + y_off) / scale
    ok = (z > 0) & (x >= 0) & (x < w) & (yy >= 0) & (yy < h)
    mask = np.zeros((h, w), np.uint8)
    mask[yy[ok].astype(int), x[ok].astype(int)] = 1
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    mask = mask.astype(bool)
    img[mask] = (0.35 * img[mask] + 0.65 * np.array(color)).astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = map(int, box)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plants", default="", help="comma separated ids (default: all)")
    args = ap.parse_args()
    from view_map3d import MapScene
    os.makedirs(args.out, exist_ok=True)
    s = MapScene(args.artifacts, use_seg=False)
    y = Yard(s)
    from .views import plant_views
    pv = plant_views(s.art, s.plants, {}, {}, y.model_hw)
    new_tags = np.load(os.path.join(args.artifacts, "plant_seg.npz"))["tags"][s.order]
    want = {int(v) for v in args.plants.split(",") if v}
    for i, p in enumerate(s.plants):
        pid = p["id"]
        if (want and pid not in want) or not pv.get(pid):
            continue
        v = max(pv[pid], key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]))
        f = s.frame_by_id[v["frame"]]
        old = draw(y, f, s.member.get(pid, np.zeros(0, np.int64)), (0, 0, 255), v["box"])
        new = draw(y, f, np.flatnonzero(new_tags == i), (0, 255, 0), v["box"])
        for im, t in ((old, "old: radius rule"), (new, "new: point transformer")):
            cv2.putText(im, t, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(args.out, f"plant_{pid:02d}.jpg"), np.hstack([old, new]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote overlays to {args.out}")


if __name__ == "__main__":
    main()
