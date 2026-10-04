"""2D detection views for each plant instance.

detect_plants.py saves one crop per lifted detection (crops/{fid}_{k}.jpg)
but not the box itself, so the box is recovered by locating the crop in its
source frame, then each box is given to the plant whose lifted points it
contains.  Newer detect_plants.py runs write the boxes directly
("boxes" on each plant) and this step is skipped.
"""
import hashlib
import json
import os
import re

import cv2
import numpy as np

from detect_plants import source_to_model_transform


def project(points, extrinsic, intrinsic, scale, y_off):
    """World points -> (src_x, src_y, depth) in source-image pixels."""
    cam = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    z = cam[:, 2]
    uv = cam[:, :2] / np.maximum(z, 1e-9)[:, None] * intrinsic[[0, 1], [0, 1]] \
        + intrinsic[[0, 1], [2, 2]]
    return uv[:, 0] / scale, (uv[:, 1] + y_off) / scale, z


def locate_crop(frame_gray, crop_gray):
    res = cv2.matchTemplate(frame_gray, crop_gray, cv2.TM_SQDIFF_NORMED)
    err, _, (x, y), _ = cv2.minMaxLoc(res)
    h, w = crop_gray.shape
    return [float(x), float(y), float(x + w), float(y + h)], float(err)


def recover_boxes(art):
    """{fid: [box, ...]} for every saved crop."""
    crops = os.path.join(art, "crops")
    by_frame = {}
    for fn in sorted(os.listdir(crops)):
        m = re.match(r"(\d+)_(\d+)\.jpg$", fn)
        if m:
            by_frame.setdefault(int(m.group(1)), []).append(fn)
    out = {}
    for fid, names in by_frame.items():
        jpg = os.path.join(art, "frames", f"{fid:06d}.jpg")
        if not os.path.exists(jpg):
            continue
        frame = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE)
        for fn in names:
            crop = cv2.imread(os.path.join(crops, fn), cv2.IMREAD_GRAYSCALE)
            if crop is None or crop.shape[0] > frame.shape[0] or crop.shape[1] > frame.shape[1]:
                continue
            box, err = locate_crop(frame, crop)
            if err < 0.02:
                out.setdefault(fid, []).append({"crop": fn, "box": box})
    return out


def plant_views(art, plants, seeds, frames, model_hw, cache=True):
    """Per plant: list of {frame, box} (source pixels) that detected it.

    plants: plants.json entries; seeds: {id: lifted points}; frames:
    {fid: dict(extrinsic, intrinsic, jpg)}.
    """
    path = os.path.join(art, "plant_views.json")
    # the cache is only valid for the plants it was built from: a new
    # detect_plants.py run renumbers them
    sig = hashlib.sha1(json.dumps([[p["id"], p.get("boxes", p.get("frames"))] for p in plants],
                                  sort_keys=True).encode()).hexdigest()
    if cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("plants_sig") == sig:
            return {int(k): v for k, v in saved["views"].items()}

    if all("boxes" in p for p in plants):
        boxes = {}
        for p in plants:
            for b in p["boxes"]:
                boxes.setdefault(int(b["frame"]), []).append(b)
    else:
        boxes = recover_boxes(art)
    views = {p["id"]: [] for p in plants}
    in_frame = {}
    for p in plants:
        for fid in p.get("frames", []):
            in_frame.setdefault(fid, []).append(p)
    for fid, cands in in_frame.items():
        f = frames.get(fid)
        if f is None or fid not in boxes:
            continue
        h, w = cv2.imread(f["jpg"], cv2.IMREAD_GRAYSCALE).shape
        scale, y_off = source_to_model_transform((h, w), model_hw)
        proj = {}
        for p in cands:
            pts = seeds.get(p["id"])
            if pts is not None and len(pts):
                proj[p["id"]] = project(pts, f["extrinsic"], f["intrinsic"],
                                        scale, y_off)
        # A box belongs to the plant whose lifted points it holds best;
        # boxes holding little of any plant's points (loosely clustered
        # detections) are dropped.
        for b in boxes[fid]:
            x0, y0, x1, y1 = b["box"]
            best, best_frac = None, 0.25
            for pid, (x, y, z) in proj.items():
                frac = ((z > 0) & (x >= x0) & (x <= x1)
                        & (y >= y0) & (y <= y1)).mean()
                if frac > best_frac:
                    best, best_frac = pid, frac
            if best is not None:
                views[best].append({"frame": int(fid), "box": b["box"],
                                    "crop": b["crop"],
                                    "hit": round(float(best_frac), 3)})
    if cache:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"plants_sig": sig, "views": views}, f)
    return views
