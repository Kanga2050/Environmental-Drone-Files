

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

DEFAULT_SPECIES = [
    ("Chrysanthemum morifolium", "florist's daisy"),
    ("Hydrangea macrophylla", "bigleaf hydrangea"),
    ("Rosa chinensis", "China rose"),
    ("Buxus sempervirens", "common boxwood"),
    ("Ligustrum ovalifolium", "garden privet"),
    ("Thuja occidentalis", "northern white cedar"),
    ("Juniperus chinensis", "Chinese juniper"),
    ("Rhododendron indicum", "azalea"),
    ("Camellia japonica", "Japanese camellia"),
    ("Hedera helix", "English ivy"),
    ("Lavandula angustifolia", "English lavender"),
    ("Pelargonium hortorum", "garden geranium"),
    ("Petunia hybrida", "garden petunia"),
    ("Begonia semperflorens", "wax begonia"),
    ("Impatiens walleriana", "busy lizzie"),
    ("Hosta plantaginea", "fragrant plantain lily"),
    ("Agapanthus africanus", "African lily"),
    ("Nandina domestica", "heavenly bamboo"),
    ("Photinia fraseri", "red tip photinia"),
    ("Pittosporum tobira", "Japanese cheesewood"),
    ("Lagerstroemia indica", "crape myrtle"),
    ("Acer palmatum", "Japanese maple"),
    ("Acer saccharum", "sugar maple"),
    ("Quercus agrifolia", "coast live oak"),
    ("Platanus acerifolia", "London plane"),
    ("Liquidambar styraciflua", "American sweetgum"),
    ("Pinus radiata", "Monterey pine"),
    ("Cupressus macrocarpa", "Monterey cypress"),
    ("Washingtonia robusta", "Mexican fan palm"),
    ("Phoenix canariensis", "Canary Island date palm"),
    ("Olea europaea", "olive"),
    ("Citrus limon", "lemon"),
    ("Prunus serrulata", "Japanese cherry"),
    ("Magnolia grandiflora", "southern magnolia"),
    ("Cordyline australis", "cabbage tree"),
    ("Phormium tenax", "New Zealand flax"),
    ("Agave americana", "century plant"),
    ("Aloe arborescens", "krantz aloe"),
    ("Yucca gloriosa", "Spanish dagger"),
    ("Festuca arundinacea", "tall fescue"),
    ("Cynodon dactylon", "Bermuda grass"),
    ("Trifolium repens", "white clover"),
    ("Bougainvillea glabra", "paper flower"),
    ("Jasminum polyanthum", "pink jasmine"),
    ("Wisteria sinensis", "Chinese wisteria"),
    ("Salvia officinalis", "common sage"),
    ("Rosmarinus officinalis", "rosemary"),
    ("Euonymus japonicus", "evergreen spindle"),
    ("Viburnum tinus", "laurustinus"),
    ("Cotoneaster horizontalis", "rock cotoneaster"),
]

DEFAULT_PROMPTS = [
    "a potted plant", "a shrub", "a bush", "a hedge",
    "a tree", "a flowering plant", "a leafy green plant",
]

def load_meta(art_dir):
    with open(os.path.join(art_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)

def load_frame(art_dir, fid):

    d = np.load(os.path.join(art_dir, "frames", f"{fid:06d}.npz"))
    jpg = os.path.join(art_dir, "frames", f"{fid:06d}.jpg")
    rgb = None
    if os.path.exists(jpg):
        rgb = cv2.cvtColor(cv2.imread(jpg), cv2.COLOR_BGR2RGB)
    return (d["depth"].astype(np.float32), d["conf"].astype(np.float32),
            d["extrinsic"], d["intrinsic"], rgb)

def source_to_model_transform(src_hw, model_hw, patch_size=14):

    hs, ws = src_hw
    hm, wm = model_hw
    scale = wm / ws
    resized_h = round(hs * scale / patch_size) * patch_size
    y_off = (resized_h - hm) // 2 if resized_h > hm else 0
    return scale, y_off

class Owlv2Detector:

    def __init__(self, model_dir, device, threshold=0.15):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.processor = Owlv2Processor.from_pretrained(model_dir)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self.threshold = threshold

    def __call__(self, rgb, prompts):
        from PIL import Image
        image = Image.fromarray(rgb)
        inputs = self.processor(text=[prompts], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target = torch.tensor([[image.height, image.width]], device=self.device)
        post = getattr(self.processor, "post_process_grounded_object_detection", None)
        if post is None:
            post = self.processor.post_process_object_detection
        res = post(outputs=outputs, threshold=self.threshold, target_sizes=target)[0]
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()
        labels = [prompts[i] for i in res["labels"].detach().cpu().numpy()]
        return boxes, scores, labels

class BioClipClassifier:

    def __init__(self, model_dir, device, species):
        import open_clip
        weights = os.path.join(model_dir, "open_clip_pytorch_model.bin")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=weights)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.device = device
        self.species = species

        texts = [f"a photo of {sci} with common name {com}." for sci, com in species]
        with torch.no_grad():
            tok = self.tokenizer(texts).to(device)
            feats = self.model.encode_text(tok)
            self.text_feats = (feats / feats.norm(dim=-1, keepdim=True)).float()

    def __call__(self, crop_rgb, topk=3):
        from PIL import Image
        img = self.preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(img)
            f = (f / f.norm(dim=-1, keepdim=True)).float()
            probs = (100.0 * f @ self.text_feats.T).softmax(dim=-1)[0].cpu().numpy()
        idx = np.argsort(-probs)[:topk]
        return [(self.species[i][0], self.species[i][1], float(probs[i])) for i in idx]

    def probs(self, crops):

        from PIL import Image
        batch = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops])
        with torch.no_grad():
            f = self.model.encode_image(batch.to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
            mean = f.mean(dim=0, keepdim=True)
            mean = mean / mean.norm(dim=-1, keepdim=True)
            per = (100.0 * f.float() @ self.text_feats.T).softmax(dim=-1)
            fused = (100.0 * mean.float() @ self.text_feats.T).softmax(dim=-1)[0]
        return per.cpu().numpy(), fused.cpu().numpy()

    def classify_views(self, crops, topk=5):

        if not crops:
            return None
        per, fused = self.probs(crops)
        order = np.argsort(-fused)[:topk]
        top = int(order[0])

        def ent(v):
            v = v[v > 0]
            return float(-(v * np.log2(v)).sum())

        return {
            "fused": [(self.species[i][0], self.species[i][1], float(fused[i]))
                      for i in order],
            "per_view": [[(self.species[i][0], self.species[i][1], float(v[i]))
                          for i in np.argsort(-v)[:topk]] for v in per],
            "at_fused": [float(v[top]) for v in per],
            "entropy": ent(fused),
            "max_entropy": float(np.log2(len(self.species))),
        }

    def classify_many(self, crops, topk=3):

        from PIL import Image
        if not crops:
            return []
        batch = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops])
        with torch.no_grad():
            f = self.model.encode_image(batch.to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0, keepdim=True)
            f = (f / f.norm(dim=-1, keepdim=True)).float()
            probs = (100.0 * f @ self.text_feats.T).softmax(dim=-1)[0].cpu().numpy()
        idx = np.argsort(-probs)[:topk]
        return [(self.species[i][0], self.species[i][1], float(probs[i])) for i in idx]

class PlantNetClassifier:

    BASE = "https://my-api.plantnet.org/v2/identify"
    PROJECT = "k-world-flora"

    def __init__(self, api_key, timeout=30, project=None):
        self.api_key = api_key
        self.timeout = timeout
        self.project = project or self.PROJECT
        self.ENDPOINT = f"{self.BASE}/{self.project}"

    def __call__(self, crop_rgb, topk=3):
        import requests
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return []
        try:
            r = requests.post(
                self.ENDPOINT,
                params={"api-key": self.api_key, "include-related-images": "false"},
                files=[("images", ("crop.jpg", buf.tobytes(), "image/jpeg"))],
                data={"organs": ["auto"]},
                timeout=self.timeout,
            )
            r.raise_for_status()
            results = r.json().get("results", [])[:topk]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return []
            print(f"    Pl@ntNet request failed: {exc}")
            return []
        except Exception as exc:
            print(f"    Pl@ntNet request failed: {exc}")
            return []
        return self._parse(results)

    @staticmethod
    def _parse(results):
        out = []
        for item in results:
            sp = item.get("species", {})
            sci = sp.get("scientificNameWithoutAuthor", "unknown")
            commons = sp.get("commonNames") or [""]
            out.append((sci, commons[0], float(item.get("score", 0.0))))
        return out

    def classify_many(self, crops, topk=3, organs=None):

        import requests
        crops = [c for c in crops if c is not None and c.size][:5]
        if not crops:
            return []
        files, data = [], []
        for i, c in enumerate(crops):
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(c, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            files.append(("images", (f"view{i}.jpg", buf.tobytes(), "image/jpeg")))
            data.append(("organs", organs[i] if organs else "auto"))
        if not files:
            return []
        try:
            r = requests.post(
                self.ENDPOINT,
                params={"api-key": self.api_key, "include-related-images": "false"},
                files=files, data=data, timeout=self.timeout)
            r.raise_for_status()
            return self._parse(r.json().get("results", [])[:topk])
        except Exception as exc:
            print(f"    Pl@ntNet multi-view request failed: {exc}")
            return []

def nms_2d(boxes, scores, labels, iou_threshold=0.5):

    if len(boxes) == 0:
        return boxes, scores, labels
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest]); iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest]); iy1 = np.minimum(y1[i], y1[rest])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        iou = inter / np.clip(areas[i] + areas[rest] - inter, 1e-9, None)
        order = rest[iou <= iou_threshold]
    keep = np.array(keep, dtype=int)
    return boxes[keep], scores[keep], [labels[i] for i in keep]

def vegetation_fraction(crop_rgb):

    if crop_rgb.size == 0:
        return 0.0
    f = crop_rgb.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    return float((exg > 8.0).mean())

def upscale_crop(crop_rgb, min_side=224, max_factor=4.0):

    if crop_rgb is None or crop_rgb.size == 0:
        return crop_rgb
    h, w = crop_rgb.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return crop_rgb
    f = min(max_factor, min_side / max(short, 1))
    if f <= 1.01:
        return crop_rgb
    return cv2.resize(crop_rgb, (int(round(w * f)), int(round(h * f))),
                      interpolation=cv2.INTER_LANCZOS4)

def stack_crops(crops, max_shift=0.25, min_corr=0.6):

    if not crops:
        return None
    ref = crops[0]
    if len(crops) == 1:
        return ref
    h, w = ref.shape[:2]
    ref_g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    acc = ref.astype(np.float32).copy()
    used = 1
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
    for c in crops[1:]:
        try:
            cr = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(cr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            warp = np.eye(2, 3, dtype=np.float32)
            corr, warp = cv2.findTransformECC(ref_g, g, warp, cv2.MOTION_AFFINE,
                                              crit, None, 5)
            if corr < min_corr:
                continue
            if (abs(warp[0, 2]) > max_shift * w) or (abs(warp[1, 2]) > max_shift * h):
                continue
            aligned = cv2.warpAffine(cr, warp, (w, h),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT)
            acc += aligned.astype(np.float32)
            used += 1
        except cv2.error:
            continue
    return np.clip(acc / used, 0, 255).astype(np.uint8)

def foliage_subcrop(crop_rgb, min_frac=0.04, min_keep_frac=0.35):

    if crop_rgb.size == 0:
        return crop_rgb
    f = crop_rgb.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    mask = (exg > max(8.0, float(np.percentile(exg, 80)))).astype(np.uint8)
    if mask.mean() < min_frac:
        return crop_rgb
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return crop_rgb
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[big, cv2.CC_STAT_LEFT], stats[big, cv2.CC_STAT_TOP],
                  stats[big, cv2.CC_STAT_WIDTH], stats[big, cv2.CC_STAT_HEIGHT])
    if w < 8 or h < 8:
        return crop_rgb
    pad_x, pad_y = int(0.1 * w), int(0.1 * h)
    y0 = max(0, y - pad_y); y1 = min(crop_rgb.shape[0], y + h + pad_y)
    x0 = max(0, x - pad_x); x1 = min(crop_rgb.shape[1], x + w + pad_x)
    sub = crop_rgb[y0:y1, x0:x1]

    area = sub.shape[0] * sub.shape[1]
    if sub.size == 0 or area < min_keep_frac * crop_rgb.shape[0] * crop_rgb.shape[1]:
        return crop_rgb
    return sub

def lift_box_to_3d(box_src, world, conf, rgb_model, scale, y_off,
                   conf_threshold, min_points, use_vegetation=True,
                   max_keep=2500, veg_erode=1, depth=None, depth_band=0.0,
                   depth_mode="nearest", veg_otsu=False):

    hm, wm = conf.shape[:2]
    x0, y0, x1, y1 = box_src
    mx0 = int(np.clip(round(x0 * scale), 0, wm - 1))
    mx1 = int(np.clip(round(x1 * scale), 0, wm))
    my0 = int(np.clip(round(y0 * scale) - y_off, 0, hm - 1))
    my1 = int(np.clip(round(y1 * scale) - y_off, 0, hm))
    if mx1 - mx0 < 2 or my1 - my0 < 2:
        return None

    patch = world[my0:my1, mx0:mx1].reshape(-1, 3)
    cpatch = conf[my0:my1, mx0:mx1].reshape(-1)
    keep = np.isfinite(patch).all(axis=1) & (cpatch > conf_threshold)

    if use_vegetation and rgb_model is not None:
        sub = rgb_model[my0:my1, mx0:mx1].astype(np.float32)
        exg = (2.0 * sub[:, :, 1] - sub[:, :, 0] - sub[:, :, 2]).reshape(-1)
        thr = 8.0
        if veg_otsu and len(exg) > 50:
            # Per-box threshold between foliage and its duller surroundings;
            # in colour-shifted video most pixels pass a fixed ExG > 8.
            e8 = np.clip(exg, -255, 255).astype(np.float32)
            e8 = ((e8 + 255) / 2).astype(np.uint8).reshape(-1, 1)
            t, _ = cv2.threshold(e8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = max(thr, float(t) * 2 - 255)
        veg = exg > thr

        if veg_erode > 0:
            vm = veg.reshape(my1 - my0, mx1 - mx0).astype(np.uint8)
            k = 2 * int(veg_erode) + 1
            vm = cv2.erode(vm, np.ones((k, k), np.uint8))
            veg = vm.astype(bool).reshape(-1)
        if int((keep & veg).sum()) < min_points:
            return None
        keep = keep & veg

    if depth is not None and depth_band > 0:

        d_all = depth[my0:my1, mx0:mx1].reshape(-1)
        d_keep = d_all[keep]
        if len(d_keep) >= min_points:
            lo_d, hi_d = np.percentile(d_keep, [2, 98])
            if hi_d > lo_d:
                hist, edges = np.histogram(d_keep, bins=32, range=(lo_d, hi_d))
                centers = 0.5 * (edges[:-1] + edges[1:])
                strong = np.flatnonzero(hist >= 0.15 * hist.max())
                if depth_mode == "peak":
                    # The depth layer holding most vegetation, so a nearer
                    # object (a fountain, a pot) in front of the plant does
                    # not win just by being closer.
                    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
                    strong = np.array([int(sm.argmax())])
                if len(strong):
                    mode = float(centers[strong[0]])
                    near = np.abs(d_all - mode) <= depth_band * max(mode, 1e-6)
                    if int((keep & near).sum()) >= min_points:
                        keep = keep & near

    pts = patch[keep]
    if len(pts) < min_points:
        return None

    centroid = np.median(pts, axis=0)

    d = np.linalg.norm(pts - centroid, axis=1)
    pts = pts[d < np.percentile(d, 97)]
    if len(pts) < min_points:
        return None

    core_lo = np.percentile(pts, 15, axis=0)
    core_hi = np.percentile(pts, 85, axis=0)
    disp_lo = np.percentile(pts, 1, axis=0)
    disp_hi = np.percentile(pts, 99, axis=0)

    if len(pts) > max_keep:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_keep, replace=False)]

    return {
        "centroid": np.median(pts, axis=0),
        "min": core_lo, "max": core_hi,
        "disp_min": disp_lo, "disp_max": disp_hi,
        "points": pts.astype(np.float32),
        "n_points": int(len(pts)),
        "box_model": [mx0, my0, mx1, my1],
    }

def instance_stats(members):

    w = np.array([m["score"] for m in members], dtype=np.float64)
    pos = np.stack([m["centroid"] for m in members])
    centroid = (pos * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    mins = np.stack([m["min"] for m in members])
    maxs = np.stack([m["max"] for m in members])
    lo = np.median(mins, axis=0)
    hi = np.median(maxs, axis=0)
    typ_diag = float(np.median(np.linalg.norm(maxs - mins, axis=1)))
    return centroid, lo, hi, typ_diag

def multiview_support_filter(members, cell, min_support, min_points=40):

    pts = [m["points"] for m in members]
    allp = np.concatenate(pts, axis=0)
    frames = np.array([m["frame"] for m in members])
    if min_support <= 1 or len(np.unique(frames)) < min_support * 2 or cell <= 0:
        return allp, 0

    origin = allp.min(axis=0)
    q = np.floor((allp - origin) / cell).astype(np.int64)
    span = q.max(axis=0) + 1
    key = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
    per_point_frame = np.repeat(frames, [len(p) for p in pts])

    pairs = np.unique(np.stack([key, per_point_frame], axis=1), axis=0)
    ukeys, counts = np.unique(pairs[:, 0], return_counts=True)
    support = counts[np.searchsorted(ukeys, key)]

    keep = support >= min_support
    if keep.sum() < max(min_points, 0.05 * len(allp)):
        return allp, 0
    return allp[keep], int((~keep).sum())

def display_box(members, pct=1.0, support_cell=0.0, min_support=1):

    pts, dropped = multiview_support_filter(members, support_cell, min_support)
    lo = np.percentile(pts, pct, axis=0)
    hi = np.percentile(pts, 100.0 - pct, axis=0)
    return lo, hi, pts, dropped

def make_instance(members):
    centroid, lo, hi, typ = instance_stats(members)
    return {"members": members, "centroid": centroid, "min": lo, "max": hi,
            "typ_diag": typ}

def cluster_detections(dets, cluster_dist):

    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    instances = []
    for i in order:
        d = dets[i]
        best, best_dist = None, cluster_dist
        for inst in instances:
            dist = float(np.linalg.norm(np.asarray(inst["centroid"]) - d["centroid"]))
            if dist < best_dist:
                best, best_dist = inst, dist
        if best is None:
            instances.append(make_instance([d]))
        else:
            best["members"].append(d)
            (best["centroid"], best["min"], best["max"],
             best["typ_diag"]) = instance_stats(best["members"])
    return instances

def box_iou_3d(a_min, a_max, b_min, b_max):

    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    if inter <= 0:
        return 0.0
    va = np.prod(np.clip(a_max - a_min, 1e-9, None))
    vb = np.prod(np.clip(b_max - b_min, 1e-9, None))
    return float(inter / (va + vb - inter))

def merge_affinity(a, b, iou_threshold, sep_frac):

    iou = box_iou_3d(a["min"], a["max"], b["min"], b["max"])
    if iou > iou_threshold:
        return 1.0 + iou
    dist = float(np.linalg.norm(np.asarray(a["centroid"]) - np.asarray(b["centroid"])))
    limit = sep_frac * max(a["typ_diag"], b["typ_diag"])
    if limit > 0 and dist < limit:
        return 1.0 - dist / limit
    return 0.0

def merge_overlapping(instances, iou_threshold, sep_frac):

    changed = True
    while changed and len(instances) > 1:
        changed = False
        best, best_score = None, 1e-9
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                aff = merge_affinity(instances[i], instances[j], iou_threshold, sep_frac)
                if aff > best_score:
                    best, best_score = (i, j), aff
        if best is not None:
            i, j = best
            a, b = instances[i], instances[j]
            a["members"].extend(b["members"])
            (a["centroid"], a["min"], a["max"],
             a["typ_diag"]) = instance_stats(a["members"])
            instances.pop(j)
            changed = True
    return instances

def crop_sharpness(crop_rgb):

    if crop_rgb.size == 0:
        return 0.0
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var() / (g.var() + 1e-6))

def crop_quality(crop_rgb):

    if crop_rgb.size == 0:
        return 0.0
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    sharp = crop_sharpness(crop_rgb)
    sharp = sharp / (sharp + 0.25)
    size = min(float(crop_rgb.shape[0] * crop_rgb.shape[1]) / (400.0 * 400.0), 1.0)
    sky = float((g > 205).mean())
    return float(max(sharp, 0.01) * max(size, 0.05) * (1.0 - sky))

def crop_from_instance(art_dir, fid, points_3d, model_hw, pad=0.10,
                       lo_pct=3.0, hi_pct=97.0, min_px=64):

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

def vote_label(members, key):

    tally = {}
    for m in members:
        for name, score in m[key]:
            tally[name] = tally.get(name, 0.0) + score * m["score"]
        if key == "species" and not m[key]:
            continue
    if not tally:
        return None, 0.0, 0.0
    total = sum(tally.values())
    name = max(tally, key=tally.get)

    raws = [sc for m in members for nm, sc in m[key] if nm == name]
    raw = float(np.mean(raws)) if raws else 0.0
    return name, float(tally[name] / max(total, 1e-9)), raw

def main():
    ap = argparse.ArgumentParser(description="Detect plants and map them into the point cloud")
    ap.add_argument("--artifacts", required=True, help="Directory from --save_artifacts")
    ap.add_argument("--out", default=None, help="Output JSON (default: <artifacts>/plants.json)")
    ap.add_argument("--models_dir", default=r"C:\Users\shaur\AppData\Local\lingbot-map\models")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS),
                    help="Comma-separated open-vocabulary detection prompts")
    ap.add_argument("--det_threshold", type=float, default=0.15)
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="Use every Nth saved frame")
    ap.add_argument("--max_frames", type=int, default=0)

    ap.add_argument("--classifier", default="bioclip", choices=["bioclip", "plantnet", "none"])
    ap.add_argument("--plantnet_project", default=PlantNetClassifier.PROJECT,
                    help="Pl@ntNet flora to search. Regional floras score far "
                         "better than 'all' because they exclude species that "
                         "cannot occur where the photo was taken; the default "
                         "is the TDWG region covering California")
    ap.add_argument("--plantnet_api_key", default=None,
                    help="Required for --classifier plantnet. NOTE: each crop is "
                         "uploaded to Pl@ntNet's servers")
    ap.add_argument("--species_list", default=None,
                    help="File of 'Scientific name|Common name' lines, replacing the default set")

    ap.add_argument("--conf_threshold", type=float, default=None,
                    help="Depth-confidence gate for 3D points (default: from meta.json)")
    ap.add_argument("--min_points", type=int, default=40)
    ap.add_argument("--cluster_dist", type=float, default=0.0,
                    help="3D merge distance (0 = auto, 4%% of scene extent)")
    ap.add_argument("--vegetation_mask", action=argparse.BooleanOptionalAction, default=True,
                    help="Keep only vegetation-coloured pixels inside a detection box when "
                         "building its 3D extent, instead of shrinking the box")
    ap.add_argument("--max_instance_points", type=int, default=60000,
                    help="Cap on pooled 3D points stored per instance for shape highlighting")
    ap.add_argument("--display_pct", type=float, default=1.0,
                    help="Percentile trim for the drawn box. 0 = exact min/max")
    ap.add_argument("--min_species_score", type=float, default=0.20,
                    help="Raw classifier score below which a plant is reported as "
                         "unidentified rather than given a name")
    ap.add_argument("--rank_crops", action=argparse.BooleanOptionalAction, default=True,
                    help="Choose which crops to classify by how identifiable they look "
                         "(vegetation, detail, size, not backlit) rather than by detection "
                         "score, which favours distant whole-plant shots")
    ap.add_argument("--min_species_agreement", type=float, default=0.5,
                    help="Share of an instance's classified crops that must agree on the "
                         "species before it is reported as identified. A closed-set "
                         "classifier scores its best-of-N confidently even when the true "
                         "species is absent, so raw score alone cannot catch that; "
                         "disagreement between views can")
    ap.add_argument("--min_species_views", type=int, default=5,
                    help="Views an instance needs before a species name is reported. "
                         "A name from two glimpses is not evidence")
    ap.add_argument("--crop_from_3d", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Re-cut each classifier view around where the fused "
                         "instance's own 3D points land in that frame, instead of "
                         "using the per-frame detection box. Off by default "
                         "because it was measured and did not help: the instance's "
                         "pooled points carry the wall and ground the plant stands "
                         "on, so the projected box drifts onto them, and it fails "
                         "outright on sparse instances. Split-half consistency "
                         "17/38 against 19/42 without it")
    ap.add_argument("--fuse_views", action=argparse.BooleanOptionalAction, default=True,
                    help="Send several views of one instance to the classifier as a "
                         "single identification, instead of classifying each crop "
                         "and voting. Both classifiers support it and it is what "
                         "they are built for; voting was this pipeline's own idea")
    ap.add_argument("--stack_views", type=int, default=0,
                    help="Build the first view by registering and averaging this "
                         "many of the best crops. Cuts the analog chroma noise, at "
                         "the risk of smearing if the views do not register. 0 off")
    ap.add_argument("--min_crop_side", type=int, default=224,
                    help="Upscale crops shorter than this with Lanczos before "
                         "classifying, so the classifier's own resize is not the "
                         "limiting filter")
    ap.add_argument("--classify_top_k", type=int, default=5,
                    help="Classify only the K best-scoring crops per plant instance. "
                         "Keeps a remote API's quota (and local runtime) proportional to "
                         "the number of plants, not the number of detections")
    ap.add_argument("--foliage_crop", action=argparse.BooleanOptionalAction, default=True,
                    help="Classify the vegetation region inside each box rather than the "
                         "whole box, so pots and walls do not drive the species guess")
    ap.add_argument("--merge_iou", type=float, default=0.15,
                    help="Merge two instances whose 3D boxes overlap by more than this")
    ap.add_argument("--merge_sep_frac", type=float, default=1.2,
                    help="Also merge when centroids are closer than this fraction of the "
                         "larger box diagonal. Absorbs reconstruction drift, which scales "
                         "with object size rather than being a fixed distance. Swept "
                         "against controls: at 1.2 a backyard settles from 12 fragments to "
                         "7 plants while a plaza stays at 15 and a single potted plant "
                         "stays at 1. It was 0.8 only because the radius used to be derived "
                         "from a growing union box, which made any wider value run away")
    ap.add_argument("--nms_iou", type=float, default=0.5,
                    help="Class-agnostic per-frame NMS. Overlapping prompts fire several "
                         "boxes on one plant; this collapses them")
    ap.add_argument("--min_vegetation", type=float, default=0.06,
                    help="Reject a detection whose crop is less than this fraction "
                         "vegetation (excess-green). Kills non-plant false positives")
    ap.add_argument("--max_box_frac", type=float, default=0.6,
                    help="Ignore detection boxes covering more than this fraction of the "
                         "frame: they do not say which plant is meant (1 keeps all)")
    ap.add_argument("--depth_mode", default="peak", choices=["peak", "nearest"],
                    help="Lift points from the depth layer with the most vegetation "
                         "(peak) or from the nearest strong layer (nearest, the old rule)")
    ap.add_argument("--veg_otsu", action=argparse.BooleanOptionalAction, default=True,
                    help="Raise the per-box vegetation threshold with Otsu's method")
    ap.add_argument("--depth_band", type=float, default=0.25,
                    help="Keep only points within this relative depth of the nearest "
                         "coherent layer in each detection box, dropping the wall or "
                         "fence behind the plant. Range separates them where colour "
                         "cannot. 0 disables")
    ap.add_argument("--veg_erode", type=int, default=1,
                    help="Erode the vegetation mask by this many pixels before lifting, "
                         "so foliage outlines whose depth belongs to the wall behind "
                         "them do not become plant points. 0 disables")
    ap.add_argument("--min_view_support", type=int, default=2,
                    help="Discard a plant's 3D points unless this many distinct frames "
                         "independently place a point in the same voxel. Removes "
                         "reconstruction floaters, which no single-frame test catches. "
                         "1 disables")
    ap.add_argument("--support_cell", type=float, default=0.0,
                    help="Voxel size for the view-support vote (0 = 2x the map voxel)")
    ap.add_argument("--min_views", type=int, default=2,
                    help="Discard instances seen in fewer than this many frames")
    args = ap.parse_args()

    art = args.artifacts
    meta = load_meta(art)
    model_hw = tuple(meta["model_hw"])
    conf_threshold = args.conf_threshold
    if conf_threshold is None:
        conf_threshold = meta.get("conf_threshold", 1.5)
    support_cell = args.support_cell or meta.get("voxel_size", 0.005) * 2.0
    frame_ids = meta["saved_frame_ids"][::max(1, args.frame_stride)]
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    print(f"Artifacts : {art}")
    print(f"Frames    : {len(frame_ids)} of {len(meta['saved_frame_ids'])} saved")
    print(f"Prompts   : {prompts}")
    print(f"Device    : {args.device}")

    detector = Owlv2Detector(os.path.join(args.models_dir, "owlv2"),
                             args.device, args.det_threshold)

    species = DEFAULT_SPECIES
    if args.species_list:
        species = []
        with open(args.species_list, encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    sci, com = line.strip().split("|", 1)
                    species.append((sci.strip(), com.strip()))

    classifier = None
    if args.classifier == "bioclip":
        classifier = BioClipClassifier(os.path.join(args.models_dir, "bioclip"),
                                       args.device, species)
        print(f"Species   : BioCLIP zero-shot over {len(species)} candidates (local)")
    elif args.classifier == "plantnet":
        if not args.plantnet_api_key:
            ap.error("--classifier plantnet requires --plantnet_api_key")
        classifier = PlantNetClassifier(args.plantnet_api_key,
                                        project=args.plantnet_project)
        print(f"  Pl@ntNet flora: {args.plantnet_project}")
        print("Species   : Pl@ntNet API")
        print("            NOTE: every detected crop is uploaded to Pl@ntNet's servers.")

    dets = []
    crops_dir = os.path.join(art, "crops")

    if os.path.isdir(crops_dir):
        for fn in os.listdir(crops_dir):
            if fn.endswith(".jpg"):
                os.remove(os.path.join(crops_dir, fn))
    os.makedirs(crops_dir, exist_ok=True)
    for n, fid in enumerate(frame_ids):
        depth, conf, extrinsic, intrinsic, rgb = load_frame(art, fid)
        if rgb is None:
            continue
        scale, y_off = source_to_model_transform(rgb.shape[:2], model_hw)

        world = unproject_depth_map_to_point_map(
            depth[None], extrinsic[None], intrinsic[None])[0]
        rgb_model = cv2.resize(rgb, (model_hw[1], model_hw[0]),
                               interpolation=cv2.INTER_AREA)
        boxes, scores, labels = detector(rgb, prompts)
        n_raw = len(boxes)
        boxes, scores, labels = nms_2d(boxes, scores, labels, args.nms_iou)
        n_nms = len(boxes)

        kept = 0
        rejected_veg = 0
        rejected_big = 0
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            frac = max(0, x1 - x0) * max(0, y1 - y0) / float(rgb.shape[0] * rgb.shape[1])
            if frac > args.max_box_frac:
                rejected_big += 1
                continue
            crop = rgb[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                continue
            veg = vegetation_fraction(crop)
            if veg < args.min_vegetation:
                rejected_veg += 1
                continue
            geo = lift_box_to_3d(box, world, conf, rgb_model, scale, y_off,
                                 conf_threshold, args.min_points,
                                 use_vegetation=args.vegetation_mask,
                                 veg_erode=args.veg_erode,
                                 depth=depth, depth_band=args.depth_band,
                                 depth_mode=args.depth_mode, veg_otsu=args.veg_otsu)
            if geo is None:
                continue
            crop_name = f"{fid:06d}_{kept}.jpg"
            cv2.imwrite(os.path.join(crops_dir, crop_name),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            dets.append({
                **geo,
                "frame": int(fid),
                "score": float(score),
                "vegetation": round(veg, 3),
                "category": [(label, float(score))],
                "species": [],
                "crop": crop_name,
                "box_src": [float(v) for v in box],
            })
            kept += 1
        print(f"  [{n + 1}/{len(frame_ids)}] frame {fid}: {n_raw} raw -> "
              f"{n_nms} after NMS -> {kept} lifted "
              f"({rejected_veg} rejected as non-vegetation, {rejected_big} as "
              f"covering most of the frame)", flush=True)

    if not dets:
        print("\nNo detections survived. Try lowering --det_threshold or --min_points.")
        return

    cloud = np.load(os.path.join(art, "cloud.npz"))["points"]
    lo, hi = np.percentile(cloud, [2, 98], axis=0)
    extent = float(np.linalg.norm(hi - lo))
    cluster_dist = args.cluster_dist or 0.04 * extent
    print(f"\nScene extent {extent:.3f}; clustering at {cluster_dist:.4f}")

    instances = cluster_detections(dets, cluster_dist)
    n_before = len(instances)
    instances = merge_overlapping(instances, args.merge_iou, args.merge_sep_frac)
    print(f"Merged overlapping boxes: {n_before} -> {len(instances)} "
          f"(IoU > {args.merge_iou} or centroids within "
          f"{args.merge_sep_frac} x box diagonal)")
    instances = [i for i in instances if len(i["members"]) >= args.min_views]
    instances.sort(key=lambda i: -max(m["score"] for m in i["members"]))

    if classifier is not None:
        n_calls = 0
        for inst in instances:

            pooled = np.concatenate([m["points"] for m in inst["members"]]) \
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
                scored.append((q, m, rgb_crop))
            scored.sort(key=lambda t: -t[0])
            chosen = scored[:args.classify_top_k]
            views = []
            for q, m, rgb_crop in chosen:
                crop = rgb_crop
                if args.foliage_crop:
                    crop = foliage_subcrop(crop)
                m["crop_quality"] = round(float(q), 4)
                views.append(upscale_crop(crop, args.min_crop_side))

            if args.stack_views > 1 and len(views) > 1:
                comp = stack_crops(views[:args.stack_views])
                if comp is not None:
                    views = [comp] + views[1:]

            if args.fuse_views and hasattr(classifier, "classify_views"):
                dist = classifier.classify_views(views, topk=5)
                if dist:
                    inst["species_fused"] = dist["fused"]
                    inst["species_dist"] = dist
                    for (q, m, _), pv in zip(chosen, dist["per_view"]):
                        m["species"] = [(f"{a}|{b}", c) for a, b, c in pv]
                n_calls += 1
            else:
                per_view = (not args.fuse_views) or args.classifier == "bioclip"
                if per_view:
                    for (q, m, _), v in zip(chosen, views):
                        m["species"] = [(f"{a}|{b}", c)
                                        for a, b, c in classifier(v)]
                        n_calls += 1

                if args.fuse_views and hasattr(classifier, "classify_many"):
                    fused = classifier.classify_many(views, topk=5)
                    if fused:
                        inst["species_fused"] = fused
                    n_calls += 1
        how = ("fused per instance" if args.fuse_views else "per crop")
        print(f"Classified {n_calls} calls, {how} "
              f"(top {args.classify_top_k} views per instance, "
              f"{len(instances)} instances)")

    out = []
    instance_points = []
    for k, inst in enumerate(instances):
        members = inst["members"]
        cat, cat_conf, _ = vote_label(members, "category")
        sp, sp_conf, sp_raw = vote_label(members, "species")
        agreement = sp_conf
        method = "vote"
        fused = inst.get("species_fused")
        if fused:
            method = "fused"
            fsci, fcom, fraw = fused[0]
            total = sum(x[2] for x in fused) or 1.0

            tops = [m["species"][0][0].split("|")[0]
                    for m in members if m.get("species")]
            agreement = (sum(t == fsci for t in tops) / len(tops)) if tops else None
            sp = f"{fsci}|{fcom}"
            sp_raw = fraw
            sp_conf = fraw / total
            if agreement is None:

                agreement = sp_conf
                method = "fused-unchecked"

        sci_top = (sp.split("|", 1)[0] if sp and "|" in sp else (sp or ""))
        dist = inst.get("species_dist") or {}
        at_fused = dist.get("at_fused") or []
        view_scores = [round(float(x), 4) for x in at_fused]
        view_top1 = [m["species"][0][0].split("|")[0]
                     for m in members if m.get("species")]
        spread = round(float(np.std(at_fused)), 4) if len(at_fused) > 1 else None
        ent_bits = (round(float(dist["entropy"]), 3)
                    if "entropy" in dist else None)
        ent_max = (round(float(dist["max_entropy"]), 3)
                   if "max_entropy" in dist else None)
        sharpening = (round(float(sp_raw / max(max(at_fused), 1e-6)), 3)
                      if at_fused else None)
        unsupported = bool(view_top1) and not any(t == sci_top for t in view_top1)

        best = max(members, key=lambda m: m["score"])
        best_crop = max(members, key=lambda m: (m.get("crop_quality", -1.0), m["score"]))
        disp_lo, disp_hi, inst_pts, n_dropped = display_box(
            members, args.display_pct, support_cell, args.min_view_support)
        if n_dropped:
            print(f"  #{k}: dropped {n_dropped:,} points unconfirmed by "
                  f"{args.min_view_support}+ views")
        if len(inst_pts) > args.max_instance_points:
            sel = np.random.default_rng(0).choice(
                len(inst_pts), args.max_instance_points, replace=False)
            inst_pts = inst_pts[sel]
        instance_points.append(inst_pts.astype(np.float32))
        sci, com = (sp.split("|", 1) if sp and "|" in sp else (sp or "", ""))
        out.append({
            "id": k,
            "category": cat,
            "category_confidence": round(cat_conf, 3),
            "species": sci,
            "species_common": com,
            "species_confidence": round(sp_conf, 3),
            "species_raw_score": round(sp_raw, 4),

            "species_method": method,
            "species_agreement": round(float(agreement), 3),
            "species_candidates": [
                {"name": a, "common": b, "score": round(float(c), 4)}
                for a, b, c in (fused or [])],
            "species_view_scores": view_scores,
            "species_view_top1": view_top1,
            "species_spread": spread,
            "species_sharpening": sharpening,
            "species_entropy": ent_bits,
            "species_entropy_max": ent_max,
            "species_unsupported": unsupported,
            "species_identified": bool(sp_raw >= args.min_species_score
                                       and agreement >= args.min_species_agreement
                                       and len(members) >= args.min_species_views),
            "centroid": [float(v) for v in inst["centroid"]],
            "bbox_min": [float(v) for v in disp_lo],
            "bbox_max": [float(v) for v in disp_hi],
            "core_min": [float(v) for v in inst["min"]],
            "core_max": [float(v) for v in inst["max"]],
            "n_points_3d": int(len(inst_pts)),
            "views": len(members),
            "detection_score": round(float(best["score"]), 3),
            "crop": best_crop["crop"],
            "frames": sorted({int(m["frame"]) for m in members}),
            "boxes": [{"frame": int(m["frame"]), "box": m["box_src"], "crop": m["crop"]}
                      for m in members],
        })

    pts_path = os.path.join(art, "plant_points.npz")
    np.savez_compressed(pts_path, **{f"pts_{o['id']}": arr for o, arr in
                                     zip(out, instance_points)})
    print(f"Wrote {pts_path} "
          f"({sum(len(a) for a in instance_points):,} points across {len(out)} instances)")

    out_path = args.out or os.path.join(art, "plants.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scene_extent": extent, "cluster_dist": cluster_dist,
                   "classifier": args.classifier, "plants": out}, f, indent=2)

    print(f"\n{len(dets)} detections -> {len(out)} plant instances (>= {args.min_views} views)")
    for p in out[:15]:
        if p["species"] and p["species_identified"]:
            name = f"{p['species']} ({p['species_common']})"
        elif p["species"]:
            name = f"unidentified (best guess {p['species']}, raw {p['species_raw_score']:.3f})"
        else:
            name = "-"
        print(f"  #{p['id']:<2} {p['category']:<22} x{p['views']:<3} views  "
              f"det {p['detection_score']:.2f}  species: {name}")
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
