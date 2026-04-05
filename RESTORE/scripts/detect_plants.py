"""Find plants in reconstructed frames and place them as 3D objects in the map.

Runs entirely off the artifacts written by ``stream_video.py --save_artifacts``,
so it never reloads the reconstruction model and can be re-run freely while you
tune prompts and thresholds.

Pipeline per selected frame:
  1. OWLv2 open-vocabulary detection against plant text prompts -> boxes.
     Open-vocabulary matters here: a fixed-class detector would only offer COCO's
     single "potted plant", while the prompt list can name hedges, shrubs and
     flowering plants separately and be edited without retraining anything.
  2. Each box's pixels are back-projected through that frame's own depth map and
     pose -- the same call the point cloud uses -- so a detection becomes a
     cluster of real 3D points rather than a ray guess.
  3. The crop goes to a classifier for a species name (BioCLIP locally, or the
     Pl@ntNet API if you opt in).

Detections from different frames are then clustered in 3D, because one plant seen
from twenty viewpoints should end up as one object. Each cluster keeps a
confidence-weighted vote over species, the union of its boxes, and its best crop.

Output: ``plants.json`` plus a ``crops/`` directory, both consumed by view_map.py.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lingbot_map.utils.geometry import unproject_depth_map_to_point_map


# =============================================================================
# Candidate species for zero-shot classification
# =============================================================================

# BioCLIP is a CLIP model: it scores an image against a *closed* list of text
# labels, so it can only ever return something named here. This list covers
# common North American suburban / garden planting; override it with
# --species_list (one "Scientific name|Common name" per line) for other settings.
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


# =============================================================================
# Artifact access
# =============================================================================

def load_meta(art_dir):
    with open(os.path.join(art_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_frame(art_dir, fid):
    """Return (depth, conf, extrinsic, intrinsic, rgb) for one saved frame."""
    d = np.load(os.path.join(art_dir, "frames", f"{fid:06d}.npz"))
    jpg = os.path.join(art_dir, "frames", f"{fid:06d}.jpg")
    rgb = None
    if os.path.exists(jpg):
        rgb = cv2.cvtColor(cv2.imread(jpg), cv2.COLOR_BGR2RGB)
    return (d["depth"].astype(np.float32), d["conf"].astype(np.float32),
            d["extrinsic"], d["intrinsic"], rgb)


def source_to_model_transform(src_hw, model_hw, patch_size=14):
    """Map source-resolution pixel coords onto the model's depth-map grid.

    ``preprocess_frame`` fixes width and rounds height to a patch multiple, then
    center-crops anything taller than image_size. Both steps are reproduced here
    rather than assumed, because a silent half-crop offset would shift every box
    by tens of pixels and quietly poison the 3D positions.
    """
    hs, ws = src_hw
    hm, wm = model_hw
    scale = wm / ws
    resized_h = round(hs * scale / patch_size) * patch_size
    y_off = (resized_h - hm) // 2 if resized_h > hm else 0
    return scale, y_off


# =============================================================================
# Detection
# =============================================================================

class Owlv2Detector:
    """Open-vocabulary box detector. Prompts are plain English, editable at will."""

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
    """Zero-shot species naming from a fixed candidate list."""

    def __init__(self, model_dir, device, species):
        import open_clip
        weights = os.path.join(model_dir, "open_clip_pytorch_model.bin")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=weights)
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.device = device
        self.species = species
        # BioCLIP was trained with taxonomic strings paired to vernacular names;
        # matching that phrasing at inference is worth a few points of accuracy.
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


class PlantNetClassifier:
    """Pl@ntNet API. Uploads each crop to an external service -- opt-in only."""

    ENDPOINT = "https://my-api.plantnet.org/v2/identify/all"

    def __init__(self, api_key, timeout=30):
        self.api_key = api_key
        self.timeout = timeout

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
        except Exception as exc:                      # network / quota / auth
            print(f"    Pl@ntNet request failed: {exc}")
            return []
        out = []
        for item in results:
            sp = item.get("species", {})
            sci = sp.get("scientificNameWithoutAuthor", "unknown")
            commons = sp.get("commonNames") or [""]
            out.append((sci, commons[0], float(item.get("score", 0.0))))
        return out


def nms_2d(boxes, scores, labels, iou_threshold=0.5):
    """Class-agnostic non-maximum suppression within one frame.

    The prompt list deliberately overlaps -- "a bush", "a shrub" and "a flowering
    plant" all describe the same object -- so a single plant fires several
    prompts and arrives as several boxes. Left alone those become separate 3D
    instances downstream. Suppression is class-agnostic precisely because the
    duplicates carry *different* labels; per-class NMS would keep every one.
    """
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
    """Share of a crop that reads as vegetation, by excess green (2G - R - B).

    Cheap sanity check on what the detector claimed: a garage door or a parked
    car prompted as "a shrub" has essentially no green, and this rejects it
    without another model in the loop.
    """
    if crop_rgb.size == 0:
        return 0.0
    f = crop_rgb.astype(np.float32)
    exg = 2.0 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]
    return float((exg > 8.0).mean())


def foliage_subcrop(crop_rgb, min_frac=0.04):
    """Narrow a detection crop down to its actual vegetation before classifying.

    A "potted plant" box is mostly pot, and a species classifier fed that image
    is largely reading terracotta -- which is exactly how you get an olive tree
    out of a chrysanthemum. Excess-green (2G - R - B) is a standard vegetation
    index and needs no extra model; the largest green component's bounding box
    is a good approximation of "the leafy part".

    Falls back to the full crop when there is too little green to trust, so a
    bare-branch or heavily shadowed plant is not reduced to noise.
    """
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
    return crop_rgb[y0:y1, x0:x1]


# =============================================================================
# 2D detection -> 3D object
# =============================================================================

def lift_box_to_3d(box_src, world, conf, rgb_model, scale, y_off,
                   conf_threshold, min_points, use_vegetation=True,
                   max_keep=2500):
    """Turn one image-space box into the 3D points of the plant inside it.

    Two boxes come back, because two different jobs need different answers:

      `core`    a tight 15-85 percentile box. Stable across viewpoints, so it is
                what drives clustering and the merge radius.
      `display` a generous 1-99 percentile box covering the whole object, which
                is what gets drawn.

    Earlier this returned only the tight box and drew that, which is why the
    highlight covered a fraction of the plant.

    Background rejection is done by vegetation colour rather than by shrinking
    the box. Shrinking discards the plant's outer foliage -- exactly the part
    that makes the extent correct -- while still admitting whatever wall sits in
    the middle of the box. An excess-green test keeps the silhouette and drops
    the driveway behind it.
    """
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
        veg = exg > 8.0
        # Only trust the colour filter when it retains a real share of the box;
        # dark or shadowed foliage can fail excess-green entirely.
        if (keep & veg).sum() >= max(min_points, 0.15 * keep.sum()):
            keep = keep & veg

    pts = patch[keep]
    if len(pts) < min_points:
        return None

    centroid = np.median(pts, axis=0)
    # One robust pass against depth's long far tail, then measure.
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
    """Summarize one cluster from its members, using statistics that do not drift.

    The box is the *median* of member boxes and `typ_diag` the median member
    diagonal -- deliberately not the union. A union grows every time a member is
    added, and when the merge radius is derived from object size (as it must be,
    to absorb drift that scales with the object) a growing box widens the radius,
    which admits more members, which grows the box again. That runaway collapsed
    a whole courthouse plaza into a single "plant". Medians stay put as views
    accumulate, because every view is measuring the same object.
    """
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
    """Keep only the 3D points that independent viewpoints agree on.

    A detection box always contains more than its object: background seen around
    and between the leaves, and -- worse -- reconstruction floaters, little slabs
    of geometry hallucinated in mid-air where the depth model was unsure. A
    floater sitting at roughly the plant's distance passes every single-frame
    test there is. It is at plausible depth, it has high depth-confidence, and
    on this footage it passes the vegetation gate too, because a tan stucco
    artifact (RGB 115/106/75) scores +22 on excess green.

    What a floater cannot do is survive parallax. It is an artifact of one
    viewpoint, so as the camera moves it fails to land in the same place twice,
    while the plant's real surface is voted for again and again. Counting how
    many *distinct frames* put a point in each voxel separates the two without
    assuming anything about colour -- which matters here, since the plant's
    leaves are bronze-pink and colour tells us nothing.

    Skipped when the instance has too few views to vote, and abandoned if it
    would leave too little behind, so a genuinely sparsely-seen plant is not
    filtered out of existence.
    """
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

    # Distinct (voxel, frame) pairs, then how many frames each voxel collected.
    pairs = np.unique(np.stack([key, per_point_frame], axis=1), axis=0)
    ukeys, counts = np.unique(pairs[:, 0], return_counts=True)
    support = counts[np.searchsorted(ukeys, key)]

    keep = support >= min_support
    if keep.sum() < max(min_points, 0.05 * len(allp)):
        return allp, 0
    return allp[keep], int((~keep).sum())


def display_box(members, pct=1.0, support_cell=0.0, min_support=1):
    """The drawn extent: percentile bounds over every point from every view.

    Pooling views is the point -- no single viewpoint sees the whole plant, so a
    box from one view (or the median of such boxes) is necessarily smaller than
    the object. Percentiles rather than min/max keep a few stray reprojections
    from inflating it.
    """
    pts, dropped = multiview_support_filter(members, support_cell, min_support)
    lo = np.percentile(pts, pct, axis=0)
    hi = np.percentile(pts, 100.0 - pct, axis=0)
    return lo, hi, pts, dropped


def make_instance(members):
    centroid, lo, hi, typ = instance_stats(members)
    return {"members": members, "centroid": centroid, "min": lo, "max": hi,
            "typ_diag": typ}


def cluster_detections(dets, cluster_dist):
    """Greedy seeding pass: highest-scoring detections anchor clusters."""
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
    """Axis-aligned 3D IoU."""
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    if inter <= 0:
        return 0.0
    va = np.prod(np.clip(a_max - a_min, 1e-9, None))
    vb = np.prod(np.clip(b_max - b_min, 1e-9, None))
    return float(inter / (va + vb - inter))


def merge_affinity(a, b, iou_threshold, sep_frac):
    """How strongly two instances look like the same physical plant.

    Two tests, either sufficient:
      * 3D box IoU above threshold -- they occupy the same space.
      * centroids closer than `sep_frac` x the larger box's diagonal -- they are
        nearer to each other than the object's own size.

    The second test is what handles reconstruction drift. Depth here is only
    weakly confident, so one static plant back-projects to slightly different
    places from different viewpoints; the spread is a fraction of the object
    rather than a fixed distance, so the criterion has to scale with the object.
    """
    iou = box_iou_3d(a["min"], a["max"], b["min"], b["max"])
    if iou > iou_threshold:
        return 1.0 + iou
    dist = float(np.linalg.norm(np.asarray(a["centroid"]) - np.asarray(b["centroid"])))
    limit = sep_frac * max(a["typ_diag"], b["typ_diag"])
    if limit > 0 and dist < limit:
        return 1.0 - dist / limit
    return 0.0


def merge_overlapping(instances, iou_threshold, sep_frac):
    """Agglomerate clusters that describe the same physical plant."""
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


def vote_label(members, key):
    """Confidence-weighted vote across every view of one plant."""
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
    # The normalized share says which candidate won; the raw score says whether
    # the classifier believed any of them. A closed-set classifier always
    # produces a confident-looking share, so the raw score is the honest signal.
    raws = [sc for m in members for nm, sc in m[key] if nm == name]
    raw = float(np.mean(raws)) if raws else 0.0
    return name, float(tally[name] / max(total, 1e-9)), raw


# =============================================================================
# Main
# =============================================================================

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
    ap.add_argument("--classify_top_k", type=int, default=5,
                    help="Classify only the K best-scoring crops per plant instance. "
                         "Keeps a remote API's quota (and local runtime) proportional to "
                         "the number of plants, not the number of detections")
    ap.add_argument("--foliage_crop", action=argparse.BooleanOptionalAction, default=True,
                    help="Classify the vegetation region inside each box rather than the "
                         "whole box, so pots and walls do not drive the species guess")
    ap.add_argument("--merge_iou", type=float, default=0.15,
                    help="Merge two instances whose 3D boxes overlap by more than this")
    ap.add_argument("--merge_sep_frac", type=float, default=0.8,
                    help="Also merge when centroids are closer than this fraction of the "
                         "larger box diagonal. Absorbs reconstruction drift, which scales "
                         "with object size rather than being a fixed distance")
    ap.add_argument("--nms_iou", type=float, default=0.5,
                    help="Class-agnostic per-frame NMS. Overlapping prompts fire several "
                         "boxes on one plant; this collapses them")
    ap.add_argument("--min_vegetation", type=float, default=0.06,
                    help="Reject a detection whose crop is less than this fraction "
                         "vegetation (excess-green). Kills non-plant false positives")
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
        classifier = PlantNetClassifier(args.plantnet_api_key)
        print("Species   : Pl@ntNet API")
        print("            NOTE: every detected crop is uploaded to Pl@ntNet's servers.")

    # ---- per-frame detection ----
    dets = []
    crops_dir = os.path.join(art, "crops")
    # Wipe first: crop filenames are indexed per run, so a shorter run leaves the
    # previous run's files behind to be mistaken for current output.
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
        # Once per frame, not once per detection -- this is the same call the
        # point cloud uses, so the boxes land in identical coordinates.
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
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            crop = rgb[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                continue
            veg = vegetation_fraction(crop)
            if veg < args.min_vegetation:
                rejected_veg += 1
                continue
            geo = lift_box_to_3d(box, world, conf, rgb_model, scale, y_off,
                                 conf_threshold, args.min_points,
                                 use_vegetation=args.vegetation_mask)
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
              f"({rejected_veg} rejected as non-vegetation)", flush=True)

    if not dets:
        print("\nNo detections survived. Try lowering --det_threshold or --min_points.")
        return

    # ---- cluster across frames ----
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

    # Classification runs here, not in the detection loop. One plant seen 96 times
    # needs one identification, not 96 -- which matters for wall-clock locally and
    # for quota when the classifier is a metered remote API.
    if classifier is not None:
        n_calls = 0
        for inst in instances:
            best_members = sorted(inst["members"], key=lambda m: -m["score"])[:args.classify_top_k]
            for m in best_members:
                img = cv2.imread(os.path.join(crops_dir, m["crop"]))
                if img is None:
                    continue
                crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if args.foliage_crop:
                    crop = foliage_subcrop(crop)
                sp = classifier(crop)
                m["species"] = [(f"{a}|{b}", c) for a, b, c in sp]
                n_calls += 1
        print(f"Classified {n_calls} crops "
              f"(top {args.classify_top_k} per instance, {len(instances)} instances)")

    out = []
    instance_points = []
    for k, inst in enumerate(instances):
        members = inst["members"]
        cat, cat_conf, _ = vote_label(members, "category")
        sp, sp_conf, sp_raw = vote_label(members, "species")
        best = max(members, key=lambda m: m["score"])
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
            "species_identified": bool(sp_raw >= args.min_species_score),
            "centroid": [float(v) for v in inst["centroid"]],
            "bbox_min": [float(v) for v in disp_lo],
            "bbox_max": [float(v) for v in disp_hi],
            "core_min": [float(v) for v in inst["min"]],
            "core_max": [float(v) for v in inst["max"]],
            "n_points_3d": int(len(inst_pts)),
            "views": len(members),
            "detection_score": round(float(best["score"]), 3),
            "crop": best["crop"],
            "frames": sorted({int(m["frame"]) for m in members}),
        })

    # The pooled points are what let the viewer highlight the plant's actual
    # shape rather than everything inside its bounding box -- a box around a
    # shrub in front of a wall contains a lot of wall.
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
