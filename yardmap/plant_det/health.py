"""Plant health from a plant's crops: healthy (green leaves), stressed
(discoloured leaves) or dead / dormant (no live leaves).

The head sits on the plant classifier's fine-tuned BioCLIP-2 backbone, so
inside detect_plants.py it reuses the classifier's network and costs one more
small layer per view.  A plant's views and their mirror images are averaged,
like the naming step; a plant is reported as "unsure" (the leading class kept
as a guess) below the confidence where held-out predictions reach 85%
precision: one threshold for a single photo, and one calibrated on
iNaturalist observations with several photos for plants seen in several views.

On a finished map (adds "health" to every plant in plants.json):

    python -m plant_det.health --artifacts ../maps/NAME --model ../models/health/health.pt
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hier import HierNet, prep_eval

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


def make_head(dim, hidden, drop, n):
    if hidden:
        return nn.Sequential(nn.Dropout(drop), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                             nn.Linear(hidden, n))
    return nn.Sequential(nn.Dropout(drop), nn.Linear(dim, n))


def load_backbone(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    v = ck["vocab"]
    net = HierNet(ck["backbone"], len(v["forms"]), len(v["families"]), len(v["genera"]),
                  len(v["species"]), train_blocks=0)
    net.load_state_dict(ck["model"], strict=False)
    return net.to(device).eval()


class HealthClassifier:

    def __init__(self, model_path, device="cuda", net=None, cls_weights=None):
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        self.classes = ck["classes"]
        self.temp, self.tau = ck["temp"], ck["tau"]
        self.tau_plant = ck.get("tau_plant", self.tau)
        self.mu, self.sd = ck["mu"].to(device), ck["sd"].to(device)
        if "backbone" in ck:                     # fine-tuned backbone of its own
            net = load_backbone(cls_weights or os.path.join(MODELS, "classifier", "best.pt"), device)
            net.backbone.load_state_dict(ck["backbone"])
        elif net is None:
            net = load_backbone(cls_weights or os.path.join(MODELS, "classifier", "best.pt"), device)
        self.net = net
        self.head = make_head(len(self.mu), ck["hidden"], ck["drop"], len(self.classes))
        self.head.load_state_dict(ck["head"])
        self.head = self.head.to(device).eval()
        self.device = device

    @torch.no_grad()
    def view_probs(self, crops):
        """crops: list of HxWx3 uint8 RGB -> (V, C) probabilities."""
        x = torch.stack([prep_eval(c) for c in crops]).to(self.device)
        n = self.net
        ps = []
        for xx in (x, torch.flip(x, dims=[3])):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = n.backbone((xx - n.mean_t) / n.std_t)
            if f.dim() == 3:
                f = f[:, 0]
            ps.append(F.softmax(self.head((f.float() - self.mu) / self.sd) / self.temp, -1))
        return 0.5 * (ps[0] + ps[1])

    def classify_views(self, crops):
        if not crops:
            return None
        p = self.view_probs(crops)
        fused = p.mean(0)
        i = int(fused.argmax())
        conf = float(fused[i])
        votes = [int(q.argmax()) for q in p]
        agree = votes.count(i) / len(votes)
        # averaged views are less peaked than one photo, so several views
        # have their own 85%-precision threshold
        sure = conf >= (self.tau_plant if len(crops) > 1 else self.tau)
        return {"label": self.classes[i] if sure else "unsure", "agree": round(agree, 3),
                "guess": self.classes[i], "conf": round(conf, 3),
                "probs": {c: round(float(fused[k]), 3) for k, c in enumerate(self.classes)},
                "view_labels": [self.classes[int(q.argmax())] for q in p],
                "views": len(crops)}


def crop_views(art, plant, k=12):
    """A plant's saved crops, best detection first (finished maps keep one
    crop per sighting under crops/)."""
    import cv2
    out = []
    for b in plant.get("boxes", [])[:64]:
        im = cv2.imread(os.path.join(art, "crops", b["crop"]))
        if im is not None and min(im.shape[:2]) >= 16:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    out.sort(key=lambda im: -float(np.var(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_32F))))
    return out[:k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--model", default=os.path.join(MODELS, "health", "health.pt"))
    ap.add_argument("--cls_weights", default=os.path.join(MODELS, "classifier", "best.pt"))
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    path = os.path.join(a.artifacts, "plants.json")
    doc = json.load(open(path, encoding="utf-8"))
    hc = HealthClassifier(a.model, a.device, cls_weights=a.cls_weights)
    counts = {}
    for p in doc["plants"]:
        h = hc.classify_views(crop_views(a.artifacts, p, a.top_k))
        if h is not None:
            p["health"] = h
            counts[h["label"]] = counts.get(h["label"], 0) + 1
    doc["health_model"] = os.path.basename(a.model)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"health of {len(doc['plants'])} plants: {counts}")


if __name__ == "__main__":
    main()
