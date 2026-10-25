"""Plant-level threshold from eval_multiview.py rows: the lowest fused
confidence reaching 85% precision on one half of the observations, scored on
the other half, written into the model as tau_plant.

    python calib_plant.py MODEL.pt MULTIVIEW.json
"""
import json, sys, zlib

import numpy as np
import torch

model, mv = sys.argv[1], sys.argv[2]
R = json.load(open(mv))["rows"]
A = [r for r in R if zlib.crc32(r["obs"].encode()) % 2 == 0]
B = [r for r in R if zlib.crc32(r["obs"].encode()) % 2 == 1]


def pr(S, t):
    s = [r for r in S if r["conf"] >= t]
    return len(s) / len(S), (float(np.mean([r["guess"] == r["cls"] for r in s])) if s else 0.0)


tau = next((round(float(t), 3) for t in np.arange(0.34, 0.99, 0.01) if pr(A, t)[1] >= 0.85), 0.99)
cov, prec = pr(B, tau)
ck = torch.load(model, map_location="cpu", weights_only=False)
ck["tau_plant"] = tau
ck["metrics"]["plant_level"] = {"tau_plant": tau, "observations": len(R), "views_mean": float(np.mean([r["n"] for r in R])),
                                "heldout_coverage": round(cov, 4), "heldout_precision": round(prec, 4),
                                "guess_acc": round(float(np.mean([r["guess"] == r["cls"] for r in R])), 4)}
torch.save(ck, model)
print(json.dumps(ck["metrics"]["plant_level"]))
