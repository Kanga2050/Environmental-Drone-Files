"""Training samples from hand-labelled plants in the real map.

Each labelled plant gives a prompt built from its real detection boxes and
lifted seeds (exactly what segment.py feeds the network) and per-point
targets: 1 for labelled points, 0 for the other points shown in the
labelling page's region for that plant, -1 (ignored) for everything the
labeller never saw.
"""
import json
import os

import numpy as np

from . import prompt as P


class RealLabelled:

    def __init__(self, art, labels_path, n_points=8192, holdout=(), only_done=True):
        from view_map3d import MapScene
        from .yard import Yard
        with open(labels_path) as f:
            lab = json.load(f)["plants"]
        regions = np.load(os.path.join(art, "plant_label_regions.npz"))
        s = MapScene(art, use_seg=False)
        y = Yard(s)
        views, seeds = y.views()
        inv = np.empty_like(s.order)
        inv[s.order] = np.arange(len(s.order))
        self.items, self.held = [], []
        for pid_s, entry in lab.items():
            pid = int(pid_s)
            if (only_done and not entry.get("done")) or f"idx_{pid}" not in regions:
                continue
            vs, sd = views.get(pid, []), seeds.get(pid)
            if not vs or sd is None or len(sd) < 20:
                continue
            lo, hi, diag = P.crop_region(sd)
            pool = np.flatnonzero(np.all((y.xyz >= lo) & (y.xyz <= hi), 1))
            target = np.full(len(y.xyz), -1, np.int8)
            target[inv[regions[f"idx_{pid}"]]] = 0
            target[inv[np.asarray(entry["points"], np.int64)]] = 1
            t = target[pool]
            if (t == 1).sum() < 20:
                continue
            feat = P.features(y.xyz[pool], y.rgb[pool], vs, sd, np.median(sd, 0), diag)
            item = {"id": pid, "feat": feat, "target": t.astype(np.float32)}
            (self.held if pid in holdout else self.items).append(item)
        self.n_points = n_points
        print(f"real labelled plants: {len(self.items)} for training, {len(self.held)} held out",
              flush=True)

    def sample(self, rng):
        it = self.items[rng.integers(len(self.items))]
        n = len(it["feat"])
        # favour points with a known label so each sample carries signal
        known = np.flatnonzero(it["target"] >= 0)
        k = min(len(known), self.n_points * 3 // 4)
        idx = np.concatenate([rng.choice(known, k, replace=False),
                              rng.choice(n, self.n_points - k, replace=n < self.n_points)])
        lab = it["target"][idx]
        veg = np.full(len(idx), -1.0, np.float32)            # vegetation unknown
        return it["feat"][idx], lab, veg, np.zeros(len(idx), np.float32), np.int8(2)
