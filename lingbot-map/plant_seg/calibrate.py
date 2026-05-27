"""Pick the probability threshold that maximises IoU on held-out scenes.

    python -m plant_seg.calibrate --weights ../datasets/plant_seg_runs/pt4/best.pt \
        --plants ../datasets/planest_npz --bank ../datasets/yard_bank.npz
"""
import argparse

import numpy as np
import torch

from .segment import load_model
from .synth import SynthScenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--plants", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    model, n_points = load_model(args.weights)
    ds = SynthScenes(args.plants, args.bank, "val", n_points, seed=1234)
    ths = np.round(np.arange(0.2, 0.75, 0.05), 2)
    ious = [[[] for _ in ths] for _ in range(2)]
    for _ in range(args.n):
        f, lab, _, _, src = ds.sample()
        x = torch.from_numpy(f)[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            m, _ = model(x[..., :3], x)
        p = torch.sigmoid(m.float()).cpu().numpy()[0]
        L = lab > 0.5
        for j, t in enumerate(ths):
            P = p > t
            i, u = (P & L).sum(), (P | L).sum()
            ious[int(src)][j].append(i / max(u, 1))
    print("threshold  IoU(all)  IoU(PLANesT targets)  IoU(yard targets)")
    best = None
    for j, t in enumerate(ths):
        a = np.mean(ious[0][j] + ious[1][j])
        print(f"   {t:.2f}     {a:.4f}      {np.mean(ious[0][j]):.4f}              {np.mean(ious[1][j]):.4f}")
        if best is None or a > best[1]:
            best = (t, a)
    print(f"best threshold {best[0]:.2f} (IoU {best[1]:.4f})")


if __name__ == "__main__":
    main()
