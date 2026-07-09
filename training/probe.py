"""Base model on a few TartanAir clips: confirms conventions and gives the
before-training numbers."""
import sys
import time

import numpy as np
import torch

import common
import tartan

envs = sys.argv[1].split(",") if len(sys.argv) > 1 else None
model = common.build().eval()
model.aggregator.to(torch.bfloat16)
seqs = tartan.sequences(envs)
rng = np.random.default_rng(0)
rows = []
for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 8):
    env, seq, n = seqs[rng.integers(len(seqs))]
    for fpv in (False, True):
        r2 = np.random.default_rng(k)
        ids = list(range(0, min(n, 24 * 4), 4))
        clip = tartan.load_clip(seq, ids, r2, fpv=fpv)
        t = time.time()
        d, c, c2w, K = common.run_stream(model, clip["images"], n_anchor=8)
        m = common.metrics(d, c2w, clip)
        m.update(env=env, fpv=fpv, sec=time.time() - t,
                 fx_pred=float(K[0, 0, 0]), fx_gt=float(clip["K"][0, 0, 0]))
        rows.append(m)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
for fpv in (False, True):
    sel = [r for r in rows if r["fpv"] == fpv]
    print("fpv" if fpv else "clean", {k: round(float(np.mean([r[k] for r in sel])), 3)
                                       for k in ("absrel", "d1", "rot", "rpe_rot", "ate", "ate_dscale")})
print("peak GB", torch.cuda.max_memory_allocated() / 1e9)
