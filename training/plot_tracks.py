"""Top view of each run's camera path after similarity alignment to GLOMAP,
coloured by time, with GLOMAP's path underneath.  Writes one PNG per run."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_yard as E


def track(art):
    run = E.load_run(art)
    ref_c2w = np.linalg.inv(np.concatenate(
        [E.REF["w2c"], np.tile([[[0, 0, 0, 1]]], (len(E.REF["w2c"]), 1, 1))], 1))
    ref = dict(zip(E.REF["src_ids"].tolist(), range(len(ref_c2w))))
    pairs = [(i, ref[s]) for i, s in enumerate(run["src"]) if s in ref]
    gi, ri = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    P, G = run["c2w"][gi, :3, 3], ref_c2w[ri, :3, 3]
    s, R, t = E.umeyama(P, G)
    return (s * (R @ P.T)).T + t, G


def main(out_dir, arts):
    os.makedirs(out_dir, exist_ok=True)
    # a common top-down plane: GLOMAP's two main axes of camera motion
    G0 = None
    for art in arts:
        Pa, G = track(art)
        if G0 is None:
            mu = G.mean(0)
            _, _, Vt = np.linalg.svd(G - mu, full_matrices=False)
            G0 = (mu, Vt[:2])
        mu, ax = G0
        g2, p2 = (G - mu) @ ax.T, (Pa - mu) @ ax.T
        fig, a = plt.subplots(figsize=(5, 5), dpi=120)
        a.plot(g2[:, 0], g2[:, 1], color="#999", lw=3, alpha=0.6, label="GLOMAP")
        a.scatter(p2[:, 0], p2[:, 1], c=np.linspace(0, 1, len(p2)), cmap="viridis", s=1.5,
                  label="GCT (colour = time)")
        a.set_aspect("equal")
        a.set_xticks([]), a.set_yticks([])
        name = os.path.basename(os.path.normpath(art))
        a.set_title(name, fontsize=11)
        a.legend(loc="lower left", fontsize=7, markerscale=4)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"track_{name}.png"))
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
