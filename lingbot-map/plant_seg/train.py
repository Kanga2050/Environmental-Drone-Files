"""Train the promptable Point Transformer on synthetic yard scenes.

    python -m plant_seg.train --plants ../datasets/planest_npz \
        --bank ../datasets/yard_bank.npz --out ../datasets/plant_seg_runs/pt1

Torch is only imported inside main(): on Windows the scene workers are
spawned processes that re-import this module, and keeping CUDA out of them
saves several GB of committed memory.
"""
import argparse
import multiprocessing as mp
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--init", default=None, help="start from these weights")
    ap.add_argument("--derived", action="store_true",
                    help="also feed lighting-independent colour and local shape cues")
    ap.add_argument("--uniform_layers", nargs="*", default=[],
                    help="layers to run without attention weights, e.g. enc2 enc3 dec0")
    ap.add_argument("--real_labels", default=None,
                    help="plant_labels.json from plant_label.html: mix hand-labelled real plants in")
    ap.add_argument("--real_art", default="../artifacts_yard_avg")
    ap.add_argument("--real_per_batch", type=int, default=2)
    ap.add_argument("--real_holdout", type=int, nargs="*", default=[],
                    help="labelled plant ids kept out of training for scoring")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/last.pt")
    args = ap.parse_args()

    from .synth import worker_init
    with mp.Pool(args.workers, initializer=worker_init,
                 initargs=(args.plants, args.bank, args.n_points)) as pool:
        from .trainer import run
        run(args, pool)


if __name__ == "__main__":
    main()
