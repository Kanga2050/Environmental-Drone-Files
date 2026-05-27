"""Convert PLANesT-3D (Mertoglu et al. 2024) PLYs into small npz files.

Reads each PLY straight out of the dataset zips, voxel-downsamples it to
~max_points and keeps xyz, rgb and the semantic label.

    python -m plant_seg.prep_planest --src ../datasets/planest --out ../datasets/planest_npz
"""
import argparse
import glob
import os
import zipfile

import numpy as np

DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                  ("r", "u1"), ("g", "u1"), ("b", "u1"),
                  ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                  ("conf", "<f4"), ("sem", "<f4"), ("inst", "<f4")])


def read_ply(fh):
    head = b""
    while not head.endswith(b"end_header\n"):
        head += fh.read(1)
    lines = head.decode("latin1").splitlines()
    n = next(int(l.split()[-1]) for l in lines if l.startswith("element vertex"))
    props = [l.split()[-1] for l in lines if l.startswith("property")]
    assert props == ["x", "y", "z", "red", "green", "blue", "nx", "ny", "nz",
                     "scalar_confidence", "scalar_semanticLabel",
                     "scalar_instanceLabel"], props
    return np.frombuffer(fh.read(n * DTYPE.itemsize), dtype=DTYPE, count=n)


def voxel_down(xyz, max_points):
    ext = np.ptp(xyz, axis=0).max()
    vox = ext / 400.0
    for _ in range(12):
        q = np.floor(xyz / vox).astype(np.int64)
        _, first = np.unique(q, axis=0, return_index=True)
        if len(first) <= max_points:
            return first
        vox *= 1.25
    return first[np.random.default_rng(0).choice(len(first), max_points, replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_points", type=int, default=120000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for zp in sorted(glob.glob(os.path.join(args.src, "*.zip"))):
        with zipfile.ZipFile(zp) as z:
            for name in sorted(n for n in z.namelist() if n.endswith(".ply")):
                out = os.path.join(args.out, os.path.basename(name)[:-4] + ".npz")
                if os.path.exists(out):
                    continue
                with z.open(name) as fh:
                    v = read_ply(fh)
                xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
                keep = voxel_down(xyz, args.max_points)
                np.savez_compressed(
                    out, xyz=xyz[keep],
                    rgb=np.stack([v["r"], v["g"], v["b"]], 1)[keep],
                    normal=np.stack([v["nx"], v["ny"], v["nz"]], 1)[keep].astype(np.float32),
                    sem=v["sem"][keep].astype(np.int8),
                    inst=v["inst"][keep].astype(np.int16))
                print(f"{name}: {len(v):,} -> {len(keep):,} pts, extent "
                      f"{np.ptp(xyz, 0).round(3)}, sem {np.unique(v['sem'])}", flush=True)


if __name__ == "__main__":
    main()
