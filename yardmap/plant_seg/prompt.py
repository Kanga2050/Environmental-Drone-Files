"""Turn a plant's 2D detections into per-point prompt features.

The same code runs on synthetic training scenes and on the real map, so the
network always sees prompts built the same way.  A "view" is a camera
(extrinsic cam-from-world, intrinsic at model resolution), a depth image at
model resolution and the detection box in model pixels.

Coordinates passed in here are already gravity aligned: z is height above
the ground plane.
"""
import numpy as np
from scipy.spatial import cKDTree

N_PROMPT = 5          # in_box, visible, centrality, seed_near, seed_far
N_FEAT = 3 + 1 + 3 + N_PROMPT


def project(xyz, E, K):
    cam = xyz @ E[:3, :3].T + E[:3, 3]
    z = cam[:, 2]
    zs = np.maximum(z, 1e-9)
    u = cam[:, 0] / zs * K[0, 0] + K[0, 2]
    v = cam[:, 1] / zs * K[1, 1] + K[1, 2]
    return u, v, z


def zbuffer(xyz, E, K, hw, down=3):
    """Depth image (hw) of a point set, rendered at 1/down resolution with
    a 3x3 min filter so sparse points close up into surfaces."""
    h, w = hw
    hs, ws = (h + down - 1) // down, (w + down - 1) // down
    u, v, z = project(xyz, E, K)
    ok = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    iu, iv = (u[ok] / down).astype(np.int64), (v[ok] / down).astype(np.int64)
    img = np.full(hs * ws, np.inf, np.float32)
    np.minimum.at(img, iv * ws + iu, z[ok].astype(np.float32))
    img = img.reshape(hs, ws)
    pad = np.pad(img, 1, constant_values=np.inf)
    filled = np.min(np.stack([pad[dy:dy + hs, dx:dx + ws]
                              for dy in range(3) for dx in range(3)]), 0)
    img = np.where(np.isfinite(img), img, filled)
    return np.repeat(np.repeat(img, down, 0), down, 1)[:h, :w]


def view_masks(xyz, view, vis_tol=0.04):
    """in_box, visible (in box and on the observed surface), centrality."""
    E, K, depth, box = view["E"], view["K"], view["depth"], view["box"]
    h, w = depth.shape
    u, v, z = project(xyz, E, K)
    x0, y0, x1, y1 = box
    inb = (z > 1e-6) & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    iu = np.clip(u, 0, w - 1).astype(np.int64)
    iv = np.clip(v, 0, h - 1).astype(np.int64)
    d = depth[iv, iu]
    vis = inb & np.isfinite(d) & (np.abs(z - d) <= vis_tol * np.maximum(d, 1e-6))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw_, hh_ = max(0.5 * (x1 - x0), 1.0), max(0.5 * (y1 - y0), 1.0)
    cen = np.where(inb, 1.0 - np.maximum(np.abs(u - cx) / hw_,
                                         np.abs(v - cy) / hh_), 0.0)
    return inb, vis, cen, z


def lift_seeds(xyz, rgb, view, min_points=40, depth_band=0.25, exg_min=8.0):
    """Mirror of detect_plants.lift_box_to_3d on a point set: visible in-box
    points that look green, kept within the first strong depth mode."""
    inb, vis, _, z = view_masks(xyz, view)
    c = rgb.astype(np.float32)
    keep = vis & ((2 * c[:, 1] - c[:, 0] - c[:, 2]) > exg_min)
    if keep.sum() < min_points:
        return np.zeros(0, np.int64)
    dk = z[keep]
    lo, hi = np.percentile(dk, [2, 98])
    if hi > lo:
        hist, edges = np.histogram(dk, bins=32, range=(lo, hi))
        strong = np.flatnonzero(hist >= 0.15 * hist.max())
        if len(strong):
            mode = 0.5 * (edges[strong[0]] + edges[strong[0] + 1])
            near = keep & (np.abs(z - mode) <= depth_band * mode)
            if near.sum() >= min_points:
                keep = near
    idx = np.flatnonzero(keep)
    p = xyz[idx]
    dist = np.linalg.norm(p - np.median(p, 0), axis=1)
    return idx[dist < np.percentile(dist, 97)]


def crop_region(seeds, pad=0.6, ground_z=None):
    """Axis-aligned search region around the seeds, in gravity frame."""
    lo, hi = np.percentile(seeds, 1, 0), np.percentile(seeds, 99, 0)
    diag = float(np.linalg.norm(hi - lo)) + 1e-6
    lo = lo - pad * diag
    hi = hi + pad * diag
    lo[2] = min(lo[2], 0.0 if ground_z is None else ground_z) - 0.1 * diag
    return lo, hi, diag


def features(xyz, rgb, views, seeds, center, diag):
    """Per-point network input (N, N_FEAT), float32.

    xyz is gravity aligned (z = height above ground)."""
    n = len(xyz)
    inb = np.zeros(n, np.float32)
    vis = np.zeros(n, np.float32)
    cen = np.zeros(n, np.float32)
    for vw in views:
        a, b, c, _ = view_masks(xyz, vw)
        inb += a
        vis += b
        cen = np.maximum(cen, c)
    nv = max(len(views), 1)
    if len(seeds):
        d, _ = cKDTree(seeds).query(xyz, k=1, workers=-1)
        d = d / diag
    else:
        d = np.full(n, 10.0)
    rel = (xyz - center) / diag
    return np.concatenate([
        rel,
        (xyz[:, 2:3] / diag),
        rgb.astype(np.float32) / 255.0,
        (inb / nv)[:, None],
        (vis / nv)[:, None],
        cen[:, None],
        np.exp(-d / 0.02)[:, None],
        np.exp(-d / 0.15)[:, None],
    ], 1).astype(np.float32)
