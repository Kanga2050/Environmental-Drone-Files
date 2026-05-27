"""Synthetic training scenes for promptable plant segmentation.

Each sample places a target plant among other plants, packed so canopies
touch, on top of real background taken from the reconstructed yard (ground,
walls, and other vegetation), all in yard units.  Targets and neighbours are
either labelled PLANesT-3D plants or real plants cut out of the yard map and
pasted somewhere else, so where a point came from never tells the network
whether it is the target -- only the prompt does.

Cameras then look at the target the way the phone video did, a noisy
detector box is drawn around it in each view, and seeds are lifted with the
same rule detect_plants.py uses.  The label is "belongs to the target".
"""
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

from . import prompt as P

HW = (420, 518)
K_YARD = np.array([[500.0, 0, 259.0], [0, 500.0, 210.0], [0, 0, 1]])
VAL_PLANTS = {"Pepper_10", "Ribes_13", "Ribes_14", "Rose_10"}


def load_plants(root, split):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "*.npz"))):
        name = os.path.basename(f)[:-4]
        if (name in VAL_PLANTS) != (split == "val"):
            continue
        d = np.load(f)
        xyz = d["xyz"].astype(np.float32)
        xyz -= np.array([*np.median(xyz[:, :2], 0), xyz[:, 2].min()], np.float32)
        out.append((name, xyz, d["rgb"]))
    return out


def look_at(cam, target, roll=0.0):
    f = target - cam
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)              # image y points down
    cr, sr = np.cos(roll), np.sin(roll)
    r, d = cr * r + sr * d, -sr * r + cr * d
    Rc = np.stack([r, d, f])
    return np.concatenate([Rc, (-Rc @ cam)[:, None]], 1)


def perturb(E, rng, diag, deg=0.6, trans=0.01):
    """Small random pose error (rotation ~deg degrees, translation ~trans*diag)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rng.normal(0, deg))
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    out = E.copy()
    out[:, :3] = R @ E[:, :3]
    out[:, 3] = R @ E[:, 3] + rng.normal(0, trans * diag, 3)
    return out


def colour_jitter(rgb, rng):
    """Scene-wide colour cast: hue rotation, saturation change and chroma
    noise (the yard video has purple/blue fringing in foliage), so that
    "green" alone is not a reliable cue."""
    c = rgb.astype(np.float32) / 255.0
    grey = c.mean(1, keepdims=True)
    chroma = c - grey
    a = np.radians(rng.normal(0, 18))
    axis = np.ones(3) / np.sqrt(3)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
    chroma = chroma @ R.T * rng.uniform(0.45, 1.3)
    noise = rng.normal(0, rng.uniform(0, 0.05), chroma.shape)
    chroma += noise - noise.mean(1, keepdims=True)
    if rng.random() < 0.3:
        # Magenta/blue fringes on a random subset of points.
        hit = rng.random(len(c)) < rng.uniform(0.05, 0.25)
        chroma[hit] += np.array([0.06, -0.08, 0.06]) * rng.uniform(0.5, 1.5)
    return (np.clip(grey + chroma, 0, 1) * 255).astype(np.uint8)


def voxel_sample(xyz, spacing, rng):
    q = np.floor((xyz + rng.uniform(0, spacing, 3)) / spacing).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return idx


class SynthScenes:

    def __init__(self, plants_root, bank_path, split="train", n_points=8192,
                 seed=None, p_yard=0.4, colour_aug=True):
        self.plants = load_plants(plants_root, split)
        self.all_plants = load_plants(plants_root, "train") + (
            self.plants if split == "val" else [])
        b = np.load(bank_path)
        self.bg_xyz, self.bg_rgb, self.bg_tags = b["xyz"], b["rgb"], b["tags"]
        # Background and yard cut-outs come from one half of the yard for
        # training and the other half for validation.
        mid = np.median(self.bg_xyz[:, 0])
        side = self.bg_xyz[:, 0] < mid if split == "train" else self.bg_xyz[:, 0] >= mid
        self.bg_xyz, self.bg_rgb, self.bg_tags = (self.bg_xyz[side], self.bg_rgb[side],
                                                  self.bg_tags[side])
        off, cen = b["cut_off"], b["cut_center"]
        self.cutouts = [(b["cut_xyz"][off[i]:off[i + 1]], b["cut_rgb"][off[i]:off[i + 1]], cen[i])
                        for i in range(len(cen)) if (cen[i][0] < mid) == (split == "train")]
        self.bg_tree = cKDTree(self.bg_xyz[:, :2])
        self.n_points = n_points
        self.p_yard = p_yard
        self.colour_aug = colour_aug
        self.rng = np.random.default_rng(seed)

    # -- scene ---------------------------------------------------------------

    def _plant(self, pool, diag, spacing, rng):
        """A plant with its base at the origin: xyz, rgb, and for a yard
        cut-out its original position (None for PLANesT)."""
        a = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        if self.cutouts and rng.random() < self.p_yard:
            xyz, rgb, origin = self.cutouts[rng.integers(len(self.cutouts))]
            xyz = xyz - np.array([*np.median(xyz[:, :2], 0), np.percentile(xyz[:, 2], 1)])
            p = (xyz @ R.T) * rng.uniform(0.8, 1.25)
            c = rgb.astype(np.float32)
        else:
            _, xyz, rgb = pool[rng.integers(len(pool))]
            origin = None
            s = rng.uniform(0.85, 1.15, 3)
            s[2] *= rng.uniform(0.6, 1.6)
            p = (xyz @ R.T) * s
            p *= diag / np.linalg.norm(np.ptp(p, 0))
            idx = voxel_sample(p, spacing, rng)
            p = p[idx]
            c = rgb[idx].astype(np.float32)
        # Lighting / white balance shift, so colour alone cannot give it away.
        c = c * rng.uniform(0.55, 1.35) * rng.uniform(0.88, 1.12, 3)
        c += rng.normal(0, 6, c.shape)
        return p.astype(np.float32), np.clip(c, 0, 255).astype(np.uint8), origin

    def _background(self, center, radius, rng, avoid=()):
        """Ground and clutter, plus a flag for points that might be
        vegetation (their vegetation label is left unknown)."""
        if rng.random() < 0.8:
            for _ in range(20):
                c0 = self.bg_xyz[rng.integers(len(self.bg_xyz)), :2]
                # Keep a pasted cut-out's original copy out of the scene.
                if all(np.linalg.norm(c0 - o[:2]) > 2 * radius for o in avoid):
                    break
            idx = np.asarray(self.bg_tree.query_ball_point(c0, radius), np.int64)
            xyz = self.bg_xyz[idx].copy()
            xyz[:, :2] += center[:2] - c0
            near = np.linalg.norm(xyz[:, :2] - center[:2], axis=1) < 0.15 * radius
            gz = np.percentile(xyz[near, 2], 10) if near.sum() > 20 else 0.0
            col = self.bg_rgb[idx].astype(np.float32)
            unk = (self.bg_tags[idx] >= 0) | ((2 * col[:, 1] - col[:, 0] - col[:, 2]) > 8)
            col = col * rng.uniform(0.8, 1.2) * rng.uniform(0.94, 1.06, 3)
            return xyz, np.clip(col, 0, 255).astype(np.uint8), float(gz), unk
        # Synthetic ground (soil, gravel, lawn or paving) with an optional wall.
        n = int(rng.integers(20000, 60000))
        xy = center[:2] + rng.uniform(-radius, radius, (n, 2))
        z = rng.normal(0, 0.003, n) + 0.02 * np.sin(xy[:, 0] * rng.uniform(5, 20))
        base = rng.choice([[110, 90, 70], [140, 140, 135], [80, 120, 60], [170, 160, 150]])
        col = np.asarray(base, np.float32) + rng.normal(0, 18, (n, 3))
        xyz = np.column_stack([xy, z])
        if rng.random() < 0.4:
            m = n // 3
            off = rng.uniform(0.4, 1.0) * radius
            wall = np.column_stack([center[0] + off + rng.normal(0, 0.003, m),
                                    center[1] + rng.uniform(-radius, radius, m),
                                    rng.uniform(0, radius, m)])
            wc = rng.uniform(60, 220, 3) + rng.normal(0, 10, (m, 3))
            xyz, col = np.vstack([xyz, wall]), np.vstack([col, wc])
        col = np.clip(col, 0, 255)
        unk = (2 * col[:, 1] - col[:, 0] - col[:, 2]) > 8
        return xyz.astype(np.float32), col.astype(np.uint8), 0.0, unk

    def scene(self, rng):
        diag = float(np.exp(rng.uniform(np.log(0.1), np.log(0.6))))
        spacing = float(rng.uniform(0.002, 0.0065))
        center = np.zeros(3)

        tp, tc, t_origin = self._plant(self.plants, diag, spacing, rng)
        diag = float(np.linalg.norm(np.ptp(tp, 0)))
        parts, origins = [(tp, tc)], [t_origin]
        r_t = 0.5 * np.ptp(tp[:, :2], 0).max()
        for _ in range(rng.integers(0, 5)):
            d2 = diag * float(np.exp(rng.normal(0, 0.4)))
            p, c, o = self._plant(self.all_plants, d2, spacing, rng)
            r_n = 0.5 * np.ptp(p[:, :2], 0).max()
            a = rng.uniform(0, 2 * np.pi)
            dist = (r_t + r_n) * rng.uniform(0.35, 1.05)
            parts.append((p + [dist * np.cos(a), dist * np.sin(a), rng.normal(0, 0.01)], c))
            origins.append(o)
        bg_xyz, bg_rgb, gz, bg_unk = self._background(
            center, 2.2 * diag, rng, [o for o in origins if o is not None])
        parts = [(p + np.float32([0, 0, gz]), c) for p, c in parts]

        # Clear background out of the space the plants occupy.
        plants_xyz = np.vstack([p for p, _ in parts])
        cell = 3 * spacing

        def key(a):
            q = np.floor(a / cell).astype(np.int64) + (1 << 20)
            return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        keep = ~np.isin(key(bg_xyz), np.unique(key(plants_xyz)))
        bg_xyz, bg_rgb, bg_unk = bg_xyz[keep], bg_rgb[keep], bg_unk[keep]
        cap = int(rng.uniform(150000, 450000))
        if len(bg_xyz) > cap:
            sel = rng.choice(len(bg_xyz), cap, replace=False)
            bg_xyz, bg_rgb, bg_unk = bg_xyz[sel], bg_rgb[sel], bg_unk[sel]

        xyz = np.vstack([p for p, _ in parts] + [bg_xyz])
        rgb = np.vstack([c for _, c in parts] + [bg_rgb])
        # Plant index, -1 for background, -2 for background that may be vegetation.
        inst = np.concatenate([np.full(len(p), i, np.int16) for i, (p, _) in enumerate(parts)]
                              + [np.where(bg_unk, -2, -1).astype(np.int16)])
        # Reconstruction noise and floaters, as in the monocular map.
        sig = rng.uniform(0.0, 0.5) * spacing
        xyz = xyz + rng.normal(0, sig, xyz.shape).astype(np.float32)
        nf = int(rng.uniform(0, 0.02) * len(xyz))
        if nf:
            src = rng.choice(len(xyz), nf)
            xyz = np.vstack([xyz, xyz[src] + rng.normal(0, 0.08 * diag, (nf, 3))])
            rgb = np.vstack([rgb, rgb[src]])
            inst = np.concatenate([inst, np.full(nf, -1, np.int16)])
        if self.colour_aug:
            rgb = colour_jitter(rgb, rng)
        return xyz.astype(np.float32), rgb, inst, diag, gz, t_origin is not None

    # -- detections ----------------------------------------------------------

    def views(self, xyz, rgb, inst, diag, gz, rng):
        tgt = inst == 0
        tc = np.median(xyz[tgt], 0)
        n_views = int(np.clip(rng.geometric(0.22), 1, 14))
        az0 = rng.uniform(0, 2 * np.pi)
        spread = np.radians(rng.uniform(5, 200))
        views = []
        for k in range(n_views * 2):
            if len(views) >= n_views:
                break
            az = az0 + spread * rng.random()
            el = np.radians(rng.uniform(-5, 40))
            dist = diag * float(np.exp(rng.uniform(np.log(0.5), np.log(3.5))))
            cam = tc + dist * np.array([np.cos(az) * np.cos(el),
                                        np.sin(az) * np.cos(el), np.sin(el)])
            cam[2] = max(cam[2], gz + 0.05 * diag)
            look = tc + rng.normal(0, 0.3 * diag, 3)
            E = look_at(cam, look, roll=rng.normal(0, 0.05))
            depth = P.zbuffer(xyz, E, K_YARD, HW)
            u, v, z = P.project(xyz[tgt], E, K_YARD)
            iu = np.clip(u, 0, HW[1] - 1).astype(int)
            iv = np.clip(v, 0, HW[0] - 1).astype(int)
            on = (z > 0) & (u >= 0) & (u < HW[1]) & (v >= 0) & (v < HW[0])
            seen = on & (np.abs(z - depth[iv, iu]) <= 0.04 * depth[iv, iu])
            if seen.sum() < 60:
                continue
            x0, x1 = np.percentile(u[seen], [1, 99])
            y0, y1 = np.percentile(v[seen], [1, 99])
            w, h = x1 - x0, y1 - y0
            if rng.random() < 0.15:
                # The detector sometimes boxes a neighbour together with it.
                other = (inst > 0)
                if other.any():
                    uo, vo, zo = P.project(xyz[other], E, K_YARD)
                    ok = (zo > 0) & (uo >= x0 - w) & (uo <= x1 + w) & (vo >= y0 - h) & (vo <= y1 + h)
                    if ok.sum() > 50:
                        x0, x1 = min(x0, np.percentile(uo[ok], 5)), max(x1, np.percentile(uo[ok], 95))
                        y0, y1 = min(y0, np.percentile(vo[ok], 5)), max(y1, np.percentile(vo[ok], 95))
                        w, h = x1 - x0, y1 - y0
            if rng.random() < 0.5:
                # Real detector boxes often frame only part of a plant.
                fw, fh = rng.uniform(0.35, 1.0, 2)
                x0 += rng.uniform(0, 1 - fw) * w
                y0 += rng.uniform(0, 1 - fh) * h
                w, h = fw * w, fh * h
                x1, y1 = x0 + w, y0 + h
            if rng.random() < 0.12:
                # ...and sometimes box most of the frame around the camera.
                x0, y0 = rng.uniform(0, 1) * x0, rng.uniform(0, 1) * y0
                x1 = x1 + rng.uniform(0, 1) * (HW[1] - 1 - x1)
                y1 = y1 + rng.uniform(0, 1) * (HW[0] - 1 - y1)
                w, h = x1 - x0, y1 - y0
            j = rng.uniform(-0.08, 0.12, 4) * np.array([w, h, w, h])
            sh = rng.normal(0, 0.1, 2) * np.array([w, h])
            box = (max(x0 - j[0] + sh[0], 0), max(y0 - j[1] + sh[1], 0),
                   min(x1 + j[2] + sh[0], HW[1] - 1), min(y1 + j[3] + sh[1], HW[0] - 1))
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            # Each frame's pose disagrees slightly with the fused map.
            views.append({"E": perturb(E, rng, diag), "K": K_YARD, "depth": depth,
                          "box": box, "E_true": E})
        if not views:
            return None, None
        # Real seeds are unprojected from each frame's own depth map, which
        # does not sit exactly on the fused cloud: offset each view's seeds.
        mis = rng.uniform(0, 0.15) * diag
        seeds = [xyz[P.lift_seeds(xyz, rgb, dict(vw, E=vw["E_true"]))]
                 + rng.normal(0, mis, 3).astype(np.float32) for vw in views]
        seeds = np.vstack(seeds) if any(len(s) for s in seeds) else np.zeros((0, 3), np.float32)
        return views, seeds

    # -- sample --------------------------------------------------------------

    def sample(self, rng=None):
        """(features, target label, vegetation label (-1 unknown),
        seed-radius baseline mask, target came from the yard)."""
        rng = rng or self.rng
        for _ in range(20):
            xyz, rgb, inst, diag, gz, from_yard = self.scene(rng)
            views, seeds = self.views(xyz, rgb, inst, diag, gz, rng)
            if views is None or len(seeds) < 40:
                continue
            out = build_input(xyz, rgb, views, seeds, self.n_points, rng)
            if out is None:
                continue
            idx, feat = out
            lab = (inst[idx] == 0).astype(np.float32)
            veg = np.where(inst[idx] >= 0, 1.0,
                           np.where(inst[idx] == -2, -1.0, 0.0)).astype(np.float32)
            if lab.sum() < 20:
                continue
            # Baseline: the rule view_map3d used (points within 2.5 voxels
            # of a lifted seed point).
            d, _ = cKDTree(seeds).query(xyz[idx], k=1)
            base = (d <= 0.0125).astype(np.float32)
            return feat, lab, veg, base, np.int8(from_yard)
        raise RuntimeError("could not build a sample")


def build_input(xyz, rgb, views, seeds, n_points, rng, idx_pool=None):
    """Crop around the seeds, subsample to n_points, compute features.
    Returns (indices into xyz, features)."""
    lo, hi, diag = P.crop_region(seeds)
    pool = np.flatnonzero(np.all((xyz >= lo) & (xyz <= hi), 1)) if idx_pool is None else idx_pool
    if len(pool) < 100:
        return None
    if len(pool) > n_points:
        pool = rng.choice(pool, n_points, replace=False)
    else:
        pool = rng.choice(pool, n_points, replace=True)
    center = np.median(seeds, 0)
    feat = P.features(xyz[pool], rgb[pool], views, seeds, center, diag)
    return pool, feat


_WORKER = None


def worker_init(plants_root, bank_path, n_points):
    import os as _os
    global _WORKER
    seed = (_os.getpid() * 7919 + int(np.random.SeedSequence().entropy % 100000))
    _WORKER = SynthScenes(plants_root, bank_path, "train", n_points, seed=seed)


def worker_sample(_):
    import time as _time
    for attempt in range(30):
        try:
            return _WORKER.sample()
        except MemoryError:
            # Other programs can briefly exhaust the commit limit; wait it out.
            _time.sleep(2 + attempt)
    return _WORKER.sample()
