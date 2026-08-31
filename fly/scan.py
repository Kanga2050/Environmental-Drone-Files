import argparse
import glob
import json
import os
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from pt import load, feats, geof, ref, NS, NQ, NG
from sim import SEM, QUAL


def cams(d):
    C, U = [], []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        E = np.load(f)["extrinsic"].astype(np.float64)
        R, t = E[:, :3], E[:, 3]
        C.append(-R.T @ t)
        U.append(R.T @ np.r_[0, -1, 0])
    return np.array(C), np.array(U)


def ground(p, up, it=400, tol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    s = p[rng.choice(len(p), min(len(p), 200000), replace=False)]
    best = (0, up, s.mean(0))
    tol = tol * np.ptp(s, 0).max()
    for _ in range(it):
        a, b, c = s[rng.choice(len(s), 3, replace=False)]
        n = np.cross(b - a, c - a)
        k = np.linalg.norm(n)
        if k < 1e-12:
            continue
        n = n / k * np.sign(n @ up)
        if n @ up < 0.9:
            continue
        m = (np.abs((s - a) @ n) < tol).sum()
        if m > best[0]:
            best = (m, n, a)
    return best[1], best[2]


def frame(up):
    z = up / np.linalg.norm(up)
    x = np.cross(z, [1, 0, 0] if abs(z[0]) < 0.9 else [0, 1, 0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def level(p, up):
    n, o = ground(p, up)
    R = frame(n)
    q = (p - o) @ R.T
    q[:, 2] -= np.percentile(q[:, 2], 2)
    return q, R, o


def voxel(p, v):
    k = np.floor(p / v).astype(np.int64)
    k -= k.min(0)
    h = (k[:, 0] * (k[:, 1].max() + 1) + k[:, 1]) * (k[:, 2].max() + 1) + k[:, 2]
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return first, inv.ravel()


def health(r):
    return float(np.clip((0.72 - r[6]) / (0.72 - 0.19), 0, 1))


class Scanner:
    def __init__(self, path, dev=None, R=5.0, npt=1024, bs=32, vox=0.08, chunks=4):
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = load(path, self.dev)
        self.R, self.npt, self.bs, self.vox, self.chunks = R, npt, bs, vox, chunks
        self.health = None

    @torch.inference_mode()
    def __call__(self, p, c, seed=0):
        rng = np.random.default_rng(seed)
        first, inv = voxel(p, self.vox)
        q, cc = p[first], c[first]
        t3 = cKDTree(q)
        r = ref(geof(q, t3, q[rng.choice(len(q), min(len(q), 6000), replace=False)]))
        self.health = health(r)
        tree = cKDTree(q[:, :2])
        lo, hi = q[:, :2].min(0), q[:, :2].max(0)
        st = self.R * 1.2
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + st, st), np.arange(lo[1], hi[1] + st, st))
        jobs = []
        for ctr in np.c_[gx.ravel(), gy.ravel()]:
            m = np.array(tree.query_ball_point(ctr, self.R), dtype=np.int64)
            if len(m) < 64:
                continue
            m = rng.permutation(m)
            for k in range(0, min(len(m), self.npt * self.chunks), self.npt):
                ch = m[k:k + self.npt]
                if len(ch) < self.npt:
                    ch = np.r_[ch, rng.choice(m, self.npt - len(ch))]
                jobs.append((ctr, ch))
        used = np.unique(np.concatenate([ch for _, ch in jobs]))
        g = np.zeros((len(q), NG), np.float32)
        g[used] = geof(q, t3, q[used])
        ps, pq, cnt = np.zeros((len(q), NS)), np.zeros((len(q), NQ)), np.zeros(len(q))
        for b in range(0, len(jobs), self.bs):
            J = jobs[b:b + self.bs]
            x = torch.from_numpy(np.stack([feats(q[ch], cc[ch], ctr, self.R, g[ch], r) for ctr, ch in J])).to(self.dev)
            ls, lq = self.m(x)
            ls, lq = ls.softmax(-1).cpu().numpy(), lq.softmax(-1).cpu().numpy()
            for i, (_, ch) in enumerate(J):
                np.add.at(ps, ch, ls[i])
                np.add.at(pq, ch, lq[i])
                np.add.at(cnt, ch, 1)
        seen = cnt > 0
        ps[seen] /= cnt[seen, None]
        pq[seen] /= cnt[seen, None]
        if (~seen).any():
            _, j = cKDTree(q[seen]).query(q[~seen], k=1)
            src = np.flatnonzero(seen)[j]
            ps[~seen], pq[~seen] = ps[src], pq[src]
        return ps[inv], pq[inv], len(jobs)


def cells(p, ps, pq, size, lo=None, hi=None):
    lo = np.floor(p[:, :2].min(0) / size) * size if lo is None else np.asarray(lo, float)
    hi = p[:, :2].max(0) if hi is None else np.asarray(hi, float)
    sh = np.maximum(np.ceil((hi - lo) / size).astype(int), 1)
    ij = np.floor((p[:, :2] - lo) / size).astype(int)
    ok = (ij >= 0).all(1) & (ij < sh).all(1)
    ij, z, ps, pq = ij[ok], p[ok, 2], ps[ok], pq[ok]
    k = ij[:, 0] * sh[1] + ij[:, 1]
    n = np.bincount(k, minlength=sh[0] * sh[1])
    out = {"lo": lo.tolist(), "size": size, "shape": sh.tolist(), "n": n.reshape(sh)}
    for i, s in enumerate(SEM):
        out[s] = (np.bincount(k, ps[:, i], len(n)) / np.maximum(n, 1)).reshape(sh)
    for i, s in enumerate(QUAL):
        out[s] = (np.bincount(k, pq[:, i], len(n)) / np.maximum(n, 1)).reshape(sh)
    out["bad"] = 1 - out["good"]
    o = np.lexsort((z, k))
    ks, zs = k[o], z[o]
    st = np.searchsorted(ks, np.arange(len(n)))
    en = np.searchsorted(ks, np.arange(len(n)), "right")
    g, t = np.full(len(n), np.nan), np.full(len(n), np.nan)
    for i in np.flatnonzero(n):
        g[i] = zs[st[i] + int(0.1 * (en[i] - st[i]))]
        t[i] = zs[st[i] + int(0.98 * (en[i] - st[i] - 1))]
    out["floor"], out["top"] = g.reshape(sh), t.reshape(sh)
    return out


def save(path, cl):
    np.savez(path, **{k: np.asarray(v) for k, v in cl.items()})


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cloud")
    a.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "pt.pth"))
    a.add_argument("--frames", default=None)
    a.add_argument("--span", type=float, default=24.0)
    a.add_argument("--cell", type=float, default=2.0)
    a.add_argument("--plants", default=None)
    a.add_argument("--live", type=int, default=0)
    a.add_argument("--out", default="scan")
    g = a.parse_args()
    d = np.load(g.cloud)
    P, Cc = d["points"].astype(np.float64), d["colors"].astype(np.float32) / 255.0
    ff = d["first_frame"] if "first_frame" in d.files else np.zeros(len(P), int)
    up = np.r_[0, 0, 1.0]
    if g.frames:
        C, U = cams(g.frames)
        up = U.mean(0)
    q, R, o = level(P, up)
    lo, hi = np.percentile(q[:, :2], [1, 99], axis=0)
    mpu = g.span / np.max(hi - lo)
    q *= mpu
    ctr = np.r_[(lo + hi) / 2 * mpu, 0]
    q -= ctr
    print(f"up {np.round(R[2], 3)}  scale {mpu:.2f} m/unit  extent {np.round(np.ptp(q, 0), 1)} m")
    sc = Scanner(g.model)
    steps = [1.0] if not g.live else list(np.linspace(1 / g.live, 1, g.live))
    res = []
    for f in steps:
        m = ff <= f * ff.max()
        t = time.time()
        ps, pq, nj = sc(q[m], Cc[m])
        dt = time.time() - t
        cl = cells(q[m], ps, pq, g.cell)
        sem = np.bincount(ps.argmax(1), minlength=NS) / m.sum()
        qu = np.bincount(pq.argmax(1), minlength=NQ) / m.sum()
        r = {"frac": round(float(f), 2), "points": int(m.sum()), "patches": nj, "sec": round(dt, 1), "health": round(sc.health, 2),
             "sem": dict(zip(SEM, np.round(sem, 3).tolist())), "qual": dict(zip(QUAL, np.round(qu, 3).tolist()))}
        print(json.dumps(r))
        res.append(r)
    np.savez(g.out + "_pts.npz", p=q.astype(np.float32), c=Cc, ps=ps.astype(np.float16), pq=pq.astype(np.float16),
             R=R, o=o, mpu=mpu, ctr=ctr)
    save(g.out + "_cells.npz", cl)
    ev = {"live": res}
    if "obs_count" in d.files or "obs_weight" in d.files:
        w = d["obs_weight"]
        bad = 1 - pq[:, 0]
        lowk = w <= np.percentile(w, 20)
        ev["bad_low_support"] = round(float(bad[lowk].mean()), 3)
        ev["bad_high_support"] = round(float(bad[w >= np.percentile(w, 80)].mean()), 3)
        o_ = np.argsort(bad)
        rk = np.empty(len(bad))
        rk[o_] = np.arange(len(bad))
        n1, n0 = lowk.sum(), (~lowk).sum()
        ev["auc_bad_vs_low_support"] = round(float((rk[lowk].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)), 3)
    if g.plants:
        pp = np.load(g.plants)
        pts = np.concatenate([pp[k] for k in pp.files]).astype(np.float64)
        pts = (pts - o) @ R.T
        pts[:, 2] -= np.percentile(((P - o) @ R.T)[:, 2], 2)
        pts = pts * mpu - ctr
        _, j = cKDTree(q).query(pts, k=1)
        veg = ps[j][:, 1] + ps[j][:, 2]
        ev["plant_pts"] = int(len(pts))
        ev["plant_recall"] = round(float((veg > 0.5).mean()), 3)
        ev["plant_as_plant"] = round(float((ps[j].argmax(1) == 2).mean()), 3)
        ev["all_veg"] = round(float(((ps[:, 1] + ps[:, 2]) > 0.5).mean()), 3)
    print(json.dumps({k: v for k, v in ev.items() if k != "live"}))
    json.dump(ev, open(g.out + "_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
