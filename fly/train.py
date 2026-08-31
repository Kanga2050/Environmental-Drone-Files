import argparse
import glob
import math
import os
import time
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn.functional as F
import sim
from scipy.spatial import cKDTree
from pt import Net, feats, geof, NS, NQ


def patches(seed, n, npt):
    rng = np.random.default_rng(seed)
    p, c, s, q = sim.plot(seed)
    k = np.floor(p / 0.08).astype(np.int64)
    _, f = np.unique(k, axis=0, return_index=True)
    p, c, s, q = p[f], c[f], s[f], q[f]
    tree = cKDTree(p)
    out = []
    tries = 0
    while len(out) < n and tries < 4 * n:
        tries += 1
        R = rng.uniform(2.5, 8)
        ctr = p[rng.integers(len(p)), :2]
        m = np.flatnonzero(np.hypot(p[:, 0] - ctr[0], p[:, 1] - ctr[1]) < R)
        if len(m) < npt // 2:
            continue
        m = rng.choice(m, npt, replace=len(m) < npt)
        gg = geof(p, tree, p[m])
        pp = p[m].copy()
        a = rng.uniform(0, 2 * np.pi)
        t = rng.normal(0, 0.07, 2)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(t[0]), -np.sin(t[0])], [0, np.sin(t[0]), np.cos(t[0])]])
        Ry = np.array([[np.cos(t[1]), 0, np.sin(t[1])], [0, 1, 0], [-np.sin(t[1]), 0, np.cos(t[1])]])
        c0 = np.r_[ctr, 0]
        pp = (pp - c0) @ (Rz @ Rx @ Ry).T * rng.uniform(0.9, 1.1, 3) + c0
        if rng.random() < 0.5:
            pp[:, 0] = 2 * ctr[0] - pp[:, 0]
        out.append((feats(pp, c[m], ctr, R, gg), s[m].astype(np.int64), q[m].astype(np.int64)))
    return out


def shard(a):
    seed, per, npt, d = a
    P = patches(seed, per, npt)
    if P:
        np.savez(f"{d}/{seed}.npz", x=np.stack([p[0] for p in P]).astype(np.float16),
                 s=np.stack([p[1] for p in P]).astype(np.uint8), q=np.stack([p[2] for p in P]).astype(np.uint8))


def make(d, n, per, npt, jobs):
    os.makedirs(d, exist_ok=True)
    with Pool(jobs) as p:
        for i, _ in enumerate(p.imap_unordered(shard, [(1000 + i, per, npt, d) for i in range(n)])):
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{n} scenes", flush=True)


def stack(d):
    fs = sorted(glob.glob(f"{d}/*.npz"))
    z = [np.load(f) for f in fs]
    return (torch.from_numpy(np.concatenate([a["x"] for a in z]).astype(np.float32)),
            torch.from_numpy(np.concatenate([a["s"] for a in z]).astype(np.int64)),
            torch.from_numpy(np.concatenate([a["q"] for a in z]).astype(np.int64)))


def aug(x):
    B = x.shape[0]
    a = torch.rand(B) * 2 * math.pi
    c, s = torch.cos(a)[:, None], torch.sin(a)[:, None]
    x = x.clone()
    u, v = x[..., 0].clone(), x[..., 1].clone()
    x[..., 0], x[..., 1] = c * u - s * v, s * u + c * v
    x[..., 3:6] = (x[..., 3:6] * (1 + 0.1 * torch.randn(B, 1, 3)) + 0.03 * torch.randn(B, 1, 1)).clamp(0, 1)
    x[..., 6] = x[..., 6] + 0.1 * torch.randn(B, 1)
    return x


def wts(counts):
    w = 1 / np.sqrt(np.maximum(counts, 1) / counts.sum())
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def evaluate(m, seeds, npt):
    cs, cq = np.zeros((NS, NS), np.int64), np.zeros((NQ, NQ), np.int64)
    with torch.no_grad():
        for sd in seeds:
            for x, s, q in patches(sd, 6, npt):
                ls, lq = m(torch.from_numpy(x)[None])
                ps, pq = ls[0].argmax(-1).numpy(), lq[0].argmax(-1).numpy()
                v = s < NS
                np.add.at(cs, (s[v], ps[v]), 1)
                np.add.at(cq, (q, pq), 1)
    iou = lambda c: np.diag(c) / np.maximum(c.sum(0) + c.sum(1) - np.diag(c), 1)
    return cs, cq, iou(cs), iou(cq)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--steps", type=int, default=3000)
    a.add_argument("--bs", type=int, default=8)
    a.add_argument("--npt", type=int, default=1024)
    a.add_argument("--lr", type=float, default=2e-3)
    a.add_argument("--data", default="data")
    a.add_argument("--make", type=int, default=0)
    a.add_argument("--per", type=int, default=16)
    a.add_argument("--jobs", type=int, default=4)
    a.add_argument("--out", default="pt.pth")
    a.add_argument("--resume", default=None)
    g = a.parse_args()
    if g.make:
        make(g.data, g.make, g.per, g.npt, g.jobs)
        return
    torch.manual_seed(0)
    X, S, Q = stack(g.data)
    print(f"{len(X)} patches", flush=True)
    m = Net()
    if g.resume:
        m.load_state_dict(torch.load(g.resume)["w"])
    opt = torch.optim.AdamW(m.parameters(), lr=g.lr, weight_decay=0.02)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(i, g.steps) / g.steps)))
    ss = S.reshape(-1).numpy()
    ws = wts(np.bincount(ss[ss < NS], minlength=NS).astype(float))
    wq = wts(np.bincount(Q.reshape(-1).numpy(), minlength=NQ).astype(float))
    print(f"{sum(p.numel() for p in m.parameters()):,} params", flush=True)
    t0 = time.time()
    run = []
    for i in range(g.steps):
        j = torch.randint(len(X), (g.bs,))
        x, s, q = aug(X[j]), S[j], Q[j]
        ls, lq = m(x)
        loss = F.cross_entropy(ls.reshape(-1, NS), s.reshape(-1), weight=ws, ignore_index=255) + \
            F.cross_entropy(lq.reshape(-1, NQ), q.reshape(-1), weight=wq)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        run.append(loss.item())
        if (i + 1) % 50 == 0:
            print(f"{i + 1} {np.mean(run[-50:]):.3f} {time.time() - t0:.0f}s", flush=True)
        if (i + 1) % 500 == 0:
            torch.save({"w": m.state_dict(), "cfg": {}}, g.out)
    torch.save({"w": m.state_dict(), "cfg": {}}, g.out)
    m.eval()
    cs, cq, a1, a2 = evaluate(m, range(10, 30), g.npt)
    print("sem iou", dict(zip(sim.SEM, np.round(a1, 3))), "miou", round(float(a1.mean()), 3))
    print("qual iou", dict(zip(sim.QUAL, np.round(a2, 3))), "miou", round(float(a2.mean()), 3))
    print("sem conf\n", cs, "\nqual conf\n", cq)


if __name__ == "__main__":
    main()
