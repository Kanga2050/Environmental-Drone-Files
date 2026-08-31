import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NS, NQ = 4, 4


NG = 8


def geof(full, tree, x):
    _, j = tree.query(x, k=48)
    nb = full[j]
    out = []
    for k in (12, 48):
        q = nb[:, :k]
        d = q - q.mean(1, keepdims=True)
        w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", d, d) / k)
        w = np.maximum(w[:, ::-1], 1e-12)
        s = np.sqrt(w)
        n = v[:, :, 0]
        off = np.abs(np.einsum("nki,ni->nk", x[:, None] - q, n)).mean(1)
        r = np.linalg.norm(q - x[:, None], axis=2).mean(1)
        out.append((r, (s[:, 0] - s[:, 1]) / s[:, 0], (s[:, 1] - s[:, 2]) / s[:, 0], s[:, 2] / s[:, 0], np.abs(n[:, 2]), off / np.maximum(r, 1e-9)))
    (r1, l1, p1, s1, n1, o1), (r2, l2, p2, s2, n2, o2) = out
    return np.c_[r1, l1, p1, s1, n1, o1, s2, o2].astype(np.float32)


def feats(p, c, ctr, R, g):
    z0 = np.percentile(p[:, 2], 2)
    x = (p - np.r_[ctr, z0]) / R
    g = g.copy()
    g[:, 0] = np.log(g[:, 0] / max(np.median(g[:, 0]), 1e-9))
    return np.c_[x, c, g].astype(np.float32)


def knn(x, k, d):
    dist = torch.cdist(x, x)
    idx = dist.topk(k * d, largest=False).indices
    return idx[..., ::d]


def gather(f, idx):
    B, N, K = idx.shape
    return f.gather(1, idx.reshape(B, N * K, 1).expand(-1, -1, f.shape[-1])).reshape(B, N, K, f.shape[-1])


class Block(nn.Module):
    def __init__(self, d, g=8):
        super().__init__()
        self.g = g
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.pe = nn.Linear(3, d)
        self.w = nn.Linear(d, g)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, f, x, idx):
        B, N, K = idx.shape
        q, k, v = self.qkv(self.n1(f)).chunk(3, -1)
        rel = x.unsqueeze(2) - gather(x, idx)
        pe = self.pe(rel)
        a = self.w(F.gelu(q.unsqueeze(2) - gather(k, idx) + pe)).softmax(2)
        vv = (gather(v, idx) + pe).reshape(B, N, K, self.g, -1)
        f = f + self.o((a.unsqueeze(-1) * vv).sum(2).reshape(B, N, -1))
        return f + self.ff(self.n2(f))


class Net(nn.Module):
    def __init__(self, d=64, k=16, dil=(1, 2, 4, 1, 2, 4)):
        super().__init__()
        self.k, self.dil = k, dil
        self.inp = nn.Sequential(nn.Linear(6 + NG, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d) for _ in dil)
        self.glob = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.mix = nn.Linear(2 * d, d)
        self.sem = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, NS))
        self.qual = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, NQ))

    def forward(self, x):
        p = x[..., :3]
        nb = {d: knn(p, self.k, d) for d in set(self.dil)}
        f = self.inp(x)
        h = len(self.blocks) // 2
        for i, b in enumerate(self.blocks):
            f = b(f, p, nb[self.dil[i]])
            if i == h - 1:
                g = self.glob(f).max(1, keepdim=True).values.expand_as(f)
                f = self.mix(torch.cat([f, g], -1))
        return self.sem(f), self.qual(f)


def load(path, dev="cpu"):
    ck = torch.load(path, map_location=dev)
    m = Net(**ck.get("cfg", {}))
    m.load_state_dict(ck["w"])
    return m.eval().to(dev)
