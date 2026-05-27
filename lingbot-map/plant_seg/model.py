"""Point Transformer (Zhao, Jiang, Jia, Torr, Koltun, ICCV 2021), pure PyTorch.

Self-attention generalised to continuous, unordered point sets: each point
attends to its k nearest neighbours with *vector* attention

    y_i = sum_j softmax(gamma(phi(x_i) - psi(x_j) + delta_ij)) * (alpha(x_j) + delta_ij)
    delta_ij = theta(p_i - p_j)

where delta is a learned encoding of the continuous relative position, so
the operator is defined for any spatial arrangement, not a grid.  The U-Net
layout (transition down by farthest point sampling + kNN max-pool,
transition up by inverse-distance interpolation) follows the paper's
segmentation network.  Two heads: the prompted plant's mask, and an
auxiliary "is vegetation" head.

With derived=True the network also computes, per point, colour that does
not depend on lighting (chromaticity, excess green, hue, saturation), how
much colour varies among its neighbours, and whether its neighbourhood is
line-like, flat or scattered (eigenvalues of the local covariance) -- cues
that separate foliage from walls, paving and stems.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(q, p, k, chunk=1024):
    """Indices (B, M, k) of the k nearest points in p (B, N, 3) for q (B, M, 3)."""
    out = []
    for s in range(0, q.shape[1], chunk):
        d = torch.cdist(q[:, s:s + chunk], p)
        out.append(d.topk(k, dim=-1, largest=False).indices)
    return torch.cat(out, 1)


def gather(x, idx):
    """x (B, N, C), idx (B, M, k) -> (B, M, k, C)."""
    B, M, k = idx.shape
    flat = idx.reshape(B, M * k, 1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, flat).reshape(B, M, k, x.shape[-1])


@torch.no_grad()
def _fps_eager(p, m):
    B, N, _ = p.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=p.device)
    dist = torch.full((B, N), float("inf"), device=p.device)
    far = torch.zeros(B, dtype=torch.long, device=p.device)   # fixed start: deterministic
    ar = torch.arange(B, device=p.device)
    for i in range(m):
        idx[:, i] = far
        c = p[ar, far].unsqueeze(1)
        dist = torch.minimum(dist, ((p - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


_FPS_GRAPHS = {}


@torch.no_grad()
def fps(p, m):
    """Farthest point sampling, batched. p (B, N, 3) -> (B, m) indices.

    The loop is thousands of tiny kernels, so on CUDA it is recorded once
    per shape as a CUDA graph and replayed (about 10x faster)."""
    if not p.is_cuda:
        return _fps_eager(p, m)
    key = (tuple(p.shape), m, p.dtype, p.device)
    if key not in _FPS_GRAPHS:
        static_p = p.detach().clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _fps_eager(static_p, m)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = _fps_eager(static_p, m)
        _FPS_GRAPHS[key] = (g, static_p, static_out)
    g, static_p, static_out = _FPS_GRAPHS[key]
    static_p.copy_(p)
    g.replay()
    return static_out.clone()


class PointTransformerLayer(nn.Module):
    """uniform=True drops the attention weights and averages over the
    neighbours instead; trained models learned uniform attention in some
    layers anyway (see plant_seg analysis), so this is a cheap equivalent."""

    def __init__(self, dim, k=16, share=8, uniform=False):
        super().__init__()
        self.k = k
        self.uniform = uniform
        self.v = nn.Linear(dim, dim)
        self.pos = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3), nn.ReLU(),
                                 nn.Linear(3, dim))
        if not uniform:
            self.q = nn.Linear(dim, dim)
            self.kk = nn.Linear(dim, dim)
            self.attn = nn.Sequential(nn.LayerNorm(dim), nn.ReLU(),
                                      nn.Linear(dim, dim // share), nn.LayerNorm(dim // share),
                                      nn.ReLU(), nn.Linear(dim // share, dim // share))
        self.share = share

    def forward(self, p, x, nbr):
        if self.uniform:
            rel = p.unsqueeze(2) - gather(p, nbr)
            return (gather(self.v(x), nbr) + self.pos(rel)).mean(2)
        q = self.q(x)
        k = gather(self.kk(x), nbr)
        v = gather(self.v(x), nbr)
        rel = p.unsqueeze(2) - gather(p, nbr)
        delta = self.pos(rel)
        w = self.attn(q.unsqueeze(2) - k + delta)          # (B, M, k, C/s)
        w = torch.softmax(w, dim=2)
        B, M, K, C = v.shape
        out = ((v + delta).view(B, M, K, self.share, C // self.share)
               * w.unsqueeze(3)).sum(2).reshape(B, M, C)
        return out


class Block(nn.Module):

    def __init__(self, dim, k, uniform=False):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.pt = PointTransformerLayer(dim, k, uniform=uniform)
        self.n2 = nn.LayerNorm(dim)
        self.l3 = nn.Linear(dim, dim)
        self.n3 = nn.LayerNorm(dim)

    def forward(self, p, x, nbr):
        y = F.relu(self.n1(self.l1(x)))
        y = F.relu(self.n2(self.pt(p, y, nbr)))
        y = self.n3(self.l3(y))
        return F.relu(x + y)


class Down(nn.Module):

    def __init__(self, cin, cout, ratio, k):
        super().__init__()
        self.ratio, self.k = ratio, k
        self.mlp = nn.Sequential(nn.Linear(cin + 3, cout), nn.LayerNorm(cout), nn.ReLU())

    def forward(self, p, x):
        m = p.shape[1] // self.ratio
        ids = fps(p, m)
        pc = torch.gather(p, 1, ids.unsqueeze(-1).expand(-1, -1, 3))
        nbr = knn(pc, p, self.k)
        g = torch.cat([gather(p, nbr) - pc.unsqueeze(2), gather(x, nbr)], -1)
        return pc, self.mlp(g).max(2).values


class Up(nn.Module):

    def __init__(self, cfine, ccoarse, cout):
        super().__init__()
        self.coarse = nn.Sequential(nn.Linear(ccoarse, cout), nn.LayerNorm(cout), nn.ReLU())
        self.fine = nn.Sequential(nn.Linear(cfine, cout), nn.LayerNorm(cout), nn.ReLU())

    def forward(self, pf, xf, pc, xc):
        nbr = knn(pf, pc, 3)
        d = (pf.unsqueeze(2) - gather(pc, nbr)).norm(dim=-1).clamp_min(1e-8)
        w = 1.0 / d
        w = w / w.sum(-1, keepdim=True)
        up = (gather(self.coarse(xc), nbr) * w.unsqueeze(-1)).sum(2)
        return up + self.fine(xf)


N_DERIVED = 12
RGB = slice(4, 7)          # where prompt.features puts the colour (0..1)


def sym3_eigvals(A):
    """Eigenvalues of symmetric 3x3 matrices (..., 3, 3), descending, in
    closed form (trigonometric method); batched cuSOLVER rejects batches
    this large."""
    a00, a11, a22 = A[..., 0, 0], A[..., 1, 1], A[..., 2, 2]
    a01, a02, a12 = A[..., 0, 1], A[..., 0, 2], A[..., 1, 2]
    q = (a00 + a11 + a22) / 3
    p1 = a01 ** 2 + a02 ** 2 + a12 ** 2
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1
    p = torch.sqrt(p2 / 6).clamp_min(1e-20)
    b00, b11, b22 = (a00 - q) / p, (a11 - q) / p, (a22 - q) / p
    b01, b02, b12 = a01 / p, a02 / p, a12 / p
    det = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
           + b02 * (b01 * b12 - b11 * b02))
    phi = torch.acos((det / 2).clamp(-1, 1)) / 3
    e1 = q + 2 * p * torch.cos(phi)
    e3 = q + 2 * p * torch.cos(phi + 2 * torch.pi / 3)
    e2 = 3 * q - e1 - e3
    return torch.stack([e1, e2, e3], -1).clamp_min(0)


def derived_features(p, rgb, nbr):
    """Per-point colour and local-shape descriptors, (B, N, N_DERIVED)."""
    with torch.autocast("cuda", enabled=False):
        p, rgb = p.float(), rgb.float()
        tot = rgb.sum(-1, keepdim=True) + 1e-3
        chrom = rgb / tot
        r, g, b = rgb.unbind(-1)
        exg = ((2 * g - r - b) / tot.squeeze(-1)).unsqueeze(-1)
        mx, mn = rgb.max(-1).values, rgb.min(-1).values
        sat = ((mx - mn) / (mx + 1e-3)).unsqueeze(-1)
        hue = torch.atan2(3 ** 0.5 * (g - b), 2 * r - g - b)
        hue = torch.stack([hue.sin(), hue.cos()], -1) * sat
        chrom_std = gather(chrom, nbr).std(2).mean(-1, keepdim=True)
        exg_std = gather(exg, nbr).std(2)
        rel = gather(p, nbr) - p.unsqueeze(2)
        cov = rel.transpose(-1, -2) @ rel / nbr.shape[-1]
        ev = sym3_eigvals(cov)
        l1 = ev[..., :1].clamp_min(1e-12)
        shape = torch.cat([(ev[..., :1] - ev[..., 1:2]) / l1,     # linearity
                           (ev[..., 1:2] - ev[..., 2:3]) / l1,    # planarity
                           ev[..., 2:3] / l1], -1)                 # scattering
        return torch.cat([chrom, exg, sat, hue, chrom_std * 10, exg_std, shape], -1)


class PointTransformerSeg(nn.Module):

    def __init__(self, in_dim, dims=(32, 64, 128, 256, 512), blocks=(1, 2, 2, 2, 1),
                 k=16, ratio=4, derived=False, uniform_layers=()):
        super().__init__()
        self.k = k
        self.derived = derived
        self.uniform_layers = tuple(uniform_layers)     # e.g. ("enc2", "enc3", "dec0")
        stem_in = in_dim + (N_DERIVED if derived else 0)
        self.stem = nn.Sequential(nn.Linear(stem_in, dims[0]), nn.LayerNorm(dims[0]), nn.ReLU())
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, (d, nb) in enumerate(zip(dims, blocks)):
            if i:
                self.down.append(Down(dims[i - 1], d, ratio, k))
            self.enc.append(nn.ModuleList([Block(d, k, f"enc{i}" in self.uniform_layers)
                                           for _ in range(nb)]))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for j, i in enumerate(range(len(dims) - 1, 0, -1)):
            self.up.append(Up(dims[i - 1], dims[i], dims[i - 1]))
            self.dec.append(Block(dims[i - 1], k, f"dec{j}" in self.uniform_layers))
        self.head_mask = nn.Sequential(nn.Linear(dims[0], dims[0]), nn.ReLU(), nn.Linear(dims[0], 1))
        self.head_veg = nn.Sequential(nn.Linear(dims[0], dims[0]), nn.ReLU(), nn.Linear(dims[0], 1))

    def forward(self, p, feat):
        """p (B, N, 3) positions (normalised), feat (B, N, C) -> two (B, N) logits."""
        nbr0 = knn(p, p, self.k)
        if self.derived:
            feat = torch.cat([feat, derived_features(p, feat[..., RGB], nbr0).to(feat.dtype)], -1)
        x = self.stem(feat)
        ps, xs = [], []
        for i, blocks in enumerate(self.enc):
            if i:
                p, x = self.down[i - 1](p, x)
            nbr = nbr0 if i == 0 else knn(p, p, self.k)
            for b in blocks:
                x = b(p, x, nbr)
            ps.append(p)
            xs.append(x)
        for j, (up, blk) in enumerate(zip(self.up, self.dec)):
            i = len(ps) - 2 - j
            x = up(ps[i], xs[i], p, x)
            p = ps[i]
            x = blk(p, x, knn(p, p, self.k))
        return self.head_mask(x).squeeze(-1), self.head_veg(x).squeeze(-1)

    def load_compatible(self, state):
        """Load weights from a checkpoint trained without some stem inputs;
        the new input columns start at zero so behaviour is unchanged."""
        w = state["stem.0.weight"]
        mine = self.stem[0].weight
        if w.shape != mine.shape:
            new = torch.zeros_like(mine)
            new[:, :w.shape[1]] = w
            state = dict(state, **{"stem.0.weight": new})
        # attention parameters of layers now running uniform are dropped
        own = self.state_dict()
        state = {k: v for k, v in state.items() if k in own}
        self.load_state_dict(state)
