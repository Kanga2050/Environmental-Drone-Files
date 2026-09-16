import colorsys
import numpy as np

SEM = ["ground", "lowveg", "plant", "struct"]
QUAL = ["good", "noisy", "sparse", "double"]


def hsv(h, s, v):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, np.clip(s, 0, 1), np.clip(v, 0, 1)))


def leaf(rng):
    if rng.random() < 0.18:
        return hsv(rng.uniform(0.0, 0.11), rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.6))
    return hsv(rng.uniform(0.18, 0.42), rng.uniform(0.2, 0.7), rng.uniform(0.15, 0.6))


def tint(rng, base, n, j=0.06):
    return np.clip(base + rng.normal(0, j, (n, 3)) * rng.uniform(0.5, 1.5, (n, 1)), 0, 1)


class Field:
    def __init__(self, rng, L):
        self.k = [(rng.uniform(-L, L, 2), rng.uniform(4, L), rng.uniform(-1.5, 1.5)) for _ in range(rng.integers(1, 5))]
        self.s = rng.normal(0, 0.03, 2)

    def __call__(self, x, y):
        z = x * self.s[0] + y * self.s[1]
        for c, w, a in self.k:
            z = z + a * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) / (2 * w * w))
        return z


def blob(rng, n, r, lump):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    f = np.ones(n)
    for _ in range(4):
        k = rng.normal(size=3)
        k /= np.linalg.norm(k)
        f += lump * rng.uniform(0.3, 1) * np.sin(d @ k * rng.uniform(2, 6) + rng.uniform(0, 6.3))
    return d * (r * f * rng.uniform(0.82, 1.0, n))[:, None]


class Scene:
    def __init__(self, rng, L=None, dens=None):
        self.rng = rng
        self.L = L or rng.uniform(12, 30)
        self.d = dens or rng.uniform(120, 400)
        self.g = Field(rng, self.L)
        self.parts = []
        self.foot = []

    def add(self, p, c, s):
        self.parts.append((p.astype(np.float32), c.astype(np.float32), np.full(len(p), s, np.uint8)))

    def area(self, a):
        return max(8, int(a * self.d * self.rng.uniform(0.6, 1.4)))

    def free(self, c, r):
        return all(np.hypot(*(c - f[0])) > r + f[1] for f in self.foot)

    def spot(self, r, tries=30):
        for _ in range(tries):
            c = self.rng.uniform(-self.L + r, self.L - r, 2)
            if self.free(c, r):
                self.foot.append((c, r))
                return c
        return None

    def ground(self):
        rng, L = self.rng, self.L
        n = self.area((2 * L) ** 2)
        x, y = rng.uniform(-L, L, (2, n))
        z = self.g(x, y)
        kind = np.full(n, rng.choice([0, 1, 2], p=[0.5, 0.25, 0.25]))
        for _ in range(rng.integers(1, 6)):
            c = rng.uniform(-L, L, 2)
            w, h = rng.uniform(2, L, 2)
            a = rng.uniform(0, np.pi)
            u = (x - c[0]) * np.cos(a) + (y - c[1]) * np.sin(a)
            v = -(x - c[0]) * np.sin(a) + (y - c[1]) * np.cos(a)
            kind[(np.abs(u) < w / 2) & (np.abs(v) < h / 2)] = rng.integers(0, 4)
        base = {0: hsv(rng.uniform(0.17, 0.3), rng.uniform(0.25, 0.6), rng.uniform(0.3, 0.6)),
                1: hsv(rng.uniform(0.05, 0.12), rng.uniform(0.2, 0.5), rng.uniform(0.25, 0.5)),
                2: hsv(rng.uniform(0, 1), rng.uniform(0, 0.12), rng.uniform(0.45, 0.85)),
                3: hsv(rng.uniform(0.2, 0.38), rng.uniform(0.35, 0.7), rng.uniform(0.2, 0.45))}
        rough = {0: 0.025, 1: 0.02, 2: 0.004, 3: 0.12}
        sem = {0: 1, 1: 0, 2: 0, 3: 1}
        for k in range(4):
            m = kind == k
            if not m.any():
                continue
            zz = z[m] + np.abs(rng.normal(0, rough[k], m.sum())) * (2.5 if k == 3 else 1)
            self.add(np.c_[x[m], y[m], zz], tint(rng, base[k], m.sum(), 0.05 if k != 2 else 0.03), sem[k])

    def shrub(self):
        rng = self.rng
        r = rng.uniform(0.4, 2.2)
        c = self.spot(r)
        if c is None:
            return
        n = self.area(4 * np.pi * r * r)
        p = blob(rng, n, r, rng.uniform(0.05, 0.25)) * np.r_[rng.uniform(0.8, 1.3, 2), rng.uniform(0.5, 1.2)]
        p = p[p[:, 2] > -0.3 * r]
        z0 = self.g(*c)
        p[:, 2] -= p[:, 2].min()
        p += np.r_[c, z0]
        self.add(p, tint(rng, leaf(rng), len(p), 0.08), 2)

    def hedge(self):
        rng, L = self.rng, self.L
        a = rng.uniform(-L, L, 2)
        ang = rng.uniform(0, 2 * np.pi)
        ln = rng.uniform(4, 1.5 * L)
        w, h = rng.uniform(0.5, 1.6), rng.uniform(0.8, 3.0)
        dv = np.r_[np.cos(ang), np.sin(ang)]
        nv = np.r_[-dv[1], dv[0]]
        n = self.area(ln * (w + 2 * h))
        t = rng.uniform(0, ln, n)
        side = rng.random(n)
        top = side < w / (w + 2 * h)
        u = np.where(top, rng.uniform(-w / 2, w / 2, n), np.where(side < (w + h) / (w + 2 * h), -w / 2, w / 2))
        zz = np.where(top, h, rng.uniform(0, h, n))
        bump = 0.12 * np.sin(t * rng.uniform(0.5, 2)) + rng.normal(0, 0.07, n)
        u = u + np.sign(u) * bump * (~top)
        zz = zz + bump * top
        xy = a + t[:, None] * dv + u[:, None] * nv
        z0 = self.g(xy[:, 0], xy[:, 1])
        self.add(np.c_[xy, zz + z0], tint(rng, leaf(rng), n, 0.07), 2)
        for k in range(0, int(ln), 3):
            self.foot.append((a + (k + 1.5) * dv, w / 2 + 0.3))

    def tree(self):
        rng = self.rng
        r = rng.uniform(1.5, 5)
        c = self.spot(r * 0.4)
        if c is None:
            return
        h = rng.uniform(2, 9)
        z0 = self.g(*c)
        tr = rng.uniform(0.08, 0.35)
        n = self.area(2 * np.pi * tr * h)
        th = rng.uniform(0, 2 * np.pi, n)
        p = np.c_[c[0] + tr * np.cos(th), c[1] + tr * np.sin(th), z0 + rng.uniform(0, h, n)]
        self.add(p, tint(rng, hsv(rng.uniform(0.05, 0.1), rng.uniform(0.2, 0.5), rng.uniform(0.15, 0.4)), n), 2)
        n = self.area(4 * np.pi * r * r * 0.7)
        q = blob(rng, n, r, rng.uniform(0.1, 0.3)) * np.r_[1, 1, rng.uniform(0.5, 1.0)]
        q = q[q[:, 2] > -0.4 * r]
        q += np.r_[c, z0 + h]
        self.add(q, tint(rng, leaf(rng), len(q), 0.09), 2)

    def box(self, c, sx, sy, sz, ang, col, roof=None):
        rng = self.rng
        faces = [(sx, sz, 0), (sx, sz, 1), (sy, sz, 2), (sy, sz, 3), (sx, sy, 4)]
        out, oc = [], []
        for a_, b_, k in faces:
            n = self.area(a_ * b_)
            u, v = rng.uniform(-0.5, 0.5, (2, n))
            if k < 2:
                p = np.c_[u * sx, np.full(n, (k - 0.5) * sy), (v + 0.5) * sz]
            elif k < 4:
                p = np.c_[np.full(n, (k - 2.5) * sx), u * sy, (v + 0.5) * sz]
            else:
                p = np.c_[u * sx, v * sy, np.full(n, sz)]
                if roof is not None:
                    p[:, 2] += (0.5 - np.abs(v)) * sy * roof
            out.append(p)
            oc.append(tint(rng, col if k < 4 or roof is None else hsv(rng.uniform(0, 0.08), 0.5, 0.45), n, 0.04))
        p = np.concatenate(out)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        p[:, :2] = p[:, :2] @ R.T + c
        p[:, 2] += self.g(*c)
        self.add(p, np.concatenate(oc), 3)

    def building(self):
        rng = self.rng
        sx, sy = rng.uniform(4, 12, 2)
        c = self.spot(max(sx, sy) / 2)
        if c is None:
            return
        col = hsv(rng.uniform(0, 1), rng.uniform(0, 0.25), rng.uniform(0.5, 0.9))
        self.box(c, sx, sy, rng.uniform(2.5, 7), rng.uniform(0, np.pi), col, rng.uniform(0.2, 0.8) if rng.random() < 0.6 else None)

    def thing(self):
        rng = self.rng
        sx, sy, sz = rng.uniform(0.4, 4.5), rng.uniform(0.4, 2), rng.uniform(0.4, 1.8)
        c = self.spot(max(sx, sy) / 2)
        if c is None:
            return
        self.box(c, sx, sy, sz, rng.uniform(0, np.pi), hsv(rng.uniform(0, 1), rng.uniform(0, 0.8), rng.uniform(0.2, 0.9)))

    def wall(self):
        rng, L = self.rng, self.L
        a = rng.uniform(-L, L, 2)
        ang = rng.uniform(0, 2 * np.pi)
        ln, h = rng.uniform(3, 1.5 * L), rng.uniform(0.6, 2.5)
        dv = np.r_[np.cos(ang), np.sin(ang)]
        nv = np.r_[-dv[1], dv[0]]
        fence = rng.random() < 0.4
        n = self.area(2 * ln * h)
        t, z = rng.uniform(0, ln, n), rng.uniform(0, h, n)
        s = rng.choice([-1, 1], n) * rng.uniform(0.02, 0.15)
        if fence:
            keep = (np.mod(t, rng.uniform(0.15, 0.4)) < rng.uniform(0.06, 0.15)) | (np.abs(z - h * 0.8) < 0.05) | (np.abs(z - h * 0.2) < 0.05)
            t, z, s = t[keep], z[keep], s[keep]
        xy = a + t[:, None] * dv + s[:, None] * nv
        col = hsv(rng.uniform(0, 1), rng.uniform(0, 0.4), rng.uniform(0.3, 0.9))
        self.add(np.c_[xy, z + self.g(xy[:, 0], xy[:, 1])], tint(rng, col, len(t), 0.05), 3)

    def build(self):
        rng = self.rng
        self.ground()
        for f, k in ((self.building, rng.integers(0, 3)), (self.tree, rng.integers(0, 8)), (self.hedge, rng.integers(0, 5)),
                     (self.wall, rng.integers(0, 5)), (self.shrub, rng.integers(2, 25)), (self.thing, rng.integers(0, 6))):
            for _ in range(k):
                f()
        p = np.concatenate([a[0] for a in self.parts])
        c = np.concatenate([a[1] for a in self.parts])
        s = np.concatenate([a[2] for a in self.parts])
        m = (np.abs(p[:, 0]) <= self.L) & (np.abs(p[:, 1]) <= self.L)
        return self.shade(p[m], c[m], s[m])

    def shade(self, p, c, s):
        rng = self.rng
        cell = np.floor(p[:, :2] / 0.5).astype(np.int64)
        _, inv = np.unique(cell[:, 0] * 100003 + cell[:, 1], return_inverse=True)
        top = np.full(inv.max() + 1, -np.inf)
        np.maximum.at(top, inv, p[:, 2])
        under = top[inv] - p[:, 2] > 0.6
        drop = under & (rng.random(len(p)) < 0.6)
        c = c * np.where(under, rng.uniform(0.45, 0.7), 1.0)[:, None]
        p, c, s = p[~drop], c[~drop], s[~drop]
        return p, c, s


def corrupt(rng, p, c, s, L):
    q = np.zeros(len(p), np.uint8)
    p = p + rng.normal(0, np.exp(rng.uniform(np.log(0.005), np.log(0.08))), p.shape)
    extra = []
    for kind in rng.permutation([1, 2, 3, 1, 2, 3, 1, 3])[: rng.integers(2, 8)]:
        ctr = rng.uniform(-L, L, 2)
        r = rng.uniform(2, max(3, L * 0.5))
        m = (np.hypot(p[:, 0] - ctr[0], p[:, 1] - ctr[1]) < r * rng.uniform(0.7, 1.0)) & (q == 0)
        if m.sum() < 50:
            continue
        if kind == 1:
            d = rng.normal(size=3) * np.r_[1, 1, 0.5]
            d /= np.linalg.norm(d)
            sd = rng.uniform(0.15, 0.6)
            k = m.sum()
            p[m] += rng.normal(0, sd, (k, 1)) * d + rng.normal(0, sd * 0.25, (k, 3))
            q[m] = 1
        elif kind == 2:
            q[m] = 2
            keep = rng.random(len(p)) < rng.uniform(0.04, 0.2)
            m2 = m & ~keep
            p, c, s, q = p[~m2], c[~m2], s[~m2], q[~m2]
        else:
            a = rng.uniform(-0.06, 0.06)
            R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            sh = np.r_[rng.uniform(-1.2, 1.2, 2), rng.uniform(-0.3, 0.3)]
            g = p[m]
            g = (g - np.r_[ctr, 0]) @ R.T + np.r_[ctr, 0] + sh
            q[m] = 3
            extra.append((g, c[m] * rng.uniform(0.85, 1.15), s[m], np.full(m.sum(), 3, np.uint8)))
    for g, cc, ss, qq in extra:
        p, c, s, q = np.r_[p, g], np.r_[c, cc], np.r_[s, ss], np.r_[q, qq]
    nf = int(len(p) * rng.uniform(0, 0.01))
    if nf:
        fp = np.c_[rng.uniform(-L, L, (nf, 2)), rng.uniform(p[:, 2].min(), p[:, 2].max() + 4, nf)]
        p = np.r_[p, fp]
        c = np.r_[c, rng.uniform(0.3, 0.9, (nf, 3))]
        s = np.r_[s, np.full(nf, 255, np.uint8)]
        q = np.r_[q, np.full(nf, 1, np.uint8)]
    return p.astype(np.float32), c, s, q


def camera(rng, c):
    c = c * rng.uniform(0.8, 1.2, 3) + rng.uniform(-0.05, 0.05, 3)
    g = c.mean(1, keepdims=True)
    c = g + (c - g) * rng.uniform(0.35, 1.1)
    c = np.clip(c, 0, 1) ** rng.uniform(0.7, 1.4)
    return np.clip(c + rng.normal(0, rng.uniform(0.01, 0.05), c.shape), 0, 1).astype(np.float32)


def plot(seed):
    rng = np.random.default_rng(seed)
    sc = Scene(rng)
    p, c, s = sc.build()
    p, c, s, q = corrupt(rng, p, c, s, sc.L)
    return p, camera(rng, c), s, q
