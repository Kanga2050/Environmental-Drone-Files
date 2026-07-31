import numpy as np, scipy.sparse as sp, pickle, json, sys, time, re
from itertools import product
from math import factorial
from scipy.spatial import cKDTree
import pypardiso


def _ps():
    s = pypardiso.PyPardisoSolver(mtype=2)
    pass
    return s


d = pickle.load(open("pre.pkl", "rb"))
X = d["X"]
EC = np.concatenate([d["E"]["can"], d["E"]["fus"]])
NE, NN = len(EC), len(X)
NC = len(d["E"]["can"])
E_, NU, RHO = 2000.0, 0.35, 1.06e-9
lam, mu = E_ * NU / ((1 + NU) * (1 - 2 * NU)), E_ / (2 * (1 + NU))
D = np.zeros((6, 6))
D[:3, :3] = lam
D[[0, 1, 2], [0, 1, 2]] += 2 * mu
D[[3, 4, 5], [3, 4, 5]] = mu
EDGE = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
FACE = {1: (0, 1, 2, 4, 5, 6), 2: (0, 3, 1, 7, 8, 4), 3: (1, 3, 2, 8, 9, 5), 4: (2, 3, 0, 9, 7, 6)}
a_, b_ = 0.5854101966249685, 0.1381966011250105
GP = np.array([[b_, b_, b_, a_], [a_, b_, b_, b_], [b_, a_, b_, b_], [b_, b_, a_, b_]])
dL = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)


def dN(L):
    g = np.zeros((10, 3))
    for i in range(4):
        g[i] = (4 * L[i] - 1) * dL[i]
    for k, (i, j) in enumerate(EDGE):
        g[4 + k] = 4 * (L[i] * dL[j] + L[j] * dL[i])
    return g


DN = np.array([dN(L) for L in GP])


def poly(i):
    p = {}
    if i < 4:
        e = [0] * 4; e[i] = 2; p[tuple(e)] = 2.0
        e = [0] * 4; e[i] = 1; p[tuple(e)] = -1.0
    else:
        a, b = EDGE[i - 4]
        e = [0] * 4; e[a] += 1; e[b] += 1; p[tuple(e)] = 4.0
    return p


def integ(p):
    return sum(c * 6 * np.prod([factorial(x) for x in e]) / factorial(sum(e) + 3) for e, c in p.items())


def mul(p, q):
    r = {}
    for (e1, c1), (e2, c2) in product(p.items(), q.items()):
        e = tuple(x + y for x, y in zip(e1, e2)); r[e] = r.get(e, 0) + c1 * c2
    return r


PL = [poly(i) for i in range(10)]
MREF = np.array([[integ(mul(PL[i], PL[j])) for j in range(10)] for i in range(10)])
NINT = np.array([integ(p) for p in PL])


def geom(ch):
    x = X[EC[ch]]
    J = np.einsum("ia,eib->eab", DN[0], x)
    det = np.linalg.det(J)
    Ji = np.linalg.inv(J)
    return x, J, det / 6.0, Ji


def Bmat(Ji, g):
    G = np.einsum("eab,ib->eia", Ji, g)
    n = len(G)
    B = np.zeros((n, 6, 30))
    B[:, 0, 0::3] = G[:, :, 0]; B[:, 1, 1::3] = G[:, :, 1]; B[:, 2, 2::3] = G[:, :, 2]
    B[:, 3, 0::3] = G[:, :, 1]; B[:, 3, 1::3] = G[:, :, 0]
    B[:, 4, 1::3] = G[:, :, 2]; B[:, 4, 2::3] = G[:, :, 1]
    B[:, 5, 0::3] = G[:, :, 2]; B[:, 5, 2::3] = G[:, :, 0]
    return B


def dofs(ch):
    return (3 * EC[ch][:, :, None] + np.arange(3)).reshape(len(ch), 30)


def assemble(mass=False):
    K = sp.csr_matrix((3 * NN, 3 * NN))
    M = sp.csr_matrix((3 * NN, 3 * NN))
    V = np.zeros(NE)
    for s in range(0, NE, 15000):
        ch = np.arange(s, min(NE, s + 15000))
        x, J, vol, Ji = geom(ch)
        V[ch] = vol
        Ke = np.zeros((len(ch), 30, 30))
        for q in range(4):
            B = Bmat(Ji, DN[q])
            Ke += np.einsum("eki,kl,elj->eij", B, D, B) * (vol / 4)[:, None, None]
        dd = dofs(ch)
        r = np.repeat(dd, 30, 1).ravel(); c = np.tile(dd, (1, 30)).ravel()
        K = K + sp.csr_matrix((Ke.ravel(), (r, c)), shape=K.shape)
        if mass:
            Me = np.kron(MREF, np.eye(3))[None] * (RHO * vol)[:, None, None]
            M = M + sp.csr_matrix((Me.ravel(), (r, c)), shape=K.shape)
    return K, M, V


def tie(kp=2e4, nk=4):
    B = d["B"]
    cn = np.unique(B["can"][2][d["dc"] < 1.0])
    fn = np.unique(B["fus"][2])
    dist, j = cKDTree(X[fn]).query(X[cn], k=nk)
    ok = dist[:, 0] < 1.2
    a, b, dd = cn[ok], fn[j[ok]], dist[ok]
    w = 1 / (dd + 0.1); w /= w.sum(1, keepdims=True)
    r, c, v = [], [], []
    for m in range(nk):
        for k in range(3):
            ia, ib = 3 * a + k, 3 * b[:, m] + k
            kk = kp * w[:, m]
            r += [ia, ib, ia, ib]; c += [ia, ib, ib, ia]; v += [kk, kk, -kk, -kk]
    TIE.update(a=a, b=b, w=w, kp=kp)
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(3 * NN, 3 * NN)), len(a)


TIE = {}
KT, NT = tie()
X0 = X.copy()


def _snap(keep):
    X[:] = X0
    a = TIE["a"][keep]
    X[a] = (X[TIE["b"][keep]] * TIE["w"][keep][..., None]).sum(1)
    for k, (i, j) in enumerate(EDGE):
        X[EC[:, 4 + k]] = 0.5 * (X[EC[:, i]] + X[EC[:, j]])
    x = X[EC[:, :4]]
    v = np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    return v


keep = np.ones(len(TIE["a"]), bool)
for _ in range(20):
    v = _snap(keep)
    bad = np.unique(EC[v <= 1e-9 * 6 * 0 + 1e-6])
    hit = np.isin(TIE["a"], bad) & keep
    if not hit.any():
        break
    keep &= ~hit
SNAPPED = int(keep.sum())
_d = np.linalg.norm(X[TIE["a"]] - (X[TIE["b"]] * TIE["w"][..., None]).sum(1), axis=1)
_ok = _d < 1e-6
_a, _b, _w = TIE["a"][_ok], TIE["b"][_ok], TIE["w"][_ok]
_r, _c, _v = [], [], []
_cf = np.c_[np.ones(len(_a)), -_w]
for k in range(3):
    _dd = np.c_[3 * _a + k, 3 * _b + k]
    for i in range(_dd.shape[1]):
        for j in range(_dd.shape[1]):
            _r.append(_dd[:, i]); _c.append(_dd[:, j]); _v.append(TIE["kp"] * _cf[:, i] * _cf[:, j])
KT = sp.csr_matrix((np.concatenate(_v), (np.concatenate(_r), np.concatenate(_c))), shape=(3 * NN, 3 * NN))
NT = int(_ok.sum())
TIE.update(a=_a, b=_b, w=_w)


def parse(path):
    L = open(path).read().splitlines()
    bc, cl, grav, pr, pm = [], [], None, [], {}
    mode, mset = None, None
    for ln in L:
        if ln.startswith("*"):
            u = ln.upper()
            mode = None
            if u.startswith("*BOUNDARY"): mode = "bc"
            elif u.startswith("*CLOAD"): mode = "cl"
            elif u.startswith("*DLOAD"): mode = "dl"
            elif u.startswith("*ELEMENT, TYPE=MASS"): mode = "me"; mset = u.split("ELSET=")[1]; pm[mset] = [[], 0]
            elif u.startswith("*MASS"): mode = "mv"; mset = u.split("ELSET=")[1]
            elif u.startswith("*FREQUENCY"): mode = "fq"
            continue
        if mode is None:
            continue
        t = [s.strip() for s in ln.split(",")]
        if mode == "bc":
            for k in range(int(t[1]), int(t[2]) + 1): bc.append(3 * (int(t[0]) - 1) + k - 1)
        elif mode == "cl":
            cl.append((3 * (int(t[0]) - 1) + int(t[1]) - 1, float(t[2])))
        elif mode == "dl":
            if t[1] == "GRAV": grav = float(t[2]) * np.array(list(map(float, t[3:6])))
            else: pr.append((int(t[0]) - 1, int(t[1][1:]), float(t[2])))
        elif mode == "me": pm[mset][0].append(int(t[1]) - 1)
        elif mode == "mv": pm[mset][1] = float(t[0])
        elif mode == "fq": nm = int(t[0])
    return dict(bc=np.array(bc, int), cl=cl, grav=grav, pr=pr, pm=pm)


def loads(p, V):
    F = np.zeros(3 * NN)
    for i, v in p["cl"]: F[i] += v
    if p["grav"] is not None:
        w = np.outer(V, NINT)
        for k in range(3):
            np.add.at(F, 3 * EC + k, RHO * w * p["grav"][k])
    if p["pr"]:
        e = np.array([q[0] for q in p["pr"]]); f = np.array([q[1] for q in p["pr"]]); pv = np.array([q[2] for q in p["pr"]])
        nodes = np.array([EC[ei][list(FACE[fi])] for ei, fi in zip(e, f)])
        x = X[nodes[:, :3]]
        n = 0.5 * np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0])
        cen = X[EC[e][:, :4]].mean(1)
        s = np.sign(np.einsum("ij,ij->i", n, x.mean(1) - cen))
        n *= s[:, None]
        fm = -(pv[:, None] * n) / 3
        for k in range(3):
            np.add.at(F, 3 * nodes[:, 3:] + k, np.repeat(fm[:, k:k + 1], 3, 1))
    return F


def stress(U):
    S = np.zeros((NE, 4, 6))
    for s in range(0, NE, 15000):
        ch = np.arange(s, min(NE, s + 15000))
        _, _, _, Ji = geom(ch)
        ue = U[dofs(ch)]
        for q in range(4):
            S[ch, q] = np.einsum("kl,elj,ej->ek", D, Bmat(Ji, DN[q]), ue)
    return S


def vm(S):
    sx, sy, sz, txy, tyz, tzx = np.moveaxis(S, -1, 0)
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))


def princ(S):
    T = np.zeros(S.shape[:-1] + (3, 3))
    T[..., 0, 0], T[..., 1, 1], T[..., 2, 2] = S[..., 0], S[..., 1], S[..., 2]
    T[..., 0, 1] = T[..., 1, 0] = S[..., 3]
    T[..., 1, 2] = T[..., 2, 1] = S[..., 4]
    T[..., 0, 2] = T[..., 2, 0] = S[..., 5]
    return np.linalg.eigvalsh(T)


def nodal(v):
    acc = np.zeros(NN); cnt = np.zeros(NN)
    ext = np.linalg.inv(GP)
    vc = v @ ext.T
    np.add.at(acc, EC[:, :4], vc); np.add.at(cnt, EC[:, :4], 1)
    out = acc / np.maximum(cnt, 1)
    for k, (i, j) in enumerate(EDGE):
        m = EC[:, 4 + k]
        out[m] = 0.5 * (out[EC[:, i]] + out[EC[:, j]])
    return out


def save(name, U, fields):
    import meshio
    pd = {"U": U.reshape(-1, 3)}
    pd.update(fields)
    meshio.write(f"{name}/{name}.vtu", meshio.Mesh(X, [("tetra10", EC)], point_data=pd, cell_data={"part": [np.r_[np.zeros(NC), np.ones(NE - NC)]]}))


def static(name):
    t0 = time.time()
    p = parse(f"{name}/{name}.inp")
    K, _, V = assemble()
    Kt, nt = KT, NT
    K = (K + Kt).tocsr()
    F = loads(p, V)
    free = np.setdiff1d(np.arange(3 * NN), p["bc"])
    A = sp.triu(K[free][:, free]).tocsr()
    A.sort_indices()
    sol = _ps()
    U = np.zeros(3 * NN)
    U[free] = sol.solve(A, F[free])
    R = K @ U - F
    S = stress(U)
    np.save(f"{name}/S.npy", S.mean(1).astype(np.float32))
    np.save(f"{name}/U.npy", U.astype(np.float32))
    v = vm(S); pr = princ(S)
    ve = v.max(1)
    res = dict(case=name, tied_nodes=int(nt), F_sum=F.reshape(-1, 3).sum(0).tolist(), reaction=R[p["bc"]].tolist(),
               umax=float(np.linalg.norm(U.reshape(-1, 3), axis=1).max()),
               vm_max=float(ve.max()), vm_999=float(np.percentile(ve, 99.9)), vm_99=float(np.percentile(ve, 99)),
               s1_max=float(pr[..., 2].max()), s3_min=float(pr[..., 0].min()),
               can_vm_max=float(ve[:NC].max()), fus_vm_max=float(ve[NC:].max()),
               loc=X[EC[int(ve.argmax())][:4]].mean(0).tolist(), secs=round(time.time() - t0))
    ex = np.r_[TIE["a"], TIE["b"].ravel(), np.unique(p["bc"] // 3)]
    near = np.zeros(NN, bool)
    for q in cKDTree(X).query_ball_point(X[ex], 2.5): near[q] = True
    xx = X[EC[:, :4]]
    Lm = np.array([np.linalg.norm(xx[:, a] - xx[:, b], axis=1) for a, b in EDGE]).max(0)
    good = V / Lm ** 3 * 6 * np.sqrt(2) > 0.02
    bulk = ~near[EC].any(1) & good
    res.update(vm_good_max=float(ve[good].max()), loc_good=X[EC[np.flatnonzero(good)[ve[good].argmax()]][:4]].mean(0).tolist(), s1_good_max=float(pr[good][..., 2].max()), s3_good_min=float(pr[good][..., 0].min()))
    U3 = U.reshape(-1, 3)
    tf = TIE["kp"] * np.linalg.norm(U3[TIE["a"]] - (U3[TIE["b"]] * TIE["w"][..., None]).sum(1), axis=1)
    tq = X[d["B"]["can"][2][d["dc"] < 1.0]]
    ta = 0.5 * np.linalg.norm(np.cross(tq[:, 1] - tq[:, 0], tq[:, 2] - tq[:, 0]), axis=1).sum()
    res.update(vm_bulk_max=float(ve[bulk].max()), vm_bulk_999=float(np.percentile(ve[bulk], 99.9)), loc_bulk=X[EC[np.flatnonzero(bulk)[ve[bulk].argmax()]][:4]].mean(0).tolist(),
               s1_bulk_max=float(pr[bulk][..., 2].max()), tie_force_max=float(tf.max()), tie_force_sum=float(tf.sum()), tie_area_mm2=float(ta), tie_stress_mean=float(tf.sum() / ta), tie_stress_peak=float(tf.max() * len(tf) / ta))
    np.save(f"{name}/ve.npy", ve)
    ve = np.where(bulk, ve, 0)
    order = np.argsort(-ve)[:2000]
    c = X[EC[order][:, :4]].mean(1)
    hot = []
    for i, cc in zip(order, c):
        if all(np.linalg.norm(cc - h[1]) > 15 for h in hot):
            hot.append((float(ve[i]), cc, int(i < NC)))
        if len(hot) == 8: break
    res["hot"] = [[round(h[0], 2), [round(x, 1) for x in h[1]], "canopy" if h[2] else "fuselage"] for h in hot]
    save(name, U, {"vm": nodal(v), "s1": nodal(pr[..., 2])})
    json.dump(res, open(f"{name}/res.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in res if k != "reaction"}))


def modal(name, nm=20):
    from scipy.sparse.linalg import eigsh, LinearOperator
    p = parse(f"{name}/{name}.inp")
    K, M, V = assemble(True)
    Kt, nt = KT, NT
    K = (K + Kt).tocsr()
    pmv = np.zeros(3 * NN)
    for s, (nodes, m) in p["pm"].items():
        for n in nodes: pmv[3 * n:3 * n + 3] += m
    M = (M + sp.diags(pmv)).tocsr()
    sig = -1e4
    A = sp.triu(K - sig * M).tocsr(); A.sort_indices()
    sol = _ps()
    sol.factorize(A)
    op = LinearOperator(K.shape, matvec=lambda b: sol.solve(A, b), dtype=float)
    w, v = eigsh(K, k=nm + 6, M=M, sigma=sig, OPinv=op, which="LM")
    f = np.sqrt(np.abs(w)) / (2 * np.pi)
    o = np.argsort(f); f, v = f[o], v[:, o]
    c0 = (X * (M.diagonal()[0::3])[:, None]).sum(0) / M.diagonal()[0::3].sum()
    R = np.zeros((3 * NN, 6))
    for k in range(3): R[k::3, k] = 1
    r = X - c0
    for k, ax in enumerate(np.eye(3)):
        R[:, 3 + k] = np.cross(ax, r).ravel()
    MR = M @ R
    G = np.linalg.inv(R.T @ MR)
    rig = [float((v[:, i] @ MR @ G @ MR.T @ v[:, i]) / (v[:, i] @ M @ v[:, i])) for i in range(v.shape[1])]
    ene = []
    for i in range(v.shape[1]):
        q = v[:, i] / np.sqrt(v[:, i] @ M @ v[:, i])
        uu = q.reshape(-1, 3)
        ene.append([float((np.abs(uu[:NC * 0 + np.unique(EC[:NC]).max() + 1]) ** 2).sum())])
    res = dict(case=name, freqs=f.tolist(), rigid_fraction=rig, mass_t=float(M.sum() / 3))
    el = [i for i in range(len(f)) if rig[i] < 0.5]
    fields = {f"m{j}_{f[i]:.0f}Hz": v[:, i].reshape(-1, 3) / np.abs(v[:, i]).max() for j, i in enumerate(el[:12])}
    save(name, np.zeros(3 * NN), fields)
    json.dump(res, open(f"{name}/res.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    (modal if sys.argv[2] == "modal" else static)(sys.argv[1])
