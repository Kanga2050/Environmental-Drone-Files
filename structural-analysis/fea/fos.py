import numpy as np, pickle, json, os, sys
from scipy.spatial import cKDTree

XT, ZT, S13 = 30.0, 15.0, 11.0
FC = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]


def frames(root="."):
    p = os.path.join(root, "normals.npy")
    if os.path.exists(p):
        return np.load(p)
    d = pickle.load(open(os.path.join(root, "pre.pkl"), "rb"))
    X = d["X"]
    out = []
    for k in ("can", "fus"):
        E = d["E"][k]
        eid, fid, f = d["B"][k]
        c = X[f].mean(1)
        n = np.cross(X[f[:, 1]] - X[f[:, 0]], X[f[:, 2]] - X[f[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
        ec = X[E[:, :4]].mean(1)
        _, j = cKDTree(c).query(ec, k=6)
        nn = n[j]
        ref = nn[:, 0:1]
        nn = nn * np.sign(np.einsum("ijk,ijk->ij", nn, np.repeat(ref, 6, 1)))[..., None]
        m = nn.mean(1)
        out.append(m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-30))
    N = np.concatenate(out)
    np.save(p, N.astype(np.float32))
    return N


def tensor(S):
    T = np.zeros((len(S), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = S[:, 0], S[:, 1], S[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = S[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = S[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = S[:, 5]
    return T


def fi(S, n):
    T = tensor(S)
    tn = np.einsum("eij,ej->ei", T, n)
    s33 = np.einsum("ei,ei->e", tn, n)
    tsh = np.linalg.norm(tn - s33[:, None] * n, axis=1)
    tr = np.trace(T, axis1=1, axis2=2)
    J2 = 0.5 * np.einsum("eij,eij->e", T, T) - tr ** 2 / 6
    vm = np.sqrt(np.maximum(3 * J2, 0))
    Pin = T - np.einsum("ei,ej->eij", tn, n) - np.einsum("ei,ej->eij", n, tn) + s33[:, None, None] * np.einsum("ei,ej->eij", n, n)
    tri = np.trace(Pin, axis1=1, axis2=2)
    vin = np.sqrt(np.maximum(1.5 * np.einsum("eij,eij->e", Pin, Pin) - 0.5 * tri ** 2, 0))
    f_in = (vin / XT) ** 2
    f_out = (np.maximum(s33, 0) / ZT) ** 2 + (tsh / S13) ** 2
    return f_in, f_out, vm


def evaluate(S, root=".", mask=None):
    Ncon = frames(root)
    Nz = np.tile([0, 0, 1.0], (len(S), 1))
    nb = np.array([0, -250, 200.0]) / np.hypot(250, 200)
    res = {}
    for tag, n in (("planar_z", Nz), ("planar_y", np.tile([0, 1.0, 0], (len(S), 1))), ("planar_belly", np.tile(nb, (len(S), 1))), ("conformal", Ncon)):
        a, b, vm = fi(S.astype(float), n)
        F = np.maximum(a, b)
        fos = 1 / np.sqrt(np.maximum(F, 1e-12))
        m = mask if mask is not None else np.ones(len(S), bool)
        i = np.flatnonzero(m)[fos[m].argmin()]
        res[tag] = dict(fos_min=float(fos[m].min()), mode="interlayer" if b[i] > a[i] else "in-layer", elem=int(i), fos_p01=float(np.percentile(fos[m], 0.1)))
        res[tag + "_map"] = fos
    vmv = fi(S.astype(float), Nz)[2]
    m = mask if mask is not None else np.ones(len(S), bool)
    res["iso"] = dict(fos_min=float((XT / np.maximum(vmv, 1e-9))[m].min()))
    return res
