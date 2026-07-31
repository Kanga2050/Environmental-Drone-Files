import fe, deck, numpy as np, scipy.sparse as sp, json, sys, os, time
import pypardiso


def _ps():
    s = pypardiso.PyPardisoSolver(mtype=2)
    pass
    return s

from scipy.sparse.linalg import eigsh, LinearOperator

NM = int(os.environ.get("NM", 80))
HOVER_META = os.environ.get('HOVER_META', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cfd/hover_sw/meta.json'))
ZETA = 0.03
CT, RHO_AIR, DP = 0.15, 1.225, 0.127
X, NN, NE = fe.X, fe.NN, fe.NE
os.makedirs("dyn", exist_ok=True)


def model():
    if os.path.exists("dyn/modes.npz"):
        z = np.load("dyn/modes.npz")
        return z["w"], z["phi"], z["mdiag"]
    p = fe.parse("modal/modal.inp")
    K, M, V = fe.assemble(True)
    K = (K + fe.KT).tocsr()
    pmv = np.zeros(3 * NN)
    for s, (nodes, m) in p["pm"].items():
        for n in nodes: pmv[3 * n:3 * n + 3] += m
    M = (M + sp.diags(pmv)).tocsr()
    sig = -1e4
    A = sp.triu(K - sig * M).tocsr(); A.sort_indices()
    sol = _ps(); sol.factorize(A)
    op = LinearOperator(K.shape, matvec=lambda b: sol.solve(A, b), dtype=float)
    w2, v = eigsh(K, k=NM, M=M, sigma=sig, OPinv=op, which="LM")
    o = np.argsort(w2); w2, v = w2[o], v[:, o]
    v /= np.sqrt(np.einsum("ij,ij->j", v, M @ v))
    w = np.sqrt(np.clip(w2, 0, None)); w[:6] = 0
    sp.save_npz("dyn/M.npz", M); sp.save_npz("dyn/K.npz", K)
    np.savez("dyn/modes.npz", w=w, phi=v.astype(np.float32), mdiag=M.diagonal())
    return w, v.astype(np.float32), M.diagonal()


def modal_stress(phi):
    if os.path.exists("dyn/sig.npy"):
        return np.load("dyn/sig.npy", mmap_mode="r")
    S = np.zeros((phi.shape[1], NE, 6), np.float32)
    for i in range(phi.shape[1]):
        S[i] = fe.stress(phi[:, i].astype(float)).mean(1)
    np.save("dyn/sig.npy", S)
    return S


def vm(s):
    sx, sy, sz, txy, tyz, tzx = np.moveaxis(s, -1, 0)
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy ** 2 + tyz ** 2 + tzx ** 2))


xx = X[fe.EC[:, :4]]
Lm = np.array([np.linalg.norm(xx[:, a] - xx[:, b], axis=1) for a, b in fe.EDGE]).max(0)
Vol = fe.geom(np.arange(NE))[2]
GOOD = Vol / Lm ** 3 * 6 * np.sqrt(2) > 0.02
near = np.zeros(NN, bool)
from scipy.spatial import cKDTree
for q in cKDTree(X).query_ball_point(X[np.r_[fe.TIE["a"], fe.TIE["b"].ravel()]], 2.5): near[q] = True
BULK = GOOD & ~near[fe.EC].any(1)


def rpm_of(T):
    return np.sqrt(max(T, 0) / (CT * RHO_AIR * DP ** 4))


def force_vec(nodes, f):
    F = np.zeros(3 * NN, complex)
    for n in nodes:
        F[3 * n:3 * n + 3] += np.asarray(f) / len(nodes)
    return F


def harmonic(w, phi, S):
    MP = deck.mn
    spin = [1, -1, -1, 1]
    mrot = 0.0046 + 0.012
    fc = deck.cs["fc"]
    ns = np.linspace(40, 560, int(os.environ.get('NSPEED', 261)))
    out = dict(rps=ns.tolist(), fc_g_1P=[], fc_g_3P=[], vm_1P=[], vm_3P=[])
    el = w > 1
    T0 = json.load(open(HOVER_META))["T"] if os.path.exists(HOVER_META) else [3.83] * 4
    for n in ns:
        Om = 2 * np.pi * n
        Tn = CT * RHO_AIR * n ** 2 * DP ** 4
        res = {}
        for tag, om, F in (("1P", Om, [force_vec(MP[k], np.array([1, -1j * spin[k], 0]) * mrot * 6.3e-3 * Om) for k in range(4)]),
                           ("3P", 3 * Om, [force_vec(MP[k], [0, 0, 0.05 * Tn]) for k in range(4)])):
            Ftot = sum(F)
            Q = (phi.T @ Ftot) / (w ** 2 - om ** 2 + 2j * ZETA * w * om)
            Q[~el] = (phi[:, ~el].T @ Ftot) / (-om ** 2)
            U = phi @ Q
            a = om ** 2 * np.abs(U.reshape(-1, 3)[fc]).mean(0)
            Qe = np.where(el, Q, 0)
            sr = np.tensordot(Qe.real, S, 1); si = np.tensordot(Qe.imag, S, 1)
            v = max(vm(np.cos(t) * sr - np.sin(t) * si)[BULK].max() for t in np.linspace(0, np.pi, 6, endpoint=False))
            out[f"fc_g_{tag}"].append(float(np.linalg.norm(a) / 9810))
            out[f"vm_{tag}"].append(float(v))
    out["hover_rps"] = [float(rpm_of(t)) for t in T0]
    out["max_rps"] = float(rpm_of(13.73))
    json.dump(out, open("dyn/harmonic.json", "w"))
    return out


def transient(w, phi, S, name, Fspace, ft, tend, v0=None, contact=None, dt=2e-6, keep=400):
    nm = len(w)
    p = phi.T.astype(float)
    fq = p @ Fspace if Fspace is not None else np.zeros(nm)
    q = np.zeros(nm); qd = np.zeros(nm)
    if v0 is not None:
        qd = p @ (np.asarray(fe_mass_vec) * np.tile(v0, NN))
    if contact is not None:
        cn, kg, dirv, gap0 = contact
        Pc = np.stack([phi[3 * cn + k] for k in range(3)], -1).astype(float)
        Pcd = np.einsum("nmk,k->nm", Pc, dirv)
    g = p @ (fe_mass_vec * np.tile([0, 0, -9810.0], NN)) if contact is not None and contact[2][2] != 0 else np.zeros(nm)
    nt = int(tend / dt)
    stride = max(1, nt // keep)
    hist_t, hist_q, hist_fc = [], [], []
    w2 = w ** 2; c = 2 * ZETA * w
    acc = lambda q, qd, t, fcn: fq * ft(t) + g + fcn - w2 * q - c * qd
    fcn = np.zeros(nm); fcont = 0.0
    a = acc(q, qd, 0, fcn)
    for i in range(nt):
        t = i * dt
        qd_half = qd + 0.5 * dt * a
        q = q + dt * qd_half
        if contact is not None:
            pen = -(Pcd @ q) - gap0
            fn = kg * np.clip(pen, 0, None) / len(cn)
            fcont = fn.sum()
            fcn = Pcd.T @ fn
        a = acc(q, qd_half, t + dt, fcn)
        qd = qd_half + 0.5 * dt * a
        if i % stride == 0:
            hist_t.append(t); hist_q.append(q.copy()); hist_fc.append(fcont)
    hist_q = np.array(hist_q); hist_t = np.array(hist_t)
    el = w > 1
    best = (0, 0)
    for j in range(len(hist_t)):
        v = vm(np.tensordot(np.where(el, hist_q[j], 0).astype(np.float32), S, 1))[BULK].max()
        if v > best[0]: best = (float(v), j)
    j = best[1]
    s = np.tensordot(np.where(el, hist_q[j], 0).astype(np.float32), S, 1)
    vv = vm(s); i = np.flatnonzero(BULK)[vv[BULK].argmax()]
    U = (phi @ np.where(el, hist_q[j], 0).astype(np.float32))
    res = dict(case=name, vm_peak=best[0], t_peak=float(hist_t[j]), loc=X[fe.EC[i, :4]].mean(0).tolist(),
               fc_peak=float(max(hist_fc)) if contact is not None else None,
               t=hist_t.tolist(), f_contact=list(map(float, hist_fc)))
    np.save(f"dyn/{name}_vm.npy", vv.astype(np.float32))
    np.save(f"dyn/{name}_U.npy", U.astype(np.float32))
    json.dump(res, open(f"dyn/{name}.json", "w"))
    print(name, {k: res[k] for k in ("vm_peak", "t_peak", "loc", "fc_peak")}, flush=True)
    return res


if __name__ == "__main__":
    t0 = time.time()
    w, phi, md = model()
    fe_mass_vec = md
    print("modes", np.round(w[:20] / 2 / np.pi, 1).tolist(), round(time.time() - t0), flush=True)
    S = modal_stress(phi)
    print("modal stress", round(time.time() - t0), flush=True)
    todo = sys.argv[1:] or ["harmonic", "drop", "step", "pole"]
    if "harmonic" in todo:
        h = harmonic(w, phi, S)
        print("harmonic", max(h["vm_1P"]), max(h["vm_3P"]), max(h["fc_g_1P"]), max(h["fc_g_3P"]), flush=True)
    z0 = X[:, 2].min()
    if "drop" in todo:
        cn = np.flatnonzero(X[:, 2] < z0 + 3)
        for hdrop in (0.3, 1.0):
            v = np.sqrt(2 * 9810 * hdrop)
            transient(w, phi, S, f"drop{int(hdrop*100)}", None, lambda t: 0, 0.03, v0=[0, 0, -v], contact=(cn, 2000.0, np.array([0, 0, 1.0]), 0.0))
    if "pole" in todo:
        cn = np.array([n for n in deck.mn[0] if X[n, 0] < X[deck.mn[0], 0].min() + 3])
        transient(w, phi, S, "pole3", None, lambda t: 0, 0.03, v0=[-3000.0, 0, 0], contact=(cn, 2000.0, np.array([1.0, 0, 0]), 0.0))
    if "step" in todo:
        m = json.load(open(HOVER_META))["T"] if os.path.exists(HOVER_META) else [3.83] * 4
        dT = [13.73 * (1.0 if k < 2 else 1.07) - t for k, t in enumerate(m)]
        F = sum(force_vec(deck.mn[k], [0, 0, dT[k]]) for k in range(4)).real
        transient(w, phi, S, "step", F, lambda t: min(t / 0.04, 1.0), 0.15, dt=1e-5)
