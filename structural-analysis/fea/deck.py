import pickle, numpy as np, json, os, sys
from scipy.spatial import cKDTree

d = pickle.load(open("pre.pkl", "rb"))
X, E, off, B, W = d["X"], d["E"], d["off"], d["B"], d["W"]
mn = d["mn"]
C = np.load("comps.npz")
G = 9810.0
RHO = 1.06e-9
MOT = [(0, 0, 0.0, 1), (200, 0, 0.0, -1), (0, 200, 250.0, -1), (200, 200, 250.0, 1)]
if os.path.exists("mot.json"):
    MOT = [tuple(m[:3]) + (s,) for m, s in zip(json.load(open("mot.json")), (1, -1, -1, 1))]
COMP = {"battery": 222.0 + 45.0, "fc": 25.0, "vtx": 9.0, "cam": 6.0, "rx": 1.5}
MM = 32.0 + 4.6

fb = np.unique(B["fus"][2])
tf = cKDTree(X[fb])
cs = {}
for k in COMP:
    for tol in (1.5, 2.5, 4.0, 6.0, 9.0):
        idx = tf.query_ball_point(C[k], tol)
        s = np.unique(np.concatenate([np.array(i, int) for i in idx if i] or [np.zeros(0, int)]))
        if len(s) >= 12:
            break
    if len(s) < 12:
        _, s = tf.query(C[k].mean(0), k=12)
    cs[k] = fb[s]


def vol(T):
    x = X[T[:, :4]]
    return np.abs(np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))) / 6


VS = {k: vol(E[k]) for k in E}
mstr = {k: VS[k].sum() * RHO * 1e6 for k in E}
cg_s = {k: (X[E[k][:, :4]].mean(1) * VS[k][:, None]).sum(0) / VS[k].sum() for k in E}
pm = [(m, X[cs[k]].mean(0)) for k, m in COMP.items()] + [(MM, X[mn[i]].mean(0)) for i in range(4)]
MT = sum(mstr.values()) + sum(p[0] for p in pm)
CG = (sum(mstr[k] * cg_s[k] for k in E) + sum(m * c for m, c in pm)) / MT


def info():
    return dict(m_struct_g=mstr, m_total_g=MT, cg=CG.tolist(), contact={k: int(len(v)) for k, v in cs.items()})


def faces_surf(k, mask):
    eid, fid, _ = B[k]
    return [(off[k] + e, f + 1) for e, f in zip(eid[mask], fid[mask])]


def tie():
    return faces_surf("can", d["dc"] < 1.0), faces_surf("fus", d["df"] < 1.0)


def pressure_faces(pfun):
    out = []
    F = np.zeros(3)
    M = np.zeros(3)
    for k in ("can", "fus"):
        ok = W[k]["ok"]
        p = pfun(W[k]["c"][ok])
        eid, fid, _ = B[k]
        u, a, c = W[k]["u"][ok], W[k]["a"][ok], W[k]["c"][ok]
        f = -(p * 1e-6 * a)[:, None] * u
        F += f.sum(0)
        M += np.cross(c - CG, f).sum(0)
        for e, fi, pp in zip(eid[ok], fid[ok], p):
            out.append((off[k] + e, fi + 1, pp * 1e-6))
    return out, F, M


def write(name, T=None, Q=None, pfun=None, fixed=None, accel=None, extra=None, modal=False, nmodes=20):
    T = T if T is not None else [0, 0, 0, 0]
    Q = Q if Q is not None else [0.011 * t for t in T]
    os.makedirs(name, exist_ok=True)
    L = []
    w = L.append
    w("*HEADING\nManta V3 " + name)
    w("*NODE, NSET=NALL")
    for i, x in enumerate(X):
        w(f"{i+1}, {x[0]:.5f}, {x[1]:.5f}, {x[2]:.5f}")
    for k, nm in (("can", "CAN"), ("fus", "FUS")):
        w(f"*ELEMENT, TYPE=C3D10, ELSET={nm}")
        for j, t in enumerate(E[k]):
            w(f"{off[k]+j}, " + ", ".join(str(v + 1) for v in t))
    w("*ELSET, ELSET=EALL\nCAN, FUS")
    w("*MATERIAL, NAME=ABS\n*ELASTIC\n2000., 0.35\n*DENSITY\n1.06E-9")
    w("*SOLID SECTION, ELSET=EALL, MATERIAL=ABS")
    sc, sf = tie()
    w("*SURFACE, NAME=SCAN, TYPE=ELEMENT")
    for e, f in sc:
        w(f"{e}, S{f}")
    w("*SURFACE, NAME=SFUS, TYPE=ELEMENT")
    for e, f in sf:
        w(f"{e}, S{f}")
    w("*TIE, NAME=CT, POSITION TOLERANCE=1.2, ADJUST=NO\nSCAN, SFUS")
    if modal:
        nid = len(X) + 1
        eid0 = len(E["can"]) + len(E["fus"]) + 1
        ms = [(m, cs[k]) for k, m in COMP.items()] + [(MM, mn[i]) for i in range(4)]
        for j, (m, s) in enumerate(ms):
            w(f"*ELEMENT, TYPE=MASS, ELSET=PM{j}")
            for n in s:
                w(f"{eid0}, {n+1}")
                eid0 += 1
            w(f"*MASS, ELSET=PM{j}\n{m*1e-6/len(s):.6E}")
        w("*STEP\n*FREQUENCY, SOLVER=SPOOLES\n" + str(nmodes))
        w("*NODE FILE\nU\n*EL FILE\nS\n*END STEP")
        open(f"{name}/{name}.inp", "w").write("\n".join(L) + "\n")
        return {}
    w("*BOUNDARY")
    if fixed is None:
        a = int(np.argmin(np.linalg.norm(X - CG, axis=1)))
        cand = cs["battery"]
        a = int(cand[np.argmin(np.linalg.norm(X[cand] - X[cand].mean(0), axis=1))])
        b = int(mn[0][0])
        c_ = int(mn[1][0])
        w(f"{a+1}, 1, 3\n{b+1}, 2, 3\n{c_+1}, 3, 3")
    else:
        for n in fixed:
            w(f"{n+1}, 1, 3")
    w("*STEP\n*STATIC, SOLVER=ITERATIVE CHOLESKY")
    Fext = np.zeros(3)
    Mext = np.zeros(3)
    cl = {}

    def add(n, v):
        cl.setdefault(n, np.zeros(3))
        cl[n] += v

    for i, (x, y, z, s) in enumerate(MOT):
        P = X[mn[i]]
        f = np.array([0, 0, T[i]]) / len(P)
        r = P[:, :2] - np.array([x, y])
        r2 = (r ** 2).sum()
        for n, p, rr in zip(mn[i], P, r):
            ft = -s * Q[i] * 1000 * np.array([-rr[1], rr[0], 0]) / r2
            add(n, f + ft)
            Fext += f + ft
            Mext += np.cross(p - CG, f + ft)
    if extra:
        for n, v in extra:
            add(n, np.array(v))
            Fext += np.array(v)
            Mext += np.cross(X[n] - CG, np.array(v))
    dl = []
    Fa = np.zeros(3)
    Ma = np.zeros(3)
    if pfun is not None:
        dl, Fa, Ma = pressure_faces(pfun)
    gvec = np.array([0, 0, -G])
    if accel is None:
        a_lin = (Fext + Fa) / (MT * 1e-6) + gvec
    else:
        a_lin = np.array(accel, float)
    geff = gvec - a_lin
    for m, s in [(m, cs[k]) for k, m in COMP.items()] + [(MM, mn[i]) for i in range(4)]:
        for n in s:
            add(n, m * 1e-6 * geff / len(s))
    w("*CLOAD")
    for n, v in cl.items():
        for j in range(3):
            if abs(v[j]) > 1e-12:
                w(f"{n+1}, {j+1}, {v[j]:.6E}")
    g = np.linalg.norm(geff)
    if g > 0:
        u = geff / g
        w(f"*DLOAD\nEALL, GRAV, {g:.4f}, {u[0]:.6f}, {u[1]:.6f}, {u[2]:.6f}")
    if dl:
        w("*DLOAD")
        for e, f, p in dl:
            w(f"{e}, P{f}, {p:.6E}")
    w("*NODE FILE\nU, RF\n*EL FILE\nS, E\n*NODE PRINT, NSET=NALL, TOTALS=ONLY\nRF\n*END STEP")
    open(f"{name}/{name}.inp", "w").write("\n".join(L) + "\n")
    meta = dict(T=list(T), Q=list(Q), F_thrust=Fext.tolist(), F_aero=Fa.tolist(), M_aero_Nmm=Ma.tolist(), M_thrust_Nmm=Mext.tolist(), a=a_lin.tolist(), g_eff=geff.tolist(), n_load=geff.tolist(), **info())
    json.dump(meta, open(f"{name}/meta.json", "w"), indent=1, default=float)
    return meta


if __name__ == "__main__":
    print(json.dumps(info(), indent=1, default=float))
