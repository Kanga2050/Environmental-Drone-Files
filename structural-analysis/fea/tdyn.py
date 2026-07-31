import fe, deck, numpy as np, scipy.sparse as sp, pypardiso, json, sys, os, time

case = sys.argv[1]
X, NN = fe.X, fe.NN
os.makedirs("dyn", exist_ok=True)
p = fe.parse("modal/modal.inp")
K, M, V = fe.assemble(True)
K = (K + fe.KT).tocsr()
pmv = np.zeros(3 * NN)
for s_, (nodes, m) in p["pm"].items():
    for n in nodes: pmv[3 * n:3 * n + 3] += m
M = (M + sp.diags(pmv)).tocsr()
z0 = X[:, 2].min()
if case.startswith("drop"):
    h = int(case[4:]) / 100
    cn = np.flatnonzero(X[:, 2] < z0 + 3); ax = 2; v0 = np.array([0, 0, -np.sqrt(2 * 9810 * h * 1000)])
    g = np.array([0, 0, -9810.0])
else:
    v = float(case[4:])
    pad = deck.mn[0]
    cn = np.array([n for n in pad if X[n, 0] < X[pad, 0].min() + 3]); ax = 0; v0 = np.array([-v * 1000, 0, 0])
    g = np.zeros(3)
kc = np.zeros(3 * NN)
BETA_D = 2 * 0.03 / (2 * np.pi * 100)
dt = float(os.environ.get("DT", 2e-5))
nb, gm = 0.25, 0.5
a0c, a1c = 1 / (nb * dt ** 2), gm / (nb * dt)
KCTOT = float(os.environ.get("KC", 2e3))
kc[3 * cn + ax] = KCTOT / len(cn)
base = sp.triu((1 + a1c * BETA_D) * K + a0c * M).tocsr(); base.sort_indices()
import gc; gc.collect()
mats = {}
def solver(on):
    if on not in mats:
        for k in list(mats):
            mats[k][1].free_memory(everything=True); del mats[k]
        A_ = (base + sp.diags(kc)).tocsr() if on else base
        A_.sort_indices()
        s_ = fe._ps(); s_.factorize(A_)
        mats[on] = (A_, s_)
    return mats[on]
t0 = time.time()
on = True
A, sol = solver(on)
print(case, "factorized", round(time.time() - t0), flush=True)
F = M @ np.tile(g, NN)
u = np.zeros(3 * NN); vv = np.tile(v0, NN).astype(float); a = np.tile(g, NN).astype(float)
hist = []
best = (0, None, 0)
tend = float(os.environ.get("TEND", 0.03))
sep = 0
for i in range(int(tend / dt)):
    rhs = F + M @ (a0c * u + vv / (nb * dt) + (1 / (2 * nb) - 1) * a) + BETA_D * (K @ (a1c * u + (gm / nb - 1) * vv + dt * (gm / (2 * nb) - 1) * a))
    un = sol.solve(A, rhs)
    pen = -u[3 * cn + ax].mean() if False else -un[3 * cn + ax].mean()
    if on and pen < 0:
        on = False; A, sol = solver(on); un = sol.solve(A, rhs)
    elif not on and pen > 0:
        on = True; A, sol = solver(on); un = sol.solve(A, rhs)
    an = a0c * (un - u) - vv / (nb * dt) - (1 / (2 * nb) - 1) * a
    vv = vv + dt * ((1 - gm) * a + gm * an)
    u, a = un, an
    fc = float(-(kc[3 * cn + ax] * u[3 * cn + ax]).sum()) if on else 0.0
    hist.append((i * dt + dt, fc))
    if fc > best[0]:
        best = (fc, u.copy(), (i + 1) * dt)

print(case, "steps", len(hist), "peak", best[0], "at", best[2], round(time.time() - t0), flush=True)
Ub = best[1]
S = fe.stress(Ub)
np.save(f"dyn/t_{case}_S.npy", S.mean(1).astype(np.float32))
v = fe.vm(S).max(1)
pr = fe.princ(S)
xx = X[fe.EC[:, :4]]
Lm = np.array([np.linalg.norm(xx[:, i] - xx[:, j], axis=1) for i, j in fe.EDGE]).max(0)
good = V / Lm ** 3 * 6 * np.sqrt(2) > 0.02
from scipy.spatial import cKDTree
near = np.zeros(NN, bool)
for q in cKDTree(X).query_ball_point(X[np.r_[fe.TIE["a"], fe.TIE["b"].ravel(), cn]], 2.5): near[q] = True
bulk = good & ~near[fe.EC].any(1)
i = np.flatnonzero(bulk)[v[bulk].argmax()]
imp = np.trapezoid([h[1] for h in hist], [h[0] for h in hist])
res = dict(case=case, F_peak=best[0], t_peak=best[2], n_peak=best[0] / (M.sum() / 3 * 9810), impulse=float(imp), momentum=float(M.sum() / 3 * np.linalg.norm(v0)),
           vm_bulk=float(v[bulk].max()), loc=X[fe.EC[i, :4]].mean(0).tolist(), s1_bulk=float(pr[bulk][..., 2].max()), vm_good=float(v[good].max()),
           vm_contact=float(v[good & near[fe.EC].any(1)].max()), umax=float(np.linalg.norm(Ub.reshape(-1, 3), axis=1).max()),
           t=[h[0] for h in hist], f=[h[1] for h in hist])
json.dump(res, open(f"dyn/t_{case}.json", "w"))
np.save(f"dyn/t_{case}_vm.npy", v.astype(np.float32))
fe.save_dir = None
import meshio
meshio.write(f"dyn/t_{case}.vtu", meshio.Mesh(X, [("tetra10", fe.EC)], point_data={"U": Ub.reshape(-1, 3), "vm": fe.nodal(fe.vm(S))}))
print(json.dumps({k: res[k] for k in res if k not in ("t", "f")}), flush=True)
