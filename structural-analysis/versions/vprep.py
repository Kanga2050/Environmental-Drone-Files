import gmsh, sys, json, os, numpy as np, pickle, meshio
from scipy.spatial import cKDTree
occ = gmsh.model.occ
def _mesh_one(tags, fragment, mode):
    gmsh.clear()
    occ.importShapes(step); occ.synchronize()
    occ.remove([(3, t) for d, t in gmsh.model.getEntities(3) if t not in tags], recursive=True); occ.synchronize()
    if fragment and len(tags) > 1:
        occ.fragment([(3, tags[0])], [(3, t) for t in tags[1:]]); occ.synchronize()
    if "heal" in mode:
        occ.healShapes(gmsh.model.getEntities(3), tolerance=1e-2, fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True, sewFaces=False, makeSolids=False); occ.synchronize()
    for k, val in [("MeshSizeMax", 2.4 if "fine" in mode else 3.0), ("MeshSizeMin", 0.7 if "fine" in mode else 0.9), ("MeshSizeFromCurvature", 0), ("Algorithm", 1 if "adapt" in mode else 6), ("Algorithm3D", 10 if "hxt" in mode else 1), ("SecondOrderLinear", 1), ("HighOrderOptimize", 0)]:
        gmsh.option.setNumber("Mesh." + k, val)
    gmsh.option.setNumber("General.NumThreads", 4 if "hxt" in mode else 1)
    gmsh.model.mesh.generate(3); gmsh.model.mesh.setOrder(2)
    nt, xyz, _ = gmsh.model.mesh.getNodes()
    X = xyz.reshape(-1, 3); idx = {t: i for i, t in enumerate(nt)}
    et, etags, enodes = gmsh.model.mesh.getElements(3)
    k = list(et).index(11)
    E = np.vectorize(idx.get)(enodes[k].reshape(-1, 10))
    return X, E[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
if sys.argv[1] == "--one":
    import gmsh as _g
    step, tags, frag, mode, out = sys.argv[2], json.loads(sys.argv[3]), bool(int(sys.argv[4])), sys.argv[5], sys.argv[6]
    _g.initialize(); _g.option.setNumber("General.Verbosity", 1)
    occ = _g.model.occ; gmsh = _g
    X, E = _mesh_one(tags, frag, mode)
    np.savez(out, X=X, E=E); _g.finalize(); sys.exit(0)
v = sys.argv[1]
cfg = json.load(open(f"{v}.json"))
os.makedirs(v, exist_ok=True)
os.chdir(v)
step = cfg.get("step", f"../{v}.step")
gmsh.initialize(); gmsh.option.setNumber("General.Verbosity", 1)
occ = gmsh.model.occ
def mesh_one(tags, fragment=False):
    import subprocess
    for mode in ("plain", "fine", "adapt", "heal", "heal_fine"):
        out = f"_m_{'_'.join(map(str, tags))}.npz"
        if os.path.exists(out): os.remove(out)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", step, json.dumps(tags), str(int(fragment)), mode, out], capture_output=True, text=True)
        if os.path.exists(out):
            z = np.load(out); os.remove(out); print("meshed", tags, "mode", mode, flush=True)
            return z["X"], z["E"]
        print("mesh attempt", mode, "failed:", (r.stdout + r.stderr).strip().splitlines()[-1][:100] if (r.stdout + r.stderr).strip() else "", flush=True)
    raise RuntimeError("all mesh attempts failed")
if not os.path.exists("manta_3.0.inp"):
    Xc, Ec = mesh_one([cfg["canopy"][0]])
    Xf, Ef = mesh_one([t for t, m in cfg["fuselage"]], fragment=True)
    X = np.vstack([Xc, Xf]); Ef = Ef + len(Xc)
    meshio.write("manta_3.0.inp", meshio.Mesh(X, [("tetra10", Ec), ("tetra10", Ef)]))
    print("meshed separately", len(Ec), len(Ef), flush=True)
gmsh.finalize()
m = meshio.read("manta_3.0.inp")
X = m.points
blocks = [c.data for c in m.cells if c.type == "tetra10"]
def vol(T):
    x = X[T[:, :4]]
    return np.abs(np.einsum("ij,ij->i", x[:, 1] - x[:, 0], np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))).sum() / 6
if len(blocks) > 2:
    vs = [vol(b) for b in blocks]
    ci = int(np.argmin([abs(x - cfg["canopy"][1]) for x in vs]))
    blocks = [blocks[ci], np.concatenate([b for i, b in enumerate(blocks) if i != ci])]
else:
    vs = [vol(b) for b in blocks]
    if abs(vs[0] - cfg["canopy"][1]) > abs(vs[1] - cfg["canopy"][1]): blocks = blocks[::-1]
E = {"can": blocks[0], "fus": blocks[1]}
print("elements", len(E["can"]), len(E["fus"]), "vol", vol(E["can"]), vol(E["fus"]), flush=True)
off = {"can": 1, "fus": 1 + len(E["can"])}
FC = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
def bfaces(T):
    f = np.concatenate([T[:, list(c)] for c in FC]); s = np.sort(f, 1)
    _, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv.ravel()] == 1
    return np.tile(np.arange(len(T)), 4)[b], np.repeat(np.arange(4), len(T))[b], f[b]
B = {k: bfaces(E[k]) for k in E}
cen = lambda k: X[B[k][2]].mean(1)
tf = cKDTree(X[np.unique(B["fus"][2])]); tc = cKDTree(X[np.unique(B["can"][2])])
dc, _ = tf.query(cen("can")); df, _ = tc.query(cen("fus"))
print("tie faces", (dc < 1.0).sum(), flush=True)
nb = np.unique(B["fus"][2])
mot, mn = [], []
for x, y, z in cfg["motors"]:
    x = min((0, 200), key=lambda a: abs(a - x)); y = min((0, 150, 200), key=lambda a: abs(a - y))
    r = np.hypot(X[nb, 0] - x, X[nb, 1] - y)
    c = nb[(r < 12) & (X[nb, 2] < z + 2) & (X[nb, 2] > z - 12)]
    zt = X[c, 2].max()
    s = nb[(r < 14) & (np.abs(X[nb, 2] - zt) < 0.6)]
    mot.append([x, y, float(zt)]); mn.append(s)
    print("motor", (x, y, round(zt, 2)), len(s), flush=True)
json.dump(mot, open("mot.json", "w"))
W = {k: dict(ok=np.zeros(len(B[k][0]), bool), u=np.zeros((len(B[k][0]), 3)), a=np.zeros(len(B[k][0])), c=cen(k), flip=np.zeros(len(B[k][0]), bool)) for k in B}
pickle.dump(dict(X=X, E=E, off=off, B=B, dc=dc, df=df, wet=None, mn=mn, W=W), open("pre.pkl", "wb"))
comps = cfg["comps"] or json.load(open("../v5.json"))["comps"]
out = {}
for k, (lo, hi) in comps.items():
    lo, hi = np.array(lo), np.array(hi)
    g = np.stack(np.meshgrid(*[np.linspace(lo[i], hi[i], 9) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    on = (np.isclose(g, lo) | np.isclose(g, hi)).any(1)
    out[k] = g[on]
np.savez("comps.npz", **out)
