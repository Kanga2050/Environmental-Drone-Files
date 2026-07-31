import gmsh, sys, json, re, numpy as np
v = sys.argv[1]
canV = float(sys.argv[2]) * 1000 if len(sys.argv) > 2 else None
gmsh.initialize(); gmsh.option.setNumber("General.Verbosity", 1)
gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
gmsh.model.occ.importShapes(f"{v}.step"); gmsh.model.occ.synchronize()
ents = []
for d, t in gmsh.model.getEntities(3):
    n = gmsh.model.getEntityName(d, t); m = gmsh.model.occ.getMass(d, t)
    ents.append((t, m, n, gmsh.model.getBoundingBox(d, t)))
KW = {"battery": r"Tattu|Battery|LiPo|1300mAh", "fc": r"FC -|F405|Flight", "vtx": r"VTX|TX1600|TX ULTRA|TX Ultra", "cam": r"Foxeer|Camera", "rx": r"RX -|RX-RP|Radiomaster|ExpressLRS|ELRS"}
comps = {}
for k, pat in KW.items():
    hit = [e for e in ents if re.search(pat, e[2]) and e[1] > 50]
    if k == "fc": hit = [e for e in hit if not re.search(KW["vtx"], e[2])]
    if hit:
        bb = np.array([e[3] for e in hit]); comps[k] = [bb[:, :3].min(0).tolist(), bb[:, 3:].max(0).tolist()]
root = [e for e in ents if (e[2].count("/") <= 1 or "=>[" in e[2] or e[2].endswith("/SOLID")) and 2e4 < e[1] < 1.2e6 and not re.search("|".join(KW.values()), e[2])]
can = min(root, key=lambda e: abs(e[1] - canV)) if canV else min(root, key=lambda e: e[1])
if canV and abs(can[1] - canV) > 0.1 * canV:
    can = (None, 0, "", None)
fus = [e for e in root if e[0] != can[0]]
mot = [e for e in ents if re.search(r"motor|Motor|2306|Velox", e[2]) and e[1] > 200]
off = 0.0
if not mot:
    mot = [e for e in ents if "Propeller" in e[2]]
    off = -21.3
cen = {}
for e in mot:
    c = ((e[3][0] + e[3][3]) / 2, (e[3][1] + e[3][4]) / 2)
    key = (round(c[0] / 50), round(c[1] / 50))
    cen.setdefault(key, []).append((c, e[3][2]))
motors = []
for key, L in cen.items():
    c = np.mean([x[0] for x in L], 0); z = min(x[1] for x in L)
    motors.append([float(c[0]), float(c[1]), float(z + off)])
motors = sorted(motors, key=lambda m: (m[1] > 100, m[0]))
cfg = dict(version=v, canopy=[can[0], can[1]], fuselage=[[e[0], e[1]] for e in fus], comps=comps, motors=motors)
json.dump(cfg, open(f"{v}.json", "w"), indent=1)
print(json.dumps(cfg))
