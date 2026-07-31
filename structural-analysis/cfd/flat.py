import trimesh, numpy as np, shapely
from shapely.ops import unary_union
from shapely.geometry import Polygon
env = trimesh.load("envelope.stl")
d = np.array([0, 200, 250.0]); d /= np.linalg.norm(d)
n = np.cross([1, 0, 0], d); n /= np.linalg.norm(n)
e1 = np.array([1.0, 0, 0]); e2 = d
V2 = np.c_[env.vertices @ e1, env.vertices @ e2]
off = (env.vertices @ n)
print("body thickness along n", off.min().round(1), off.max().round(1))
tris = [Polygon(V2[f]) for f in env.faces]
poly = unary_union([t.buffer(0.05) for t in tris if t.area > 1e-6]).buffer(1.0).buffer(-1.0).simplify(0.3)
if poly.geom_type != "Polygon": poly = max(poly.geoms, key=lambda g: g.area)
poly = Polygon(poly.exterior)
print("planform cm2", round(poly.area / 100, 1))
pl = trimesh.creation.extrude_polygon(poly, 10.0)
pl.apply_translation([0, 0, -5.0])
T = np.eye(4); T[:3, 0] = e1; T[:3, 1] = e2; T[:3, 2] = n
pl.apply_transform(T)
D = [(0, 0, 21.0, 61.0), (200, 0, 21.0, 61.0), (0, 200, 271.0, 63.5), (200, 200, 271.0, 63.5)]
for x, y, z, R in D:
    cyl = trimesh.creation.cylinder(radius=R + 4, height=7 + 10, sections=96)
    cyl.apply_translation([x, y, z + 3.5])
    pl = pl.difference(cyl)
print("watertight", pl.is_watertight, "volume cm3", round(pl.volume / 1000, 1), "area cm2", round(pl.area / 100, 1), pl.bounds.round(0).tolist())
pl = pl.subdivide_to_size(4.5)
pl.export("flat.stl")
print(len(pl.faces))
