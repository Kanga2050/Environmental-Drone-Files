import pyvista as pv, numpy as np, sys, json, os
pv.OFF_SCREEN = True
S = os.path.dirname(os.path.abspath(__file__))
case = sys.argv[1]
vmax = float(sys.argv[2]) if len(sys.argv) > 2 else None
body_stl = sys.argv[3] if len(sys.argv) > 3 else None
C = f"{S}/cfd/{case}"
meta = json.load(open(f"{C}/meta.json"))
u = np.array(meta["u"], float)
V = np.linalg.norm(u)
if V > 1:
    ev = -u / V
    up = np.array([0, -ev[2], ev[1]]); up /= np.linalg.norm(up)
else:
    ev = np.array([0, 1.0, 0]); up = np.array([0, 0, 1.0])
side = np.cross(ev, up)
v = pv.read(f"{C}/vol_solution.vtu")
v.points *= 1000
v = v.clip_box([-1100, 1300, -1100, 1300, -1300, 1500], invert=False)
v["|V|"] = np.linalg.norm(v["Velocity"], axis=1)
v.set_active_vectors("Velocity")
s = pv.read(f"{C}/surface.vtu"); s.points *= 1000; s = s.cell_data_to_point_data()
props = pv.read(f"{S}/cfd/props_fusion.stl")
motors = pv.read(f"{S}/cfd/motors_fusion.stl")
D = [(0, 0, 21, 61), (200, 0, 21, 61), (0, 200, 271, 63.5), (200, 200, 271, 63.5)]
rng = np.random.default_rng(3)
pts = []
c0 = np.array([100, 110, 135.0])
for x, y, z, r in D:
    n = 220
    rr = r * np.sqrt(rng.uniform(0.05, 0.96, n)); a = rng.uniform(0, 2 * np.pi, n)
    pts += list(np.c_[x + rr * np.cos(a), y + rr * np.sin(a), np.full(n, z - 2.5)])
    pts += list(np.c_[x + rr * np.cos(a), y + rr * np.sin(a), np.full(n, z + 9.5)])
if V > 1:
    g = rng.uniform(-330, 330, (1100, 2))
    pts += list(c0 - 650 * ev + g[:, :1] * side + g[:, 1:] * up)
else:
    g = rng.uniform(-380, 580, (700, 2))
    pts += list(np.c_[g, np.full(700, 430.0)])
sl = v.streamlines_from_source(pv.PolyData(np.array(pts)), vectors="Velocity", integration_direction="both", max_steps=40000, initial_step_length=0.5, step_unit="cl", max_length=9000)
vm = vmax or float(np.percentile(sl["|V|"], 99))
lo = 0.0 if V < 1 else max(0.0, 2 * V - vm)
print(case, sl.n_lines, vm)
views = {"nose": 0.78 * ev + 0.38 * up + 0.5 * side, "nose_l": 0.78 * ev + 0.38 * up - 0.5 * side, "top": 0.3 * ev + 0.9 * up + 0.3 * side, "under": 0.6 * ev - 0.45 * up + 0.65 * side}
pc = s["Pressure"]
q = 0.5 * 1.225 * max(V, 10) ** 2
for name, vw in views.items():
    p = pv.Plotter(off_screen=True, window_size=(1800, 1200)); p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.add_mesh(s, scalars="Pressure", cmap="viridis", clim=(-0.6 * q, 0.6 * q), smooth_shading=True, show_scalar_bar=False, specular=0.3)
    p.add_mesh(props, color="#2b2b2b", smooth_shading=True, specular=0.5)
    p.add_mesh(motors, color="#8a8f96", smooth_shading=True, specular=0.6)
    p.add_mesh(sl, scalars="|V|", cmap="turbo", clim=(lo, vm), line_width=1.4, opacity=0.85, scalar_bar_args=dict(title="Airspeed (m/s)", color="black", vertical=True, position_x=0.91, position_y=0.12, height=0.6))
    vw = vw / np.linalg.norm(vw)
    p.camera.focal_point = c0; p.camera.position = c0 + 760 * vw; p.camera.up = up
    p.screenshot(f"{S}/img/flow3_{case}_{name}.png")
