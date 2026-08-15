import pyvista as pv, numpy as np, sys
pv.OFF_SCREEN = True
S, case, mode = sys.argv[1], sys.argv[2], sys.argv[3]
vmax = float(sys.argv[4]) if len(sys.argv) > 4 else 14
v = pv.read(f"{S}/cfd/{case}/vol_solution.vtu")
v.points *= 1000
v = v.clip_box([-1000, 1200, -1000, 1200, -1500, 1600], invert=False)
v["|V|"] = np.linalg.norm(v["Velocity"], axis=1)
v.set_active_vectors("Velocity")
s = pv.read(f"{S}/cfd/{case}/surface.vtu"); s.points *= 1000; s = s.cell_data_to_point_data()
D = [(0, 0, 21, 61), (200, 0, 21, 61), (0, 200, 271, 63.5), (200, 200, 271, 63.5)]
rng = np.random.default_rng(1)
pts = []
if mode == "props":
    for x, y, z, r in D:
        n = 260
        rr = r * np.sqrt(rng.uniform(0.02, 0.97, n)); a = rng.uniform(0, 2 * np.pi, n)
        for zz in (z + 20, z - 3):
            pts += list(np.c_[x + rr * np.cos(a), y + rr * np.sin(a), np.full(n, zz)])
    a = rng.uniform(-350, 550, (500, 2))
    pts += list(np.c_[a, np.full(500, 420.0)])
    pts += list(np.c_[a[:, 0] * 1.2, np.full(500, -380.0), rng.uniform(-200, 500, 500)])
else:
    d = np.array(list(map(float, mode.split(",")))); d /= np.linalg.norm(d)
    c = np.array([100, 100, 130.0]) - 700 * d
    e1 = np.cross(d, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 0.1 else np.cross(d, [0, 0, 1]) / np.linalg.norm(np.cross(d, [0, 0, 1]))
    e2 = np.cross(d, e1)
    g = rng.uniform(-420, 420, (2400, 2))
    pts = list(c + g[:, :1] * e1 + g[:, 1:] * e2)
    for x, y, z, r in D:
        rr = r * np.sqrt(rng.uniform(0.02, 0.97, 150)); a = rng.uniform(0, 2 * np.pi, 150)
        pts += list(np.c_[x + rr * np.cos(a), y + rr * np.sin(a), np.full(150, z - 3)])
sl = v.streamlines_from_source(pv.PolyData(np.array(pts)), vectors="Velocity", integration_direction="both", max_steps=30000, initial_step_length=0.5, step_unit="cl", max_length=8000)
print(case, sl.n_lines)
for name, view in (("isoA", (1, -1, 0.75)), ("isoB", (-1, -1, 0.75)), ("isoC", (1, 1, 0.75)), ("isoD", (1, -1, -0.6))):
    p = pv.Plotter(off_screen=True, window_size=(1600, 1100)); p.set_background("white")
    p.enable_anti_aliasing("ssaa")
    p.add_mesh(s, scalars="Pressure", cmap="viridis", clim=(-40, 60), smooth_shading=True, show_scalar_bar=False)
    p.add_mesh(sl, scalars="|V|", cmap="turbo", clim=(0, vmax), line_width=1.6, opacity=0.9, scalar_bar_args=dict(title="Airspeed (m/s)", color="black", vertical=True, position_x=0.9, position_y=0.12, height=0.6))
    c = np.array([100, 100, 125.0])
    p.camera.focal_point = c; p.camera.position = c + 820 * np.array(view) / np.linalg.norm(view); p.camera.up = (0, 0, 1)
    p.screenshot(f"{S}/img/flow_{case}_{name}.png")
