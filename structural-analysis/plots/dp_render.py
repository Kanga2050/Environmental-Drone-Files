import pyvista as pv, numpy as np, json, sys
pv.OFF_SCREEN = True
a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
sa = pv.read(f"cfd/{a}/surface.vtu"); sb = pv.read(f"cfd/{b}/surface.vtu")
sa.points *= 1000
sa = sa.cell_data_to_point_data(); sb = sb.cell_data_to_point_data()
sa["dp"] = sa["Pressure"] - sb["Pressure"]
meta = json.load(open(f"cfd/{a}/meta.json")); u = np.array(meta["u"]); V = np.linalg.norm(u)
ev = -u / V; up = np.array([0, -ev[2], ev[1]]); up /= np.linalg.norm(up); side = np.cross(ev, up)
props = pv.read("cfd/props_fusion.stl"); motors = pv.read("cfd/motors_fusion.stl")
c0 = np.array([100, 110, 135.0])
for name, vw in {"top": 0.25 * ev + 0.9 * up + 0.35 * side, "under": 0.25 * ev - 0.9 * up + 0.35 * side}.items():
    p = pv.Plotter(off_screen=True, window_size=(1600, 1150)); p.set_background("white"); p.enable_anti_aliasing("ssaa")
    p.add_mesh(sa, scalars="dp", cmap="RdBu_r", clim=(-60, 60), smooth_shading=True, scalar_bar_args=dict(title="Pressure change from prop wash (Pa)", color="black", vertical=True, position_x=0.88, position_y=0.12, height=0.65))
    p.add_mesh(props, color="#2b2b2b", opacity=0.6); p.add_mesh(motors, color="#8a8f96", opacity=0.7)
    vw = vw / np.linalg.norm(vw)
    p.camera.focal_point = c0; p.camera.position = c0 + 700 * vw; p.camera.up = ev if name == "top" else ev
    p.add_text("Viewed from above (flight direction up the page)" if name == "top" else "Viewed from below", position="upper_left", font_size=12, color="black")
    p.screenshot(f"img/{out}_{name}.png")
print("dp range", np.percentile(sa["dp"], [1, 50, 99]))
