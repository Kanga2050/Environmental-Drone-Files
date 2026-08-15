import pyvista as pv, numpy as np, sys, os, json
pv.OFF_SCREEN = True
S = os.path.dirname(os.path.abspath(__file__))
props = pv.read(f"{S}/cfd/props_fusion.stl")
motors = pv.read(f"{S}/cfd/motors_fusion.stl")
VIEWS = {"front": ((0.62, 0.75, 0.45), (0, 0, 1)), "back": ((-0.55, -0.8, 0.35), (0, 0, 1)), "under": ((0.5, -0.6, -0.65), (0, 0, 1))}
C0 = np.array([100, 110, 130.0])


def surf(vtu, field, warp=None):
    m = pv.read(vtu)
    if warp is not None:
        m = m.warp_by_vector(warp[0], factor=warp[1])
    s = m.extract_surface(nonlinear_subdivision=1)
    return s


def shot(s, field, out, clim, cmap="turbo", title="", views=("front", "back"), hardware=True, label=None, dist=720, log=False):
    for v in views:
        d, up = VIEWS[v]
        p = pv.Plotter(off_screen=True, window_size=(1600, 1150)); p.set_background("white")
        p.enable_anti_aliasing("ssaa")
        p.add_mesh(s, scalars=field, cmap=cmap, clim=clim, smooth_shading=True, specular=0.25, log_scale=log,
                   scalar_bar_args=dict(title=title, color="black", vertical=True, position_x=0.9, position_y=0.15, height=0.65, fmt="%.3g"))
        if hardware:
            p.add_mesh(props, color="#2b2b2b", opacity=0.35, smooth_shading=True)
            p.add_mesh(motors, color="#8a8f96", opacity=0.5, smooth_shading=True)
        if label:
            p.add_text(label, position="upper_left", font_size=13, color="black")
        dv = np.array(d) / np.linalg.norm(d)
        c0 = np.array(list(map(float, os.environ["FOC"].split(",")))) if os.environ.get("FOC") else C0
        dd = float(os.environ.get("DIST", dist))
        p.camera.focal_point = c0; p.camera.position = c0 + dd * dv; p.camera.up = up
        p.screenshot(out.replace(".png", f"_{v}.png"))


if __name__ == "__main__":
    vtu, field, out, lo, hi = sys.argv[1:6]
    title = sys.argv[6] if len(sys.argv) > 6 else field
    label = sys.argv[7] if len(sys.argv) > 7 else None
    s = surf(vtu, field)
    if field == "fos":
        pass
    shot(s, field, out, (float(lo), float(hi)), title=title, label=label)
