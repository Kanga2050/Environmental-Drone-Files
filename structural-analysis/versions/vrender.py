import pyvista as pv, numpy as np, meshio, os, sys, json
pv.OFF_SCREEN = True
D = os.path.dirname(os.path.abspath(__file__))
C0 = np.array([100, 110, 130.0])
d = np.array([0.62, 0.75, 0.45]); d /= np.linalg.norm(d)


def surf(v, field=None, case=None):
    m = meshio.read(f"{D}/{v}/manta_3.0.inp")
    cells = [c.data for c in m.cells if c.type == "tetra10"]
    g = pv.UnstructuredGrid({pv.CellType.QUADRATIC_TETRA: np.concatenate(cells)}, m.points)
    g.cell_data["part"] = np.r_[np.zeros(len(cells[0])), np.ones(sum(len(c) for c in cells[1:]))]
    if case and os.path.exists(f"{D}/{v}/{case}/{case}.vtu"):
        r = pv.read(f"{D}/{v}/{case}/{case}.vtu")
        g.point_data["vm"] = r.point_data["vm"]
    if os.environ.get("CUT"):
        g = g.clip(normal=(1, 0, 0), origin=(100, 0, 0), invert=True)
    return g.extract_surface(nonlinear_subdivision=0)


if __name__ == "__main__":
    vs = sys.argv[1].split(",")
    case = sys.argv[2] if len(sys.argv) > 2 else None
    hi = float(sys.argv[3]) if len(sys.argv) > 3 else 6
    n = len(vs); cols = min(4, n); rows = (n + cols - 1) // cols
    p = pv.Plotter(off_screen=True, shape=(rows, cols), window_size=(560 * cols, 520 * rows), border=False); p.set_background("white")
    for k, v in enumerate(vs):
        p.subplot(k // cols, k % cols)
        s = surf(v, case=case)
        if case and "vm" in s.point_data:
            p.add_mesh(s, scalars="vm", cmap="turbo", clim=(0, hi), smooth_shading=True, show_scalar_bar=(k == n - 1),
                       scalar_bar_args=dict(title="von Mises (MPa)", color="black", vertical=True, position_x=0.82, position_y=0.1, height=0.7))
        else:
            p.add_mesh(s, scalars="part", cmap=["#b8c4d1", "#5d7ea8"], smooth_shading=True, show_scalar_bar=False)
        info = json.load(open(f"{D}/{v}/info.json")) if os.path.exists(f"{D}/{v}/info.json") else None
        lab = v.upper() if info is None else f"{v.upper()}  {info['mass_g']:.0f} g"
        p.add_text(lab, position="upper_left", font_size=14, color="black")
        dd = np.array([1.0, 0.25, 0.2]) / np.linalg.norm([1.0, 0.25, 0.2]) if os.environ.get("CUT") else d
        p.camera.focal_point = C0; p.camera.position = C0 + 900 * dd; p.camera.up = (0, 0, 1)
    out = f"{D}/../img/versions_{case or 'geom'}{'_cut' if os.environ.get('CUT') else ''}.png"
    p.screenshot(out); print(out)
