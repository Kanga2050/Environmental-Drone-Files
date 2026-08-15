import pyvista as pv, numpy as np
pv.OFF_SCREEN = True
z = np.load("fea/dyn/modes.npz"); f = z["w"] / 2 / np.pi; phi = z["phi"]
m = pv.read("fea/hover/hover.vtu")
base = m.extract_surface(nonlinear_subdivision=0)
ids = base["vtkOriginalPointIds"] if "vtkOriginalPointIds" in base.point_data else None
el = [i for i in range(len(f)) if f[i] > 1][:6]
p = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(2100, 1300), border=False); p.set_background("white")
C0 = np.array([100, 110, 130.0]); d = np.array([0.62, 0.75, 0.45]); d /= np.linalg.norm(d)
for k, i in enumerate(el):
    U = phi[:, i].reshape(-1, 3).astype(float)
    mm = m.copy(); mm["U"] = U; mm["|U|"] = np.linalg.norm(U, axis=1)
    sc = 25.0 / np.abs(U).max()
    w = mm.warp_by_vector("U", factor=sc).extract_surface(nonlinear_subdivision=0)
    p.subplot(k // 3, k % 3)
    p.add_mesh(base, color="#d0d4d9", opacity=0.18)
    p.add_mesh(w, scalars="|U|", cmap="plasma", smooth_shading=True, show_scalar_bar=False)
    lab = {0: "flight controller on its mount", 1: "front arm, bending", 2: "front arms, twisting", 3: "rear arm, bending", 4: "rear arm, bending", 5: "flight controller, rocking"}[k]
    if k in (0, 5):
        p.add_mesh(base, color="#d0d4d9", opacity=0.08)
        p.add_mesh(pv.Sphere(radius=14, center=(100, 133, 155)), color="#e74c3c")
        p.add_mesh(w.clip_box([60, 140, 100, 170, 120, 190], invert=False), scalars="|U|", cmap="plasma", show_scalar_bar=False)
    p.add_text(f"Mode {k+1}: {f[i]:.1f} Hz: {lab}", position="upper_left", font_size=13, color="black")
    p.camera.focal_point = C0; p.camera.position = C0 + 820 * d; p.camera.up = (0, 0, 1)
p.screenshot("img/manta_modes.png")
print([round(f[i], 1) for i in el])
