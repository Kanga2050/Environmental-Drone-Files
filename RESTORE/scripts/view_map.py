"""Serve the reconstructed point cloud with detected plants as 3D objects.

Reads only what the earlier stages wrote -- ``cloud.npz``, ``plants.json`` and
``crops/`` -- so it starts in a second and can be restarted freely while you
re-run detection with different prompts.

Each plant instance gets four linked representations:
  * a wireframe box at its 3D extent, in that instance's colour
  * a floating label with category, species and confidence
  * its own points in the cloud recoloured to match, so the geometry itself is
    identified rather than merely boxed
  * a sidebar entry with the source crop the label came from, and a button that
    flies the camera to it

The confidence slider drives all four together, so raising it prunes weak
detections from the scene and the sidebar at once.

If ``build_surface.py`` has been run, a fifth representation appears: the cloud
meshed into a solid surface, with each plant marked on it by growing along the
surface from the detected points. Because a surface carries connectivity that a
point cloud does not, this fills in parts of a plant that were never
well-observed, and the spread slider retunes it live.
"""

import argparse
import colorsys
import json
import os
import time

import cv2
import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from build_surface import (close_mask, diffuse_glb, keep_contiguous, mesh_edges,
                           srgb_to_linear)


def instance_colors(n):
    """Evenly spaced, saturated hues -- distinguishable against grey point cloud."""
    out = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 1.0)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def box_edges(lo, hi):
    """The 12 edges of an axis-aligned box as (24, 3) line-segment endpoints."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    idx = [(0, 1), (1, 2), (2, 3), (3, 0),
           (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([np.stack([c[a], c[b]]) for a, b in idx])


def species_text(p):
    """Name the plant only when the classifier actually backed the name.

    `species_confidence` is a share of the vote and looks high even when every
    candidate was a poor match; `species_raw_score` is what the classifier
    actually assigned. Reporting a confident-sounding binomial off a raw score of
    0.06 would be the most misleading thing this viewer could do.
    """
    if not p.get("species"):
        return None
    if not p.get("species_identified", True):
        return (f"*unidentified* — best guess {p['species']} "
                f"(score {p.get('species_raw_score', 0):.2f})")
    common = p.get("species_common") or ""
    return f"*{p['species']}*" + (f" — {common}" if common else "")


def plant_label(p, show_species):
    parts = [f"#{p['id']} {p['category']}"]
    txt = species_text(p) if show_species else None
    if txt:
        parts.append(txt.replace("*", ""))
    else:
        parts.append(f"det {p['detection_score']:.2f}")
    return "  |  ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="View the map with detected plants")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--plants", default=None, help="Default: <artifacts>/plants.json")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point_size", type=float, default=0.004)
    ap.add_argument("--highlight_grid", type=float, default=2.0,
                    help="Voxel size for shape-based highlighting, in multiples of the "
                         "map's own voxel size. Larger fills gaps, smaller hugs the plant")
    ap.add_argument("--surface", action="store_true",
                    help="Start with the surface shown instead of the point cloud")
    ap.add_argument("--spread", type=float, default=8.0,
                    help="Default surface growth distance in cm from the detected "
                         "points. The viewer's slider retunes this live")
    ap.add_argument("--max_spread", type=float, default=12.0,
                    help="Upper end of the spread slider, in cm. Cannot exceed what "
                         "build_surface.py precomputed")
    ap.add_argument("--ambient", type=float, default=1.8,
                    help="Ambient light intensity, which sets how bright the lit "
                         "surface renders relative to the unlit point cloud")
    ap.add_argument("--close_rings", type=int, default=2,
                    help="Rings of morphological closing on the surface highlight, "
                         "sealing pinholes and seams. 0 disables")
    ap.add_argument("--bridge", type=float, default=5.0,
                    help="In cm: keep highlight fragments within this distance of the "
                         "main body. The reconstructed mesh breaks one plant into "
                         "several pieces, so 0 (largest piece only) loses real parts")
    ap.add_argument("--max_points", type=int, default=2_000_000,
                    help="Subsample the cloud before sending it to the browser")
    args = ap.parse_args()

    art = args.artifacts
    cloud = np.load(os.path.join(art, "cloud.npz"))
    points = cloud["points"].astype(np.float32)
    colors = cloud["colors"].astype(np.uint8)

    if len(points) > args.max_points:
        # The whole cloud crosses a websocket on every redraw, so the browser --
        # not RAM -- is the binding constraint on how many points are worth sending.
        sel = np.random.default_rng(0).choice(len(points), args.max_points, replace=False)
        points, colors = points[sel], colors[sel]
        print(f"Subsampled cloud to {args.max_points:,} points for the browser")

    plants_path = args.plants or os.path.join(art, "plants.json")
    with open(plants_path, encoding="utf-8") as f:
        data = json.load(f)
    plants = data["plants"]
    print(f"Loaded {len(points):,} points and {len(plants)} plant instances")

    palette = instance_colors(len(plants))

    # Which cloud points belong to each plant.
    #
    # Preferred route is by *shape*: voxelize the plant's own pooled 3D points and
    # keep cloud points landing in the same voxels. A bounding box around a shrub
    # standing in front of a wall contains a great deal of wall, and colouring
    # that wall makes the highlight look both too big and wrong. Matching
    # occupancy follows the silhouette instead.
    #
    # Falls back to the box when the shape match comes up thin -- which happens if
    # the pooled points and the cloud were built with different filtering.
    inst_pts_file = os.path.join(art, "plant_points.npz")
    inst_pts = np.load(inst_pts_file) if os.path.exists(inst_pts_file) else None

    voxel = float(data.get("voxel_size", 0.0)) or None
    if voxel is None:
        meta_path = os.path.join(art, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                voxel = float(json.load(f).get("voxel_size", 0.005))
        else:
            voxel = 0.005
    grid = voxel * args.highlight_grid

    def voxel_keys(arr, cell):
        q = np.floor(arr / cell).astype(np.int64)
        q -= q.min(axis=0, keepdims=True)
        m = q.max(axis=0) + 1
        return q[:, 0] + m[0] * (q[:, 1] + m[1] * q[:, 2]), q.min(axis=0), m

    member_idx = []
    for p in plants:
        lo = np.array(p["bbox_min"], dtype=np.float32)
        hi = np.array(p["bbox_max"], dtype=np.float32)
        idx = np.zeros(0, dtype=np.int64)
        key = f"pts_{p['id']}"
        if inst_pts is not None and key in inst_pts:
            ip = inst_pts[key]
            if len(ip):
                # One shared lattice for both sets, so the keys are comparable.
                origin = np.minimum(ip.min(axis=0), points.min(axis=0))
                qa = np.floor((ip - origin) / grid).astype(np.int64)
                qb = np.floor((points - origin) / grid).astype(np.int64)
                span = np.maximum(qa.max(axis=0), qb.max(axis=0)) + 1
                ka = qa[:, 0] + span[0] * (qa[:, 1] + span[1] * qa[:, 2])
                kb = qb[:, 0] + span[0] * (qb[:, 1] + span[1] * qb[:, 2])
                idx = np.flatnonzero(np.isin(kb, np.unique(ka)))
        box_inside = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
        if len(idx) < 0.02 * max(len(box_inside), 1):
            idx = box_inside
            how = "box fallback"
        else:
            how = "shape match"
        member_idx.append(idx)
        print(f"  #{p['id']} {p['category']:<20} {len(idx):>8,} cloud points "
              f"({how}; box would give {len(box_inside):,})")

    # ---------------- surface (optional) ----------------
    #
    # Distances along the surface are precomputed out to a ceiling, so the spread
    # slider is a threshold on an existing field rather than a re-run of growth.
    surf_path = os.path.join(art, "surface.npz")
    surf = None
    if os.path.exists(surf_path):
        sf = np.load(surf_path)
        surf = {
            "vertices": sf["vertices"].astype(np.float32),
            "faces": sf["faces"].astype(np.int32),
            "colors": sf["colors"].astype(np.uint8),
            "reach": {}, "block": {}, "free": {},
        }
        for p in plants:
            k = p["id"]
            if f"plant_{k}_idx" in sf:
                surf["reach"][k] = sf[f"plant_{k}_idx"]
                surf["block"][k] = sf[f"plant_{k}_block"]
                surf["free"][k] = sf[f"plant_{k}_free"]
        surf["src"], surf["dst"] = mesh_edges(surf["faces"], len(surf["vertices"]))
        print(f"Loaded surface: {len(surf['vertices']):,} vertices, "
              f"{len(surf['faces']):,} faces")
    else:
        print("No surface.npz -- run build_surface.py to enable the surface view")

    def surface_members(pid, spread, blocked, refine=False):
        """Vertices within ``spread`` metres of the detection, along the surface.

        With ``refine``, the raw threshold is tidied using the fact that a plant
        is one physical object: closing seals the pinholes and hairline seams a
        distance cutoff leaves behind, and the contiguity pass drops any piece
        too far from the main body to belong to it.
        """
        if surf is None or pid not in surf["reach"]:
            return np.zeros(0, dtype=np.int32)
        d = surf["block" if blocked else "free"][pid]
        idx = surf["reach"][pid][np.isfinite(d) & (d <= spread)]
        if not refine or len(idx) == 0:
            return idx
        mask = np.zeros(len(surf["vertices"]), dtype=bool)
        mask[idx] = True
        mask = close_mask(mask, surf["src"], surf["dst"], args.close_rings)
        mask = keep_contiguous(mask, surf["vertices"], surf["src"], surf["dst"],
                               args.bridge / 100.0)
        return np.flatnonzero(mask).astype(np.int32)

    crops = {}
    for p in plants:
        cp = os.path.join(art, "crops", p.get("crop", ""))
        if os.path.exists(cp):
            img = cv2.cvtColor(cv2.imread(cp), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            if w > 320:
                img = cv2.resize(img, (320, max(1, int(h * 320 / w))))
            crops[p["id"]] = img

    start_surface = bool(args.surface) and surf is not None

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    if surf is not None and args.ambient > 0:
        # Diffuse shading multiplies albedo by an N.L term well below 1, so a lit
        # mesh lands darker than the unlit point cloud showing the same colours.
        # This intensity was measured to put the two within 1% of each other.
        server.scene.add_light_ambient("/ambient", color=(255, 255, 255),
                                       intensity=args.ambient)

    # ---------------- scene ----------------
    def render_cloud(highlight, threshold):
        cols = colors
        if highlight:
            cols = colors.copy()
            for p, idx, col in zip(plants, member_idx, palette):
                if p["detection_score"] >= threshold and len(idx):
                    cols[idx] = np.array(col, dtype=np.uint8)
        server.scene.add_point_cloud(
            name="/cloud", points=points, colors=cols,
            point_size=gui_psize.value, point_shape="rounded",
            visible=gui_points.value,
        )

    surf_state = {"handle": None}

    def render_surface():
        """Rebuild the surface mesh for the current toggles.

        Re-exporting the whole mesh on every change costs about ten milliseconds,
        which is cheaper than any scheme for patching colours in place would be.
        """
        if surf is None:
            return
        if not gui_surface.value:
            if surf_state["handle"] is not None:
                surf_state["handle"].visible = False
            return

        if gui_photo.value:
            cols = surf["colors"].copy()
        else:
            # Neutral grey reads geometry far better than photo colour does, and
            # gives the highlight something to stand out against.
            cols = np.full_like(surf["colors"], 185)

        if gui_srecolor.value:
            spread = gui_spread.value / 100.0
            for p, col in zip(plants, palette):
                if p["detection_score"] < gui_conf.value:
                    continue
                idx = surface_members(p["id"], spread, gui_sblock.value,
                                      gui_refine.value)
                if len(idx):
                    cols[idx] = np.array(col, dtype=np.uint8)

        # glTF wants COLOR_0 linear; these are sRGB out of JPEG frames.
        rgba = np.concatenate(
            [srgb_to_linear(cols), np.full((len(cols), 1), 255, dtype=np.uint8)], axis=1)
        mesh = trimesh.Trimesh(vertices=surf["vertices"], faces=surf["faces"],
                               vertex_colors=rgba, process=False)
        # Not add_mesh_trimesh: that path leaves the glTF material unset, and the
        # spec's default material is fully metallic, which renders the surface
        # about 30% darker than the same colours drawn as points.
        surf_state["handle"] = server.scene.add_glb(
            name="/surface", glb_data=diffuse_glb(mesh))

    def render_plants(threshold, show_boxes, show_labels, show_species):
        for p, col in zip(plants, palette):
            vis = p["detection_score"] >= threshold
            lo = np.array(p["bbox_min"], dtype=np.float32)
            hi = np.array(p["bbox_max"], dtype=np.float32)
            server.scene.add_line_segments(
                name=f"/plants/{p['id']}/box",
                points=box_edges(lo, hi),
                colors=np.tile(np.array(col, dtype=np.uint8), (12, 2, 1)),
                thickness=2.0, thickness_units="screen",
                visible=vis and show_boxes,
            )
            server.scene.add_label(
                name=f"/plants/{p['id']}/label",
                text=plant_label(p, show_species),
                position=(float((lo[0] + hi[0]) / 2),
                          float((lo[1] + hi[1]) / 2),
                          float(hi[2])),
                visible=vis and show_labels,
            )

    def fly_to(p):
        lo = np.array(p["bbox_min"], dtype=np.float32)
        hi = np.array(p["bbox_max"], dtype=np.float32)
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo)) + 1e-6
        for client in server.get_clients().values():
            client.camera.look_at = tuple(float(v) for v in center)
            client.camera.position = tuple(
                float(v) for v in center + np.array([radius * 1.8, radius * 1.2, radius * 1.8])
            )

    # ---------------- GUI ----------------
    with server.gui.add_folder("Display"):
        gui_points = server.gui.add_checkbox("Show point cloud", not start_surface)
        gui_psize = server.gui.add_slider("Point size", 0.0005, 0.02, 0.0005, args.point_size)
        gui_conf = server.gui.add_slider("Min detection conf", 0.0, 1.0, 0.01, 0.15)
        gui_boxes = server.gui.add_checkbox("3D boxes", True)
        gui_labels = server.gui.add_checkbox("Labels", True)
        gui_species = server.gui.add_checkbox("Show species guess", True)
        gui_highlight = server.gui.add_checkbox("Recolor plant points", True)

    with server.gui.add_folder("Surface", expand_by_default=surf is not None):
        gui_surface = server.gui.add_checkbox(
            "Show surface", start_surface, disabled=surf is None)
        gui_photo = server.gui.add_checkbox("Photo colors", True, disabled=surf is None)
        gui_srecolor = server.gui.add_checkbox(
            "Recolor plant on surface", True, disabled=surf is None)
        gui_spread = server.gui.add_slider(
            "Plant spread (cm)", 0.0, args.max_spread, 0.5, args.spread,
            disabled=surf is None)
        gui_sblock = server.gui.add_checkbox(
            "Stop at flat surfaces", True, disabled=surf is None)
        gui_refine = server.gui.add_checkbox(
            "Fill gaps, keep contiguous", True, disabled=surf is None)
        if surf is None:
            server.gui.add_markdown("Run `build_surface.py` to enable these.")

    def on_surface_toggle(_=None):
        # Points and surface occupy the same space, so showing both is mush.
        # Swapping them is what you almost always want; re-tick to overlay.
        gui_points.value = not gui_surface.value
        refresh()

    server.gui.add_markdown(
        f"**{len(plants)} plants** · classifier `{data.get('classifier', '?')}`\n\n"
        f"A name appears only when the classifier's **raw** score clears the "
        f"threshold; below it the plant reads *unidentified*. A closed-set "
        f"classifier always returns its best candidate, so a high vote share "
        f"on its own means nothing."
    )

    for p, col in zip(plants, palette):
        with server.gui.add_folder(f"#{p['id']}  {p['category']}", expand_by_default=False):
            if p["id"] in crops:
                server.gui.add_image(crops[p["id"]], label="source crop", format="jpeg")
            species_line = species_text(p) or "*no species candidate*"
            # Plain markdown only: viser's renderer rejects raw HTML, and a single
            # bad tag blanks the whole block. Instance colour is already carried by
            # the 3D box and the recoloured points.
            server.gui.add_markdown(
                f"**{p['category']}** ({p['category_confidence']:.2f})\n\n"
                f"{species_line}\n\n"
                f"classifier score **{p.get('species_raw_score', 0):.3f}** "
                f"(vote share {p['species_confidence']:.2f})\n\n"
                f"detection **{p['detection_score']:.2f}** · "
                f"seen in **{p['views']}** views\n\n"
                f"cloud points: **{len(member_idx[p['id']]):,}**"
                + (f"\n\nsurface vertices: "
                   f"**{len(surface_members(p['id'], args.spread / 100.0, True, True)):,}** "
                   f"at {args.spread:g}cm spread" if surf is not None else "")
            )
            btn = server.gui.add_button("Fly to", icon=viser.Icon.PLANE)
            btn.on_click(lambda _, pp=p: fly_to(pp))

    def refresh(_=None):
        render_cloud(gui_highlight.value, gui_conf.value)
        render_plants(gui_conf.value, gui_boxes.value, gui_labels.value, gui_species.value)
        render_surface()

    for handle in (gui_points, gui_psize, gui_conf, gui_boxes, gui_labels,
                   gui_species, gui_highlight, gui_photo, gui_srecolor,
                   gui_spread, gui_sblock, gui_refine):
        handle.on_update(refresh)
    gui_surface.on_update(on_surface_toggle)

    refresh()
    print(f"\n  Viewer serving at http://localhost:{args.port}   (Ctrl-C to exit)\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
