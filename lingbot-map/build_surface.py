"""Turn the fused point cloud into a surface, and decide on that surface which
parts are plant.

Written to ``<artifacts>/surface.npz`` so the viewer loads it instantly and
toggles it with a checkbox.

**Meshing.** The cloud is rasterised into a voxel occupancy grid, blurred, and
iso-surfaced with marching cubes. The blur is what turns a cloud of samples into
something continuous: an unblurred occupancy grid iso-surfaces into a field of
disconnected cubes, whereas a light blur lets neighbouring occupied voxels merge
into one sheet.

**Membership.** A surface knows which of its parts touch each other; a point
cloud does not. So instead of asking "is this point inside the plant's box" --
which sweeps in the wall behind and misses leaves the detector never saw clearly
-- we seed from the detected points and dilate outward *along the mesh*. Leaves
nobody detected get included because they are continuous with ones that were,
and reaching the wall means crossing empty space the dilation will not pay for.

Two things this learned the hard way on real footage, both encoded below:

* **Clean the seeds first.** The pooled detection points carry a few percent of
  scattered outliers. Under a growth rule every outlier is a fresh nucleus, so
  contamination does not merely add error, it blooms. Trimming seeds to the
  detection's own display box first is the difference between an 8% and an 80%
  highlight.
* **Do not trust colour or local roughness as barriers.** They are the obvious
  choices and both failed here: a plant with bronze-pink leaves scores no higher
  on a vegetation index than concrete under a green colour cast, and a surface
  built from noisy video is foamy everywhere, so flat things do not measure flat
  locally. Flatness *is* recoverable, but only by fitting planes globally --
  which is what the RANSAC pass does.

Distances are stored per vertex rather than a fixed mask, so the viewer can
re-threshold the spread live instead of re-running this.

Run after ``detect_plants.py``, before ``view_map.py``.
"""

import argparse
import glob
import json
import os
import struct
import time

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree


def build_mesh(points, colors, voxel, sigma, level, min_points):
    """Voxel occupancy -> blurred field -> marching cubes -> coloured mesh."""
    origin = points.min(axis=0) - voxel * 4.0
    q = np.floor((points - origin) / voxel).astype(np.int32)
    dims = (q.max(axis=0) + 5).astype(np.int64)

    flat = (q[:, 0].astype(np.int64) * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    counts = np.bincount(flat, minlength=int(dims.prod()))
    vol = (counts >= min_points).astype(np.float32).reshape(dims)
    occupied = int((counts >= min_points).sum())
    print(f"  grid {tuple(int(d) for d in dims)} = {dims.prod() / 1e6:.1f}M voxels, "
          f"{occupied:,} occupied ({100.0 * occupied / dims.prod():.2f}%)")

    # Imported here rather than at module scope so view_map.py can borrow the
    # graph helpers below without paying for scikit-image at startup.
    from skimage import measure

    vol = ndimage.gaussian_filter(vol, sigma=sigma, mode="constant")
    if vol.max() <= level:
        raise SystemExit(
            f"Nothing crosses the iso-level: field peaks at {vol.max():.3f} but "
            f"--level is {level}. Lower --level or raise --smooth.")

    verts, faces, normals, _ = measure.marching_cubes(
        vol, level=level, spacing=(voxel, voxel, voxel), gradient_direction="ascent")
    verts = verts.astype(np.float32) + origin.astype(np.float32)
    print(f"  marching cubes: {len(verts):,} vertices, {len(faces):,} faces")

    # Colour each vertex from the cloud points nearest it, by *median* of only
    # three. Averaging six was washing the surface out: it is a second blur on
    # top of the one the cloud already applies by fusing many views, and it cost
    # a measured 4.7 points of mean saturation and 4 of luminance contrast
    # against the cloud it samples. A median of three rejects the same outliers
    # without regressing everything toward grey.
    tree = cKDTree(points)
    _, idx = tree.query(verts, k=min(3, len(points)), workers=-1)
    vcol = np.median(colors[idx].astype(np.float32), axis=1)
    return (verts, faces.astype(np.int32), normals.astype(np.float32),
            np.clip(vcol, 0, 255).astype(np.uint8))


def match_source_colors(vcol, art_dir, l_strength, chroma_strength,
                        frame_stride=4, per_frame=4000):
    """Restore the tonal range that multi-view fusion averaged away.

    Fusing dozens of views of the same surface is a mean, and a mean of noisy
    samples regresses toward grey. Measured on this scene, the source frames
    carry a luminance spread of sigma 48.4; the fused cloud keeps only 34.8.
    That loss is why the surface looks dull, and no amount of care in sampling
    the cloud can recover it, because it happened before the cloud existed.

    So match moments against the frames themselves in CIELAB -- shift and scale
    each channel so the surface's distribution matches the photographs'. This is
    a correction toward the measured truth rather than a saturation slider: the
    target is what the camera actually recorded.

    Chroma is matched at lower strength by default. Stretching a and b to full
    source variance overshoots apparent saturation (56.7 against the source's
    43.6), because the surface's mean chroma differs from the frames' and the
    scaling compounds with it.
    """
    files = sorted(glob.glob(os.path.join(art_dir, "frames", "*.jpg")))[::frame_stride]
    if not files or (l_strength <= 0 and chroma_strength <= 0):
        return vcol

    rng = np.random.default_rng(0)
    sample = []
    for f in files:
        im = cv2.imread(f)
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        take = min(per_frame, len(im))
        sample.append(im[rng.choice(len(im), take, replace=False)])
    if not sample:
        return vcol

    def to_lab(a):
        return cv2.cvtColor(a.reshape(-1, 1, 3).astype(np.uint8),
                            cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    tgt = to_lab(np.concatenate(sample))
    cur = to_lab(vcol)
    mu_t, sd_t = tgt.mean(axis=0), tgt.std(axis=0)
    mu_c, sd_c = cur.mean(axis=0), cur.std(axis=0)
    full = (cur - mu_c) / np.maximum(sd_c, 1e-6) * sd_t + mu_t

    out = cur.copy()
    out[:, 0] = cur[:, 0] + (full[:, 0] - cur[:, 0]) * l_strength
    out[:, 1:] = cur[:, 1:] + (full[:, 1:] - cur[:, 1:]) * chroma_strength
    clipped = float(((out[:, 0] < 0) | (out[:, 0] > 255)).mean())
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8).reshape(-1, 1, 3),
                       cv2.COLOR_LAB2RGB).reshape(-1, 3)
    print(f"  colour matched to {len(files)} source frames: "
          f"L sigma {cur[:, 0].std():.1f} -> {to_lab(rgb)[:, 0].std():.1f} "
          f"(source {tgt[:, 0].std():.1f}), {clipped:.1%} clipped")
    return rgb


def excess_green(colors):
    """2G-R-B vegetation index in [-1, 1], computed per vertex."""
    c = colors.astype(np.float32) / 255.0
    return 2.0 * c[:, 1] - c[:, 0] - c[:, 2]


def colour_barrier(vcol, seeds, margin, pct):
    """A colour cutoff for growth -- but only where colour actually separates.

    Colour was rejected as a barrier once already, and correctly: a plant with
    bronze-pink leaves scored no higher on a vegetation index than the concrete
    behind it, so any fixed green threshold was noise. The wrong lesson to draw
    is that colour never works. In a backyard of green foliage against terracotta
    and stucco it separates cleanly, and without it growth walks straight off a
    shrub onto the fountain it sits beside -- which a plane barrier cannot stop,
    because a fountain is not flat.

    So measure the separation per plant instead of assuming it either way: compare
    this plant's seed vertices against the rest of the mesh, and use a colour
    cutoff only when the seeds are actually greener by ``margin``. Returns None
    when they are not, leaving geometry as the only barrier.
    """
    if len(seeds) == 0:
        return None
    exg = excess_green(vcol)
    seed_med = float(np.median(exg[seeds]))
    scene_med = float(np.median(exg))
    if seed_med - scene_med < margin:
        return None, seed_med, scene_med
    return float(np.percentile(exg[seeds], pct)), seed_med, scene_med


def srgb_to_linear(colors):
    """Convert sRGB bytes to linear, as glTF requires for COLOR_0.

    Our colours come from JPEG frames, so they are sRGB-encoded. glTF defines
    vertex colours as *linear*, and handing sRGB numbers to a renderer that
    reads them as linear lifts the midtones and pulls the channels together --
    the surface comes out brighter and noticeably washed out. Measured against
    the same scene drawn as points, it cost 16 points of mean saturation, which
    is far more than any of the colour work upstream was worth.

    Point clouds are unaffected, which is why the two disagreed on screen while
    carrying identical colours.
    """
    x = colors.astype(np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return np.clip(lin * 255.0 + 0.5, 0, 255).astype(np.uint8)


def diffuse_glb(mesh, double_sided=True, brightness=1.0):
    """Export a mesh to GLB with an explicitly non-metallic material.

    trimesh writes vertex colours as COLOR_0 and no material at all. glTF says
    that a primitive without a material takes the *default* material -- and the
    default has ``metallicFactor`` 1.0. A fully metallic surface has no diffuse
    albedo: its colour comes from reflected environment instead of from its own
    base colour, so the vertex colours are crushed and the whole mesh renders
    dark and grey with a specular sheen on it.

    That is a rendering bug, not a colour-grading problem, and it is worth about
    30% of the brightness -- measured against the same scene drawn as points,
    which take no material and so are unaffected.

    Declaring metallic 0 / roughness 1 makes the surface purely diffuse, so
    lighting still conveys its shape while the albedo stays the colour we
    actually computed.
    """
    glb = mesh.export(file_type="glb")
    total = struct.unpack("<I", glb[8:12])[0]

    chunks, off = [], 12
    while off < total:
        clen, ctype = struct.unpack("<II", glb[off:off + 8])
        chunks.append([ctype, glb[off + 8:off + 8 + clen]])
        off += 8 + clen

    json_chunk = 0x4E4F534A
    for chunk in chunks:
        if chunk[0] != json_chunk:
            continue
        doc = json.loads(chunk[1].decode("utf-8"))
        k = float(brightness)
        doc.setdefault("materials", []).append({
            "name": "diffuse",
            "doubleSided": bool(double_sided),
            "pbrMetallicRoughness": {
                "baseColorFactor": [k, k, k, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        })
        index = len(doc["materials"]) - 1
        for entry in doc.get("meshes", []):
            for prim in entry.get("primitives", []):
                prim["material"] = index
        chunk[1] = json.dumps(doc, separators=(",", ":")).encode("utf-8")

    body = b""
    for ctype, data in chunks:
        pad = (4 - len(data) % 4) % 4
        data += (b" " if ctype == json_chunk else b"\x00") * pad
        body += struct.pack("<II", len(data), ctype) + data
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


# =============================================================================
# Surface graph operations -- also used by view_map.py
# =============================================================================

def mesh_edges(faces, n_verts):
    """Directed edge arrays (src, dst) covering both directions of every edge."""
    indptr, indices = vertex_adjacency(faces, n_verts)
    src = np.repeat(np.arange(n_verts, dtype=np.int32), np.diff(indptr))
    return src, indices


def close_mask(mask, src, dst, rings):
    """Morphological closing on the surface: dilate then erode by ``rings``.

    Seals the pinholes and hairline seams that a growth threshold leaves behind
    without moving the region's outer boundary, since whatever the dilation adds
    on the outside the erosion takes straight back off.
    """
    if rings <= 0:
        return mask
    m = mask.copy()
    for _ in range(rings):
        nxt = m.copy()
        nxt[dst[m[src]]] = True
        m = nxt
    for _ in range(rings):
        outside = np.zeros(len(mask), dtype=bool)
        outside[src[~m[dst]]] = True
        m = m & ~outside
    return m


def keep_contiguous(mask, verts, src, dst, bridge):
    """Keep the main blob plus any piece close enough to be the same object.

    A plant is one connected thing, so its highlight should be too -- but the
    *reconstructed* surface is not the object. On this scene the plant meshes as
    three pieces separated by gaps of 1.6 and 3.9cm, so keeping only the largest
    connected component throws away a third of a correct highlight.

    Physical contiguity is the right test, not mesh contiguity: keep components
    within ``bridge`` of the largest, drop the rest. That still discards a
    genuine satellite sitting off on its own.
    """
    n = len(mask)
    if not mask.any() or bridge < 0:
        return mask
    sel = mask[src] & mask[dst]
    graph = sparse.csr_matrix(
        (np.ones(int(sel.sum()), dtype=np.int8), (src[sel], dst[sel])), shape=(n, n))
    ncomp, lab = csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(lab[mask], minlength=ncomp)
    main = int(np.argmax(sizes))
    out = mask & (lab == main)
    if bridge == 0:
        return out
    tree = cKDTree(verts[out])
    for c in np.flatnonzero(sizes > 0):
        if c == main:
            continue
        part = mask & (lab == c)
        if tree.query(verts[part], k=1, workers=-1)[0].min() <= bridge:
            out |= part
    return out


def vertex_adjacency(faces, n_verts):
    """Undirected vertex graph as CSR-style (indptr, indices)."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    keep = np.ones(len(e), dtype=bool)
    keep[1:] = np.any(e[1:] != e[:-1], axis=1)
    e = e[keep]
    indptr = np.zeros(n_verts + 1, dtype=np.int64)
    np.add.at(indptr, e[:, 0].astype(np.int64) + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, e[:, 1].astype(np.int32)


def drop_small_components(verts, faces, min_faces):
    """Delete islands smaller than ``min_faces``.

    Marching cubes over a noisy cloud always leaves a scatter of detached crumbs
    from stray points -- individually invisible, but they inflate the mesh.
    """
    indptr, indices = vertex_adjacency(faces, len(verts))
    graph = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.int8), indices, indptr),
        shape=(len(verts), len(verts)))
    ncomp, comp = csgraph.connected_components(graph, directed=False)

    face_comp = comp[faces[:, 0]]
    sizes = np.bincount(face_comp, minlength=ncomp)
    good = np.flatnonzero(sizes >= min_faces)
    keep_face = np.isin(face_comp, good)
    if keep_face.all():
        return verts, faces, np.ones(len(verts), dtype=bool)
    faces = faces[keep_face]
    used = np.zeros(len(verts), dtype=bool)
    used[faces.ravel()] = True
    remap = np.full(len(verts), -1, dtype=np.int32)
    remap[used] = np.arange(int(used.sum()), dtype=np.int32)
    print(f"  dropped {ncomp - len(good):,} of {ncomp:,} components "
          f"({int((~used).sum()):,} vertices)")
    return verts[used], remap[faces], used


def find_planes(verts, normals, args):
    """Mark vertices lying on a large flat structure.

    Driveways, garage doors and walls are planes; a plant is not. Local curvature
    cannot see this on a surface built from noisy video -- at 3cm scale the whole
    mesh is bumpy -- but a plane fitted across thousands of vertices averages that
    noise away, which is exactly why this runs globally rather than per vertex.

    Requiring the surface normal to agree with the plane's own normal keeps a
    plane from being threaded edge-on through foliage, which is otherwise an easy
    way to accumulate a large but meaningless inlier count.
    """
    rng = np.random.default_rng(0)
    remaining = np.ones(len(verts), dtype=bool)
    is_plane = np.zeros(len(verts), dtype=bool)
    found = 0
    for _ in range(args.max_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < 1200:
            break
        best = None
        for _ in range(args.plane_iters):
            tri = verts[rng.choice(idx, 3, replace=False)]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            ln = np.linalg.norm(n)
            if ln < 1e-8:
                continue
            n = n / ln
            d = -float(n @ tri[0])
            inl = (np.abs(verts[idx] @ n + d) < args.plane_tol) & \
                  (np.abs(normals[idx] @ n) > 0.80)
            c = int(inl.sum())
            if best is None or c > best[0]:
                best = (c, n, d, inl)
        if best is None or best[0] < len(verts) * args.plane_min_frac:
            break
        _, n, d, inl = best
        # A plane is a global structure, so claim every vertex on it, not only the
        # ones not yet claimed. Stricter tolerance here than for fitting: the fit
        # wants reach, the claim wants to avoid swallowing foliage.
        #
        # The claim deliberately does *not* re-check the normal, though the fit
        # does. Right where the ground meets the pot -- exactly where growth needs
        # to be stopped -- reconstruction noise tilts the surface enough to fail a
        # normal test, so requiring one punched a hole in the barrier and let the
        # highlight spill across the driveway. Lying on the plane is the reliable
        # signal; pointing along with it is not.
        claim = np.abs(verts @ n + d) < args.plane_tol * 0.85
        if args.plane_claim_normal > 0:
            claim &= np.abs(normals @ n) > args.plane_claim_normal
        is_plane |= claim
        remaining[idx[inl]] = False
        found += 1
    print(f"  {found} planes cover {int(is_plane.sum()):,} vertices "
          f"({is_plane.mean():.1%} of the mesh)")
    return is_plane


def geodesic_spread(seeds, indptr, indices, verts, blocked, max_dist, max_rounds):
    """Distance from the seed set measured *along* the surface.

    Dijkstra by expanding frontiers rather than a priority queue -- edge lengths
    on a marching-cubes mesh are all within a factor of two of each other, so the
    frontier order is near enough to sorted that this converges in a few hundred
    rounds while staying fully vectorised.

    Returns +inf for anything unreachable within ``max_dist`` or behind a barrier.
    """
    dist = np.full(len(verts), np.inf, dtype=np.float32)
    if len(seeds) == 0:
        return dist
    dist[seeds] = 0.0
    deg = np.diff(indptr)
    frontier = np.asarray(seeds, dtype=np.int32)
    for _ in range(max_rounds):
        frontier = frontier[deg[frontier] > 0]
        if frontier.size == 0:
            break
        dst = np.concatenate([indices[indptr[f]:indptr[f + 1]] for f in frontier])
        src = np.repeat(frontier, deg[frontier])
        nd = dist[src] + np.linalg.norm(verts[src] - verts[dst], axis=1)
        ok = (nd < dist[dst] - 1e-6) & (nd <= max_dist) & (~blocked[dst])
        if not ok.any():
            break
        dst, nd = dst[ok], nd[ok]
        order = np.argsort(nd)
        dst, nd = dst[order], nd[order]
        uniq, first = np.unique(dst, return_index=True)
        best = nd[first]
        better = best < dist[uniq] - 1e-6
        uniq = uniq[better]
        dist[uniq] = best[better]
        frontier = uniq.astype(np.int32)
    return dist


def main():
    ap = argparse.ArgumentParser(description="Mesh the cloud and label the plants on it")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--plants", default=None, help="Default: <artifacts>/plants.json")
    ap.add_argument("--out", default=None, help="Default: <artifacts>/surface.npz")

    ap.add_argument("--mesh_voxel", type=float, default=0.0,
                    help="Iso-surface voxel size. 0 = 2x the map's own voxel. Smaller "
                         "keeps detail and multiplies triangles")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Blur radius in voxels before iso-surfacing. Larger closes "
                         "gaps between leaves but rounds off fine structure")
    ap.add_argument("--level", type=float, default=0.28,
                    help="Iso-level in the blurred field. Lower inflates the surface "
                         "and closes more holes")
    ap.add_argument("--min_points", type=int, default=1,
                    help="Points needed in a voxel before it counts as occupied")
    ap.add_argument("--min_faces", type=int, default=400,
                    help="Discard mesh islands smaller than this")
    ap.add_argument("--color_match", type=float, default=1.0,
                    help="Strength of luminance moment-matching against the source "
                         "frames, recovering contrast lost to multi-view averaging. "
                         "0 leaves the fused colours as they are")
    ap.add_argument("--color_match_chroma", type=float, default=0.5,
                    help="Same, for the a/b chroma channels. Full strength overshoots "
                         "apparent saturation, so this defaults to half")

    ap.add_argument("--seed_radius", type=float, default=2.0,
                    help="Vertices within this many mesh voxels of a detected plant "
                         "point become seeds")
    ap.add_argument("--seed_trim", action="store_true", default=True,
                    help="Drop detected points outside the plant's display box before "
                         "seeding. Strongly recommended: outliers bloom under growth")
    ap.add_argument("--no_seed_trim", dest="seed_trim", action="store_false")
    ap.add_argument("--spread_ceiling", type=float, default=0.45,
                    help="How far to precompute growth along the surface, as a fraction "
                         "of each plant's own diagonal. The viewer thresholds inside this, "
                         "so it is a ceiling, not a setting")
    ap.add_argument("--min_spread", type=float, default=0.06,
                    help="Floor on that ceiling in metres, so a small plant still has room")
    ap.add_argument("--max_spread", type=float, default=0.40,
                    help="Cap on that ceiling in metres")

    ap.add_argument("--max_planes", type=int, default=12)
    ap.add_argument("--plane_iters", type=int, default=700)
    ap.add_argument("--plane_tol", type=float, default=0.012,
                    help="Distance for a vertex to count as on a plane")
    ap.add_argument("--plane_claim_normal", type=float, default=0.0,
                    help="Also require the surface normal to align this closely with "
                         "a plane before blocking on it. 0 disables; raising it makes "
                         "the barrier leakier, not tighter")
    ap.add_argument("--plane_min_frac", type=float, default=0.015,
                    help="Smallest plane worth keeping, as a fraction of the mesh")
    ap.add_argument("--colour_margin", type=float, default=0.025,
                    help="How much greener than the rest of the scene a plant's seeds "
                         "must be before colour is trusted as a growth barrier. Guards "
                         "against scenes where foliage is not green")
    ap.add_argument("--colour_pct", type=float, default=15.0,
                    help="Percentile of the seeds' own greenness used as the cutoff "
                         "when the colour barrier is enabled")
    ap.add_argument("--max_rounds", type=int, default=4000)
    args = ap.parse_args()

    art = args.artifacts
    t0 = time.time()

    cloud = np.load(os.path.join(art, "cloud.npz"))
    points = cloud["points"].astype(np.float32)
    colors = cloud["colors"].astype(np.uint8)

    map_voxel = 0.005
    meta_path = os.path.join(art, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            map_voxel = float(json.load(f).get("voxel_size", 0.005))
    if args.mesh_voxel <= 0:
        args.mesh_voxel = map_voxel * 2.0

    print(f"Meshing {len(points):,} points at voxel {args.mesh_voxel:.4f} "
          f"(map voxel {map_voxel:.4f})")
    verts, faces, normals, vcol = build_mesh(
        points, colors, args.mesh_voxel, args.smooth, args.level, args.min_points)
    verts, faces, kept = drop_small_components(verts, faces, args.min_faces)
    normals, vcol = normals[kept], vcol[kept]
    # marching_cubes returns gradients, not unit vectors.
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    print(f"  final mesh: {len(verts):,} vertices, {len(faces):,} faces")

    vcol = match_source_colors(vcol, art, args.color_match, args.color_match_chroma)

    is_plane = find_planes(verts, normals, args)

    plants_path = args.plants or os.path.join(art, "plants.json")
    with open(plants_path, encoding="utf-8") as f:
        data = json.load(f)
    plants = data["plants"]

    ip_path = os.path.join(art, "plant_points.npz")
    inst_pts = np.load(ip_path) if os.path.exists(ip_path) else None

    indptr, indices = vertex_adjacency(faces, len(verts))
    vtree = cKDTree(verts)
    free = np.zeros(len(verts), dtype=bool)

    out = {
        "vertices": verts,
        "faces": faces,
        "normals": normals.astype(np.float32),
        "colors": vcol,
        "is_plane": is_plane,
    }
    for p in plants:
        lo = np.array(p["bbox_min"], dtype=np.float32)
        hi = np.array(p["bbox_max"], dtype=np.float32)

        src_pts = None
        key = f"pts_{p['id']}"
        if inst_pts is not None and key in inst_pts and len(inst_pts[key]):
            src_pts = inst_pts[key].astype(np.float32)
            if args.seed_trim:
                inside = np.all((src_pts >= lo) & (src_pts <= hi), axis=1)
                dropped = int((~inside).sum())
                src_pts = src_pts[inside]
                if dropped:
                    print(f"  #{p['id']}: trimmed {dropped:,} stray detection points "
                          f"({dropped / len(inst_pts[key]):.1%})")

        seed_mask = np.zeros(len(verts), dtype=bool)
        if src_pts is not None and len(src_pts):
            hits = vtree.query_ball_point(
                src_pts, r=args.seed_radius * args.mesh_voxel, workers=-1)
            for h in hits:
                if len(h):
                    seed_mask[h] = True
        if not seed_mask.any():
            seed_mask = np.all((verts >= lo) & (verts <= hi), axis=1)
            print(f"  #{p['id']}: no detection points near the surface, seeding from box")
        seeds = np.flatnonzero(seed_mask).astype(np.int32)

        # Growth distance has to scale with the plant. A fixed 8cm was right for a
        # single potted shrub, but this scene spans a 5.5x range of sizes, and a
        # distance that closes the gaps in a tree canopy would swallow a seedling
        # whole.
        diag = float(np.linalg.norm(hi - lo))
        ceiling = float(np.clip(args.spread_ceiling * diag,
                                args.min_spread, args.max_spread))
        blocked = is_plane
        barrier = colour_barrier(vcol, seeds, args.colour_margin, args.colour_pct)
        cutoff, seed_med, scene_med = barrier if barrier else (None, 0.0, 0.0)
        if cutoff is not None:
            blocked = is_plane | (excess_green(vcol) < cutoff)
            print(f"  #{p['id']}: colour barrier on (seed green {seed_med:+.3f} vs "
                  f"scene {scene_med:+.3f}, cutoff {cutoff:+.3f})")
        d_block = geodesic_spread(seeds, indptr, indices, verts, blocked,
                                  ceiling, args.max_rounds)
        d_free = geodesic_spread(seeds, indptr, indices, verts, free,
                                 ceiling, args.max_rounds)
        reach = np.flatnonzero(np.isfinite(d_free)).astype(np.int32)
        out[f"plant_{p['id']}_idx"] = reach
        out[f"plant_{p['id']}_diag"] = np.float32(diag)
        out[f"plant_{p['id']}_block"] = d_block[reach].astype(np.float32)
        out[f"plant_{p['id']}_free"] = d_free[reach].astype(np.float32)

        mid = np.isfinite(d_block) & (d_block <= 0.16 * diag)
        print(f"  #{p['id']} {p['category']:<20} diag {diag:4.2f} seeds {len(seeds):>6,} -> "
              f"{int(mid.sum()):>6,} vertices at 16% spread "
              f"({int(mid.sum()) / len(verts):.1%} of mesh), "
              f"{len(reach):,} reachable within {ceiling * 100:.0f}cm")

    out_path = args.out or os.path.join(art, "surface.npz")
    np.savez_compressed(out_path, **out)
    print(f"\nWrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB) "
          f"in {time.time() - t0:.1f}s")
    print("Restart view_map.py and tick 'Show surface'.")


if __name__ == "__main__":
    main()
