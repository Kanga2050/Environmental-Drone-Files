# The 3D viewer

```
.venv310/Scripts/python.exe lingbot-map/view_map3d.py \
    --artifacts artifacts_yard_final \
    --port 8090 \
    --classify bioclip          # optional: enables the Identify button
```

Then open <http://localhost:8090>. Startup is a few seconds and reads nothing but
the artifacts directory, so it can run beside a detection re-run.

`view_map.py` (viser) still works and is unchanged. This one exists because the
two things most wanted here are things viser does not expose: per-frame keyboard
state, so no WASD, and a camera driven from a recorded intrinsic matrix, so no
aligned photo overlay.

## Getting around

| | |
|---|---|
| `W A S D` | move, relative to where you are looking |
| `Space` / `Shift` | up / down along gravity |
| drag mouse | look |
| scroll | move forward / back |
| `Ctrl` (held) | move slowly |
| `-` / `=` | slower / faster |
| `R` | reset the view |

Gravity is taken from the mean camera down-axis over the whole flight, not from
a ground-plane fit — a plane fit can land on a fence and flip, and a hand-flown
drone is level on average even when no single frame is.

## Analysis tools

**Plants tab** — every instance, filterable by name or species and sortable by
confidence, size, height, volume or number of views. Clicking one selects it,
flies to it, and fills the Selection tab with its measurements: height above the
fitted ground, crown width and depth from the principal axes of its footprint
(a canopy is rarely axis-aligned), occupied volume by voxel count rather than
box volume, mean colour, and the frames it was seen in. **Export table (CSV)**
writes the lot.

**Frames tab** — step through the recorded cameras. *Stand here* puts the
virtual camera exactly where that one was, with the same projection built from
its intrinsic matrix, and lays the photo over the render at an opacity you
control. If the points sit on the photo, the reconstruction is right there; if
they slide, it is not. This is the most direct check available, and the 3D boxes
are drawn into the same view, so a box that sits on a plant in one frame and
drifts off it in another is a merge error rather than a detection.

Two more layers, which answer different questions:

- *its own depth map* — what this one frame produced, unprojected. The fused
  cloud is hidden while it is on, because standing at that camera the two
  project onto the same pixels and you would see nothing new. Fly away from the
  camera afterwards and you can see the slab it contributed.
- *points it created* — the cloud points whose first observation was this frame.
  Exact, and needs `--fuse average` (see below).

**Selection** — hold `Ctrl` and drag a rectangle over the cloud. The nearest
coherent depth layer is kept, because a rectangle over a shrub also catches the
fence behind it. You get the same measurements a detected plant gets, which
instances it overlaps, and the frames that see it best — ranked by how much of
the selection each frame sees *and* how much of the frame it fills, so the crop
you could take is the largest available. Click a thumbnail to go stand in it.

**Identify** runs the classifier on that selection: it crops the region out of
its best five views and sends them as one identification rather than classifying
each and voting. This works on anything, including plants the detector missed
entirely.

**Measure** (`M`) — click two points for the distance, split into components
along and across gravity. **Inspect** — click a point for its position, height,
colour and which instance owns it.

**Colour by** — photo colour, height above ground, times observed, or the frame
that last saw each point. The last two need `--fuse average`; *times observed*
is the one to look at when you doubt a piece of geometry, since a surface seen
once is a guess and one seen fifty times is a measurement.

## Units

A monocular reconstruction fixes shape but not size, so everything starts in
arbitrary units and is labelled `u`. To get metres: press `M`, measure something
you know (a fence panel, a brick course, a doorway), and type its real length in
**View → Scale**. Every length, volume and speed in the viewer then converts,
and the factor is remembered per artifacts directory. Exported PLY files record
in their header which convention they were written in.

Nothing is reported as centimetres until you have done this once. Guessing would
be worse than saying so.

## `--fuse average`

`stream_video.py --fuse average` changes how a revisited voxel is updated: a
confidence-weighted running mean of every observation instead of the newest
sample. It also records, per point, the first and last frame that saw it and how
much evidence stands behind it — which is what the *points it created*,
*times observed* and *last seen* views read.

Rebuild with `lingbot-map/run_yard.sh <output-dir>`, which carries the full flag
set and the reason for each one.

## Species confidence: the distribution, not the number

A plant's `species_confidence` is one number from a fused call that averages
every view's embedding before the softmax. Measured over this scene, that number
is not representative: the median instance's per-view scores for its **own**
reported name have sd 0.30, and only ~40% of views independently pick it.

So the plant panel draws two spreads, and `plants.json` records them:

| field | meaning |
|---|---|
| `species_candidates` | fused belief across species (top 5 of 50) |
| `species_view_scores` | each view's own score for the reported name |
| `species_view_top1` | what each view would have said on its own |
| `species_agreement` | share of views whose top-1 is the reported name |
| `species_spread` | sd of `species_view_scores` |
| `species_sharpening` | fused ÷ best single view (≈1 = views carried it) |
| `species_entropy` | bits, against `species_entropy_max` (5.64 for 50) |
| `species_unsupported` | no view's top-1 is the reported name |

Plant #0 is the worked example: reported *Olea europaea* 0.317 on a red autumn
shrub, with views scoring 0.007 / 0.006 / 0.033 / 0.997. One view carried it.

Note `species_sharpening` is reported, not gated. A fused score above every
individual view is normally fusion working — it fires at 1.04–1.68 on unanimous
calls — so `min_species_agreement` remains the check that rejects a name.

## Stand here (G)

The camera takes the recorded frame's pose and intrinsics outright, so the photo
overlay lies on the geometry it produced. Verified by edge-correlating the
photo-only render against the points-only render inside the letterbox: the best
match is at shift (0, 0) in both axes, i.e. no systematic offset.

Three bugs used to make it feel otherwise, all in the handover rather than the
frame view itself:

- **Aimed backwards.** `enterFrame` derived its forward from the basis's `+Z`
  column, but a three.js camera looks down `-Z`. Measured over all 264 poses
  that vector was 179.97–180.00° from the true view direction, so stepping out
  of a frame spun the scene around.
- **Horizon snapped flat.** The fly camera had yaw and pitch but no roll, and
  this footage rolls a median 5.0° (p90 9.6°, max 20.6°; half the frames past
  5°). `adoptBasis` now decomposes the pose into yaw/pitch/roll, which
  round-trips to float32 precision, so the pose is a state the camera can hold.
- **Nothing redrew.** `leaveFrame` never called `invalidate`, so under
  render-on-demand the canvas kept showing the frame-locked image until the next
  input — which is why the jump appeared to happen *when you moved* rather than
  when you left.

Field of view is handed over by preserving pixels-per-degree, not by copying the
47° the camera recorded. In frame the render is letterboxed into `r.h` pixels;
outside it fills the canvas. Those differ whenever the window is narrower than
the frame's 1.23 aspect, and the scene jumped up to 1.7x. Measured as
in-frame/out-of-frame image correlation at 1000x900: **-0.08 → +0.85** (a wide
1500x900 window was already +0.99, which is why this only showed up windowed).

`L` levels the horizon if the roll you inherited is not wanted.

### Looking around from a frame

Dragging the mouse in frame view used to call `leaveFrame()`, which set the
overlay `<img>` to `display:none`. The camera was continuous across that — a
fixed world point moves 2-3 px, which is exactly the 0.18° a one-pixel drag
asks for, and the radial scale factor about the centre is 1.0012 — but a
photo-realistic view became a sparse point cloud in a single frame, which reads
as being teleported somewhere else.

The photo is now pinned into the world instead of deleted: a quad sized to fill
that camera's frustum exactly, at the frame's pose. Mouse-look only rotates and
never moves `cam.pos` (verified: 0.000000 u over a 32° turn), and rotation about
the optical centre produces no parallax, so a frustum-filling quad stays correct
at any distance. It simply slides out of view as you turn.

Same-rect image correlation across the first drag went **+0.60 → +0.89**.

Textures are tagged `SRGBColorSpace`. three.js treats a texture as linear unless
told, then encodes to sRGB on output, so without it the pinned photo rendered
visibly washed out against the `<img>` it replaces — mean abs difference 47 vs
10 per channel. The camera photo ribbon had the same bug and is fixed with it.

`O` drops the pinned photo, or re-pins the current frame's.

### When the pinned photo goes away

Turning keeps it; moving drops it. The quad reproduces the photo exactly only
from the point it was taken from — rotation about the optical centre has no
parallax, translation does, and a flat image against real geometry comes apart
immediately. So W/A/S/D, space/shift, the wheel and any fly-to call
`clearAnchorPhoto()`, while mouse-look leaves it alone.

Verified: a 40 px drag leaves `anchored` at the frame id; holding W (0.190 u of
travel) and a wheel notch both take it to -1.

### Selecting a plant stands in its frame

Clicking a plant in the Plants tab now enters the first recorded frame that saw
it — the same frame the detail panel's first thumbnail shows — instead of flying
to a free-camera view of the box. That is the view where its 3D box can be
checked against the photo it was detected in, and the species distribution is
right there in the panel while you look at it.

Falls back to `flyToPlant` if none of the plant's frames was saved to disk.

### window.__dbg()

Returns the camera's lock state, fov, position, yaw/pitch/roll, world and
projection matrices, the active viewport rect and the pinned-photo id. Read-only
— the viewer never reads it back. It is what settled the "stand here" bugs:
screenshot comparison kept conflating letterbox and field-of-view changes with
camera motion, and only the matrices showed the camera was fine all along.

## Caching

Only frame photos (`.jpg`) are cached hard — a frame's image is fixed by its id.
`app.js`, `index.html`, `style.css` and the vendored three.js send `no-cache`
with an mtime+size ETag, and `_file()` answers `If-None-Match` with a 304.

This matters more than it sounds. These files previously went out with
`public, max-age=3600`, so the browser served reloads from its own copy without
asking, and viewer edits silently did not appear for up to an hour. Headless
probes launch with an empty cache and never saw it.
