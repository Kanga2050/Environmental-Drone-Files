# GCT streaming approach — parked

This branch preserves the LingBot-Map GCT (causal streaming) pipeline at the
point it was set aside in favour of a global reconstruction method.

## What was built on top of upstream

- `lingbot-map/view_map3d.py` + `lingbot-map/viewer/` — self-hosted Three.js
  point-cloud viewer. WASD/mouse flight with roll, standing in a recorded frame
  using its own intrinsics with a world-pinned photo overlay, rectangle/box
  selection, per-plant metrics, CSV/PLY export, render-on-demand.
- `lingbot-map/detect_plants.py` — multi-view species fusion, the full per-view
  confidence distribution (`species_view_scores` / `spread` / `sharpening` /
  `entropy`) instead of one number, Pl@ntNet regional floras.
- `lingbot-map/stream_video.py` — `--fuse average` (confidence-weighted
  per-voxel mean, −33% local colour roughness), `obs_count` provenance, and
  `--pin_fx`/`--pin_fy` to hold focal length fixed rather than re-predicting it.

## Why it was parked

GCT is causal: a 16-frame sliding window plus a 12-frame anchor from the opening
~29 s. It never revisits a decision.

Measured on this footage:

| | |
|---|---|
| path length | 34.74 u inside a 2.47 × 0.86 × 1.43 u box |
| revisit pairs (>1500 frames apart, <0.155 u) | 248 |
| focal length spread across frames | 26% (fx 433–560) |
| focal length trend across the flight | +7.7% (+6.0% half to half) |
| rigid-ICP drift at a revisit | 0.35 u translation, residual 0.042 → 0.027 u |

The camera flies over the same ground ~14 times and the reconstruction deposits
geometry twice at the revisits.

**The error is a non-rigid warp, not a displacement.** ICP between revisit
chunks of the accumulated map is asymmetric by 1.8–15.0° — aligning A→B and B→A
gives transforms that are not inverses. So no rigid loop-closure edge exists,
and a pose graph cannot express the correction. 0 of 14 candidate constraints
survived symmetry and stability gating.

Two further measurements worth keeping:

- 32% of saved frames carry <1000 confident points and 67 carry <100, so
  single-frame ICP had nothing to work with. Chunking the map by `first_frame`
  fixed the sparsity but not the asymmetry.
- Free-scale ICP is unusable here: it shrinks a cloud to 0.58× volume to fit a
  dense region and reports a low residual. Any scale figure from it is an
  artifact. Rigid-only is the trustworthy measurement.

Fixing this needs a method that solves all frames jointly, which is why the work
moved to COLMAP / GLOMAP.
