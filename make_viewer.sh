#!/usr/bin/env bash
# Video -> labelled, coloured 3D plant map -> one offline viewer HTML.
#
#   bash make_viewer.sh VIDEO NAME
#   bash make_viewer.sh --from-map NAME     (steps 2-4 on an existing maps/NAME,
#                                            e.g. one recorded by yardmap/live.py)
#
# Writes maps/NAME/ (cloud, per-frame depth and poses, plants) and
# viewers/NAME.html.
# Short clips (under a minute) need a smaller anchor: ANCHOR_STRIDE=6.
set -euo pipefail
cd "$(dirname "$0")"

VIDEO="$1"
NAME="$2"
GCT=models/gct.pt
ANCHOR_STRIDE="${ANCHOR_STRIDE:-72}"
PY=.venv310/Scripts/python.exe
M=models
MAP="maps/$NAME"

# 1. Reconstruct: streaming GCT -> depth + poses -> fused coloured cloud
[ "$VIDEO" = "--from-map" ] || "$PY" -u yardmap/reconstruct.py \
  --video_path "$VIDEO" --model_path "$GCT" --use_sdpa --fps 30 \
  --num_scale_frames 12 --anchor_stride "$ANCHOR_STRIDE" --kv_cache_sliding_window 16 \
  --keyframe_interval 30 \
  --voxel_size 0.005 --max_points 4000000 --downsample_factor 4 --conf_threshold 1.5 \
  --crop_margins 0.05,0.14,0.22,0.22 --clahe --min_frame_luma 14 --min_frame_std 18 \
  --mask_sky --sky_model "$M/skyseg.onnx" --fuse average \
  --save_artifacts "$MAP" --artifact_every 30 --no_viewer

cd yardmap
# 2. Find plants: detector ensemble -> lift boxes to 3D -> cluster -> name, health
#    (growth form -> family -> genus -> species, stopping when unsure)
"../$PY" detect_plants.py --artifacts "../$MAP" \
  --detector yolo --det_weights "../$M/detector_fpv.pt,../$M/detector_yard.pt" \
  --det_threshold 0.10 --det_tta --max_box_frac 0.85 \
  --cluster_frac 0.026 --merge_iou 0.25 --merge_sep_frac 0.8 \
  --classifier hier --cls_weights "../$M/classifier/best.pt" --tau "../$M/classifier/tau85.json" \
  --classify_top_k 12 --health "../$M/health/health.pt"

# 3. Assign map points to plants (Point Transformer)
"../$PY" -m plant_seg.segment --artifacts "../$MAP" --weights "../$M/plant_seg.pt" --threshold 0.4

# 4. One self-contained viewer
"../$PY" export_viewer.py --artifacts "../$MAP" --out "../viewers/$NAME.html" --min_conf 0.0
