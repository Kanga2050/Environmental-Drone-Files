#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv310/Scripts/python.exe
VIDEO="C:\\Users\\shaur\\Downloads\\VID_20260913_192111.mp4"
MODEL="C:\\Users\\shaur\\AppData\\Local\\lingbot-map\\lingbot-map.pt"
OUT="${1:-artifacts_yard_avg}"
EXTRA="${2:-}"

"$VENV" -u lingbot-map/stream_video.py \
  --video_path "$VIDEO" \
  --model_path "$MODEL" \
  --use_sdpa \
  --fps 30 \
  --num_scale_frames 12 \
  --anchor_stride 72 \
  --kv_cache_sliding_window 16 \
  --keyframe_interval 30 \
  --voxel_size 0.005 \
  --max_points 4000000 \
  --downsample_factor 4 \
  --conf_threshold 1.5 \
  --crop_margins 0.05,0.14,0.22,0.22 \
  --clahe \
  --min_frame_luma 14 \
  --min_frame_std 18 \
  --mask_sky \
  --fuse average \
  --save_artifacts "$OUT" \
  --artifact_every 30 \
  --no_viewer \
  $EXTRA
