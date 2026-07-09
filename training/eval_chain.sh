#!/usr/bin/env bash
# Final validation: reconstruct the whole 7-minute garden video with each
# model (same settings as make_viewer.sh, depth saved every 10 frames for the
# consistency check), then score every run against GLOMAP.
#
#   bash training/eval_chain.sh NAME=WEIGHTS [NAME=WEIGHTS ...]
cd "$(dirname "$0")/.."
PY=.venv310/Scripts/python.exe
OUT="${EVAL_DIR:-C:/Users/shaur/gct_data/evalmaps}"
mkdir -p "$OUT"
for pair in "$@"; do
  name="${pair%%=*}"; w="${pair#*=}"
  [ -f "$OUT/$name/meta.json" ] && continue
  "$PY" -u yardmap/reconstruct.py --video_path video/VID_20260913_192111.mp4 --model_path "$w" \
    --use_sdpa --fps 30 --num_scale_frames 12 --anchor_stride 72 --kv_cache_sliding_window 16 \
    --keyframe_interval 30 --voxel_size 0.005 --max_points 4000000 --downsample_factor 4 \
    --conf_threshold 1.5 --crop_margins 0.05,0.14,0.22,0.22 --clahe --min_frame_luma 14 \
    --min_frame_std 18 --mask_sky --sky_model models/skyseg.onnx --fuse average \
    --save_artifacts "$OUT/$name" --artifact_every 10 --no_viewer --stats_every 1000 \
    > "$OUT/$name.log" 2>&1
  "$PY" training/eval_yard.py "$OUT/$name" | tee -a "$OUT/scores.jsonl"
done
