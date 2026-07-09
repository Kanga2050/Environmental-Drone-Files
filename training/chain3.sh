#!/usr/bin/env bash
# After chain2: garden-video validation of the first models, then V4, then its validation.
cd "$(dirname "$0")"
until grep -q "saved" runs/v3_drone.log 2>/dev/null; do sleep 60; done
sleep 60
PY=../.venv310/Scripts/python.exe
R="$(pwd)/runs"
bash eval_chain.sh base="$(pwd)/../models/gct.pt" v1_clean="$R/v1_clean/gct_merged.pt" \
  v2_fpv="$R/v2_fpv/gct_merged.pt" v3_drone="$R/v3_drone/gct_merged.pt" > runs/eval1.log 2>&1
$PY -u train.py --name v4_drone_r32 --steps 4000 --eval_every 1000 --rank 32 --lora_camera \
  --train_depth_head --stream_p 0.5 --fpv_p 0.9 --drone_p 0.9 --k1 0.1 > runs/v4_drone_r32.log 2>&1
bash eval_chain.sh v4_drone_r32="$R/v4_drone_r32/gct_merged.pt" > runs/eval2.log 2>&1
