#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=../.venv310/Scripts/python.exe
COMMON="--steps 2500 --eval_every 500 --rank 16 --lora_camera --train_depth_head --stream_p 0.3"
$PY -u train.py --name v2_fpv   $COMMON --fpv_p 0.7 --drone_p 0.0 > runs/v2_fpv.log 2>&1
$PY -u train.py --name v3_drone $COMMON --fpv_p 0.7 --drone_p 0.7 --k1 0.1 > runs/v3_drone.log 2>&1
