#!/usr/bin/env bash
# V4: aimed at long-range drift - more and longer streaming sequences.
cd "$(dirname "$0")"
F=C:/Users/shaur/gct_data/evalmaps/scores.jsonl
true
sleep 30
PY=../.venv310/Scripts/python.exe
R="$(pwd)/runs"
$PY -u train.py --name v4_stream --steps 3000 --eval_every 1000 --rank 16 --lora_camera \
  --train_depth_head --stream_p 0.6 --stream_anchor 4 --stream_len 8 --stream_grad_last 4 --stride 2,10 \
  --lr 5e-5 --fpv_p 0.8 --drone_p 0.8 --k1 0.1 > runs/v4_stream.log 2>&1
bash eval_chain.sh v4_stream="$R/v4_stream/gct_merged.pt" > runs/eval2.log 2>&1
