"""Fold a saved LoRA adapter (and depth head) into a plain GCT checkpoint.

    python merge_adapter.py runs/NAME [out.pt]"""
import os
import sys

import torch

import common
import lora

run = sys.argv[1]
ad = torch.load(os.path.join(run, "adapter.pt"), map_location="cpu", weights_only=False)
a = ad["args"]
model = common.build(device="cpu")
model.aggregator.to(torch.bfloat16)
lora.add_lora(model.aggregator.frame_blocks, a["rank"], a["rank"])
lora.add_lora(model.aggregator.global_blocks, a["rank"], a["rank"])
if a.get("lora_camera"):
    lora.add_lora(model.camera_head, a["rank"], a["rank"])
missing = model.load_state_dict(ad["lora"], strict=False)
assert not missing.unexpected_keys, missing.unexpected_keys[:3]
if ad.get("depth_head") is not None:
    model.depth_head.load_state_dict(ad["depth_head"])
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, "gct_merged.pt")
torch.save({"model": lora.merged_state_dict(model.eval())}, out)
print("step", ad["step"], "->", out)
