
import sys

import numpy as np
import torch

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
import stream_video as sv

VID = r"C:\Users\shaur\Downloads\VID_20260913_192111.mp4"
CKPT = r"C:\Users\shaur\AppData\Local\lingbot-map\lingbot-map.pt"

class A:
    pass

a = A()
a.image_size, a.patch_size = 518, 14
a.max_frame_num, a.kv_cache_sliding_window = 20000, 32
a.max_context_frames, a.use_sdpa, a.camera_num_iterations = 256, True, 4
a.num_scale_frames = 32
a.model_path = CKPT

dev = torch.device("cuda")
model = sv.build_model(a, dev)
print("model ready\n", flush=True)

SCALE, ASTRIDE, KF, FPS, NMAX = 32, 10, 10, 30.0, 700
LUMA, STD = 14.0, 18.0

CONFIGS = [
    ("A  old-style crop 0.75 + clahe", dict(crop=0.75, clahe=True)),
    ("B  margins t.05 b.14, full width + clahe",
     dict(clahe=True, margins=(0.05, 0.14, 0.0, 0.0))),
    ("C  margins t.05 b.14 l.08 r.08 + clahe",
     dict(clahe=True, margins=(0.05, 0.14, 0.08, 0.08))),
    ("D  margins t.05 b.14 l.15 r.15 + clahe",
     dict(clahe=True, margins=(0.05, 0.14, 0.15, 0.15))),
    ("E  margins t.05 b.14, full width, no clahe",
     dict(margins=(0.05, 0.14, 0.0, 0.0))),
]

for label, kw in CONFIGS:
    prep = sv.make_frame_prep(**kw)
    src = sv.iter_video_frames(VID, FPS, 1, NMAX, False, 518, 14, prep,
                               min_luma=LUMA, min_std=STD)

    anchor, consumed = [], 0
    for t, _rgb in src:
        if consumed % ASTRIDE == 0:
            anchor.append(t)
            if len(anchor) >= SCALE:
                consumed += 1
                break
        consumed += 1
    if len(anchor) < SCALE:
        print(f"{label}: only {len(anchor)} anchor frames, skipped")
        continue

    batch = torch.stack(anchor).unsqueeze(0).to(dev)
    hw = tuple(batch.shape[-2:])
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward(batch, num_frame_for_scale=SCALE,
                            num_frame_per_block=SCALE, causal_inference=True)
    del out, batch

    p90s, fracs, i = [], [], 0
    for t, _rgb in src:
        is_kf = (i % KF == 0)
        if not is_kf:
            model._set_skip_append(True)
        fb = t.unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model.forward(fb, num_frame_for_scale=SCALE,
                              num_frame_per_block=1, causal_inference=True)
        if not is_kf:
            model._set_skip_append(False)
        c = o["depth_conf"][0, 0].float().cpu().numpy()
        p90s.append(np.percentile(c, 90))
        fracs.append((c > 1.5).mean())
        del o, fb
        i += 1

    px = hw[0] * hw[1]
    print(f"{label:44s} {hw[1]}x{hw[0]} ({px/1000:.0f}k px) | streamed {i:3d} | "
          f"conf p90 {np.mean(p90s):6.3f} | frac>1.5 {np.mean(fracs):.3f}", flush=True)
    torch.cuda.empty_cache()
