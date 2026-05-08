
import sys

import numpy as np
import torch

REPO = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map"
sys.path.insert(0, REPO)
import stream_video as sv

VID = r"C:\Users\shaur\Downloads\VID_20260913_192111.mp4"
MODELS = [
    ("balanced  anchor24/win32", r"C:\Users\shaur\AppData\Local\lingbot-map\lingbot-map.pt"),
    ("long      anchor24/win32", r"C:\Users\shaur\AppData\Local\lingbot-map\lingbot-map-long.pt"),
]

SCALE, ASTRIDE, KF, FPS, NMAX = 24, 12, 10, 10.0, 1000

class A:
    pass

for label, ckpt in MODELS:
    a = A()
    a.image_size, a.patch_size = 518, 14
    a.max_frame_num, a.kv_cache_sliding_window = 20000, 32
    a.max_context_frames, a.use_sdpa, a.camera_num_iterations = 32, True, 4
    a.num_scale_frames = SCALE
    a.model_path = ckpt
    dev = torch.device("cuda")
    model = sv.build_model(a, dev)

    prep = sv.make_frame_prep(clahe=True, margins=(0.05, 0.14, 0.22, 0.22))
    src = sv.iter_video_frames(VID, FPS, 1, NMAX, False, 518, 14, prep,
                               min_luma=14.0, min_std=18.0)
    anchor, consumed = [], 0
    for t, _ in src:
        if consumed % ASTRIDE == 0:
            anchor.append(t)
            if len(anchor) >= SCALE:
                consumed += 1
                break
        consumed += 1
    batch = torch.stack(anchor).unsqueeze(0).to(dev)
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.forward(batch, num_frame_for_scale=SCALE,
                      num_frame_per_block=SCALE, causal_inference=True)
    del batch

    p90s, fracs, dead, i = [], [], 0, 0
    for t, _ in src:
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
        f = float((c > 1.5).mean())
        p90s.append(np.percentile(c, 90))
        fracs.append(f)
        dead += (f < 0.05)
        del o, fb
        i += 1
    print(f"{label:32s} streamed {i:4d} | conf p90 {np.mean(p90s):6.3f} | "
          f"usable {np.mean(fracs):.3f} | dead frames {dead:3d} ({dead / max(i,1):.1%}) | "
          f"VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    del model
    torch.cuda.empty_cache()
