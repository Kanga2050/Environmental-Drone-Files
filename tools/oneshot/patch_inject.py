import ast
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\stream_video.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    parser.add_argument("--pin_fx", type=float, default=0.0,''',
    '''    parser.add_argument("--inject_pairs", default=None,
                        help="JSON list of {a,b} source-frame pairs; frame a is "
                             "replayed into context just before frame b")
    parser.add_argument("--inject_repeat", type=int, default=1,
                        help="How many times to replay each injected frame")
    parser.add_argument("--pin_fx", type=float, default=0.0,'''))

subs.append((
    '''    print("Streaming... (Ctrl-C to stop and keep the viewer up)")
    for tensor, rgb in source:''',
    '''    inject = {}
    if args.inject_pairs:
        import json as _json
        pairs = _json.load(open(args.inject_pairs, encoding="utf-8"))
        want = sorted({int(x["a"]) for x in pairs})
        for x in pairs:
            inject.setdefault(int(x["b"]), []).append(int(x["a"]))
        cap2 = cv2.VideoCapture(args.video_path)
        store, got = {}, 0
        for idx in want:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap2.read()
            if not ok:
                continue
            rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if prep is not None:
                rgb2 = prep(rgb2)
            store[idx] = preprocess_frame(rgb2, args.image_size, args.patch_size)
            got += 1
        cap2.release()
        print(f"Injection: {len(pairs)} pairs, {got} context frames preloaded")

    def feed(t, keyframe):
        fb = t.unsqueeze(0).unsqueeze(0).to(device)
        if not keyframe:
            model._set_skip_append(True)
        with torch.no_grad(), autocast:
            model.forward(fb, num_frame_for_scale=scale_n,
                          num_frame_per_block=1, causal_inference=True)
        if not keyframe:
            model._set_skip_append(False)
        del fb

    print("Streaming... (Ctrl-C to stop and keep the viewer up)")
    for tensor, rgb in source:
        if inject:
            here = src_ids[frame_idx] if frame_idx < len(src_ids) else -1
            for a in inject.pop(here, []):
                if a in store:
                    for _ in range(max(1, args.inject_repeat)):
                        feed(store[a], True)'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:160])
    s = s.replace(old, new, 1)
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("injection wired")
