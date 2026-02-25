import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\detect_plants.py"
s = io.open(p, encoding="utf-8").read()
s2 = s.replace('    PROJECT = "k-southwestern-u-s-a"',
               '    PROJECT = "k-world-flora"', 1)
assert s2 != s, "PROJECT default not found"
io.open(p, "w", encoding="utf-8", newline="\n").write(s2)

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
s2 = s.replace('ap.add_argument("--plantnet_project", default="k-southwestern-u-s-a")',
               'ap.add_argument("--plantnet_project", default="k-world-flora")', 1)
assert s2 != s, "viewer plantnet default not found"
io.open(p, "w", encoding="utf-8", newline="\n").write(s2)
print("Pl@ntNet default -> k-world-flora")

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\stream_video.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''def frames_to_world(out, images_bchw, conf_threshold, downsample, conf_percentile=0.0,
                    sky_mask=None):''',
    '''def frames_to_world(out, images_bchw, conf_threshold, downsample, conf_percentile=0.0,
                    sky_mask=None, pin_k=None):'''))

subs.append((
    '''    intrinsic = intrinsic[0].detach().cpu().numpy()             # [S, 3, 3]''',
    '''    intrinsic = intrinsic[0].detach().cpu().numpy()             # [S, 3, 3]
    if pin_k is not None:
        # One physical lens has one focal length. The model re-predicts it every
        # frame and it wanders 26% with a 7.7% trend across this footage, which
        # rescales the unprojected depth and accumulates into trajectory drift.
        # Pinning it costs nothing and removes that source.
        intrinsic[:, 0, 0] = pin_k[0]
        intrinsic[:, 1, 1] = pin_k[1]'''))

subs.append((
    '''    parser.add_argument("--keyframe_interval", type=int, default=1,''',
    '''    parser.add_argument("--pin_fx", type=float, default=0.0,
                        help="Force this focal length (px, model grid) on every "
                             "frame instead of the per-frame prediction")
    parser.add_argument("--pin_fy", type=float, default=0.0,
                        help="Vertical focal length; defaults to --pin_fx")
    parser.add_argument("--keyframe_interval", type=int, default=1,'''))

for old, new in subs:
    if old not in s:
        sys.exit("stream_video NOT FOUND:\n" + old[:200])
    s = s.replace(old, new, 1)

n = s.count('''        pts, cols, extrinsic, intrinsic, raw = frames_to_world(
            out, frame_b, args.conf_threshold, args.downsample_factor,
            args.conf_percentile, sky)''')
s = s.replace('''        pts, cols, extrinsic, intrinsic, raw = frames_to_world(
            out, frame_b, args.conf_threshold, args.downsample_factor,
            args.conf_percentile, sky)''',
              '''        pts, cols, extrinsic, intrinsic, raw = frames_to_world(
            out, frame_b, args.conf_threshold, args.downsample_factor,
            args.conf_percentile, sky, pin_k)''')
print(f"threaded pin_k through {n} call site(s)")

subs2 = [(
    '''    kf = max(1, args.keyframe_interval)''',
    '''    pin_k = ((args.pin_fx, args.pin_fy or args.pin_fx)
             if args.pin_fx > 0 else None)
    if pin_k:
        print(f"Pinning intrinsics: fx={pin_k[0]:.1f} fy={pin_k[1]:.1f}")
    kf = max(1, args.keyframe_interval)''')]
for old, new in subs2:
    if old not in s:
        sys.exit("stream_video NOT FOUND:\n" + old[:160])
    s = s.replace(old, new, 1)

import ast
ast.parse(s)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("stream_video: --pin_fx / --pin_fy added")
