
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\stream_video.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    raw = {"depth": depth, "conf": conf_full, "extrinsic": extrinsic,
           "intrinsic": intrinsic, "images": images}
    return pts[mask], cols[mask], extrinsic, intrinsic, raw''',
    '''    raw = {"depth": depth, "conf": conf_full, "extrinsic": extrinsic,
           "intrinsic": intrinsic, "images": images}
    # The surviving confidences ride along so fusion can weight by them. They
    # are the model's own certainty per pixel, which is the only per-observation
    # quality signal available without a second pass.
    raw["kept_conf"] = conf.reshape(-1)[mask]
    return pts[mask], cols[mask], extrinsic, intrinsic, raw'''))

subs.append((
    '''        point_map.add(pts, cols, frame_idx)
        frame_idx += 1''',
    '''        point_map.add(pts, cols, frame_idx, raw.get("kept_conf"))
        frame_idx += 1'''))

subs.append((
    '''    parser.add_argument("--voxel_size", type=float, default=0.0,''',
    '''    parser.add_argument("--fuse", default="latest", choices=["latest", "average"],
                        help="How a revisited voxel is updated. 'latest' takes the "
                             "newest observation; 'average' accumulates a "
                             "confidence-weighted mean of every observation, which "
                             "is what removes analog chroma noise, and records the "
                             "first and last frame that saw each point")
    parser.add_argument("--voxel_size", type=float, default=0.0,'''))

subs.append((
    '''    point_map = BoundedPointMap(voxel, args.max_points, args.compact_every)''',
    '''    point_map = BoundedPointMap(voxel, args.max_points, args.compact_every,
                                average=(args.fuse == "average"))
    if args.fuse == "average":
        print("Fusion   : confidence-weighted mean per voxel, with provenance")'''))

subs.append((
    '''        np.savez_compressed(os.path.join(art_dir, "cloud.npz"), points=p, colors=c)''',
    '''        cloud_extra = {}
        obs = point_map.observations()
        if obs is not None:
            first, last, weight = obs
            # Provenance is what lets a viewer answer "which frames made this
            # point" exactly, instead of re-deriving it by reprojection against
            # the handful of frames that happened to be saved to disk.
            cloud_extra = {"first_frame": first, "last_frame": last,
                           "obs_weight": weight.astype(np.float32)}
        np.savez_compressed(os.path.join(art_dir, "cloud.npz"), points=p, colors=c,
                            **cloud_extra)'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:220])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("wired: frames_to_world, add(), --fuse, save")
