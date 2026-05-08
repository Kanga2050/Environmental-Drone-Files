

import argparse
import os
import signal
import sys
import time

if sys.platform == "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as TF

from lingbot_map.utils.geometry import (
    closed_form_inverse_se3,
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

def make_frame_prep(crop=1.0, undistort_k1=0.0, undistort_fov=120.0, clahe=False,
                    margins=(0.0, 0.0, 0.0, 0.0)):

    def prep(rgb: np.ndarray) -> np.ndarray:
        if undistort_k1:
            h, w = rgb.shape[:2]
            f = w / (2.0 * np.tan(np.radians(undistort_fov) / 2.0))
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
            rgb = cv2.undistort(rgb, K, np.array([undistort_k1, 0, 0, 0, 0], np.float64))
        if any(m > 0 for m in margins):
            h, w = rgb.shape[:2]
            t, b, l, r = margins
            y0, y1 = int(h * t), h - int(h * b)
            x0, x1 = int(w * l), w - int(w * r)
            if y1 - y0 >= 16 and x1 - x0 >= 16:
                rgb = rgb[y0:y1, x0:x1]
        if crop < 1.0:
            h, w = rgb.shape[:2]
            ch, cw = int(h * crop), int(w * crop)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            rgb = rgb[y0:y0 + ch, x0:x0 + cw]
        if clahe:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return rgb
    return prep

def preprocess_frame(rgb: np.ndarray, image_size: int, patch_size: int) -> torch.Tensor:

    img = Image.fromarray(rgb)
    width, height = img.size
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = TF.ToTensor()(img)
    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        tensor = tensor[:, start_y:start_y + image_size, :]
    return tensor

def frame_is_usable(rgb, min_luma, min_std):

    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(g.mean()) >= min_luma and float(g.std()) >= min_std

def iter_video_frames(video_path, fps, stride, max_frames, loop, image_size, patch_size,
                      prep=None, min_luma=0.0, min_std=0.0, on_skip=None,
                      emitted_ids=None):

    emitted = 0
    pass_idx = 0
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, round(src_fps / fps))
        raw_idx = 0
        kept = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if raw_idx % interval == 0:
                if kept % stride == 0:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    if prep is not None:
                        rgb = prep(rgb)
                    if (min_luma > 0 or min_std > 0) and \
                            not frame_is_usable(rgb, min_luma, min_std):
                        if on_skip is not None:
                            on_skip(raw_idx)
                        kept += 1
                        raw_idx += 1
                        continue
                    if emitted_ids is not None:
                        emitted_ids.append(raw_idx)
                    yield preprocess_frame(rgb, image_size, patch_size), rgb
                    emitted += 1
                    if max_frames and emitted >= max_frames:
                        cap.release()
                        return
                kept += 1
            raw_idx += 1
        cap.release()
        pass_idx += 1
        if not loop:
            return
        print(f"[source] --loop: restarting video "
              f"(pass {pass_idx + 1}, {emitted} frames so far)")

def iter_image_folder(image_folder, stride, max_frames, loop, image_size, patch_size,
                      prep=None):

    import glob
    paths = []
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
    paths = sorted(set(paths))[::stride]
    if not paths:
        raise RuntimeError(f"No images found in {image_folder}")
    emitted = 0
    while True:
        for path in paths:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            rgb = np.asarray(img)
            if prep is not None:
                rgb = prep(rgb)
            yield preprocess_frame(rgb, image_size, patch_size), rgb
            emitted += 1
            if max_frames and emitted >= max_frames:
                return
        if not loop:
            return

class BoundedPointMap:

    _BITS = 21
    _MASK = (1 << _BITS) - 1
    _HALF = 1 << (_BITS - 1)

    def __init__(self, voxel_size: float, max_points: int, compact_every: int = 30,
                 average: bool = False):
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        self.compact_every = int(compact_every)
        self.average = bool(average)
        self._pos = np.zeros((0, 3), dtype=np.float32)
        self._col = np.zeros((0, 3), dtype=np.uint8)
        self._age = np.zeros((0,), dtype=np.int64)
        self._key = np.zeros((0,), dtype=np.int64)
        self._psum = np.zeros((0, 3), dtype=np.float32)
        self._csum = np.zeros((0, 3), dtype=np.float32)
        self._w = np.zeros((0,), dtype=np.float32)
        self._n = np.zeros((0,), dtype=np.int32)
        self._first = np.zeros((0,), dtype=np.int32)
        self._pending_pos = []
        self._pending_col = []
        self._pending_age = []
        self._pending_w = []
        self._since_compact = 0
        self.dropped = 0

    def _keys(self, pos: np.ndarray) -> np.ndarray:
        q = np.floor(pos / self.voxel_size).astype(np.int64)
        q = np.clip(q, -self._HALF, self._HALF - 1) + self._HALF
        return ((q[:, 0] & self._MASK)
                | ((q[:, 1] & self._MASK) << self._BITS)
                | ((q[:, 2] & self._MASK) << (2 * self._BITS)))

    def add(self, pos: np.ndarray, col: np.ndarray, age: int,
            weight: np.ndarray = None) -> None:
        if len(pos) == 0:
            return
        self._pending_pos.append(pos.astype(np.float32, copy=False))
        self._pending_col.append(col.astype(np.uint8, copy=False))
        self._pending_age.append(np.full(len(pos), age, dtype=np.int64))
        if self.average:
            self._pending_w.append(
                np.ones(len(pos), np.float32) if weight is None
                else np.asarray(weight, np.float32))
        self._since_compact += 1
        if self._since_compact >= self.compact_every:
            self.compact()

    def compact(self) -> None:

        if not self._pending_pos:
            return
        if self.average:
            return self._compact_average()
        pos = np.concatenate([self._pos] + self._pending_pos, axis=0)
        col = np.concatenate([self._col] + self._pending_col, axis=0)
        age = np.concatenate([self._age] + self._pending_age, axis=0)
        self._pending_pos, self._pending_col, self._pending_age = [], [], []
        self._since_compact = 0

        keys = self._keys(pos)

        _, rev_idx = np.unique(keys[::-1], return_index=True)
        keep = len(keys) - 1 - rev_idx

        if len(keep) > self.max_points:

            newest = np.argpartition(age[keep], -self.max_points)[-self.max_points:]
            self.dropped += len(keep) - self.max_points
            keep = keep[newest]

        self._pos = np.ascontiguousarray(pos[keep])
        self._col = np.ascontiguousarray(col[keep])
        self._age = np.ascontiguousarray(age[keep])

    def _compact_average(self) -> None:

        new_pos = np.concatenate(self._pending_pos, axis=0)
        new_col = np.concatenate(self._pending_col, axis=0).astype(np.float32)
        new_age = np.concatenate(self._pending_age, axis=0)
        new_w = (np.concatenate(self._pending_w, axis=0) if self._pending_w
                 else np.ones(len(new_pos), np.float32))
        self._pending_pos, self._pending_col = [], []
        self._pending_age, self._pending_w = [], []
        self._since_compact = 0

        n_old = len(self._key)
        keys = np.concatenate([self._key, self._keys(new_pos)])
        wsum = np.concatenate([self._w, new_w])
        nsum = np.concatenate([self._n.astype(np.float32),
                               np.ones(len(new_pos), np.float32)])
        psum = np.concatenate([self._psum, new_pos * new_w[:, None]])
        csum = np.concatenate([self._csum, new_col * new_w[:, None]])

        uniq, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(-1)
        m = len(uniq)
        w = np.bincount(inv, weights=wsum, minlength=m).astype(np.float32)
        n = np.bincount(inv, weights=nsum, minlength=m).astype(np.int32)
        ps = np.stack([np.bincount(inv, weights=psum[:, k], minlength=m)
                       for k in range(3)], axis=1).astype(np.float32)
        cs = np.stack([np.bincount(inv, weights=csum[:, k], minlength=m)
                       for k in range(3)], axis=1).astype(np.float32)

        first = np.zeros(m, np.int32)
        last = np.zeros(m, np.int32)
        inv_new = inv[n_old:]
        age32 = new_age.astype(np.int32)
        last[inv_new] = age32
        first[inv_new[::-1]] = age32[::-1]
        if n_old:
            inv_old = inv[:n_old]
            first[inv_old] = self._first
            older = np.zeros(m, np.int32)
            older[inv_old] = self._age.astype(np.int32)
            last = np.maximum(last, older)

        if m > self.max_points:
            newest = np.argpartition(last, -self.max_points)[-self.max_points:]
            self.dropped += m - self.max_points
            uniq, w, n, ps, cs, first, last = (
                uniq[newest], w[newest], n[newest], ps[newest], cs[newest],
                first[newest], last[newest])

        self._key = np.ascontiguousarray(uniq)
        self._w = np.ascontiguousarray(w)
        self._n = np.ascontiguousarray(n)
        self._psum = np.ascontiguousarray(ps)
        self._csum = np.ascontiguousarray(cs)
        self._first = np.ascontiguousarray(first)
        self._age = last.astype(np.int64)
        safe = np.maximum(w, 1e-6)[:, None]
        self._pos = np.ascontiguousarray(ps / safe)
        self._col = np.ascontiguousarray(np.clip(cs / safe, 0, 255).astype(np.uint8))

    def observations(self):

        if not self.average:
            return None
        return (self._first, self._age.astype(np.int32), self._w, self._n)

    def snapshot(self):
        self.compact()
        return self._pos, self._col

    def __len__(self) -> int:
        return len(self._pos) + sum(len(p) for p in self._pending_pos)

    def nbytes(self) -> int:
        return (self._pos.nbytes + self._col.nbytes + self._age.nbytes
                + self._key.nbytes + self._psum.nbytes + self._csum.nbytes
                + self._w.nbytes + self._n.nbytes + self._first.nbytes)

class LiveViewer:

    def __init__(self, port: int, point_size: float, max_traj: int = 4000):
        import viser
        import viser.transforms as vtf
        self._vtf = vtf
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.point_size = point_size
        self.max_traj = max_traj
        self.traj = []
        self._setup_gui()

    def _setup_gui(self):
        self.gui_frames = self.server.gui.add_number("Frames", 0, disabled=True)
        self.gui_points = self.server.gui.add_number("Map points", 0, disabled=True)
        self.gui_fps = self.server.gui.add_number("FPS", 0.0, disabled=True)
        self.gui_vram = self.server.gui.add_number("VRAM (GB)", 0.0, disabled=True)
        self.gui_psize = self.server.gui.add_slider(
            "Point size", min=0.0005, max=0.05, step=0.0005,
            initial_value=self.point_size,
        )
        self.gui_show_traj = self.server.gui.add_checkbox("Show trajectory", True)

    def update(self, points, colors, cam_R=None, cam_t=None, fov=1.0, aspect=1.5,
               frames=0, fps=0.0, vram_gb=0.0):
        self.gui_frames.value = int(frames)
        self.gui_points.value = int(len(points))
        self.gui_fps.value = round(float(fps), 2)
        self.gui_vram.value = round(float(vram_gb), 2)

        if len(points):
            self.server.scene.add_point_cloud(
                name="/map",
                points=points,
                colors=colors,
                point_size=self.gui_psize.value,
            )

        if cam_R is not None:
            quat = self._vtf.SO3.from_matrix(cam_R).wxyz
            self.server.scene.add_camera_frustum(
                name="/camera",
                fov=float(fov),
                aspect=float(aspect),
                wxyz=quat,
                position=cam_t,
                scale=0.08,
                color=(255, 90, 60),
            )
            self.traj.append(np.asarray(cam_t, dtype=np.float32))

            if len(self.traj) > self.max_traj * 2:
                self.traj = self.traj[::2]

        if self.gui_show_traj.value and len(self.traj) > 1:
            self.server.scene.add_spline_catmull_rom(
                name="/trajectory",
                positions=np.stack(self.traj),
                color=(255, 200, 60),
                line_width=2.0,
            )

def build_model(args, device):
    from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        kv_cache_max_context_frames=args.max_context_frames,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    model = model.to(device)

    print(f"Loading checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    del ckpt, state_dict

    if device.type == "cuda":
        dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model.aggregator = model.aggregator.to(dtype=dt)
        torch.cuda.empty_cache()
    return model.eval()

class SkyMasker:

    def __init__(self, model_path, enabled=True, threshold=0.5):
        self.session = None
        self.threshold = float(threshold)
        if not enabled:
            return
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.h = inp.shape[2] if isinstance(inp.shape[2], int) else 384
        self.w = inp.shape[3] if isinstance(inp.shape[3], int) else 384
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"Sky masking: {os.path.basename(str(model_path))} "
              f"on {self.session.get_providers()[0]}")

    def __call__(self, rgb, out_hw):

        if self.session is None:
            return None
        img = cv2.resize(rgb, (self.w, self.h)).astype(np.float32) / 255.0
        inp = ((img - self.mean) / self.std).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: inp})[0]
        if out.ndim == 4 and out.shape[1] == 2:
            prob = out[:, 1]
        elif out.ndim == 4 and out.shape[1] == 1:
            prob = out[:, 0]
        else:
            prob = out
        m = (prob[0] > self.threshold).astype(np.uint8)
        return cv2.resize(m, (out_hw[1], out_hw[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

def frames_to_world(out, images_bchw, conf_threshold, downsample, conf_percentile=0.0,
                    sky_mask=None, pin_k=None):

    hw = images_bchw.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(out["pose_enc"].float(), hw)

    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4),
                                device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic = closed_form_inverse_se3_general(extrinsic_4x4)[..., :3, :4]

    extrinsic = extrinsic[0].detach().cpu().numpy()
    intrinsic = intrinsic[0].detach().cpu().numpy()
    if pin_k is not None:
        intrinsic[:, 0, 0] = pin_k[0]
        intrinsic[:, 1, 1] = pin_k[1]
    depth = out["depth"][0].detach().float().cpu().numpy()
    conf = out["depth_conf"][0].detach().float().cpu().numpy()

    world = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    images = images_bchw[0].detach().float().cpu().numpy()
    colors = np.transpose(images, (0, 2, 3, 1))
    conf_full = conf

    world = world[:, ::downsample, ::downsample, :]
    colors = colors[:, ::downsample, ::downsample, :]
    conf = conf[:, ::downsample, ::downsample]

    pts = world.reshape(-1, 3)
    cols = (np.clip(colors.reshape(-1, 3), 0.0, 1.0) * 255).astype(np.uint8)

    if sky_mask is not None:

        sub = sky_mask[::downsample, ::downsample]
        conf = np.where(sub[None, :, :], 0.0, conf)

    conf_flat = conf.reshape(conf.shape[0], -1)
    if conf_percentile > 0:

        thr = np.percentile(conf_flat, conf_percentile, axis=1, keepdims=True)
        keep = conf_flat > thr
    else:
        keep = conf_flat > conf_threshold

    mask = np.isfinite(pts).all(axis=1) & keep.reshape(-1)

    raw = {"depth": depth, "conf": conf_full, "extrinsic": extrinsic,
           "intrinsic": intrinsic, "images": images}
    raw["kept_conf"] = conf.reshape(-1)[mask]
    return pts[mask], cols[mask], extrinsic, intrinsic, raw

def save_frame_artifacts(frames_dir, frame_ids, raw, rgb_list):

    os.makedirs(frames_dir, exist_ok=True)
    for i, fid in enumerate(frame_ids):
        np.savez_compressed(
            os.path.join(frames_dir, f"{fid:06d}.npz"),
            depth=np.asarray(raw["depth"][i], dtype=np.float16),
            conf=np.asarray(raw["conf"][i], dtype=np.float16),
            extrinsic=np.asarray(raw["extrinsic"][i], dtype=np.float32),
            intrinsic=np.asarray(raw["intrinsic"][i], dtype=np.float32),
        )
        rgb = rgb_list[i]
        if rgb is not None:
            cv2.imwrite(os.path.join(frames_dir, f"{fid:06d}.jpg"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def _trajectory_memory_frames(model) -> int:

    agg = getattr(model, "aggregator", None)
    mgr = getattr(agg, "kv_cache_manager", None)
    if mgr is not None:
        return int(mgr.get_cache_stats(0).get("memory_frames", 0))
    cache = getattr(agg, "kv_cache", None)
    if isinstance(cache, dict) and cache.get("k_0_special") is not None:
        return int(cache["k_0_special"].shape[2])
    return 0

def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(len(points), dtype=dtype)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())

def build_parser():
    parser = argparse.ArgumentParser(
        description="LingBot-Map: open-ended streaming reconstruction with a live web viewer",
    )

    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames per second to sample from the source video")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after N frames (0 = run until the source is exhausted)")
    parser.add_argument("--loop", action="store_true",
                        help="Replay the source forever -- the memory-stability soak test")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_sdpa", action=argparse.BooleanOptionalAction, default=None,
                        help="SDPA attention backend. Default: auto (SDPA unless FlashInfer imports)")

    parser.add_argument("--max_context_frames", type=int, default=256,
                        help="Cap on trajectory-memory frames each frame may attend to. "
                             "-1 disables the cap and lets KV memory grow without bound")
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64,
                        help="Full-detail local window, in keyframes")
    parser.add_argument("--num_scale_frames", type=int, default=8,
                        help="Anchor frames processed bidirectionally up front; never evicted")
    parser.add_argument("--anchor_stride", type=int, default=1,
                        help="Take every Nth source frame when building the anchor pass. "
                             "The anchor is what the whole reconstruction is pinned to, and it "
                             "needs real baselines between its frames -- at high --fps, "
                             "consecutive frames span almost no motion and the anchor is "
                             "degenerate. Set this so num_scale_frames x anchor_stride covers "
                             "several seconds of movement. Skipped frames are consumed, not "
                             "reconstructed")
    parser.add_argument("--inject_pairs", default=None,
                        help="JSON list of {a,b} source-frame pairs; frame a is "
                             "replayed into context just before frame b")
    parser.add_argument("--inject_repeat", type=int, default=1,
                        help="How many times to replay each injected frame")
    parser.add_argument("--pin_fx", type=float, default=0.0,
                        help="Force this focal length (px on the model grid) on "
                             "every frame instead of the per-frame prediction")
    parser.add_argument("--pin_fy", type=float, default=0.0,
                        help="Vertical focal length; defaults to --pin_fx")
    parser.add_argument("--keyframe_interval", type=int, default=1,
                        help="Cache every N-th frame. Non-keyframes still yield predictions. "
                             "Raising this stretches how far --max_frame_num reaches")
    parser.add_argument("--max_frame_num", type=int, default=20000,
                        help="3D-RoPE position table size, counted in KEYFRAMES. Costs ~512 B "
                             "per slot; the run stops when keyframes reach it")

    parser.add_argument("--max_points", type=int, default=3_000_000,
                        help="Hard ceiling on points kept in the global map (~19 B each)")
    parser.add_argument("--fuse", default="latest", choices=["latest", "average"],
                        help="How a revisited voxel is updated. 'latest' takes the "
                             "newest observation; 'average' accumulates a "
                             "confidence-weighted mean of every observation, which "
                             "is what removes analog chroma noise, and records the "
                             "first and last frame that saw each point")
    parser.add_argument("--voxel_size", type=float, default=0.0,
                        help="Voxel edge for map dedup. 0 = auto from the initial scene extent")
    parser.add_argument("--downsample_factor", type=int, default=4,
                        help="Pixel-grid stride when sampling each frame's depth map")
    parser.add_argument("--conf_threshold", type=float, default=1.5,
                        help="Absolute confidence cutoff. Calibrated for clean footage")
    parser.add_argument("--conf_percentile", type=float, default=0.0,
                        help="Keep the top (100-P)%% most confident points per frame, "
                             "instead of an absolute cutoff. Use for footage where "
                             "confidence collapses toward 1.0 (fisheye / action cams) "
                             "and --conf_threshold would drop every point. 0 = off")

    parser.add_argument("--crop", type=float, default=1.0,
                        help="Center-crop this fraction of each source frame before "
                             "preprocessing. Drops burned-in OSD and the worst lens distortion")
    parser.add_argument("--undistort_k1", type=float, default=0.0,
                        help="Radial undistortion coefficient (negative corrects barrel/fisheye)")
    parser.add_argument("--undistort_fov", type=float, default=120.0,
                        help="Assumed horizontal FOV in degrees, used to build the "
                             "undistortion camera matrix")
    parser.add_argument("--crop_margins", default="",
                        help="Trim edges independently as fractions: TOP,BOTTOM,LEFT,RIGHT. "
                             "An OSD is rarely centred, so this clears it without the "
                             "symmetric loss --crop forces. e.g. 0.05,0.14,0,0")
    parser.add_argument("--clahe", action="store_true",
                        help="Local contrast equalization -- helps washed-out analog video")
    parser.add_argument("--min_frame_luma", type=float, default=0.0,
                        help="Skip frames whose mean brightness is below this (0-255). "
                             "Catches the black dropouts an FPV video link produces")
    parser.add_argument("--min_frame_std", type=float, default=0.0,
                        help="Skip frames whose brightness spread is below this. Catches "
                             "washed-out interference, which black-frame tests miss")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Drop sky pixels using the skyseg ONNX model. Strongly "
                             "recommended outdoors: sky reconstructs as a confident "
                             "false surface that later frames then register against")
    parser.add_argument("--sky_model",
                        default=os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                             "lingbot-map", "skyseg_batch.onnx"),
                        help="Path to the sky segmentation ONNX model")
    parser.add_argument("--sky_threshold", type=float, default=0.5,
                        help="Sky probability above which a pixel is dropped. The model "
                             "is only weakly confident on dull overcast sky, so lowering "
                             "this catches hazy horizons at the cost of bright walls")
    parser.add_argument("--compact_every", type=int, default=30)

    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.004)
    parser.add_argument("--vis_every", type=int, default=5,
                        help="Push a viewer update every N frames")
    parser.add_argument("--stats_every", type=int, default=50)
    parser.add_argument("--save_ply", type=str, default=None,
                        help="Write the final map to this .ply path on exit")
    parser.add_argument("--save_artifacts", type=str, default=None,
                        help="Directory to write per-frame depth/confidence/pose plus the "
                             "conditioned frame, and the final cloud. This is what lets "
                             "detection be re-run without redoing reconstruction")
    parser.add_argument("--artifact_every", type=int, default=10,
                        help="Save artifacts for every Nth frame (1 = all). Each saved "
                             "frame costs roughly 0.5 MB")
    parser.add_argument("--no_viewer", action="store_true")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    if not (args.video_path or args.image_folder):
        parser.error("Provide --video_path or --image_folder")

    if args.use_sdpa is None:
        try:
            import flashinfer
            args.use_sdpa = False
        except ImportError:
            args.use_sdpa = True
    print(f"Attention backend: {'SDPA' if args.use_sdpa else 'FlashInfer'}")

    if args.max_context_frames <= 0:
        print("WARNING: --max_context_frames <= 0 -- trajectory memory is unbounded, "
              "so KV memory grows with sequence length and the run will eventually OOM.")
    else:
        kf = max(1, args.keyframe_interval)
        reach = args.max_frame_num * kf
        print(f"Bounded context: {args.num_scale_frames} anchor + "
              f"{args.kv_cache_sliding_window} local window + "
              f"{args.max_context_frames} trajectory memory (keyframes).")
        print(f"RoPE ceiling: {args.max_frame_num} keyframes x keyframe_interval {kf} "
              f"= {reach} frames "
              f"(~{reach / max(args.fps, 1e-6) / 60:.0f} min at {args.fps} fps).")

    pin_k = (args.pin_fx, args.pin_fy or args.pin_fx) if args.pin_fx > 0 else None
    if pin_k:
        print(f"Pinning intrinsics: fx={pin_k[0]:.1f} fy={pin_k[1]:.1f}")
    kf = max(1, args.keyframe_interval)
    win_sec = args.kv_cache_sliding_window * kf / max(args.fps, 1e-6)
    anchor_sec = args.num_scale_frames * max(1, args.anchor_stride) / max(args.fps, 1e-6)
    mem_sec = args.max_context_frames * kf / max(args.fps, 1e-6)

    span = (f"Context in wall-clock terms: anchor {anchor_sec:.1f}s, "
            f"local window {win_sec:.1f}s")
    if args.max_context_frames > 0:
        span += f", trajectory memory {mem_sec:.0f}s"
    print(span)
    if anchor_sec < 3.0:
        print(f"  NOTE: the anchor spans only {anchor_sec:.1f}s. Frames that close "
              f"together carry little parallax, which starves depth confidence. "
              f"Raise --anchor_stride or lower --fps.")
    if win_sec < 2.0:
        print(f"  NOTE: the local window spans only {win_sec:.1f}s of footage. "
              f"Raise --keyframe_interval so each cached slot covers more time.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    import gc
    gc.collect()

    if torch.cuda.is_available():

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    margins = (0.0, 0.0, 0.0, 0.0)
    if args.crop_margins.strip():
        parts = [float(v) for v in args.crop_margins.split(",")]
        if len(parts) != 4:
            parser.error("--crop_margins needs four values: TOP,BOTTOM,LEFT,RIGHT")
        margins = tuple(parts)
    prep = make_frame_prep(args.crop, args.undistort_k1, args.undistort_fov, args.clahe,
                           margins)
    sky_masker = None
    if args.mask_sky:
        if not os.path.exists(args.sky_model):
            parser.error(f"--mask_sky needs the ONNX model; not found at {args.sky_model}")
        sky_masker = SkyMasker(args.sky_model, threshold=args.sky_threshold)
    conditioning = [n for n, on in (
        (f"margins t/b/l/r {'/'.join(f'{m:g}' for m in margins)}", any(margins)),
        (f"crop {args.crop:g}", args.crop < 1.0),
        (f"undistort k1={args.undistort_k1:g} fov={args.undistort_fov:g}", bool(args.undistort_k1)),
        ("CLAHE", args.clahe),
        ("sky masking", bool(args.mask_sky)),
        (f"skip frames under luma {args.min_frame_luma:g} / std {args.min_frame_std:g}",
         args.min_frame_luma > 0 or args.min_frame_std > 0)) if on]
    if conditioning:
        print("Source conditioning: " + ", ".join(conditioning))
    if args.conf_percentile > 0:
        print(f"Confidence gate: top {100 - args.conf_percentile:g}% per frame "
              f"(percentile mode; --conf_threshold ignored)")

    if args.video_path:
        skipped = {"n": 0}

        def note_skip(idx, _st=skipped):
            _st["n"] += 1
            if _st["n"] <= 5 or _st["n"] % 100 == 0:
                print(f"[source] skipped unusable frame {idx} "
                      f"({_st['n']} so far)", flush=True)

        src_ids = []
        source = iter_video_frames(args.video_path, args.fps, args.stride,
                                   args.max_frames, args.loop,
                                   args.image_size, args.patch_size, prep,
                                   args.min_frame_luma, args.min_frame_std, note_skip,
                                   src_ids)
    else:
        src_ids = []
        source = iter_image_folder(args.image_folder, args.stride, args.max_frames,
                                   args.loop, args.image_size, args.patch_size, prep)

    viewer = None
    if not args.no_viewer:
        viewer = LiveViewer(args.port, args.point_size)
        print(f"\n  Viewer ready:  http://localhost:{args.port}\n")

    stop = {"flag": False}

    def _on_sigint(signum, frame):
        if stop["flag"]:
            sys.exit(1)
        stop["flag"] = True
        print("\nStopping after this frame (Ctrl-C again to force quit)...")

    signal.signal(signal.SIGINT, _on_sigint)

    scale_n = args.num_scale_frames
    if device.type == "cuda":
        autocast = torch.amp.autocast("cuda", dtype=dtype)
    else:
        autocast = torch.amp.autocast("cpu", enabled=False)

    want_artifacts = args.save_artifacts is not None
    art_dir = args.save_artifacts
    art_frames_dir = os.path.join(art_dir, "frames") if want_artifacts else None
    saved_ids = []
    if want_artifacts:
        os.makedirs(art_dir, exist_ok=True)
        print(f"Saving artifacts every {args.artifact_every} frames -> {art_dir}")

    model.clean_kv_cache()
    frame_idx = 0
    t_start = time.time()
    t_window = time.time()
    fps_est = 0.0

    anchor_stride = max(1, args.anchor_stride)
    scale_frames, scale_rgb = [], []
    consumed = 0
    for tensor, rgb in source:
        if consumed % anchor_stride == 0:
            scale_frames.append(tensor)
            scale_rgb.append(rgb if want_artifacts else None)
            if len(scale_frames) >= scale_n:
                consumed += 1
                break
        consumed += 1
    if anchor_stride > 1:
        print(f"Anchor: {len(scale_frames)} frames taken every {anchor_stride} "
              f"({consumed} source frames consumed, "
              f"~{consumed / max(args.fps, 1e-6):.1f}s of motion)")
    if not scale_frames:
        print("No frames decoded -- nothing to do.")
        return
    scale_n = len(scale_frames)
    scale_batch = torch.stack(scale_frames).unsqueeze(0).to(device)
    del scale_frames

    print(f"Processing {scale_n} anchor frames...")
    with torch.no_grad(), autocast:
        out = model.forward(scale_batch, num_frame_for_scale=scale_n,
                            num_frame_per_block=scale_n, causal_inference=True)
    pts, cols, extrinsic, intrinsic, raw = frames_to_world(
        out, scale_batch, args.conf_threshold, args.downsample_factor,
        args.conf_percentile, None, pin_k)
    del out

    if want_artifacts:

        ids = list(range(scale_n))
        save_frame_artifacts(art_frames_dir, ids, raw, scale_rgb)
        saved_ids.extend(ids)

    if args.voxel_size > 0:
        voxel = args.voxel_size
        print(f"Voxel size: {voxel:.5f}")
    else:

        if len(pts):
            lo, hi = np.percentile(pts, [5, 95], axis=0)
            extent = float(np.linalg.norm(hi - lo))
        else:
            extent = 1.0
        voxel = max(extent / 512.0, 1e-6)
        print(f"Voxel size: {voxel:.5f} (auto, from a {extent:.2f} initial extent)")

    point_map = BoundedPointMap(voxel, args.max_points, args.compact_every,
                                average=(args.fuse == "average"))
    if args.fuse == "average":
        print("Fusion   : confidence-weighted mean per voxel, with provenance")
    point_map.add(pts, cols, frame_idx, raw.get("kept_conf"))
    frame_idx = scale_n

    H, W = scale_batch.shape[-2:]
    aspect = W / H
    fov = 2 * float(np.arctan(intrinsic[-1, 1, 2] / intrinsic[-1, 1, 1]))
    del scale_batch, pts, cols

    def cam_pose(extr):

        c2w = closed_form_inverse_se3(extr[None])[0]
        return c2w[:3, :3], c2w[:3, 3]

    if viewer:
        R, t = cam_pose(extrinsic[-1])
        p, c = point_map.snapshot()
        viewer.update(p, c, R, t, fov, aspect, frames=frame_idx)

    inject = {}
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
                        feed(store[a], True)
        if stop["flag"]:
            break

        agg_pos = getattr(model.aggregator, "total_frames_processed", 0)
        if agg_pos >= args.max_frame_num - 1:
            print(f"\nReached the 3D-RoPE ceiling at {agg_pos} keyframes "
                  f"(--max_frame_num={args.max_frame_num}). Stopping.\n"
                  f"To go further: raise --max_frame_num (~512 B per slot) or raise "
                  f"--keyframe_interval so each slot covers more frames.")
            break

        is_keyframe = (args.keyframe_interval <= 1) or \
            ((frame_idx - scale_n) % args.keyframe_interval == 0)
        if not is_keyframe:
            model._set_skip_append(True)

        frame_b = tensor.unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad(), autocast:
            out = model.forward(frame_b, num_frame_for_scale=scale_n,
                                num_frame_per_block=1, causal_inference=True)
        if not is_keyframe:
            model._set_skip_append(False)

        sky = sky_masker(rgb, tuple(frame_b.shape[-2:])) if sky_masker else None
        pts, cols, extrinsic, intrinsic, raw = frames_to_world(
            out, frame_b, args.conf_threshold, args.downsample_factor,
            args.conf_percentile, sky, pin_k)
        del out, frame_b

        if want_artifacts and (frame_idx % args.artifact_every == 0):
            save_frame_artifacts(art_frames_dir, [frame_idx], raw, [rgb])
            saved_ids.append(frame_idx)

        point_map.add(pts, cols, frame_idx, raw.get("kept_conf"))
        frame_idx += 1

        if viewer and frame_idx % args.vis_every == 0:
            now = time.time()
            fps_est = args.vis_every / max(now - t_window, 1e-6)
            t_window = now
            R, t = cam_pose(extrinsic[-1])
            p, c = point_map.snapshot()
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            viewer.update(p, c, R, t, fov, aspect,
                          frames=frame_idx, fps=fps_est, vram_gb=vram)

        if args.stats_every and frame_idx % args.stats_every == 0:
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"[{frame_idx:6d}] {fps_est:5.2f} fps | "
                  f"VRAM {vram:5.2f} GB (peak {peak:5.2f}) | "
                  f"map {len(point_map):>9,} pts / {point_map.nbytes() / 1e6:6.1f} MB "
                  f"(dropped {point_map.dropped:,}) | "
                  f"traj-mem {_trajectory_memory_frames(model)} frames")

    elapsed = time.time() - t_start
    p, c = point_map.snapshot()
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.2f} fps average)")
    print(f"Final map: {len(p):,} points ({point_map.nbytes() / 1e6:.1f} MB), "
          f"{point_map.dropped:,} voxels aged out")

    if args.save_ply:
        save_ply(args.save_ply, p, c)
        print(f"Wrote {args.save_ply}")

    if want_artifacts:
        import json
        cloud_extra = {}
        obs = point_map.observations()
        if obs is not None:
            first, last, weight, count = obs
            cloud_extra = {"first_frame": first, "last_frame": last,
                           "obs_weight": weight.astype(np.float32),
                           "obs_count": count.astype(np.int32)}
        np.savez_compressed(os.path.join(art_dir, "cloud.npz"), points=p, colors=c,
                            **cloud_extra)
        meta = {
            "model_hw": [int(H), int(W)],
            "num_frames": int(frame_idx),
            "saved_frame_ids": saved_ids,
            "source_frame_ids": [int(x) for x in src_ids],
            "voxel_size": float(point_map.voxel_size),
            "conf_threshold": float(args.conf_threshold),
            "conf_percentile": float(args.conf_percentile),
            "source": args.video_path or args.image_folder,
            "fps": float(args.fps),
            "crop": float(args.crop),
            "crop_margins": list(margins),
            "clahe": bool(args.clahe),
            "num_scale_frames": int(scale_n),
            "anchor_stride": int(anchor_stride),
            "keyframe_interval": int(args.keyframe_interval),
        }
        with open(os.path.join(art_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        size_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(art_dir) for fn in fns
        ) / 1e6
        print(f"Artifacts: {len(saved_ids)} frames + cloud -> {art_dir} ({size_mb:.1f} MB)")

    if viewer:
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        R, t = cam_pose(extrinsic[-1])
        viewer.update(p, c, R, t, fov, aspect, frames=frame_idx,
                      fps=fps_est, vram_gb=vram)
        print(f"\n  Viewer still serving at http://localhost:{args.port}  (Ctrl-C to exit)\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
