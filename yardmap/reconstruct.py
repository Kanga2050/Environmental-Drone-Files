

import argparse
import json
import os
import signal
import socket
import sys
import time
from collections import deque

if sys.platform == "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

from gct.utils.geometry import (
    closed_form_inverse_se3,
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)
from gct.utils.pose_enc import pose_encoding_to_extri_intri

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
    from gct.models.gct_stream import GCTStream

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
        description="Streaming GCT reconstruction of a video, with an optional live web viewer",
    )

    parser.add_argument("--video_path", type=str, default=None,
                        help="Video file, or a camera index (0, 1, ...) for a live device")
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
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             "models", "skyseg.onnx"),
                        help="Path to the sky segmentation ONNX model")
    parser.add_argument("--sky_threshold", type=float, default=0.5,
                        help="Sky probability above which a pixel is dropped. The model "
                             "is only weakly confident on dull overcast sky, so lowering "
                             "this catches hazy horizons at the cost of bright walls")
    parser.add_argument("--compact_every", type=int, default=30)
    parser.add_argument("--tap", type=str, default=None,
                        help="HOST:PORT to send per-frame map quality to fly/ctl.py")
    parser.add_argument("--snap", type=str, default=None,
                        help="Write the live map to this .npz for fly/ctl.py scans")
    parser.add_argument("--snap_every", type=int, default=60)
    parser.add_argument("--snap_win", type=int, default=120)
    parser.add_argument("--snap_points", type=int, default=600000)

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

def iter_video_rgb(video_path, fps, stride=1, max_frames=0, loop=False):
    """Raw RGB frames of a video file (or camera index) sampled at about `fps`.
    Yields (source frame index, rgb)."""
    emitted = 0
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, round(src_fps / fps))
        raw_idx = kept = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if raw_idx % interval == 0:
                if kept % stride == 0:
                    yield raw_idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    emitted += 1
                    if max_frames and emitted >= max_frames:
                        cap.release()
                        return
                kept += 1
            raw_idx += 1
        cap.release()
        if not loop:
            return


def _c2w(extr):
    """Camera-to-world 4x4 from a world-to-camera 3x4 extrinsic."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :4] = extr
    return np.linalg.inv(m).astype(np.float32)


class StreamMapper:
    """Frame-at-a-time GCT reconstruction.

    push(rgb) conditions a raw frame, collects the anchor (num_scale_frames
    frames taken every anchor_stride), then streams every later frame through
    the model and fuses its confident depth into a voxel map.  Used by the
    command line below, the notebook and the live camera script."""

    def __init__(self, model_path=None, model=None, device=None, image_size=518, patch_size=14,
                 num_scale_frames=12, anchor_stride=72, kv_cache_sliding_window=16,
                 max_context_frames=256, keyframe_interval=30, max_frame_num=20000,
                 camera_num_iterations=4, use_sdpa=True, voxel_size=0.005,
                 max_points=4_000_000, fuse="average", downsample_factor=4,
                 conf_threshold=1.5, conf_percentile=0.0, crop=1.0, undistort_k1=0.0,
                 undistort_fov=120.0, crop_margins=(0.0, 0.0, 0.0, 0.0), clahe=False,
                 min_frame_luma=0.0, min_frame_std=0.0, sky_model=None, sky_threshold=0.5,
                 compact_every=30, pin_k=None, save_artifacts=None, artifact_every=30,
                 source=None, fps=30.0, gpu_fraction=0.8):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda":
            # The KV cache is rebuilt by concatenation every frame, which leaves
            # PyTorch's cache of freed blocks growing until, on Windows, the
            # driver starts paging GPU memory to system RAM and a frame takes
            # 10x longer.  A cap makes the allocator reuse its cache instead.
            torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        self.cfg = dict(image_size=image_size, patch_size=patch_size,
                        num_scale_frames=num_scale_frames, anchor_stride=max(1, anchor_stride),
                        keyframe_interval=max(1, keyframe_interval), max_frame_num=max_frame_num,
                        downsample=downsample_factor, conf_threshold=conf_threshold,
                        conf_percentile=conf_percentile, min_luma=min_frame_luma,
                        min_std=min_frame_std, artifact_every=artifact_every,
                        voxel_size=voxel_size, max_points=max_points, fuse=fuse,
                        compact_every=compact_every, crop=crop, crop_margins=list(crop_margins),
                        clahe=clahe, source=source, fps=fps)
        if model is None:
            ns = argparse.Namespace(image_size=image_size, patch_size=patch_size,
                                    max_frame_num=max_frame_num,
                                    kv_cache_sliding_window=kv_cache_sliding_window,
                                    num_scale_frames=num_scale_frames,
                                    max_context_frames=max_context_frames, use_sdpa=use_sdpa,
                                    camera_num_iterations=camera_num_iterations,
                                    model_path=model_path)
            model = build_model(ns, self.device)
        self.model = model
        self.dtype = (torch.bfloat16 if self.device.type == "cuda"
                      and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
        self.prep = make_frame_prep(crop, undistort_k1, undistort_fov, clahe, tuple(crop_margins))
        self.sky = SkyMasker(sky_model, threshold=sky_threshold) if sky_model else None
        self.pin_k = pin_k
        self.art = save_artifacts
        if self.art:
            os.makedirs(os.path.join(self.art, "frames"), exist_ok=True)
        self.reset()

    def reset(self):
        self.model.clean_kv_cache()
        if hasattr(self.model.aggregator, "total_frames_processed"):
            self.model.aggregator.total_frames_processed = 0
        self.anchor, self.anchor_rgb, self.anchor_src = [], [], []
        self.consumed = 0
        self.frame_idx = 0              # frames reconstructed so far
        self.point_map = None
        self.saved_ids, self.frame_src = [], []
        self.src_ids = []               # every usable source frame seen
        self.skipped = 0
        self.poses = []                 # camera-to-world 4x4 per reconstructed frame
        self.K = None
        self.hw = None
        self.t0 = time.time()

    @property
    def started(self):
        return self.point_map is not None

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=self.dtype)
        return torch.amp.autocast("cpu", enabled=False)

    def push(self, rgb, src_id=None):
        """Feed one raw RGB frame.  Returns None while the anchor is still being
        collected (or the frame was unusable), else a dict for the newest
        reconstructed frame: points, colors, c2w, K, depth, conf, frame_idx."""
        c = self.cfg
        src_id = len(self.src_ids) + self.skipped if src_id is None else int(src_id)
        rgb = self.prep(rgb)
        if (c["min_luma"] > 0 or c["min_std"] > 0) and \
                not frame_is_usable(rgb, c["min_luma"], c["min_std"]):
            self.skipped += 1
            return None
        self.src_ids.append(src_id)
        tensor = preprocess_frame(rgb, c["image_size"], c["patch_size"])
        if not self.started:
            if self.consumed % c["anchor_stride"] == 0:
                self.anchor.append(tensor)
                self.anchor_rgb.append(rgb)
                self.anchor_src.append(src_id)
            self.consumed += 1
            if len(self.anchor) >= c["num_scale_frames"]:
                return self._run_anchor()
            return None
        return self._run_frame(tensor, rgb, src_id)

    def flush(self):
        """Start from a short anchor if the source ended before it filled."""
        if not self.started and self.anchor:
            return self._run_anchor()
        return None

    def _keep(self, raw, sky_masks):
        """Confident, finite, non-sky points of every frame in raw."""
        c = self.cfg
        d = c["downsample"]
        world = unproject_depth_map_to_point_map(raw["depth"], raw["extrinsic"],
                                                 raw["intrinsic"])[:, ::d, ::d]
        colors = np.transpose(raw["images"], (0, 2, 3, 1))[:, ::d, ::d]
        conf = raw["conf"][:, ::d, ::d].astype(np.float32).copy()
        for i, m in enumerate(sky_masks):
            if m is not None:
                conf[i][m[::d, ::d]] = 0.0
        p = world.reshape(-1, 3)
        col = (np.clip(colors.reshape(-1, 3), 0, 1) * 255).astype(np.uint8)
        cf = conf.reshape(conf.shape[0], -1)
        if c["conf_percentile"] > 0:
            ok = cf > np.percentile(cf, c["conf_percentile"], axis=1, keepdims=True)
        else:
            ok = cf > c["conf_threshold"]
        ok = ok.reshape(-1) & np.isfinite(p).all(1)
        return p[ok], col[ok], cf.reshape(-1)[ok]

    def _infer(self, batch, n_block):
        with torch.no_grad(), self._autocast():
            out = self.model.forward(batch, num_frame_for_scale=self.cfg["num_scale_frames"]
                                     if n_block == 1 else n_block,
                                     num_frame_per_block=n_block, causal_inference=True)
        _, _, _, _, raw = frames_to_world(out, batch, self.cfg["conf_threshold"], 1, 0.0,
                                          None, self.pin_k)
        return raw

    def _run_anchor(self):
        c = self.cfg
        n = len(self.anchor)
        c["num_scale_frames"] = n
        batch = torch.stack(self.anchor).unsqueeze(0).to(self.device)
        self.hw = tuple(batch.shape[-2:])
        raw = self._infer(batch, n)
        del batch
        skies = [self.sky(r, self.hw) if self.sky else None for r in self.anchor_rgb]
        pts, cols, w = self._keep(raw, skies)
        voxel = c["voxel_size"]
        if voxel <= 0:
            ext = float(np.linalg.norm(np.subtract(*np.percentile(pts, [95, 5], 0)))) \
                if len(pts) else 1.0
            voxel = max(ext / 512.0, 1e-6)
        self.point_map = BoundedPointMap(voxel, c["max_points"], c["compact_every"],
                                         average=(c["fuse"] == "average"))
        self.point_map.add(pts, cols, 0, w)
        ids = list(range(n))
        if self.art:
            save_frame_artifacts(os.path.join(self.art, "frames"), ids, raw, self.anchor_rgb)
            self.saved_ids.extend(ids)
        self.frame_src.extend(self.anchor_src)
        self.poses.extend(_c2w(e) for e in raw["extrinsic"])
        self.K = raw["intrinsic"][-1]
        self.frame_idx = n
        self.last_rgb = self.anchor_rgb[-1]
        self.anchor, self.anchor_rgb = [], []
        return {"points": pts, "colors": cols, "c2w": self.poses[-1], "K": self.K,
                "frame_idx": n - 1, "anchor": True, "depth": raw["depth"][-1],
                "conf": raw["conf"][-1], "rgb": self.last_rgb}

    def _run_frame(self, tensor, rgb, src_id):
        c = self.cfg
        if getattr(self.model.aggregator, "total_frames_processed", 0) >= c["max_frame_num"] - 1:
            raise RuntimeError("Reached the 3D-RoPE ceiling; raise max_frame_num")
        is_key = (c["keyframe_interval"] <= 1) or \
            ((self.frame_idx - c["num_scale_frames"]) % c["keyframe_interval"] == 0)
        if not is_key:
            self.model._set_skip_append(True)
        fb = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        raw = self._infer(fb, 1)
        if not is_key:
            self.model._set_skip_append(False)
        sky = self.sky(rgb, tuple(fb.shape[-2:])) if self.sky else None
        del fb
        pts, cols, w = self._keep(raw, [sky])
        fid = self.frame_idx
        if self.art and fid % c["artifact_every"] == 0:
            save_frame_artifacts(os.path.join(self.art, "frames"), [fid], raw, [rgb])
            self.saved_ids.append(fid)
        self.point_map.add(pts, cols, fid, w)
        self.frame_src.append(src_id)
        self.poses.append(_c2w(raw["extrinsic"][-1]))
        self.K = raw["intrinsic"][-1]
        self.frame_idx += 1
        self.last_rgb = rgb
        return {"points": pts, "colors": cols, "c2w": self.poses[-1], "K": self.K,
                "frame_idx": fid, "anchor": False, "depth": raw["depth"][-1],
                "conf": raw["conf"][-1], "keyframe": is_key, "rgb": rgb, "w": w}

    def snapshot(self):
        if self.point_map is None:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
        return self.point_map.snapshot()

    def save(self, art_dir=None, ply=None):
        """Write cloud.npz, meta.json and poses.npy (what the plant steps read)."""
        import json
        art_dir = art_dir or self.art
        p, col = self.snapshot()
        if ply:
            save_ply(ply, p, col)
        if not art_dir:
            return
        os.makedirs(art_dir, exist_ok=True)
        extra = {}
        obs = self.point_map.observations() if self.point_map is not None else None
        if obs is not None:
            first, last, weight, count = obs
            extra = {"first_frame": first, "last_frame": last,
                     "obs_weight": weight.astype(np.float32), "obs_count": count.astype(np.int32)}
        np.savez_compressed(os.path.join(art_dir, "cloud.npz"), points=p, colors=col, **extra)
        c = self.cfg
        meta = {"model_hw": [int(self.hw[0]), int(self.hw[1])] if self.hw else None,
                "num_frames": int(self.frame_idx), "saved_frame_ids": self.saved_ids,
                "source_frame_ids": self.src_ids, "frame_source_ids": self.frame_src,
                "voxel_size": float(self.point_map.voxel_size) if self.point_map else None,
                "conf_threshold": float(c["conf_threshold"]),
                "conf_percentile": float(c["conf_percentile"]), "source": c["source"],
                "fps": float(c["fps"]), "crop": float(c["crop"]),
                "crop_margins": c["crop_margins"], "clahe": bool(c["clahe"]),
                "num_scale_frames": int(c["num_scale_frames"]),
                "anchor_stride": int(c["anchor_stride"]),
                "keyframe_interval": int(c["keyframe_interval"])}
        with open(os.path.join(art_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.poses:
            np.save(os.path.join(art_dir, "poses.npy"), np.stack(self.poses).astype(np.float32))


def mapper_from_args(args):
    """StreamMapper configured from the command-line arguments below."""
    margins = (0.0, 0.0, 0.0, 0.0)
    if args.crop_margins.strip():
        margins = tuple(float(v) for v in args.crop_margins.split(","))
    if args.use_sdpa is None:
        try:
            import flashinfer  # noqa: F401
            args.use_sdpa = False
        except ImportError:
            args.use_sdpa = True
    pin_k = (args.pin_fx, args.pin_fy or args.pin_fx) if args.pin_fx > 0 else None
    return StreamMapper(
        args.model_path, image_size=args.image_size, patch_size=args.patch_size,
        num_scale_frames=args.num_scale_frames, anchor_stride=args.anchor_stride,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        max_context_frames=args.max_context_frames, keyframe_interval=args.keyframe_interval,
        max_frame_num=args.max_frame_num, camera_num_iterations=args.camera_num_iterations,
        use_sdpa=args.use_sdpa, voxel_size=args.voxel_size, max_points=args.max_points,
        fuse=args.fuse, downsample_factor=args.downsample_factor,
        conf_threshold=args.conf_threshold, conf_percentile=args.conf_percentile,
        crop=args.crop, undistort_k1=args.undistort_k1, undistort_fov=args.undistort_fov,
        crop_margins=margins, clahe=args.clahe, min_frame_luma=args.min_frame_luma,
        min_frame_std=args.min_frame_std,
        sky_model=args.sky_model if args.mask_sky else None, sky_threshold=args.sky_threshold,
        compact_every=args.compact_every, pin_k=pin_k, save_artifacts=args.save_artifacts,
        artifact_every=args.artifact_every, source=args.video_path, fps=args.fps)


class Feed:
    def __init__(self, args):
        self.a = args
        self.sock, self.iq = None, None
        self.win = deque(maxlen=max(1, args.snap_win)) if args.snap else None
        if args.tap:
            host, port = args.tap.rsplit(":", 1)
            self.sock, self.addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fly"))
            try:
                from iq import raw as iq
                self.iq = iq
            except ImportError:
                pass

    def __call__(self, r, t):
        a, fi = self.a, r["frame_idx"]
        if self.sock:
            n = r["conf"][::a.downsample_factor, ::a.downsample_factor].size
            w = r["w"]
            msg = {"t": t, "f": fi, "q": len(r["points"]) / max(n, 1),
                   "c": float(w.mean()) if len(w) else 0.0,
                   "p": [float(v) for v in r["c2w"][:3, 3]]}
            if self.iq is not None:
                msg["iq"] = self.iq(r["rgb"])
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        if self.win is not None:
            self.win.append((t, r["points"].copy(), r["colors"].copy()))
            if fi % a.snap_every == 0:
                sp = np.concatenate([x[1] for x in self.win])
                sc = np.concatenate([x[2] for x in self.win])
                if len(sp) > a.snap_points:
                    keep = np.random.default_rng(fi).choice(len(sp), a.snap_points, replace=False)
                    sp, sc = sp[keep], sc[keep]
                tmp = a.snap + ".tmp.npz"
                np.savez(tmp, points=sp, colors=sc, frame=fi, t=t, t0=self.win[0][0])
                os.replace(tmp, a.snap)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.video_path:
        parser.error("Provide --video_path (a file, or a camera index such as 0)")
    if args.mask_sky and not os.path.exists(args.sky_model):
        parser.error(f"--mask_sky needs the ONNX model; not found at {args.sky_model}")
    src = int(args.video_path) if args.video_path.isdigit() else args.video_path
    m = mapper_from_args(args)
    print(f"Anchor: {args.num_scale_frames} frames every {args.anchor_stride} "
          f"(~{args.num_scale_frames * args.anchor_stride / args.fps:.1f}s of motion); "
          f"keyframe every {args.keyframe_interval}")

    feed = Feed(args) if args.tap or args.snap else None
    viewer = None if args.no_viewer else LiveViewer(args.port, args.point_size)
    if viewer:
        print(f"\n  Viewer ready:  http://localhost:{args.port}\n")
    stop = {"flag": False}

    def _on_sigint(signum, frame):
        if stop["flag"]:
            sys.exit(1)
        stop["flag"] = True
        print("\nStopping after this frame (Ctrl-C again to force quit)...")

    signal.signal(signal.SIGINT, _on_sigint)
    t_win, fps_est = time.time(), 0.0
    for raw_idx, rgb in iter_video_rgb(src, args.fps, args.stride, args.max_frames, args.loop):
        if stop["flag"]:
            break
        t_frame = time.time()
        try:
            r = m.push(rgb, raw_idx)
        except RuntimeError as e:
            print(e)
            break
        if r is None:
            continue
        if feed and not r["anchor"]:
            feed(r, t_frame)
        fi = m.frame_idx
        if viewer and fi % args.vis_every == 0:
            now = time.time()
            fps_est = args.vis_every / max(now - t_win, 1e-6)
            t_win = now
            p, c = m.snapshot()
            H, W = m.hw
            viewer.update(p, c, r["c2w"][:3, :3], r["c2w"][:3, 3],
                          2 * float(np.arctan(r["K"][1, 2] / r["K"][1, 1])), W / H,
                          frames=fi, fps=fps_est)
        if args.stats_every and fi % args.stats_every == 0:
            el = time.time() - m.t0
            vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            resv = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"[{fi:6d}] {fi / max(el, 1e-6):5.2f} fps avg | VRAM {vram:5.2f} GB "
                  f"(reserved {resv:5.2f}) | "
                  f"map {len(m.point_map):>9,} pts", flush=True)
    m.flush()
    el = time.time() - m.t0
    print(f"\nProcessed {m.frame_idx} frames in {el:.1f}s "
          f"({m.frame_idx / max(el, 1e-6):.2f} fps)")
    m.save(ply=args.save_ply)
    if args.save_artifacts:
        print(f"Artifacts: {len(m.saved_ids)} frames + cloud -> {args.save_artifacts}")
    if viewer:
        print(f"\n  Viewer still serving at http://localhost:{args.port}  (Ctrl-C to exit)\n")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
