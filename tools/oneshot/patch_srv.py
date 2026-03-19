
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''        self.points = np.ascontiguousarray(pts[order])
        self.colors = np.ascontiguousarray(cols[order])
        del pts, cols''',
    '''        self.points = np.ascontiguousarray(pts[order])
        self.colors = np.ascontiguousarray(cols[order])
        del pts, cols

        # Provenance, when the reconstruction was run with --fuse average. It
        # turns "which frames produced this point" from a reprojection estimate
        # over the frames that happen to be on disk into an exact lookup over
        # every frame in the video.
        self.has_prov = "first_frame" in cloud
        if self.has_prov:
            self.first_frame = cloud["first_frame"].astype(np.int32)[order]
            self.last_frame = cloud["last_frame"].astype(np.int32)[order]
            self.obs = cloud["obs_weight"].astype(np.float32)[order]
            self.obs_p99 = float(np.percentile(self.obs, 99))
        else:
            self.first_frame = self.last_frame = self.obs = None
            self.obs_p99 = 1.0'''))

subs.append((
    '''    _blob = None

    def points_blob(self):
        """positions f32 | plant tag i16 | colors u8, concatenated.

        Order is alignment, not taste: the browser wraps this one buffer in
        three typed-array views without copying, and an Int16Array cannot start
        at an odd byte. Putting the 3-byte-per-point colours last is the only
        arrangement where every view lands on a legal boundary for any n.
        """
        if self._blob is None:
            self._blob = (self.points.tobytes() + self.tags.tobytes()
                          + self.colors.tobytes())
        return self._blob''',
    '''    _blob = None

    def channels(self):
        """Names of the per-point blocks in points.bin, in order."""
        base = ["pos", "tag"]
        if self.has_prov:
            base += ["obs", "frame"]
        return base + ["color"]

    def points_blob(self):
        """One buffer the browser can view without copying.

        Order is alignment, not taste: an Int16Array cannot start at an odd byte
        offset, so every 2-byte block comes before the 3-byte-per-point colours.
        With colours last, every view lands on a legal boundary for any n.
        """
        if self._blob is None:
            parts = [self.points.tobytes(), self.tags.tobytes()]
            if self.has_prov:
                # Counts and frame indices both fit in 16 bits here: a voxel is
                # seen a few hundred times at most, and the video is 7,564
                # frames long.
                parts.append(np.clip(self.obs, 0, 65535).astype(np.uint16).tobytes())
                parts.append(np.clip(self.last_frame, 0, 65535)
                             .astype(np.uint16).tobytes())
            parts.append(self.colors.tobytes())
            self._blob = b"".join(parts)
        return self._blob

    def frame_created(self, fid):
        """Cloud points whose first observation was this frame -- exact."""
        if not self.has_prov:
            return np.zeros(0, np.uint32)
        return np.flatnonzero(self.first_frame == fid).astype(np.uint32)'''))

subs.append((
    '''            "classifier": self.classifier_name,
            "live_classifier": self.classifier_kind,''',
    '''            "classifier": self.classifier_name,
            "live_classifier": self.classifier_kind,
            "channels": self.channels(),
            "provenance": bool(self.has_prov),
            "obs_p99": self.obs_p99,
            "num_frames": int(self.meta.get("num_frames", 0)),'''))

subs.append((
    '''                if parts[3] == "seen.bin":
                    return self._send(s.frame_seen(fid).tobytes())''',
    '''                if parts[3] == "seen.bin":
                    return self._send(s.frame_seen(fid).tobytes())
                if parts[3] == "created.bin":
                    return self._send(s.frame_created(fid).tobytes())'''))

subs.append((
    '''    def analyse(self, idx):''',
    '''    def point_info(self, i):
        """Everything known about one point, provenance included."""
        i = int(i)
        out = {"index": i,
               "position": [float(x) for x in self.points[i]],
               "color": [int(x) for x in self.colors[i]],
               "height": float(self.height_above_ground(self.points[i])),
               "tag": int(self.tags[i])}
        if self.has_prov:
            out["first_frame"] = int(self.first_frame[i])
            out["last_frame"] = int(self.last_frame[i])
            out["observations"] = float(self.obs[i])
        return out

    def analyse(self, idx):'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("server patched for provenance")
