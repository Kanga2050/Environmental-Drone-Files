
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\stream_video.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''        self._w = np.zeros((0,), dtype=np.float32)
        self._first = np.zeros((0,), dtype=np.int32)''',
    '''        self._w = np.zeros((0,), dtype=np.float32)
        self._n = np.zeros((0,), dtype=np.int32)
        self._first = np.zeros((0,), dtype=np.int32)'''))

subs.append((
    '''        wsum = np.concatenate([self._w, new_w])''',
    '''        wsum = np.concatenate([self._w, new_w])
        nsum = np.concatenate([self._n.astype(np.float32),
                               np.ones(len(new_pos), np.float32)])'''))

subs.append((
    '''        w = np.bincount(inv, weights=wsum, minlength=m).astype(np.float32)''',
    '''        w = np.bincount(inv, weights=wsum, minlength=m).astype(np.float32)
        n = np.bincount(inv, weights=nsum, minlength=m).astype(np.int32)'''))

subs.append((
    '''            uniq, w, ps, cs, first, last = (uniq[newest], w[newest], ps[newest],
                                            cs[newest], first[newest],
                                            last[newest])''',
    '''            uniq, w, n, ps, cs, first, last = (
                uniq[newest], w[newest], n[newest], ps[newest], cs[newest],
                first[newest], last[newest])'''))

subs.append((
    '''        self._w = np.ascontiguousarray(w)
        self._psum = np.ascontiguousarray(ps)''',
    '''        self._w = np.ascontiguousarray(w)
        self._n = np.ascontiguousarray(n)
        self._psum = np.ascontiguousarray(ps)'''))

subs.append((
    '''    def observations(self):
        """Per-point (first frame, last frame, accumulated weight), or None."""
        if not self.average:
            return None
        return self._first, self._age.astype(np.int32), self._w''',
    '''    def observations(self):
        """Per-point (first frame, last frame, summed weight, pixel count).

        The last two are different questions. `weight` is the sum of the
        confidences that landed in the voxel and is what the colour and position
        means are divided by; `count` is how many pixels landed there at all.
        Confidence averages well above 1 on this footage, so reporting the
        weight as if it were a count overstates the evidence by that factor.
        """
        if not self.average:
            return None
        return (self._first, self._age.astype(np.int32), self._w, self._n)'''))

subs.append((
    '''            first, last, weight = obs''',
    '''            first, last, weight, count = obs'''))

subs.append((
    '''            cloud_extra = {"first_frame": first, "last_frame": last,
                           "obs_weight": weight.astype(np.float32)}''',
    '''            cloud_extra = {"first_frame": first, "last_frame": last,
                           "obs_weight": weight.astype(np.float32),
                           "obs_count": count.astype(np.int32)}'''))

subs.append((
    '''                + self._key.nbytes + self._psum.nbytes + self._csum.nbytes
                + self._w.nbytes + self._first.nbytes)''',
    '''                + self._key.nbytes + self._psum.nbytes + self._csum.nbytes
                + self._w.nbytes + self._n.nbytes + self._first.nbytes)'''))

for old, new in subs:
    if old not in s:
        sys.exit("stream_video NOT FOUND:\n" + old[:220])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("stream_video: obs_count recorded")

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''            self.obs = cloud["obs_weight"].astype(np.float32)[order]
            self.obs_p99 = float(np.percentile(self.obs, 99))''',
    '''            # Prefer the true pixel count when the map has one. Older maps
            # only stored summed confidence, which runs a factor of ~2.4 high
            # on this footage; converting by the mean confidence is the best
            # that can be done after the fact, and is labelled as an estimate.
            if "obs_count" in cloud:
                self.obs = cloud["obs_count"].astype(np.float32)[order]
                self.obs_exact = True
            else:
                self.obs = (cloud["obs_weight"].astype(np.float32)[order]
                            / MEAN_CONFIDENCE)
                self.obs_exact = False
            self.obs_p99 = float(np.percentile(self.obs, 99))'''))

subs.append((
    '''        else:
            self.first_frame = self.last_frame = self.obs = None
            self.obs_p99 = 1.0''',
    '''        else:
            self.first_frame = self.last_frame = self.obs = None
            self.obs_p99 = 1.0
            self.obs_exact = False'''))

subs.append((
    '''            "obs_p99": self.obs_p99,''',
    '''            "obs_p99": self.obs_p99,
            "obs_exact": bool(getattr(self, "obs_exact", False)),'''))

subs.append((
    '''            out["observations"] = float(self.obs[i])''',
    '''            out["observations"] = float(self.obs[i])
            out["observations_exact"] = bool(self.obs_exact)'''))

subs.append((
    '''HERE = os.path.dirname(os.path.abspath(__file__))''',
    '''# Mean per-pixel depth confidence measured over this footage. Only used to
# turn the older maps' summed-confidence into an approximate pixel count.
MEAN_CONFIDENCE = 2.43

HERE = os.path.dirname(os.path.abspath(__file__))'''))

for old, new in subs:
    if old not in s:
        sys.exit("view_map3d NOT FOUND:\n" + old[:220])
    s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("view_map3d: prefers obs_count, converts obs_weight otherwise")
