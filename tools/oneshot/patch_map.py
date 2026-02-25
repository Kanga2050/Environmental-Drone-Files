
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\stream_video.py"
s = io.open(p, encoding="utf-8").read()

OLD_INIT = '''    def __init__(self, voxel_size: float, max_points: int, compact_every: int = 30):
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        self.compact_every = int(compact_every)
        self._pos = np.zeros((0, 3), dtype=np.float32)
        self._col = np.zeros((0, 3), dtype=np.uint8)
        self._age = np.zeros((0,), dtype=np.int64)
        self._pending_pos = []
        self._pending_col = []
        self._pending_age = []
        self._since_compact = 0
        self.dropped = 0'''

NEW_INIT = '''    def __init__(self, voxel_size: float, max_points: int, compact_every: int = 30,
                 average: bool = False):
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        self.compact_every = int(compact_every)
        self.average = bool(average)
        self._pos = np.zeros((0, 3), dtype=np.float32)
        self._col = np.zeros((0, 3), dtype=np.uint8)
        self._age = np.zeros((0,), dtype=np.int64)
        # Averaging mode keeps running sums instead of the last sample, plus the
        # canonical voxel key: once a position is a mean it can drift over a cell
        # boundary, and re-deriving the key from it would split one voxel in two.
        self._key = np.zeros((0,), dtype=np.int64)
        self._psum = np.zeros((0, 3), dtype=np.float32)
        self._csum = np.zeros((0, 3), dtype=np.float32)
        self._w = np.zeros((0,), dtype=np.float32)
        self._first = np.zeros((0,), dtype=np.int32)
        self._pending_pos = []
        self._pending_col = []
        self._pending_age = []
        self._pending_w = []
        self._since_compact = 0
        self.dropped = 0'''

OLD_ADD = '''    def add(self, pos: np.ndarray, col: np.ndarray, age: int) -> None:
        if len(pos) == 0:
            return
        self._pending_pos.append(pos.astype(np.float32, copy=False))
        self._pending_col.append(col.astype(np.uint8, copy=False))
        self._pending_age.append(np.full(len(pos), age, dtype=np.int64))
        self._since_compact += 1
        if self._since_compact >= self.compact_every:
            self.compact()'''

NEW_ADD = '''    def add(self, pos: np.ndarray, col: np.ndarray, age: int,
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
            self.compact()'''

OLD_COMPACT = '''    def compact(self) -> None:
        """Merge pending points in: dedup by voxel, then enforce the budget."""
        if not self._pending_pos:
            if len(self._pos) == 0:
                return
            return
        pos = np.concatenate([self._pos] + self._pending_pos, axis=0)'''

NEW_COMPACT = '''    def compact(self) -> None:
        """Merge pending points in: dedup by voxel, then enforce the budget."""
        if not self._pending_pos:
            return
        if self.average:
            return self._compact_average()
        pos = np.concatenate([self._pos] + self._pending_pos, axis=0)'''

OLD_SNAP = '''    def snapshot(self):
        self.compact()
        return self._pos, self._col'''

NEW_SNAP = '''    def _compact_average(self) -> None:
        """Merge pending points by accumulating them, not by replacing.

        Keeping the newest observation of a voxel throws away every earlier look
        at the same surface. On analog video that is the expensive choice: the
        chroma noise is close to independent between frames, so the mean over k
        observations of one voxel carries roughly sqrt(k) less of it, and a
        surface revisited thirty times comes back with a colour that is actually
        its colour. Position averages too, which puts the point in the middle of
        the voxel rather than wherever the last depth sample happened to land.

        Confidence weights the mean, so a frame that was unsure about a pixel
        moves it less than a frame that was not.
        """
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
        psum = np.concatenate([self._psum, new_pos * new_w[:, None]])
        csum = np.concatenate([self._csum, new_col * new_w[:, None]])

        uniq, inv = np.unique(keys, return_inverse=True)
        inv = inv.reshape(-1)
        m = len(uniq)
        w = np.bincount(inv, weights=wsum, minlength=m).astype(np.float32)
        ps = np.stack([np.bincount(inv, weights=psum[:, k], minlength=m)
                       for k in range(3)], axis=1).astype(np.float32)
        cs = np.stack([np.bincount(inv, weights=csum[:, k], minlength=m)
                       for k in range(3)], axis=1).astype(np.float32)

        # First and last frame per voxel without a scatter-reduce. Frames arrive
        # in order, so a plain fancy assignment in forward order leaves the
        # newest value behind and in reverse order the oldest -- exact, and far
        # cheaper than np.maximum.at over a few million elements.
        first = np.zeros(m, np.int32)
        last = np.zeros(m, np.int32)
        inv_new = inv[n_old:]
        age32 = new_age.astype(np.int32)
        last[inv_new] = age32
        first[inv_new[::-1]] = age32[::-1]
        if n_old:
            inv_old = inv[:n_old]
            first[inv_old] = self._first          # an existing voxel is older
            older = np.zeros(m, np.int32)
            older[inv_old] = self._age.astype(np.int32)
            last = np.maximum(last, older)

        if m > self.max_points:
            newest = np.argpartition(last, -self.max_points)[-self.max_points:]
            self.dropped += m - self.max_points
            uniq, w, ps, cs, first, last = (uniq[newest], w[newest], ps[newest],
                                            cs[newest], first[newest],
                                            last[newest])

        self._key = np.ascontiguousarray(uniq)
        self._w = np.ascontiguousarray(w)
        self._psum = np.ascontiguousarray(ps)
        self._csum = np.ascontiguousarray(cs)
        self._first = np.ascontiguousarray(first)
        self._age = last.astype(np.int64)
        safe = np.maximum(w, 1e-6)[:, None]
        self._pos = np.ascontiguousarray(ps / safe)
        self._col = np.ascontiguousarray(np.clip(cs / safe, 0, 255).astype(np.uint8))

    def observations(self):
        """Per-point (first frame, last frame, accumulated weight), or None."""
        if not self.average:
            return None
        return self._first, self._age.astype(np.int32), self._w

    def snapshot(self):
        self.compact()
        return self._pos, self._col'''

OLD_LEN = '''    def nbytes(self) -> int:
        return self._pos.nbytes + self._col.nbytes + self._age.nbytes'''

NEW_LEN = '''    def nbytes(self) -> int:
        return (self._pos.nbytes + self._col.nbytes + self._age.nbytes
                + self._key.nbytes + self._psum.nbytes + self._csum.nbytes
                + self._w.nbytes + self._first.nbytes)'''

for old, new in ((OLD_INIT, NEW_INIT), (OLD_ADD, NEW_ADD),
                 (OLD_COMPACT, NEW_COMPACT), (OLD_SNAP, NEW_SNAP),
                 (OLD_LEN, NEW_LEN)):
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:200])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("BoundedPointMap patched")
