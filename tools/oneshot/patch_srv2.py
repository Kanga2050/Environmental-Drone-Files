
import io
import sys

p = r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\view_map3d.py"
s = io.open(p, encoding="utf-8").read()
subs = []

subs.append((
    '''    def channels(self):
        """Names of the per-point blocks in points.bin, in order."""
        base = ["pos", "tag"]
        if self.has_prov:
            base += ["obs", "frame"]
        return base + ["color"]''',
    '''    def channels(self):
        """Names of the per-point blocks in points.bin, in order."""
        base = ["pos", "tag"]
        if self.has_prov:
            base += ["obs", "frame", "first"]
        return base + ["color"]'''))

subs.append((
    '''                parts.append(np.clip(self.last_frame, 0, 65535)
                             .astype(np.uint16).tobytes())''',
    '''                parts.append(np.clip(self.last_frame, 0, 65535)
                             .astype(np.uint16).tobytes())
                parts.append(np.clip(self.first_frame, 0, 65535)
                             .astype(np.uint16).tobytes())'''))

subs.append((
    '''            "provenance": bool(self.has_prov),''',
    '''            "provenance": bool(self.has_prov),
            "created_frames": (sorted(int(x) for x in
                                      np.unique(self.first_frame))[:20000]
                               if self.has_prov else []),'''))

for old, new in subs:
    if old not in s:
        sys.exit("NOT FOUND:\n" + old[:240])
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("server exposes first_frame")
