
import sys, time
import numpy as np
sys.path.insert(0, r"C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map")
from stream_video import BoundedPointMap

rng = np.random.default_rng(0)
VOXEL = 0.005
for existing in (200_000, 800_000, 1_800_000):
    for mode in ("latest", "average"):
        m = BoundedPointMap(VOXEL, 4_000_000, compact_every=10**9,
                            average=(mode == "average"))

        base = rng.normal(0, 0.6, (existing, 3)).astype(np.float32)
        m.add(base, rng.integers(0, 255, (existing, 3), dtype=np.uint8), 0,
              np.ones(existing, np.float32))
        m.compact()

        chunk = 30 * 13_600
        idx = rng.integers(0, existing, chunk)
        newp = base[idx] + rng.normal(0, VOXEL * 0.3, (chunk, 3)).astype(np.float32)
        newc = rng.integers(0, 255, (chunk, 3), dtype=np.uint8)
        t = []
        for r in range(5):
            m.add(newp, newc, r + 1, np.ones(chunk, np.float32))
            t0 = time.perf_counter()
            m.compact()
            t.append(time.perf_counter() - t0)
        print(f"{existing:>9,} voxels  {mode:<8s} "
              f"{1000*np.median(t):7.1f} ms per compaction "
              f"({1000*np.median(t)/30:5.2f} ms per frame)")
    print()
