# Restore point — 2026-08-30

Disk was freed by deleting **only things that can be downloaded or rebuilt**.
Everything expensive to reproduce was kept.

To come back:

```powershell
powershell -ExecutionPolicy Bypass -File RESTORE\restore.ps1
```

Needs network, ~7 GB of downloads, 20–40 min. Then:

```powershell
cd C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map
..\.venv310\Scripts\python.exe view_map.py --artifacts ..\artifacts --port 8080 --surface
```

---

## What was kept

The reconstruction is the only thing here that cost real compute, and it is
small, so it stayed:

| Kept | Size | Why |
|---|---|---|
| `artifacts\` | 59 MB | The 586-frame reconstruction: fused cloud, 87 depth maps with poses, detections, surface. Regenerating means re-running the model over the video. |
| `artifacts_courthouse\` | 45 MB | The control scene that caught the over-merging bug. |
| `lingbot-map\*.py`, `lingbot_map\` | 227 MB | All source, including uncommitted work. |
| `RESTORE\scripts\` | 221 KB | Second copy of the scripts I wrote, in case the repo folder is disturbed. |

Because `artifacts\` survived, **no reconstruction or detection re-run is
needed** — the viewer works as soon as the venv and checkpoint are back.

## What was deleted

| Deleted | Freed | Restored by |
|---|---|---|
| `.venv310\` | 7.8 GB | step 1–3 |
| `lingbot-map.pt` checkpoint | 4.6 GB | step 4 |
| `owlv2` + `bioclip` weights | 1.2 GB | step 4 |
| Playwright Chromium | 702 MB | step 3 |
| `lingbot-map\.git\` | 218 MB | see below |
| pip cache | 52 MB | rebuilds itself |
| `fpv_map.ply`, `fpv_map_30fps.ply` | 33 MB | see below |

## The two that need explaining

**`.git` was deleted, the working tree was not.** Every source file is intact,
including all uncommitted modifications. Only the upstream history is gone. If
you want git back:

```powershell
cd $env:TEMP
git clone https://github.com/Robbyant/lingbot-map.git lm-tmp
git -C lm-tmp checkout 1740f18ead3cca8e0e07dd0ec6b7030d22f0894c
Move-Item lm-tmp\.git C:\Users\shaur\OneDrive\Desktop\Test\lingbot-map\.git
```

`git status` will then show the same 10 modified files and 4 new ones as before.
`uncommitted.patch` in this folder is an independent record of those changes, in
case the clone is ever unavailable.

**The `.ply` exports were deleted** — they are just `cloud.npz` in another
format:

```powershell
..\.venv310\Scripts\python.exe -c "import numpy as np, trimesh; d=np.load(r'..\artifacts\cloud.npz'); trimesh.PointCloud(d['points'], d['colors']).export(r'..\fpv_map.ply')"
```

## Not touched

`C:\Users\shaur\.cache\huggingface\hub\models--allenai--specter2_base` is
**840 MB** and belongs to some other project, not this one. It is a cache and
would re-download on demand, but it is not mine to delete — remove it yourself
if you want the space:

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--allenai--specter2_base"
```

The source video `Downloads\VID_20260822_183412.mp4` (63 MB) was left alone.
It is the one input that cannot be regenerated at all.

## Pinned versions

Python 3.10.11 · torch 2.8.0+cu128 (CUDA 12.8) · viser 1.1.0 · trimesh 4.12.2 ·
transformers 5.15.1 · scikit-image 0.25.2 · open_clip_torch 3.3.0 ·
playwright 1.62.0. Full set in `requirements-frozen.txt`.

`requirements-nolocal.txt` is the same list minus torch/torchvision (which must
come from PyTorch's CUDA index, not PyPI) and minus the editable `lingbot_map`
line (which points at a git URL and would overwrite the local changes).

## Settings worth not rediscovering

The viewer and pipeline defaults that were arrived at by measurement:

```
stream_video.py   --fps 30 --keyframe_interval 10 --anchor_stride 10
                  --crop 0.75 --clahe --voxel_size 0.0035
                  --max_context_frames 256 --kv_cache_sliding_window 32
detect_plants.py  --min_view_support 2   (drops reconstruction floaters)
build_surface.py  --mesh_voxel 0.007 --level 0.28 --color_match 1.0
view_map.py       --spread 8.0 --ambient 1.8 --close_rings 2 --bridge 5.0
```
