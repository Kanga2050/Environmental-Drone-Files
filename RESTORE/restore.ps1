# Rebuild everything that was deleted to free disk space.
#
# Nothing here is irreplaceable -- it is all downloads and installs. What was
# kept is the part that cannot be re-downloaded: the reconstruction in
# artifacts\ and artifacts_courthouse\, and every .py file in lingbot-map\.
#
# Needs network. Budget ~7 GB of downloads and 20-40 minutes.
#
#   powershell -ExecutionPolicy Bypass -File RESTORE\restore.ps1
#
# Steps are independent and safe to re-run; each skips work already done.

$ErrorActionPreference = "Stop"
$Base = "C:\Users\shaur\OneDrive\Desktop\Test"
$Venv = "$Base\.venv310"
$Repo = "$Base\lingbot-map"
$Models = "$env:LOCALAPPDATA\lingbot-map"
$Py310 = "C:\Users\shaur\AppData\Local\Programs\Python\Python310\python.exe"

Write-Host "=== 1/5  virtualenv (Python 3.10.11) ===" -ForegroundColor Cyan
if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    if (-not (Test-Path $Py310)) {
        throw "Python 3.10 not found at $Py310. Install Python 3.10.11 first."
    }
    & $Py310 -m venv $Venv
}
$VPy = "$Venv\Scripts\python.exe"
& $VPy -m pip install --upgrade pip

Write-Host "=== 2/5  PyTorch 2.8.0 + CUDA 12.8 ===" -ForegroundColor Cyan
# The CUDA build only exists on PyTorch's own index, so it installs separately
# from the rest. Installing torch from PyPI instead silently gets a CPU build,
# and the reconstruction then runs unusably slowly.
& $VPy -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 `
    --index-url https://download.pytorch.org/whl/cu128

Write-Host "=== 3/5  remaining packages ===" -ForegroundColor Cyan
& $VPy -m pip install -r "$PSScriptRoot\requirements-nolocal.txt"
# Editable install of the local working tree. The frozen file pins this to a git
# URL, which would fetch upstream and throw away the local modifications.
& $VPy -m pip install -e $Repo --no-deps
& $VPy -m playwright install chromium

Write-Host "=== 4/5  model weights (~5.8 GB) ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $Models | Out-Null
$Ckpt = "$Models\lingbot-map.pt"
if (-not (Test-Path $Ckpt)) {
    # local_dir, not the default cache: the default layout uses symlinks, which
    # need privileges this account does not have (WinError 1314).
    & $VPy -c @"
from huggingface_hub import hf_hub_download
p = hf_hub_download('robbyant/lingbot-map', 'lingbot-map.pt',
                    local_dir=r'$Models')
print('checkpoint ->', p)
"@
} else {
    Write-Host "  checkpoint already present, skipping"
}

if (-not (Test-Path "$Models\models\owlv2\config.json")) {
    & $VPy -c @"
from huggingface_hub import snapshot_download
# allow_patterns skips the duplicate .bin next to the safetensors; without it
# this downloads the same weights twice.
snapshot_download('google/owlv2-base-patch16-ensemble',
                  local_dir=r'$Models\models\owlv2',
                  allow_patterns=['*.json','*.txt','*.safetensors','*.model'])
print('owlv2 ready')
"@
}
if (-not (Test-Path "$Models\models\bioclip\open_clip_config.json")) {
    & $VPy -c @"
from huggingface_hub import snapshot_download
snapshot_download('imageomics/bioclip',
                  local_dir=r'$Models\models\bioclip',
                  allow_patterns=['*.json','*.txt','*.safetensors','*.bin'])
print('bioclip ready')
"@
}

Write-Host "=== 5/5  verify ===" -ForegroundColor Cyan
& $VPy -c @"
import torch, viser, trimesh, skimage, cv2
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('viser', viser.__version__, '| trimesh', trimesh.__version__)
import os
art = r'$Base\artifacts'
for f in ['cloud.npz', 'plants.json', 'plant_points.npz', 'surface.npz', 'meta.json']:
    p = os.path.join(art, f)
    print(('  OK  ' if os.path.exists(p) else '  MISSING '), f)
"@

Write-Host ""
Write-Host "Done. Bring the viewer back with:" -ForegroundColor Green
Write-Host "  cd $Repo"
Write-Host "  ..\.venv310\Scripts\python.exe view_map.py --artifacts $Base\artifacts --port 8080 --surface"
Write-Host ""
Write-Host "artifacts\ was preserved, so no reconstruction or detection re-run is needed."
