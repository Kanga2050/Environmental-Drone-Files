# plant_det — plant detection and hierarchical naming

`detect_plants.py` finds plants in the saved frames of a reconstruction and
names them:

1. **Plant candidates.** Two YOLO11m box detectors (classes `plant`, `tree`)
   run with flip/scale test-time augmentation; their boxes are pooled and
   merged by NMS.
   - `models/detector_fpv.pt`: fine-tuned from COCO on Open Images plant/tree
     boxes, half of them degraded to look like the analog FPV video (low line
     count, sideways colour smear, cross-colour on foliage, CLAHE).
   - `models/detector_yard.pt`: the same detector fine-tuned on 3D-consistent
     pseudo-labels, i.e. the plants of a finished map projected back into
     every frame.
2. **Growth form → family → genus → species** (`hier.py`). BioCLIP-2 (ViT-L)
   fully fine-tuned with four heads on iNaturalist 2021, PlantNet-300K and
   ~776k California iNaturalist photos (cultivated plants and hybrids such as
   *Photinia × fraseri* included; species not restricted by region). A plant's
   best views and their mirror images are averaged; the answer walks down the
   tree and stops at the first rank whose confidence is below its threshold
   (`models/classifier/tau85.json`: about 85% precision per rank on
   FPV-degraded and never-seen-species test images; `tau90.json` is stricter).
3. The boxes are lifted to 3D and handed to the Point Transformer
   (`plant_seg`).

The whole chain from a video is `bash make_viewer.sh VIDEO NAME` (project root).

The training and evaluation code (data preparation, the FPV degradation,
detector and classifier training, threshold calibration, pseudo-labels and the
candidate comparison page) is in git history, commits 4fe416b8 to ae11e584.

## Sources

- Models: YOLO11 (Ultralytics, 2024); BioCLIP 2 (Gu et al. 2025,
  arXiv:2505.23883).
- Data: Open Images V6/V7 boxes (Kuznetsova et al. 2020, IJCV);
  iNaturalist 2021 (Van Horn et al., CVPR 2021); PlantNet-300K (Garcin et al.,
  NeurIPS 2021 Datasets & Benchmarks); iNaturalist Open Data (AWS);
  GIFT growth forms (Weigelt et al. 2020, J. Veg. Sci. 31:315).
