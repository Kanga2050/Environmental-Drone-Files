# plant_det — trained plant detector + hierarchical naming

Replaces the OWLv2 text-prompt detector and the 50-species BioCLIP list in
`detect_plants.py` with models trained here:

1. **Plant candidates** — a YOLO11 box detector (two classes: `plant`, `tree`)
   fine-tuned from COCO weights on Open Images plant/tree boxes, then on a copy
   of those images degraded to look like the yard's analog FPV video
   (`fpv.py`: low line count, sideways colour smear, cross-colour rainbow on
   foliage, CLAHE, JPEG).
2. **Growth form → family → genus → species** — one image backbone with four
   heads, trained on iNaturalist 2021 (plants), PlantNet-300K and California
   iNaturalist observations (any quality grade, so cultivated garden plants
   are included). Species are not restricted by region. Inference averages the
   plant's best views, then walks down the tree and stops at the first rank
   whose confidence is below τ (default 0.5): a plant is named "shrub ·
   Rosaceae" when the model is sure of the family but not the genus.
3. The boxes are lifted to 3D and handed to the Point Transformer
   (`plant_seg`) exactly as before.

Current best (`yard_viewer_v3.html`): an ensemble of the FPV detector and a
copy fine-tuned on **3D-consistent pseudo-labels** (`pseudo3d.py`: each
finished plant's points are projected into every saved frame, with a depth
test, to give its box there), boxes up to 85% of the frame, and classifier
C9 = BioCLIP-2 ViT-L fully fine-tuned (gradient checkpointing) on iNat 2021 +
PlantNet-300K + ~776k California iNaturalist photos including garden hybrids
(`Photinia × fraseri` etc. are kept as species), averaged with a mirrored
view, thresholds calibrated on held-out species for 85% precision per rank.

Growth form labels come from the GIFT database (species value; else a clear
genus/family majority), with fixed rules for ferns, mosses, palms, succulents
and grasses.

## Commands (from `lingbot-map/`)

| Step | Command |
|---|---|
| Detector data | `python -m plant_det.prep_oi --raw ../datasets/raw --out ../datasets/oi_plants` |
| FPV-style copy | `python -m plant_det.make_fpv_set --src ../datasets/oi_plants --dst ../datasets/oi_plants_fpv` |
| Train detector | `python -m plant_det.train_det --data ../datasets/oi_plants_fpv/data.yaml --model <init.pt> --name y11m_fpv` |
| Classifier images | `python -m plant_det.prep_cls inat21 \| plantnet \| open_select \| open_fetch` |
| Labels + vocabulary | `python -m plant_det.train_cls labels --tag _all` |
| Train classifier | `python -m plant_det.train_cls train --tag _all --backbone bioclip2 --train_blocks 4 --name c2` |
| Score classifier | `python -m plant_det.eval_cls --ckpt ../datasets/cls_runs/c2/best.pt --tag _all` |
| Run a candidate on the yard | `python -m plant_det.run_candidate --name NAME --detect "--detector yolo --det_weights W --classifier hier --cls_weights C --classify_top_k 10"` |
| 3D pseudo-labels | `python -m plant_det.pseudo3d --cand ../cands/NAME --out ../datasets/yard3d` |
| Detector ensemble | `--det_weights a.pt,b.pt` (boxes pooled, then NMS) |
| Comparison page | `python -m plant_det.compare_cands --out ../plant_candidates.html` |

`detect_plants.py` keeps its old defaults (OWLv2 + BioCLIP); the new parts
are `--detector yolo --det_weights`, `--classifier hier --cls_weights`
(or `zeroshot` with a vocab JSON for the untrained BioCLIP-2 baseline) and
`--tau`.

## Sources

- Models: YOLO11 (Ultralytics, 2024); BioCLIP 2 (Gu et al. 2025,
  arXiv:2505.23883); DINOv2 (Oquab et al. 2023, arXiv:2304.07193).
- Data: Open Images V6/V7 boxes (Kuznetsova et al. 2020, IJCV);
  iNaturalist 2021 (Van Horn et al., CVPR 2021); PlantNet-300K (Garcin et al.,
  NeurIPS 2021 Datasets & Benchmarks); iNaturalist Open Data (AWS);
  GIFT growth forms (Weigelt et al. 2020, J. Veg. Sci. 31:315).
