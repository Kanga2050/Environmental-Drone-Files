# Plant health model

Rates each plant **healthy** (green leaves), **stressed** (discoloured leaves)
or **dead / dormant** (no live leaves). Inference is `yardmap/plant_det/health.py`;
`detect_plants.py --health models/health/health.pt` runs it on the same views a
plant is named from, and `make_viewer.sh` turns it on.

## Data

iNaturalist research-grade plant observations with the community "Leaves"
annotation (green / coloured / no live leaves), CC0, CC-BY and CC-BY-NC photos,
half California and half worldwide, at most 25 per species. 20,097 photos
after removing 425 observations marked both green and coloured. Split by
species (no species in two splits): 15,762 train, 1,735 validation, 2,175 test.

## Steps (from this folder, data in `data/` or `$HEALTH_DATA`)

| Step | Command |
|---|---|
| Download | `python get_data.py 7000` |
| Backbone features, clean + mirrored + 2 FPV-degraded views | `python feats.py data/feats.npz` |
| Head on frozen features | `python train_head.py data/feats.npz runs/h5.pt --drop 0.5 --epochs 80 --drop_obs data/conflict_obs.txt` |
| Fine-tune the last 8 backbone blocks, FPV look on the fly | `python finetune.py runs/h5.pt runs/f2.pt --blocks 8 --epochs 4 --drop_obs data/conflict_obs.txt` |
| Multi-photo test observations | `python get_multiview.py` |
| Plant-level threshold | `python eval_multiview.py runs/f2.pt runs/f2_mv.json && python calib_plant.py runs/f2.pt runs/f2_mv.json` |

## Results (held-out species)

| Run | Change | Macro-F1 clean | Macro-F1 FPV-degraded |
|---|---|---|---|
| H1 | frozen backbone, MLP head | 0.750 | 0.736 |
| H5 | dropout 0.5, 80 epochs | 0.759 | 0.749 |
| F1 | fine-tune last 4 blocks | 0.800 | 0.764 |
| **F2** | **last 8 blocks, 4 epochs (shipped)** | **0.804** | **0.772** |
| F3 | F2 with FPV look on 85% of images | 0.804 | 0.767 |

F2 on degraded photos: 77.9% accuracy; at its 85%-precision confidence it answers
for 74% of single photos. For plants seen in several views a separate threshold
(0.59) is calibrated on 1,192 held-out observations with ~3 photos each: 76% of
them are labelled, at 91% precision.

Limits: the labels describe leaf colour, so autumn colour, red-leaved plants and
summer-dormant natives read as stressed or dormant; training photos are close-ups
while drone crops are small and blurry, and no drone footage has health labels yet.
