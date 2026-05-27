# plant_seg — assigning map points to detected plants

A promptable Point Transformer (Zhao et al., "Point Transformer", ICCV 2021,
arXiv:2012.09164) decides which points of the reconstructed cloud belong to
each plant found by `detect_plants.py`. The prompt for a plant is built from
its 2D detection boxes (in which views each point falls inside the box, is
visible, and how central it is) and the 3D points lifted from those boxes.

Run from `lingbot-map/`, with `A=../artifacts_yard_avg`.

| Step | Command |
|---|---|
| Detect plants (boxes saved per view) | `python detect_plants.py --artifacts $A` |
| Assign points (writes `plant_seg.npz`, used by `view_map3d.py`) | `python -m plant_seg.segment --artifacts $A --weights ../datasets/plant_seg_runs/pt7_slim/best.pt --threshold 0.4` |
| Label-free checks vs the old radius rule | `python -m plant_seg.eval_yard --artifacts $A` |
| Photos with old/new points drawn | `python -m plant_seg.overlay --artifacts $A --out overlays` |
| Comparison page | `python -m plant_seg.compare_view --artifacts $A --overlays overlays --out ../plant_seg_compare.html` |
| Hand-labelling page | `python -m plant_seg.label_tool --artifacts $A --out ../plant_label.html` |
| Score against hand labels | `python -m plant_seg.score_labels --artifacts $A --labels $A/plant_labels.json --seg radius plant_seg.npz` |

`view_map3d.py --no-seg` falls back to the radius rule.

## Training

Training scenes are synthetic: labelled PLANesT-3D plants (Mertoğlu et al.
2024, arXiv:2407.21150) and plants cut out of the real map, packed together on
real yard background, with detector boxes, pose errors, misplaced seeds and
colour casts simulated to match the real detections.

```
python -m plant_seg.prep_planest --src <PLANesT zips> --out ../datasets/planest_npz
python -c "from plant_seg.yard import write_bank; write_bank('$A', '../datasets/yard_bank.npz')"
python -m plant_seg.train --plants ../datasets/planest_npz --bank ../datasets/yard_bank.npz \
    --out ../datasets/plant_seg_runs/NAME --init <previous best.pt> \
    --derived --uniform_layers enc2 enc3 dec0
```

Add `--real_labels $A/plant_labels.json --real_holdout <ids>` to mix
hand-labelled real plants into each batch; score the held-out ids with
`score_labels --ids <ids>`. `python -m plant_seg.calibrate` picks the
probability threshold on held-out synthetic scenes.

`--derived` adds lighting-independent colour, colour texture and local-shape
inputs; `--uniform_layers` runs the layers whose attention the trained model
left uniform as plain neighbourhood averages.
