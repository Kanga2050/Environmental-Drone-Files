# plant_seg — assigning map points to detected plants

A promptable Point Transformer (Zhao et al., "Point Transformer", ICCV 2021,
arXiv:2012.09164) decides which points of the reconstructed cloud belong to
each plant found by `detect_plants.py`. The prompt for a plant is built from
its 2D detection boxes (in which views each point falls inside the box, is
visible, and how central it is) and the 3D points lifted from those boxes.

```
python -m plant_seg.segment --artifacts ../maps/NAME --weights ../models/plant_seg.pt --threshold 0.4
```

writes `plant_seg.npz`, which `view_map3d.py` and `export_viewer.py` use.
`../../make_viewer.sh` runs this as step 3 of the whole pipeline.

The weights were trained on synthetic scenes built from labelled PLANesT-3D
plants (Mertoğlu et al. 2024, arXiv:2407.21150) and plants cut out of the yard
map. The training, evaluation and labelling code is in git history
(commit f1283155, "Assign map points to plants with a promptable Point
Transformer").
