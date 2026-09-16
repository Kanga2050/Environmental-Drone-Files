# Environmental Drone Analysis

Video from the Manta V3, a 5-inch FPV VTOL drone, turned into a 3D plant map
in which each plant is outlined, named and rated for health, viewable offline
in one HTML file. Also the drone's flight setup and the airframe's CFD and
structural analysis.

```
make_viewer.sh          video -> maps/NAME -> viewers/NAME.html (the whole chain)
pipeline.ipynb          the same chain step by step
yardmap/
  live.py               live camera / stream version with a browser viewer
  reconstruct.py        streaming 3D reconstruction
  gct/                  the Geometric Context Transformer model code
  calib/                lens calibration of the DVR video
  detect_plants.py      plant boxes -> 3D plants -> names and health
  plant_det/            hierarchical classifier, plant health head
  plant_seg/            Point Transformer: which points belong to which plant
  export_viewer.py      one self-contained viewer HTML (web/ holds its page)
  view_map3d.py         the same viewer served live
training/health/        plant health data, training and calibration
tools/survey/           pictures, collages and flight reconstructions of finished maps
auton/
  drone/manta.parm      ArduPilot parameters for the Manta
  radio/SCRIPTS/        EdgeTX Lua scripts for the radio
  computer/             ground computer: MAVLink, flashing, survey planning, live control
structural-analysis/    CFD and FEA of the Manta V3 airframe
models/, video/, maps/, viewers/   weights, source video, maps and viewers (not in git)
```
