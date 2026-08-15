# Manta V3 — structural and CFD analysis

CFD and finite-element analysis of the Manta V3 airframe (Fusion design "Manta V3 - 5in FPV VTOL"), plus the same structural analysis run on earlier saved versions of the design.

## Layout

| Folder | What it does |
|---|---|
| `cfd/` | Watertight envelope from the CAD (`wrap.py`), farfield mesh with actuator-disk props (`mesh.py`), SU2 case writer with momentum-theory inflow (`run.py`) and motor-torque swirl (`swirl.py`), forward-flight trim (`trim.py`, `trim2.py`, `ff.py`), flat-plate comparison body (`flat.py`), force summaries (`summary.py`), case queue (`queue.py`) |
| `fea/` | Quadratic-tet mesh of fuselage + canopy (`extract.py`, `fmesh.py`, `pre.py`), component and wetted-surface mapping (`comps.py`, `wet.py`), load decks with CFD pressures, thrust, torque and inertia (`deck.py`, `go.py`, `run_fea.py`), direct sparse solver with bonded canopy joint (`fe.py`), modal/harmonic analysis (`dyn.py`), implicit transient impact with unilateral contact (`tdyn.py`), print-direction failure index (`fos.py`), regional post-processing (`post.py`) |
| `versions/` | Per-version extraction, meshing, analysis and comparison of saved Fusion versions (`vcfg.py`, `vprep.py`, `vrun.py`, `vpost.py`, `vrender.py`) |
| `plots/` | Streamline renders with props, stress and mode renders, prop-wash pressure maps, Campbell/FRF, power and comparison charts |
| `results/` | Summary numbers (CFD forces and trim, FEA regional stresses and factors of safety, version comparison) |

## Tools
SU2 8.5 (incompressible RANS, SST), gmsh 4.15, numpy/scipy, PARDISO (via pypardiso), meshio, trimesh, pymeshlab, PyVista, matplotlib.

## Main results
- **Hover:** 38% of thrust is lost to download on the body, so hover takes 1.62× weight.
- **Cruise:** 37–44% less power than hover at 10–15 m/s.
- **Prop wash in cruise:** costs the body about 1 N of lift.
- **Flat plate of the same outline:** 21% hover download, and 50–100% more lift at similar drag.
- **Full throttle:** factor of safety at least 3.6.
- **1 m drop:** rear arm root factor of safety 0.78 with planar layers, 1.12 with conformal (5-axis) layers.
- **3 m/s pole strike:** the arm tip fails first and the body survives.
- **Vibration:** first frame modes 31–62 Hz, against prop excitation at 280 Hz and blade pass at 850 Hz in hover.

Write-ups: `Manta V3 — CFD Analysis` and `Manta V3 — Structural Analysis` in my notes.
