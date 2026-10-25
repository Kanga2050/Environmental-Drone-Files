# Survey tools

Pictures and flight reconstructions for finished maps (`maps/NAME`), used for
the Oct 23 trail survey.

| Script | What |
|---|---|
| `survey_media.py --map M --video V --mount down\|forward --name N --out DIR` | Top-down plant map with the flight path, labelled 3D view, camera / cloud / plant points panel, identification grid with health, camera frames, plant crops and `summary.json` |
| `collage.py DIR NAME TITLE SUBTITLE OUT.jpg` | One portfolio collage from `survey_media.py` output |
| `flight_track.py MAP OUT` | Rebuilds a flight without its log from the GCT poses: height above the fitted ground, footprint cells, passes and revisits, new map points and confidence over time |
| `flight_video.py MAP VIDEO OUT` | Video of that reconstruction: map with path and passes, camera inset and timelines |
| `calib/detect_corners.py`, `calib/fit_lens.py` | Checkerboard corners from a DVR video and the lens fit (rational model) behind `yardmap/calib/foxeer_dvr_1080.json` |
