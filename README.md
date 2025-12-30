Padel highlights pipeline (WIP).

## Project - Plan
1) Per-frame parquet: run YOLO + TrackNetV2 to extract ball, players, net, and any other useful metadata per frame.
2) Ticking: resample detections every fixed tick (e.g., every N frames) and compute features using the current and recent frames (ball speed, player activity, visibility, distances, etc.).
3) Model: train a Random Forest on tick-level features to estimate probability of “in-play”.
4) Pattern detection: use the in-play signal and trajectories to spot highlight-worthy patterns.

## TrackNetV2
Use the external repo’s patched model (apply `tf2torch/diff.txt`) and run detections with your video:
```
python external/TrackNetV2-pytorch/detect.py --source data/raw/match.mp4 --weights external/TrackNetV2-pytorch/tf2torch/track.pt --save-txt
```
Or a minimal smoke test with the handler:
```
python -m src.padel.pipelines.run_tracknet_demo --video data/raw/match.mp4 --max-frames 100
```
