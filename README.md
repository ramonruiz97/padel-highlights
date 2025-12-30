Padel highlights pipeline (WIP).

## Project - Plan
1) Per-frame parquet: run YOLO + TrackNetV2 to extract ball, players, net, and any other useful metadata per frame.
2) Ticking: resample detections every fixed tick (e.g., every N frames) and compute features using the current and recent frames (ball speed, player activity, visibility, distances, etc.).
3) Model: train a Random Forest on tick-level features to estimate probability of “in-play”.
4) Pattern detection: use the in-play signal and trajectories to spot highlight-worthy patterns.
