# OBS-Detect Vision – Sports Edition (zoom) – v5.3.0

This bundle contains the **updated `detect-filter.cpp`** for the *Safe ROI + Group clustering* workflow.

> Note: this ZIP is a **source patch bundle** (it does not include the full repository tree).  
> Drop `src/detect-filter.cpp` into your repo (OpenVINO branch / Windows build) replacing the existing file.

## What changed in v5.0.1

- **Preview group clusters now uses `groupMaxDistFrac`** (same value as the crop logic).
- **Removed duplicated Safe ROI defaults** in `detect_filter_defaults()`.
- **Auto-snap velocity reset cleanup** (removed redundant resets).
- **Clustering allocation/perf tweaks** (reused buffers; less per-frame churn).
- **Better cluster selection for basketball:** choose **highest people count first**, then **largest area**.
- configurable zoom 

## Settings recap (Tracking group)

- `GroupMaxDistFrac` (0.05–0.50): max distance between people as a fraction of frame width.
- `Group min people` + `Strict min people`
- Safe ROI margins: Left/Right/Top/Bottom (%)
- `Safe ROI Hold (ms)`
- `Cluster inertia (ms)`
- `Preview group cluster` + optional label
