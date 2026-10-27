# Pipeline scripts – running them separately

Normally `../pipeline.py` runs these steps in order (fisheye corrector → trajectory tracker → background extractor + segmentation → feature extractor) (see the
[main README](../README.md)). This page is for describing each of the files more in detail and provides instructions for running **one part of the pipeline at a time**,
e.g. to re-segment a scene after hand correction without re-tracking people.

Run every command from this folder (`video_processing/video_processing/`).
Scripts that take videos or images use the paths given on the command line, or
the `VIDEOS` / image list at the top of the file when none are given.

## Contents

1. [Files per clip](#1-files-per-clip)
2. [fisheye_correction.py](#2-fisheye_correctionpy) – step 1
3. [video_processor.py](#3-video_processorpy) – step 2
4. [background_extraction.py](#4-background_extractionpy) – step 3
5. [background_segmentation.py / video_segmentation.py](#5-background_segmentationpy--video_segmentationpy) – step 3
6. [extract_features_o.py](#6-extract_features_opy) – step 4
7. [3D / depth scripts](#7-3d--depth-scripts)

## 1. Files per clip

`<video>` is the source file name without extension; `<clip>` =
`<video>_<action>` with action ∈ `enter | pass | exit`.

| File | Written by | Content |
|---|---|---|
| `<video>_lens.json` | fisheye_correction | `distorted`, `reason`, `K`, `D`, `new_K` |
| `<video>_undistorted.mp4` | fisheye_correction | only when distortion was found |
| `<clip>.json` | video_processor | trajectory + COCO-17 keypoints + metadata |
| `<clip>.mp4` | video_processor | the clip itself |
| `<clip>_pose.json` | video_processor | keypoints as normalised, named landmarks |
| `<video>_background.png` | background_extraction | image of the empty room |
| `<clip>_background.json` | background_segmentation | scene polygons (pixels) |
| `<clip>_background_overlay.png` | background_segmentation | polygons drawn on the background |
| `exits_<clip>.json` | by hand, optional | exits that replace the automatic ones |
| `features_<clip>.json` | extract_features_o | all features, see [FEATURES.md](../FEATURES.md) |
| `features_<clip>_debug.png` | extract_features_o `--debug` | scene, exits and trajectory on one image |
| `feature_catalog.json` | extract_features_o | one entry per feature |

The scene is stored **once**, in its own file, rather than repeated inside every
clip JSON.

## 2. `fisheye_correction.py`

```bash
python fisheye_correction.py ../source/clip.mp4
```

Straight lines in a room (wall edges, door frames) must be straight in an
undistorted image. The script finds long edges on the median background and
searches for the focal length that makes them straightest. If that changes the
image noticeably, every frame is undistorted.

- **In:** video
- **Out:** `<video>_lens.json` (always — records the decision), `<video>_undistorted.mp4` (only if distorted)
- **Thresholds:** `MIN_DISTORTION`, `MIN_GAIN`, `MIN_STRAIGHT`, `MIN_CHAINS` at the top of the file.
- `new_K` in the lens JSON is the camera matrix of the undistorted video; the
  3D steps need it.

## 3. `video_processor.py`

Jesper's collector. Detects and tracks people, labels each clip and writes the
trajectory JSON. Point it at the **undistorted** video from step 1 when one was
written, so the trajectory and the scene share the same geometry. Self-contained: needs only `ultralytics`, `opencv-python`,
`numpy`.

```bash
python video_processor.py               # no arguments; settings in main()
```

Settings in `main()`:

- `save_clips=True` — record and save clips (default).
- `save_clips=False` — track, label and show, but write nothing (use this to
  aim the camera and check detections before recording).
- `print_labels=True` — print a line each time a person is labelled.
- `headless=True` — no window (screenless Jetson; stop with Ctrl+C).
- `model="..."` — a `.pt` for testing, or the exported `.engine` on the Jetson.
- `source_dir` / `source_video` — process a whole folder or one file.
- plus `camera_index`, `ellipse_axes`, `distort`, `fullscreen`,
  `status_every_s`, `session_note`.

Other knobs (grace/tail, minimum duration, confidence, capture size,
pose sidecar on/off) live in `CollectorConfig` at the top of the script.

### Output layout

```
data/
  2026-05-29/
    enter/    <clip>.json  <clip>.mp4  <clip>_pose.json
    pass/
    exit/
    removed/  too_short/  too_few_points/  partial_bottom/  corrupt/
    _sessions/   session_<date>_<time>.json   (metadata + counts per session)
    _temp/       scratch, emptied as clips finalise
```

Same-day recordings are appended into one dated folder. Multiple tracks from
one source get `_2`, `_3`, … suffixes. Nothing is deleted: unusable clips are
moved to `removed/<reason>/`.

### Trajectory JSON (`<clip>.json`)

```jsonc
{
  "id": 7,
  "label": "enter",                  // enter | pass | exit
  "fps": 28.94,                      // measured effective fps
  "frame_width": 640,
  "frame_height": 480,
  "door_center": [320, 480],         // the door this camera controls
  "ellipse_axes": [480, 130],
  "cleaning_result": "keep",         // keep | exit | removed
  "cleaning_reason": "accepted",
  "n_frames": 64,
  "n_detected": 61,
  "detection_rate": 0.9531,
  "meta": {
    "schema_version": "2.0",
    "session_id": "2026-05-29_083015",
    "created_utc": "2026-05-29T06:30:42Z",
    "duration_s": 2.18,
    "effective_fps": 28.94,
    "record_fps": 30.0,              // fps the .mp4 is encoded at
    "n_gap_frames": 3,               // frames where the person was briefly missing
    "n_pose_frames": 61,
    "entry_edge": "top", "exit_edge": "bottom",
    "model": "yolo11n-pose.pt", "tracker": "botsort.yaml",
    "conf_threshold": 0.3, "imgsz": 640,
    "computer": "Jetson", "camera_index": 0,
    "capture_resolution": [640, 480], "distort": false,
    "session_note": "main entrance",
    "software": { "python": "...", "opencv": "...", "numpy": "...", "ultralytics": "..." },
    "git_commit": "abc1234",
    "keypoint_format": "COCO-17",
    "keypoint_names": ["nose", "left_eye", ...],
    "keypoint_coords": "pixels (divide by frame_width/height to normalise)"
  },
  "frames": [
    {
      "frame_index": 0,
      "t": 0.0,                      // seconds since clip start
      "center": [320, 240],          // pixels (null when not detected)
      "bbox": [300, 120, 340, 470],  // x1,y1,x2,y2 pixels (null when not detected)
      "keypoints": [[x, y, conf], ... x17]   // COCO-17 pixels (null when not detected)
    }
  ]
}
```

Frames where the person was briefly missing keep `center` / `bbox` /
`keypoints` = `null`, so frame indices stay continuous.

### Pose sidecar (`<clip>_pose.json`)

Per frame a `detected` flag and, when detected, a `landmarks` dict mapping each
name to `{x, y, visibility}` with x/y normalised to `[0, 1]`:

| landmark | COCO idx | | landmark | COCO idx |
|---|---|---|---|---|
| nose | 0 | | left_wrist | 9 |
| left_ear | 3 | | right_wrist | 10 |
| right_ear | 4 | | left_hip | 11 |
| left_shoulder | 5 | | right_hip | 12 |
| right_shoulder | 6 | | left_knee | 13 |
| left_elbow | 7 | | right_knee | 14 |
| right_elbow | 8 | | left_ankle | 15 |
|  |  | | right_ankle | 16 |

Turn it off with `write_pose_sidecar=False`.

### Notes

- **People leaving the frame.** A track counts as gone after `grace_frames`
  (default 15) missing frames; shorter dropouts are bridged. Then
  `tail_seconds` (default 1 s) more is recorded. A recycled track id starts a new
  clip; an id that reappears far from where it vanished is split into a new clip.
- **fps.** Top-level `fps` is measured from the `t` timestamps. If the camera
  reports no fps and `record_fps` is not set, the script stops instead of
  guessing.
- **Sorting.** Thresholds for enter / pass / exit / removed live in
  `CollectorConfig`; each clip records its own `cleaning_result` and
  `cleaning_reason`.

## 4. `background_extraction.py`

```bash
python background_extraction.py ../source/clip_undistorted.mp4
```

Takes every `STEP`-th frame (default 3) and keeps the per-pixel median. People
move, so the median at each pixel is the empty room.

- **In:** video (the undistorted one, if step 1 produced it)
- **Out:** `<video>_background.png` next to the video

## 5. `background_segmentation.py` / `video_segmentation.py`

Both segment the background with SAM 3 using the text prompts
`floor, wall, door, window` (`PROMPTS` at the top of each file). Requires the
SAM 3 setup from the main README.

| | `background_segmentation.py` | `video_segmentation.py` |
|---|---|---|
| Input | background images | videos (makes the background itself) |
| GUI | none | FiftyOne app for hand correction |
| Merging | none | touching floor / wall pieces merged into one object |
| Use when | batch runs, servers | checking or fixing a new scene by hand |

```bash
python background_segmentation.py      # edit the image list in __main__
python video_segmentation.py           # edit VIDEOS / EXISTING_DATASETS
```

- **Out:** a JSON with `image_size` and `objects[]`, each with `label`,
  `confidence` and `polygon` (pixels), plus an `_overlay.png`.
- `background_segmentation.py` writes to `../data/segmentation_results/` by
  default. Rename/move the JSON to `<clip>_background.json` next to the
  trajectory so `extract_features_o.py` finds it.
- `THRESHOLD` controls how many objects are kept (higher = fewer, surer).

## 6. `extract_features_o.py`

```bash
# one clip – side files are found by name next to <clip>.json
python extract_features_o.py --traj ../data/<date>/enter/<clip>.json --debug

# a whole folder
python extract_features_o.py --input-dir ../data/<date>/enter --out-dir ../data/features --debug
```

- **Needs:** `<clip>.json` and `<clip>_background.json` (clips without a
  segmentation are skipped).
- **Optional:** `<clip>_pose.json` (keypoint fallback), `<clip>_background.png`
  (lighting features + debug image), `exits_<clip>.json` (hand-made exits).
- **Out:** `features_<clip>.json`, `feature_catalog.json`, and with `--debug`
  `features_<clip>_debug.png`.
- Override any side file with `--scene`, `--pose`, `--background`, `--exits`,
  `--out`.
- All thresholds are in `PARAMS` at the top of the file and copied into every
  output's `info.params`.

Hand-made exits format:

```json
{"exits": [{"type": "door", "segment_px": [[x1, y1], [x2, y2]]}]}
```

## 7. 3D / depth scripts

In the parent folder; optional, needs `requirements-background-depth.txt`.

| Script | Does |
|---|---|
| `track_single_video.py` | tracks one video without trimming or re-indexing frames |
| `background_depth.py` | background depth from saved LingBot predictions |
| `segment_background_3d.py` | lifts scene polygons into 3D using that depth |
| `door_distance_3d.py` | fits the floor plane, lifts foot points, door distances in metres (scale from an assumed 0.825 m door leaf) |

Each script prints its options with `--help`.
