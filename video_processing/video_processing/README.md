# video_processing

`video_processor.py` processes pedestrian videos from `../source/` and writes a
labelled, pose-enriched dataset into `../data/`. It is self-contained (only needs `ultralytics`, `opencv-python`, `numpy`)

The `video_processor.py` originally from previous work ([pose_data_collection.py](https://github.com/MirianaPassarotto/SeamAI/blob/main/automatic_door_project/intent_prediction_2026/data_collection_v2/data_collection/pose_data_collection.py)), credits to Jesper Malmgren.

## What it does

For each person tracked through the camera view it saves:

- a per-frame trajectory: center point, bounding box, and the 17 COCO keypoints;
- the matching video clip (`.mp4`);
- a JSON file with the trajectory and metadata;
- a `<name>_pose.json` sidecar with the skeleton as normalized, named landmarks.

Each finished clip is labelled `enter`, `pass`, or `exit` and sorted into its
folder. Clips that look unusable are not deleted, only moved to
`removed/<reason>/`. Same-day recordings are appended into one dated folder.

## Running it

Set the options in `main()` at the bottom of `video_processor.py`, then run
it — press Run in your editor, or:

```bash
python video_processor.py
```

No command-line arguments are needed. The settings in `main()`:

- `save_clips=True` — record and save clips (default).
- `save_clips=False` — track, label and show, but write nothing (monitor / setup;
  use this to aim the camera and check detections before recording).
- `print_labels=True` — print a line each time a person is labeled enter/pass/exit.
- `headless=True` — no window (screenless Jetson; stop with Ctrl+C).
- `model="..."` — a `.pt` for testing, or the exported `.engine` on the Jetson.
- plus `camera_index`, `ellipse_axes`, `distort`, `fullscreen`, `status_every_s`, `session_note`.

Other knobs (grace/tail, minimum duration, confidence, capture size, pose-sidecar
on/off) live in `CollectorConfig` at the top of the script.

## Output layout

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

## JSON schema (per clip)

The flat top-level keys describe the trajectory; everything else is under `meta`.

```jsonc
{
  "id": 7,
  "label": "enter",                  // enter | pass | exit
  "fps": 28.94,                      // measured effective fps (see "fps" below)
  "frame_width": 640,
  "frame_height": 480,
  "door_center": [320, 480],
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
    "effective_fps": 28.94,          // measured from per-frame timestamps (== top-level fps)
    "record_fps": 30.0,              // fps the .mp4 is encoded at (--record-fps or camera-reported)
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
      "center": [320, 240],          // pixels  (null when the person is not detected)
      "bbox": [300, 120, 340, 470],  // x1,y1,x2,y2 pixels  (null when not detected)
      "keypoints": [[x, y, conf], ... x17]   // COCO-17 in pixels (null when not detected)
    }
  ]
}
```

`center` and `bbox` keep the same meaning and units as the original schema; `t`
and `keypoints` are added. Frames where the person was briefly missing keep
`center`/`bbox`/`keypoints` = `null` so the frame indices stay continuous.

## Pose sidecar

The same COCO-17 keypoints are also written to `<name>_pose.json` (for kept
clips): per frame a `detected` flag and, when detected, a `landmarks` dict
mapping each name to `{x, y, visibility}` with x/y normalised to `[0, 1]`. The
landmark names are a named subset of COCO-17:

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

Turn it off by setting `write_pose_sidecar=False` if you only want the inline keypoints.

## Notes

**People leaving the frame.** A track counts as gone once it has been missing
for `grace_frames` (default 15); a short dropout (a missed detection or brief
occlusion) is bridged inside the same clip. After that, an extra `tail_seconds`
(default 1 s) of footage is recorded before the clip is closed, so each clip ends
with a short tail. If a track id is recycled afterwards, a new clip starts — a
finished track is never resumed. A still-active id that reappears far from where
it vanished is treated as a different person and split into a new clip.

**fps.** The top-level `fps` is the measured effective rate, computed from the
per-frame `t` timestamps. There is no guessed fallback: if the camera reports no
fps and `record_fps` is not set, the script stops and asks for it rather than
assuming a value. `meta.record_fps` is the fps the `.mp4` is encoded at.

**Sorting.** Each finished clip is sorted into `enter` / `pass` / `exit` /
`removed/<reason>`; the thresholds live in `CollectorConfig` at the top of the
script, and every clip records its own `cleaning_result` / `cleaning_reason`.

## Privacy

This records identifiable video and pose data of people; handle it accordingly.
