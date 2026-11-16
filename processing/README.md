# processing/

The 2D steps of the pipeline. `pipeline.py` runs them in this order; each can
also be run on its own with `python processing/<script>.py` from the
repository root.

| Step | File | What it does | Main output |
|---|---|---|---|
| 1 | `clean_datasource.py` | Moves duplicate videos and very short videos (time is configurable) out of the input folder. | `<source-dir>/deleted/duplicated/`, `<source-dir>/deleted/too_short/` |
| 2 | `fisheye_correction.py` | Checks if the video has fisheye distortion and undistorts it if so. | `<video>_undistorted.mp4`, `<video>_lens.json` |
| 3 | `video_processor.py` | Tracks people (YOLO pose), cuts one clip per person and labels it enter / pass / exit. Based on Jesper's code. | `<clip>.mp4`, `<clip>.json` (trajectory + keypoints), `<clip>_pose.json` |
| 4 | `background_extraction.py` | Makes an image of the empty room by masking out people and taking the median over frames. Pixelates anyone still visible. | `<clip>_background.png` |
| 5 | `background_segmentation.py` | Finds floor, walls, doors and windows in that image with SAM 3. | `<clip>_background.json` (polygons), `_overlay.png` |
| 6 | `feature_extractor.py` | Combines the person's trajectory with the scene into per-frame features. | `<clip>_features.json` |

`<clip>` is the video name, with `_2`, `_3`, … when one video has several
people. All files of a clip share this name, which is how each step finds the
output of the previous one.

## Good to know

- **Undistort first.** Steps 3–6 use the undistorted video, so the person and
  the scene are in the same geometry.
- **Clean first.** The pipeline moves duplicate and sub-0.5-second videos
  from the root of the configured source directory into its `deleted/`
  subfolders. Nested directories are not scanned. Run the cleaner by itself
  with `python processing/clean_datasource.py --source-dir DIR`.
- **Settings** (thresholds, model, device) are constants at the top of each file.
- **Labels** are stored inside `<clip>.json` and in `clips.csv`. Fix a wrong
  label by filling in `manual_label` in `clips.csv`.
- **Removed clips** (too short, person only partly visible) stay on disk but
  are not passed on to the next step.
- **Check the result** with `feature_extractor.py --debug`: it draws the scene,
  the exits and the trajectory on one image.

Every feature is explained in [FEATURES.md](../FEATURES.md); the optional 3D
step is in [geometry3d/](../geometry3d/README.md).
