# 3D background geometry and door-distance pipeline

This guide covers the optional 3D extension to the video-processing pipeline:
video input, tracking, scene segmentation, LingBot depth, background geometry,
per-person door distances, visual validation, and the combined feature output.

The main entry point is [`pipeline.py`](../pipeline.py).
Run the example commands **from the repository root**. Paths in commands are
relative to that root; links in this guide are relative to this file.

All 3D/LingBot source and optional requirements are in `geometry3d/`. The 2D
stages are in `processing/`; `data/`, `videos/`, `source/`, `reports/` and
`tests/` are directly under the project root. The main pipeline imports both
packages. Standalone geometry commands use `python -m geometry3d.MODULE`.

## Purpose and project scope

The intent-prediction project compares motion representations across camera
placements and office layouts. Image-space distances change with perspective.
This extension explores whether reconstructed floor positions and distances to
door references provide useful geometric features alongside the existing 2D
trajectory and pose features.

The pipeline extracts features; it does not train or evaluate an intent model.
It does not demonstrate camera-independent generalisation, measured metric
accuracy, or automatic time-to-event labels. Those require separate evaluation.

**Assumptions:** a fixed camera, usable floor and door segmentation, and enough
stable background depth to fit the scene. The current integrated distance stage
also requires at least one uncropped segmented door for its metric scale.

## What happens at each stage

| Stage | Implementation | Behaviour |
| --- | --- | --- |
| 1. Prepare video | `processing/fisheye_correction.py` | In raw-video mode, estimates lens distortion from scene edges and prepares corrected/copied inputs. `--skip-undistort` bypasses this. |
| 2. Track pedestrians | `processing/video_processor.py` | Runs YOLO pose and ByteTrack, writes per-person clips and trajectories, and creates a processed-clip manifest. Existing manifests skip stages 1 and 2. |
| 3. Prepare scene labels | `processing/background_extraction.py`, `processing/background_segmentation.py` | Reuses supplied scene labels, or runs SAM 3 on a temporal-median RGB image for floor, wall, door and window polygons. A missing RGB background is generated even when labels are supplied. |
| 4. Prepare features and timing | `pipeline_geometry.py`, `processing/feature_extractor.py` | Adapts saved detections without rerunning YOLO, normalises time to frame index / encoded FPS, and extracts the existing 2D features. |
| 5. Obtain depth | `lingbot_runtime.py`, `lingbot_inference.py` | Reuses an exact matching depth export or launches LingBot in its own environment locally or over SSH/Docker. Exports every frame. |
| 6. Estimate background depth | `background_depth.py` | Aggregates per-pixel depth over the clip, saving a temporal median, temporal mean, stability mask and diagnostics. |
| 7. Reconstruct labelled geometry | `segment_background_3d.py` | Aligns scene labels with the depth grid and back-projects pixels into camera/world coordinates. Exports labelled point clouds and a viewer. |
| 8. Measure and join | `door_distance_3d.py`, `pipeline_geometry.py` | Fits the floor and door references, lifts person contacts to the floor, computes distances/rates, and appends 3D fields to the 2D feature JSON. |

Processing runs per manifest record. With raw inputs, this can mean a saved
person clip rather than the entire original recording. Camera, depth, labels
and trajectory must describe that same processed clip.

## Dependencies and the public LingBot repository

**LingBot is an external project:**
[Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map).
This repository contains our adapter and geometry processing, not LingBot's
implementation or checkpoint. Cloning this project does not install LingBot.

Use two Python environments:

| Environment | Required for | Setup |
| --- | --- | --- |
| This project's environment | Pipeline orchestration, tracking/segmentation when needed, feature extraction, geometry and viewers | Install the applicable requirements below. |
| External LingBot environment | New depth inference only | Clone LingBot separately, follow its upstream installation instructions, and download `lingbot-map-long.pt` from its model-download links. |

For the full local pipeline:

```sh
python -m pip install -r requirements.txt
python -m pip install -r geometry3d/requirements-background-depth.txt
```

New tracking needs YOLO weights; missing scene labels need SAM 3 weights and any
model access required by its upstream provider. Installing Python packages
alone does not guarantee those weights are available. `--device` controls new
YOLO runs; the SAM 3 module selects CUDA when available, otherwise CPU.

When reusing trajectories, scene labels and cached depth, a smaller environment
can avoid loading perception models:

```sh
python -m pip install -r geometry3d/requirements-background-depth.txt
python -m pip install "opencv-python>=4.8,<4.10"
```

The adapter's model construction and checkpoint loading were checked against
LingBot revision `3c637397e288fa9429d43e9c9ec17ad7069589c3` with PyTorch
`2.8.0+cu128`. Follow upstream instructions for its environment; changing the
upstream API or checkpoint may require adapter changes. The current adapter
uses streaming inference with PyTorch SDPA and does not invoke LingBot's batch
renderer or interactive demo.

### Configure a local LingBot installation

Copy [`lingbot.local.example.json`](lingbot.local.example.json) to a personal
file, for example `data/lingbot.local.json`. Replace the
`python`, `repo`, and `checkpoint` placeholders with absolute paths:

```json
{
  "runtime": "local",
  "python": "/absolute/path/to/lingbot-env/bin/python",
  "repo": "/absolute/path/to/lingbot-map",
  "checkpoint": "/absolute/path/to/lingbot-map-long.pt",
  "device": "cuda",
  "dtype": "auto",
  "scale_frames": 8,
  "cache_window": 64,
  "keyframe_interval": 1
}
```

On Windows, `python` can point to the LingBot environment's `python.exe`. Use
forward slashes or escaped backslashes in JSON paths. Relative local paths are
resolved from the configuration file's directory. The pipeline starts this
Python as a separate process; it does not activate or modify that environment.
`device` here controls LingBot independently of the pipeline's `--device`.
The default requires functioning CUDA; CPU can be selected explicitly but has
not been benchmarked for practical video inference.

### Configure an existing SSH/Docker installation

Local LingBot setup can be replaced by a remote installation:

```json
{
  "runtime": "ssh-docker",
  "host": "user@server-address",
  "port": 22,
  "identity_file": "~/.ssh/id_ed25519",
  "repo": "/srv/projects/lingbot-map",
  "checkpoint": "/srv/projects/lingbot-map/data/models/lingbot-map-long.pt",
  "image": "YOUR_EXISTING_LINGBOT_IMAGE:TAG",
  "remote_workdir": "/srv/projects/lingbot-map/data/pipeline_jobs",
  "device": "cuda"
}
```

The laptop needs OpenSSH `ssh` and `scp` clients and working non-interactive
authentication. The server needs the checkout, weights, a compatible Docker
image, permission to run Docker, a writable job directory, and GPU/container
support. Remote paths must be absolute POSIX paths. No particular shared
server is required by this project.

The runtime checks the GPU before uploading, transfers the adapter and exact
clip, mounts the checkout and checkpoint read-only, runs inference, and
downloads depth exports. Background aggregation and feature extraction then
continue locally. Remote job folders are retained for diagnostics. This mode
does not install Docker, build an image, or repair GPU drivers.

## Running the integrated pipeline

### Option A: Start from raw videos

Put the intended inputs in a dedicated directory and run:

```sh
python pipeline.py --source-dir videos --output data/pipeline/run_3d --with-3d --lingbot-config data/lingbot.local.json --device cpu
```

This enables new tracking and automatic scene segmentation. The command's
`--device cpu` applies to YOLO; LingBot still uses the device in its runtime
JSON. Omit `--with-3d` for the existing 2D path. Add `--skip-undistort` when
inputs are already corrected or should retain their supplied image geometry.

The tracker still has configured entrance-region assumptions. Enabling 3D
does not automatically relocate the entrance or adapt tracking labels to a
new camera placement. Review the tracker's configuration and scene labels.

### Option B: Reuse existing tracks and segmentation

Create `inputs_3d.json`, for example:

```json
[
  {
    "video": "videos/clip.mp4",
    "trajectory": "data/existing/clip_track1.json",
    "pose": "data/existing/clip_track1_pose.json",
    "scene": "source/scene_clip.json"
  }
]
```

Paths in this manifest are relative to the manifest itself. The video and
trajectory are mandatory. `pose` is optional when keypoints are available in
the trajectory; `scene` and `background` can supply existing segmentation and
RGB references. Otherwise missing scene labels trigger SAM 3. A string list of
video paths is also accepted, with sibling `.json` trajectories inferred.

```sh
python pipeline.py --processed-manifest inputs_3d.json --output data/pipeline/run_3d --with-3d --lingbot-config data/lingbot.local.json
```

`--processed-manifest` skips fisheye processing and YOLO. The adapter expects a
per-track trajectory with `frame_width`, `frame_height`, `frames`, and contiguous
`frame_index` values covering the exact clip. It rejects missing frames or
coordinate mismatches instead of guessing a clip offset. Other trajectory
schemas need an explicit conversion.

### Option C: Use cached LingBot predictions only

```sh
python pipeline.py --processed-manifest inputs_3d.json --output data/pipeline/run_cached_3d --with-3d --lingbot-runs data/lingbot
```

No LingBot environment or runtime configuration is needed when matching exports
exist for every clip. Without a match or configured runtime, the pipeline fails
with an explanation. If both cache and runtime are supplied, an exact cached
match takes precedence. Selecting a different checkpoint in the configuration
does not force replacement of an existing matching cache entry; use separate
cache/output locations for reconstruction comparisons.

### Relevant arguments

| Argument | Purpose |
| --- | --- |
| `--source-dir` | Raw inputs; used when no processed manifest is supplied. |
| `--processed-manifest` | Reuse saved clip/trajectory records; skip fisheye and tracking. |
| `--output` | Local output root; default `data/pipeline`. |
| `--with-3d` | Enable the geometry extension. |
| `--lingbot-runs` | Search saved inference manifests; default `data/lingbot`. |
| `--lingbot-config` | Local or SSH/Docker runtime JSON for missing predictions. |
| `--lingbot-command-config` | Legacy alias; also accepts custom command-array providers. |
| `--door-width-m` | Assumed leaf width for estimated metres; default `0.825`. |
| `--skip-undistort` | Bypass the raw-video lens-correction stage. |
| `--device` | YOLO device for new tracking; independent of LingBot's runtime device. |
| `--fail-fast` | Stop at the first failing clip; otherwise record failures and continue. |
| `--log-level` | Console logging level; the timestamped log also captures DEBUG detail. |

Run `python pipeline.py --help` for CLI help.

## Depth, the binary mask and pixel alignment

New inference processes every frame in order with no frame subsampling. LingBot
resizes the input to width 518, with height rounded to a multiple of 14. The
current geometry mapping supports landscape/square inputs without centre
cropping. Examples: 640 x 480 becomes 518 x 392; 1280 x 720 becomes 518 x 294.
Portrait inputs requiring cropping are rejected.

The background estimate takes the temporal median of valid positive depth at
each prediction pixel. It also saves the mean and median absolute deviation.
The integrated defaults mark a pixel stable when:

- At least 90% of frames have valid depth there.
- At least 80% of its valid depths lie within 10% of its temporal median.

The **binary stability mask** says whether a pixel passes these consistency
checks. Depth itself remains continuous, not binary. There are no person masks
in this aggregation method. A stationary person or persistent foreground
object can remain in the background; temporal consistency alone cannot identify
what an object is.

Scene polygons are rasterised/aligned to the depth grid using a verified resize
mapping that preserves class and instance IDs. Depth, labels and projected XYZ
then share the same pixel grid. The scene uses median predicted intrinsics and
the first frame's pose as its fixed-camera reference. New adapter exports use
world-to-camera extrinsics.

Video SHA-256, dimensions, FPS, frame count and trajectory indices constrain
reuse. A renamed byte-identical clip can reuse depth; an undistorted, cropped,
trimmed or re-encoded clip needs its own matching export. Matching filenames
alone are insufficient. Review supplied scene labels too: dimension agreement
does not prove that an old un-hashed segmentation belongs to the same scene.

## What the distances actually measure

The person reference is the midpoint of two visible ankle keypoints, one
visible ankle, or a bounding-box bottom-centre proxy when ankles are unavailable
and the lower image boundary does not crop the feet. Its camera ray is
intersected with a robustly fitted floor plane. This avoids using a person's
foreground body depth as a background floor measurement.

For each segmented door, the pipeline fits its panel plane, intersects it with
the floor plane, and estimates a floor-level segment spanning the visible
support. Distance is the shortest Euclidean distance from the person's fitted
floor point to that segment. These are door-panel proxies, not surveyed opening
thresholds. Insufficient stable door support causes a failure rather than an
invented reference.

The special `main_bottom_centre` reference is the floor intersection of the
original bottom-centre image pixel. Its distance is point-to-point. It is not
the centre of the person's bounding box to the centre of a door, and it is not
automatically a measured physical entrance. A new deployment must review this
reference for its camera placement.

Missing people, cropped feet, or failed floor intersections produce unavailable
3D rows and null distances. Gaps are not interpolated. Closure rate is the
change in distance divided by elapsed source-video time for consecutive track
frames when both contacts are valid: **negative means approaching**. It is not itself a
time-to-event prediction.

### Model units and estimated metres

Reconstructed depth and XYZ use **LingBot model units**, not image pixels and
not measured metres. The original and prediction images have pixel units;
their reconstructed coordinates have model units.

For estimated metres, each usable uncropped door supplies a factor:

```text
metres_per_model_unit = assumed_door_leaf_width_m / reconstructed_panel_span
```

The median factor scales every coordinate, distance and rate uniformly.
`--door-width-m 0.825` is an assumption, not a measurement or a universal door
size. A clipped door can remain a distance reference but is excluded from the
default scale estimate. The integrated stage currently fails if no uncropped
door can provide a scale, even if only model-unit distances are of interest.
Scaling cannot correct reconstruction shape or door-reference errors.

## Outputs and visual inspection

Outputs have the following structure; `<clip-key>` contains the video stem,
the first 10 video-hash characters and first 8 trajectory-hash characters:

```text
OUTPUT/
  logs/pipeline_<timestamp>.log
  pipeline_run.json
  feature_catalog.json
  processed_video_paths.json                 # generated in raw-video mode
  tracks/                                    # generated in raw-video mode
  <clip-key>/
    features_<video-stem>.json
    <video-stem>_background.png               # if not supplied elsewhere
    <video-stem>_background.json              # if generated here
    geometry_3d/
      lingbot_run/                            # only when new inference runs here
        processing_manifest.json
        inference.log
        <video-stem>/frame_000000.npz ...
      depth/<depth-stem>/
        background_depth.npy
        stable_background_depth.npy
        background_depth.npz
        stable_mask.png
        background_depth_heatmap.png
        validation_preview.png
        metadata.json
      segments/<depth-stem>/
        aligned_scene.json
        aligned_label_map.npy
        aligned_instance_map.npy
        segmented_background.npz
        segmented_background.ply
        segmented_background.glb
        viewer.html
        metadata.json
      distances/
        tracking/
        trajectory_3d.json
        ground_geometry.json
        metric_calibration.json
        distances.csv
        floor_fit_validation.png
        trajectory_3d_preview.png
        viewer.html
        summary.json
      postprocess.log
```

Cached predictions stay in their original location. The `<depth-stem>` can
differ from the current video stem when identical video bytes were renamed.
Additional preview/debug exports may also be present. If another application
locks `distances.csv`, the exporter can write `distances_with_estimated_metres.csv`.

Start visual validation with:

1. `depth/.../validation_preview.png`: median RGB, depth, temporal variation and
   the stability mask; check whether people or depth artefacts remain.
2. `segments/.../viewer.html`: switch between labelled 3D geometry, the original
   2D segmentation and pixel-depth heatmap; check alignment.
3. `distances/viewer.html`: inspect the synchronised track, floor contact,
   door references and distance curves. Switch model units/estimated metres,
   adjust the assumed width, or select reference doors. Viewer changes affect
   the displayed values and its downloadable CSV; saved pipeline JSON/CSV keep
   the original CLI calibration.

`pipeline_run.json` records completion/failure and output paths. Per-clip errors
are listed in `failed_clips`; exit code 2 means a run completed with failures.
Per-stage
metadata records geometry checks and provenance. New inference manifests
include the external repository revision, checkpoint hash and inference settings.

### Combined feature format

The existing `feat-0.3` feature JSON retains its 2D groups and adds a
`geometry_3d` object to each `data.frames` row. Joining checks frame IDs,
timestamps and source provenance. It does not assume list order alone.

| Added field | Meaning |
| --- | --- |
| `available` | Whether a valid floor contact exists. |
| `contact_source`, `contact_confidence` | Which pose/proxy observation was used and its available confidence. |
| `point_ground_model` | Position in the fitted floor coordinate frame, in model units. |
| `point_ground_estimated_m` | The same position after the assumed uniform metric scale. |
| `distances_model`, `distances_estimated_m` | Dictionaries of per-door distances, including `main_bottom_centre`. |
| `closure_rates_model_per_s`, `closure_rates_estimated_m_per_s` | Per-door distance-change rates. |

Calibration and causality metadata live in `info.depth`; units are recorded in
`info.units`. `feature_catalog.json` describes the extension. Door keys are
segmentation-instance references, separate from the existing 2D exit ordering;
they are not stable physical door IDs across videos. The integrated feature
join selects the record's track; standalone distance output can contain all
saved tracks and identifies a primary track by its number of observed frames.

## Limits for intent prediction and new cameras

- **Whole-clip calibration:** depth aggregation, RGB background and camera
  estimates use the entire clip. The 3D extension is marked `causal: false`.
  For online prediction or time-to-event benchmarking, estimate scene geometry
  from an independent prior calibration recording and freeze it before evaluating
  pedestrian clips. Otherwise future-frame information can enter the features.
- **Fixed-camera assumption:** temporal medians and one reference pose do not
  model a moving camera. LingBot's reconstruction may also struggle with a fixed
  camera and moving people; useful geometry must be checked visually.
- **Placement assumptions:** floor coordinates reduce image-space perspective
  effects but do not establish a common world frame across cameras. Entrance
  selection, tracking regions, lens correction, scene labels and scale still
  need validation for each installation. Do not compare raw model-unit values
  across independently reconstructed clips as if they shared a scale.
- **Offline workflow:** inference/post-processing retain clip-level arrays and
  exports. Near-real-time performance has not been established.
- **Privacy:** input videos, RGB previews, pose tracks and point clouds remain
  potentially identifying. The implementation saves them for validation; it
  does not implement the final system's deletion/retention policy or prove
  anonymity. Model training can select motion-only features separately.

## Troubleshooting and validation status

| Symptom | Check |
| --- | --- |
| No matching depth | Use the exact processed clip; configure `--lingbot-config` or supply matching exports. |
| Trajectory frame/dimension mismatch | Provide the corresponding per-person clip and full matching track timeline; do not mix original and trimmed videos. |
| Wrong segmentation alignment | Confirm scene provenance and coordinate system. Do not combine distorted labels with undistorted depth. |
| Missing model/environment | Check the external checkout, environment Python and checkpoint paths; consult upstream LingBot setup. |
| CUDA unavailable / driver-library mismatch | Repair the environment or server GPU/container runtime. The adapter does not silently switch a CUDA configuration to CPU. Inspect `lingbot_run/inference.log`. |
| No uncropped calibration door | Improve the segmentation/reference scene; current integrated metric calibration needs at least one uncropped usable panel. |
| Floor/door fit fails | Inspect stable support, labels and depth previews before changing thresholds. |
| Null distances | Inspect `contact_source`, feet visibility and floor intersection; nulls are intentional when no usable contact exists. |

Contract tests require the small cached-processing environment, not model weights:

```sh
python -m unittest discover -s tests -v
```

After the root-folder reorganisation, 37 tests passed, including path relocation,
manifest resolution and offline-viewer asset checks. The integrated pipeline
completed on the existing id93 clip using cached LingBot depth, and the real
checkpoint loaded with no missing/unexpected weights. New adapter GPU inference
has **not** been validated end-to-end: the available remote GPU reported a
driver/library mismatch. Mocked model/runtime tests and CPU checkpoint loading
are not a substitute for that pending real-inference validation.

Existing local manifests and background-depth metadata can still refer to the
former outer `video_processing/` path. `project_paths.py` resolves those recorded
paths to the moved files without changing historical provenance or hashes.
New commands and generated output paths use the root layout. Reopen old saved
viewers under `data/` and update external bookmarks to their new locations.

Personal runtime JSONs, SSH settings, model weights, recordings and generated
exports should stay outside code commits. A future clone needs its own inputs
and configuration; this guide and the portable example describe that setup.
