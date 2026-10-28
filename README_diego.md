# Privacy-preserving pedestrian intent prediction

Extract pedestrian trajectories, pose, scene-relative motion features and
optional 3D door distances from fixed-camera video. The main entry point is
[`pipeline.py`](pipeline.py). Run these commands from the repository root.

## Repository layout

```text
privacy-preserving-intent-prediction/
  pipeline.py                 Main pipeline CLI and orchestration
  project_paths.py            Project paths and older local metadata support
  processing/                 Fisheye correction, tracking, segmentation, 2D features
  geometry3d/                 LingBot adapters, depth, 3D segmentation and distances
    README.md                 Complete 3D workflow and setup
    lingbot.local.example.json Portable external-environment configuration
  tests/                      Pipeline and geometry contract tests
  source/                     Input videos, scene labels and recorder helper
  videos/                     Local video collection (not committed)
  data/                       Tracks, cached inference and generated outputs
  docs/                       Local project references
  reports/                    Local reports
  requirements.txt            Main perception dependencies
  check_env.py                Basic environment check
```

The former outer `video_processing/` directory has been removed. The 2D stages
are now in `processing/`; 3D code and its optional requirements are together in
`geometry3d/`. Private recordings and model weights are not included in Git.

## Setup

Create a Python environment, then install the perception dependencies:

```sh
python -m pip install -r requirements.txt
python check_env.py
```

New tracking uses YOLO pose weights. New automatic scene segmentation uses
SAM 3 and requires its weights and upstream model access. For the 3D extension,
also install:

```sh
python -m pip install -r geometry3d/requirements-background-depth.txt
```

New LingBot inference uses a **separate** checkout and Python environment for
[Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map). See
[`geometry3d/README.md`](geometry3d/README.md) for local/SSH Docker setup, cached
depth, units, calibration assumptions and visual validation.

## Run from raw videos

```sh
python pipeline.py --source-dir videos --output data/pipeline/run_2d --device cpu
```

The pipeline prepares lens-corrected inputs, tracks people, extracts a median
RGB background, obtains scene segmentation and writes 2D features. The tracker
can produce per-person clips rather than one output for an entire recording.
Review its configured entrance region when changing camera placement. Add
`--skip-undistort` to keep the supplied image geometry.

## Reuse saved clips and trajectories

Create an `inputs.json` manifest, with paths relative to that file:

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

The video and trajectory are required. Pose, scene and RGB `background` paths
can be supplied separately; otherwise the pipeline uses its sidecar conventions
and generates missing scene inputs. Supplying saved tracks skips fisheye
correction and YOLO. Supplying matching scene labels also avoids rerunning SAM 3.

```sh
python pipeline.py --processed-manifest inputs.json --output data/pipeline/run_2d
```

## Enable 3D features

With matching cached LingBot predictions:

```sh
python pipeline.py --processed-manifest inputs.json --output data/pipeline/run_3d --with-3d --lingbot-runs data/lingbot
```

For missing predictions, copy
[`geometry3d/lingbot.local.example.json`](geometry3d/lingbot.local.example.json)
to a personal JSON, fill in the external environment/checkout/checkpoint paths,
and pass `--lingbot-config data/lingbot.local.json`. This can also select an
existing SSH/Docker installation. Cached depth takes precedence when it matches
the exact video. 3D features supplement the existing 2D JSON fields.

See the [3D guide](geometry3d/README.md) for the full workflow. Reconstruction
uses model units; metre values are estimates based on an assumed door width.
Whole-clip scene calibration is not causal and needs an independent prior
calibration before online intent-prediction evaluation.

## Outputs and logging

Each run writes `pipeline_run.json`, `feature_catalog.json`, a timestamped log
under `OUTPUT/logs/`, and per-clip workspaces containing feature JSON and any
generated scene/3D outputs. Failed clips are recorded in `failed_clips`; other
clips continue by default. Use `--fail-fast` to stop at the first error, and
`--log-level DEBUG` for more console detail. Exit codes are 0 for complete,
2 for complete with failed clips, and 1 for pipeline failure.

For old local runs, manifest input paths and saved depth `source_run` paths
under the removed outer folder are resolved to their new locations. Historical
metadata and its checksums are preserved. Newly generated exports use the new
layout. Update external scripts/bookmarks to the new entry points and folders.

## Updating an existing checkout

Git relocates the tracked source files, not ignored local recordings or results.
After switching to this layout, existing private assets may therefore remain
under `video_processing/`. Move the contents of its `data/`, `videos/`, `source/`,
`docs/` and `reports/` folders into the corresponding root folders. Merge folder
contents without overwriting existing files; retain the new tracked READMEs.
The old nested code folder is replaced by `processing/` and the root entry point,
so it should not be copied back over the new code.

The old outer folder remains ignored to keep any leftover private assets out of
Git while each collaborator migrates their own files. Remove it once empty.
Personal LingBot runtime configuration remains local; update its location in
your command and any external shortcuts to the new layout.

## More detail and tests

- [2D processing, recorder behaviour and JSON schema](processing/README.md)
- [3D/LingBot pipeline and validation](geometry3d/README.md)
- [Collected-data layout](data/README.md)

```sh
python pipeline.py --help
python -m unittest discover -s tests -v
```

Recordings, annotated previews, pose tracks and reconstructions may contain
identifying information. The current pipeline retains them for validation;
it does not implement a production retention/deletion policy or prove anonymity.
