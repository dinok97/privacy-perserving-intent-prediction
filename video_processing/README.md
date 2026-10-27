# Privacy-preserving pedestrian intent prediction

## 1. Project

An automatic door should open for people who are about to **enter**, and stay
shut for people who only **pass** by or **exit**. Jesper Malmgren's master's
thesis (SeamAI) showed that this intent can be predicted from a person's pose
and trajectory, using a single camera above one door.

This project extends that work so the prediction **generalises to new
locations**: cameras mounted at different heights and angles, rooms with
different lighting, and places with more or less traffic. To do that, the
pipeline here turns raw videos into JSON files that describe:

- **the person** – position, speed, heading, body orientation, gait;
- **the scene** – floor, walls, windows, other doors and other ways out;
- **how the two relate** – distance to walls, which exit the person is heading
  for, how long until they reach the door.

The aim is breadth: extract as many features as possible and document each one,
so the supervisors and the later classification model can choose which to use.

## 2. Table of contents

1. [Project](#1-project)
2. [Table of contents](#2-table-of-contents)
3. [Structure](#3-structure)
4. [Setup](#4-setup)
   - [4.1 General setup](#41-general-setup)
   - [4.2 SAM 3 setup](#42-sam-3-setup)
5. [Running](#5-running)
   - [5.1 Running the whole pipeline](#51-running-the-whole-pipeline)
   - [5.2 Running scripts separately](#52-running-scripts-separately)
6. [Output](#6-output)
7. [Known gaps](#7-known-gaps)
8. [Credits and privacy](#8-credits-and-privacy)

## 3. Structure

```
video_processing/
  README.md                     this file
  FEATURES.md                   every feature in the output JSON
  requirements*.txt             Python dependencies (see Setup)
  check_env.py                  checks that the main packages import
  pipeline.py                   runs all steps below in order   (coming soon)
  video_processing/             the pipeline steps, one script per step
    README.md                   how to run each script on its own
    fisheye_correction.py       1. fisheye corrector
    video_processor.py          2. trajectory tracker
    background_extraction.py    3. background extractor
    background_segmentation.py  3. segmentation (batch, no GUI)
    video_segmentation.py       3. segmentation (with hand correction)
    extract_features_o.py       4. feature extractor
  track_single_video.py         ┐
  background_depth.py           │ 3D / depth steps (optional)
  segment_background_3d.py      │
  door_distance_3d.py           ┘
  source/                       input videos go here (git-ignored)
  data/                         all output is written here (git-ignored)
```

### Pipeline

Every step is its own script, so any step can be re-run or replaced alone.

```
video.mp4
  │ 1. fisheye corrector         detect lens distortion, undistort the whole video
  │    (fisheye_correction.py)
  │ 2. trajectory tracker        track people, label enter / pass / exit
  │    (video_processor.py)      → <clip>.json, <clip>_pose.json
  │ 3. background extractor      per-pixel median → image of the empty room
  │    + segmentation            SAM 3 → floor / wall / door / window polygons
  │    (background_extraction.py, background_segmentation.py)
  │                              → <clip>_background.png, <clip>_background.json
  │ 4. feature extractor         combine person + scene → features per frame
  │    (extract_features_o.py)
  ▼
features_<clip>.json
  │ 5. (optional) depth + 3D scripts → distances in metres
```

Why this order:

- **Undistort first.** Depth and 3D mapping assume a straight-line (pinhole)
  camera. If only the background were undistorted, the person's trajectory and
  the scene would live in different geometries and every distance between them
  would be wrong. So both the tracker and the background step read the
  undistorted video.
- **Track before the background.** The tracker cuts the video into clips and
  names them `<clip>` = `<video>_<action>`. The scene files are named after
  those clips (`<clip>_background.json`), so the clip names have to exist first.
- **Median background.** People move, so at each pixel the most common value
  over the video is the empty room. Segmenting that one image gives the static
  scene once per camera position instead of once per frame.
- **Features before aggregation.** Per-frame values are kept, so any window or
  summary statistic can be computed later without re-running the pipeline.

## 4. Setup

### 4.1 General setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt                       # steps 1–5
pip install -r requirements-background-depth.txt     # step 6 (3D), optional
python3 check_env.py
```

For CPU-only Linux, install the CPU PyTorch wheels first if needed:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Download `yolo11n-pose.pt` and place it in this folder (model weights are
git-ignored).

### 4.2 SAM 3 setup

The background segmentation scripts use the `facebook/sam3` model through Hugging Face Transformers. 

### Prerequisites

The model is gated: user must request and receive access before the scripts can download its weights. What to do are:

1. Sign in to Hugging Face or [create an account](https://huggingface.co/join).
2. Open the [`facebook/sam3` model page](https://huggingface.co/facebook/sam3),
   select **Agree and access repository** (or the equivalent access request
   button), and review/accept the model's terms. Wait for access to be granted.
   Approval is tied to your Hugging Face account; installing the Python package
   alone does not grant access.

**Storage requirements**: The pretrained SAM 3 model requires approximately **6.9 GB** for the full set of model files. 
Additional disk space is needed for Python dependencies, cached files, datasets, and segmentation outputs. Ensure sufficient free storage before the first run.

### Steps

1. Install all required dependency by run command `pip install -r requirements.txt`.
2. Get Hugging Face access token. Read more [here](https://huggingface.co/docs/hub/main/en/security-tokens#how-to-manage-user-access-tokens).
3. Authenticate Hugging Face by run command `hf auth login`. Then enter you Hugging
Face access token when prompted. 
4. Done. The model will be downloaded at the first time of running the pipeline. Note that the model

### Suggested system requirements

- **Operating system**: Linux, Windows, or another environment supported by the required Python dependencies.
- **Processing hardware**: A GPU is recommended for faster inference. CPU execution may be possible but significantly slower.
- **RAM**: 16 GB is a reasonable starting point; 32 GB or more is preferable.
- **Storage**: More than 10GB free storage available, since the `SAM3` model itself will take around 7GB of storage.


## 5. Running

### 5.1 Running the whole pipeline

> `pipeline.py` is not in the repository yet. This section describes how it
> will work; update the commands once it is merged.

```bash
python pipeline.py <video or folder>          # final JSON only
python pipeline.py <video or folder> --debug  # all intermediate files too
```

| Mode | Writes |
|---|---|
| default | `features_<clip>.json` + `feature_catalog.json` |
| `--debug` | the above, plus every intermediate file: lens JSON, background image, segmentation JSON + overlay, trajectory and pose JSON, feature debug overlay |

Use `--debug` when setting up a new camera location: the overlay images show at
a glance whether the floor, walls and exits were found correctly.

### 5.2 Running scripts separately

Each step can also be run on its own, e.g. to re-segment a scene without
re-tracking. Commands, inputs and the files each script writes are listed in
[video_processing/README.md](video_processing/README.md).

## 6. Output

The final output per clip is **`features_<clip>.json`**, plus one
**`feature_catalog.json`** per output folder that lists every feature with its
unit, the inputs it needs and whether it is safe for online prediction.

Every feature – layout, exits, units and meaning – is described in
**[FEATURES.md](FEATURES.md)**.

## 7. Known gaps

- **TODO**

## 8. Credits and privacy

`video_processor.py` is based on
[pose_data_collection.py](https://github.com/MirianaPassarotto/SeamAI/blob/main/automatic_door_project/intent_prediction_2026/data_collection_v2/data_collection/pose_data_collection.py)
by Jesper Malmgren.

Videos and pose data can identify people. `source/`, `data/`, videos and
images are git-ignored — never commit recordings.
