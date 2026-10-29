# Privacy-preserving pedestrian intent prediction

## 1. Project

An automatic door should open for people who are about to **enter**, and stay
shut for people who only **pass** by or **exit**. Jesper Malmgren's master's
thesis showed this intent can be predicted from a person's pose and trajectory,
seen by one camera above the door.

This project makes that work **generalise to new locations**: other camera
heights and angles, other lighting, more or less traffic. The pipeline turns
raw videos into JSON features that describe:

- **the person** – position, speed, heading, body orientation, gait;
- **the scene** – floor, walls, windows, other doors and other ways out;
- **how the two relate** – distance to walls, which exit the person is heading
  for, time until they reach the door.

The goal is breadth: extract as many documented features as possible, so the
supervisors and the later classifier can choose which ones to use.

## 2. Contents

1. [Project](#1-project)
2. [Contents](#2-contents)
3. [Structure](#3-structure)
4. [Setup](#4-setup)
5. [Running](#5-running)
6. [Output](#6-output)
7. [More details and tests](#7-more-details-and-tests)
8. [Credits and privacy](#8-credits-and-privacy)

## 3. Structure

```
privacy-perserving-intent-prediction/
  pipeline.py          main entry point: runs all steps below
  FEATURES.md          every feature in the output JSON
  processing/          the 2D steps, one script each      → processing/README.md
  geometry3d/          optional 3D step (depth, distances) → geometry3d/README.md
  source/              input videos + a simple recorder    → source/README.md
  data/                all output (git-ignored)            → data/README.md
  tests/               pipeline and geometry tests
  project_paths.py     path helpers, keeps old local runs working
  requirements.txt     Python dependencies
  check_env.py         checks that the main packages import
```

### Pipeline

```
video
  │ 1. fisheye correction      undistort the video if the lens bends straight lines
  │ 2. trajectory tracking     YOLO pose → one clip per person, labelled enter / pass / exit
  │ 3. background extraction   image of the empty room, people masked out
  │ 4. segmentation            SAM 3 → floor / wall / door / window polygons
  │ 5. feature extraction      person + scene → features per frame
  ▼
<clip>_features.json
  │ 6. (optional) 3D           depth → distances to doors in estimated metres
```

The video is undistorted **first**, so the person's trajectory and the scene
are in the same geometry. What each script does is in
[processing/README.md](processing/README.md).

## 4. Setup

### 4.1 Clone the repository

```bash
git clone https://github.com/dinok97/privacy-perserving-intent-prediction.git
cd privacy-perserving-intent-prediction
```

All commands in this README are run from this folder. Get updates later with
`git pull`. Videos, model weights and outputs are not in the repository.

### 4.2 Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 check_env.py
```

- CPU-only Linux: install the CPU PyTorch wheels first,
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- The YOLO pose model (`yolo11n-pose.pt`) is downloaded into this folder
  automatically on the first run.

### 4.3 SAM 3

Segmentation uses the gated `facebook/sam3` model from Hugging Face.

1. Sign in to Hugging Face (or [create an account](https://huggingface.co/join)).
2. On the [`facebook/sam3` model page](https://huggingface.co/facebook/sam3),
   select **Agree and access repository** and wait for access to be granted.
   Access is tied to your account; installing the package alone is not enough.
3. Create an [access token](https://huggingface.co/docs/hub/main/en/security-tokens#how-to-manage-user-access-tokens)
   and log in with `hf auth login`.
4. The model (~7 GB) is downloaded the first time the pipeline segments a scene.

**Suggested system:** a GPU (CPU works but is much slower), 16 GB RAM (32 GB
preferred), and more than 10 GB of free disk space.

### 4.4 3D extension (optional)

```bash
pip install -r geometry3d/requirements-background-depth.txt
```

New depth predictions need [LingBot-Map](https://github.com/Robbyant/lingbot-map)
in a **separate** checkout and Python environment. Setup (local or SSH/Docker),
units and calibration are described in [geometry3d/README.md](geometry3d/README.md).

## 5. Running

### 5.1 From raw videos

Put videos in `source/`, then:

```bash
python pipeline.py
```

This runs steps 1–5 on every video and writes to `data/pipeline/`.

| Option | Use |
|---|---|
| `--source-dir DIR` | read videos from another folder |
| `--output DIR` | write somewhere other than `data/pipeline/` |
| `--device cuda` | run YOLO on the GPU |
| `--skip-undistort` | keep the original image geometry |
| `--fail-fast` | stop at the first failing clip instead of skipping it |
| `--show-output` | also show the raw output of YOLO / SAM 3 in the console |

When moving the camera, check the door entrance region (`ellipse_axes` in
`processing/video_processor.py`), since the enter / pass / exit labels depend on it.

### 5.2 Reusing saved clips

To skip fisheye correction and tracking (and SAM 3, if scenes are given), list
existing clips in a manifest. Paths are relative to the manifest file:

```json
[
  {
    "video": "videos/clip.mp4",
    "trajectory": "data/existing/clip.json",
    "pose": "data/existing/clip_pose.json",
    "scene": "data/existing/clip_background.json"
  }
]
```

Only `video` and `trajectory` are required; missing files are found by name or
generated.

```bash
python pipeline.py --processed-manifest inputs.json
```

### 5.3 With 3D features

```bash
python pipeline.py --processed-manifest inputs.json --with-3d --lingbot-runs data/lingbot
```

Cached depth in `data/lingbot/` is used when it matches the video. For new
predictions, copy `geometry3d/lingbot.local.example.json`, fill in your
LingBot paths and add `--lingbot-config <your file>`.

The 3D values use the whole clip, so they are **not causal**, and metres are
estimated from an assumed door width (`--door-width-m`, default 0.825).

### 5.4 Single steps

Each step can also run on its own, e.g. to re-segment a scene without
re-tracking. See [processing/README.md](processing/README.md).

## 6. Output

Everything goes to `data/pipeline/` (layout in [data/README.md](data/README.md)):

- `<clip>_features.json` per clip – the features, explained in [FEATURES.md](FEATURES.md);
- `feature_catalog.json` – every feature with its unit, inputs and whether it is causal;
- `pipeline_run.json` – run status, including `failed_clips`;
- `logs/pipeline_<timestamp>.log` – full log of the run.

A failing clip is logged and skipped. Exit codes: `0` complete, `2` complete
with failed clips, `1` pipeline failed.

## 7. More details and tests

- [processing/README.md](processing/README.md) – what each 2D step does
- [FEATURES.md](FEATURES.md) – every feature in the output JSON
- [geometry3d/README.md](geometry3d/README.md) – 3D / LingBot setup and validation
- [data/README.md](data/README.md) – where the output is written
- [source/README.md](source/README.md) – input videos and the recorder

```bash
python pipeline.py --help                      # all pipeline options
python -m unittest discover -s tests -v        # run the tests
```

## 8. Credits and privacy

`processing/video_processor.py` is based on
[pose_data_collection.py](https://github.com/MirianaPassarotto/SeamAI/blob/main/automatic_door_project/intent_prediction_2026/data_collection_v2/data_collection/pose_data_collection.py)
by Jesper Malmgren.

Videos, pose tracks, previews and 3D reconstructions can identify people. They
are git-ignored – never commit recordings. The pipeline keeps them for
validation; it has no retention or deletion policy and does not guarantee
anonymity.
