# Privacy-preserving pedestrian intent prediction

## 1. Project

An automatic door should open for people who are about to **enter**, and stay
shut for people who only **pass** by. Jesper Malmgren's master's
thesis showed that this intent can be predicted from a person's pose
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
   - [4.1 Clone the repository](#41-clone-the-repository)
   - [4.2 General setup](#42-general-setup)
   - [4.3 SAM 3 setup](#43-sam-3-setup)
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

Every step is its own script, so any step can later be re-run.

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

## 4. Setup

### 4.1 Clone the repository
 
All code lives in the group's GitHub repository. Clone it and move into the
`video_processing/` folder; every command below is run from there.
 
```bash
git clone https://github.com/dinok97/privacy-perserving-intent-prediction.git
cd privacy-perserving-intent-prediction/video_processing
```
 
To get the latest changes later:
 
```bash
git pull
```
 
Videos, model weights and outputs are not in the repository (they are
git-ignored), so after cloning you still need to add your own videos to
`source/` and download the model weights (see below).
 
### 4.2 General setup
 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt                       # steps 1–5
pip install -r requirements-background-depth.txt     # step 5 (3D), optional
python3 check_env.py
```
 
For CPU-only Linux, install the CPU PyTorch wheels first if needed:
 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
 
Download `yolo11n-pose.pt` and place it in this folder (model weights are
git-ignored).
 
### 4.3 SAM 3 setup

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

***TODO***

### 5.2 Running scripts separately

Each step can also be run on its own, e.g. to re-segment a scene without
re-tracking. Commands, inputs and the files each script writes are listed in
[video_processing/README.md](video_processing/README.md).

## 6. Output

***TODO***

## 7. Known gaps

- **TODO**

## 8. Credits and privacy

`video_processor.py` is based on
[pose_data_collection.py](https://github.com/MirianaPassarotto/SeamAI/blob/main/automatic_door_project/intent_prediction_2026/data_collection_v2/data_collection/pose_data_collection.py)
by Jesper Malmgren.

Videos and pose data can identify people. `source/`, `data/`, videos and
images are git-ignored — never commit recordings.
