# Features

## Contents

1. [What this file describes](#1-what-this-file-describes)
2. [Layout](#2-layout)
3. [Exits](#3-exits)
4. [Features](#4-features)
   - [4.1 Scene summary](#41-scene-summary-static-per-clip)
   - [4.2 Position (`pos`)](#42-position-pos)
   - [4.3 Movement (`kin`)](#43-movement-kin)
   - [4.4 Pose (`pose`)](#44-pose-pose)
   - [4.5 Person × scene (`scene_rel`)](#45-person--scene-scene_rel)
   - [4.6 Track summary](#46-track-summary)
5. [Parameters](#5-parameters)

## 1. What this file describes

The content of **`features_<clip>.json`** (schema `feat-0.3`), the final output
of the pipeline. It is produced by `extract_features_o.py` from three inputs:

- the trajectory `<clip>.json` (where the person is, per frame),
- the pose sidecar `<clip>_pose.json` (body keypoints),
- the scene segmentation `<clip>_background.json` (floor, walls, doors, windows).

Every run also writes **`feature_catalog.json`**: one entry per feature with its
name, level (frame / track / scene), unit, required inputs and whether it is
causal. Use it to select features programmatically.

### Units (suffix in every name)

| suffix | unit |
|---|---|
| `_px` | pixels |
| `_bh` | body heights (median of the last 3 untruncated bbox heights) |
| `_s` | seconds |
| `_deg` | degrees, image coordinates: 0 = right, 90 = down |
| `_norm` | fraction of frame width / height |

## 2. Layout

```jsonc
{
  "info": {
    "schema_version", "clip", "label",
    "sources",          // which input files were used
    "segmentation",     // labels present in the scene
    "trajectory_meta",  // session, fps, tracker, git commit
    "frame_size", "params", "units", "notes"
  },
  "data": {
    "scene_summary": { ... },                         // 4.1 – once per clip
    "track_summary": { "causal": {...}, "hindsight": {...} },   // 4.6
    "frames": [
      { "frame_index", "t", "detected",
        "pos": {...},        // 4.2
        "kin": {...},        // 4.3
        "pose": {...},       // 4.4
        "scene_rel": {...}   // 4.5
      }
    ]
  }
}
```

Frames where the person was not detected keep only `frame_index`, `t` and
`detected: false`, so frame indices stay continuous.

## 3. Exits

An **exit** is anywhere a person can leave the scene other than the target
door. Doors alone are not enough: outside a shopping mall there may be no other
door, and someone walking along a corridor leaves through the edge of the image.
So there are three automatic types:

| type | found from | example |
|---|---|---|
| `door` | door polygon; its threshold is where the outline touches the floor | a side door in the room |
| `edge` | a gap in the frame border that the floor reaches and no wall/window blocks | a corridor running out of the image |
| `floor_tip` | a part of the floor that sticks out and is cut off by a wall | the floor turning round a corner |

- `status`: `detected` (door polygon), `inferred` (edge / floor tip) or
  `manual` (from `exits_<clip>.json`, which replaces the automatic exits).
- The **target** — `door_center` in the trajectory JSON, the door the camera
  controls — is never an exit.
- Exit indices are only valid **within one file**; there are no stable IDs
  across clips.
- Check exits visually with `--debug`: the `_debug.png` draws every exit with
  its index.

## 4. Features

### 4.1 Scene summary (static, per clip)

`data.scene_summary`

| field | meaning |
|---|---|
| `target.*` | target centre, threshold segment, whether its door polygon is visible |
| `exits[]` | every exit: type, edge, anchor, segment, width, status, direction |
| `counts.*` | `n_exits`, `n_door`, `n_edge`, `n_floor_tip`, `n_manual` |
| `areas.*` | share of the image that is floor / wall / window / door / unlabelled |
| `floor_top_y_px` | highest floor row — proxy for camera tilt |
| `brightness_mean`, `brightness_contrast` | lighting of the background (0–1) |
| `n_polygons_used`, `n_polygons_dropped` | segmentation quality |

### 4.2 Position (`pos`)

| field | meaning |
|---|---|
| `ground_px`, `ground_norm` | ground contact: mid-ankle, else bbox bottom (3-frame causal mean) |
| `ground_src` | `ankles` or `bbox_bottom` |
| `bbox_h_px`, `bh_ref_px` | bbox height; reference body height used for `_bh` |
| `bbox_truncated` | bbox touches the top/bottom of the frame |
| `reliable` | false when the ground point comes from a truncated bbox |
| `on_floor` | ground point lies on the segmented floor |
| `inside_exit_idx` | exit whose area contains the ground point |

### 4.3 Movement (`kin`)

| field | meaning |
|---|---|
| `speed_px_s`, `speed_bh_s` | ground speed |
| `accel_bh_s2` | change of speed |
| `heading_deg` | walking direction (`null` below 0.1 bh/s – direction is noise when standing) |
| `turn_rate_deg_s`, `curvature_per_px` | how sharply the path bends |
| `looming_rate_per_s` | d ln(bbox height)/dt; > 0 = getting bigger = approaching the camera |
| `tau_s` | time-to-contact = 1 / looming (Lee, 1976) |

Looming covers what pixel speed misses: someone walking straight towards the
camera barely moves in the image, but grows quickly.

### 4.4 Pose (`pose`)

| field | meaning |
|---|---|
| `kp_conf_mean`, `n_kp_visible` | keypoint quality – also a lighting proxy |
| `body_height_px` | nose to mid-ankle |
| `shoulder_width_bh`, `hip_width_bh` | shrink when the body turns sideways |
| `shoulder_angle_deg` | orientation of the shoulder line |
| `torso_vs_motion_deg` | 0 = facing the walking direction, 90 = sideways |
| `head_yaw_proxy`, `ear_vis_asym` | head turn (looking at the door?) |
| `ankle_sep_bh`, `leg_length_bh` | gait phase and stride |
| `wrist_spread_bh` | arm swing / carrying something |

### 4.5 Person × scene (`scene_rel`)

| field | meaning |
|---|---|
| `dist_block_px`, `dist_block_bh` | distance to the nearest wall or window |
| `occlusion_frac` | share of the bbox covered by wall / window / door |
| `target.dist_px`, `target.dist_bh` | distance to the target threshold |
| `target.align_deg` | 0 when the walking direction points into the target opening |
| `target.align_center_deg` | angle to the opening's centre (separates wide openings) |
| `target.ttr_s` | time to reach the target at the current approach speed |
| `competitors[0..1]` | the two best-aligned exits, same fields + type/status, `null`-padded |
| `most_aligned` | `"target"` or `"competitor"` |
| `target_minus_best_competitor_align_deg` | < 0 → the target explains the motion better |
| `exits[i]` | the same distance/alignment block for every exit |

`exits[]` has a different length in every scene, while most classifiers need a
fixed input size; `competitors` is that fixed-size version.

### 4.6 Track summary

`data.track_summary.causal` – safe for online use:

| field | meaning |
|---|---|
| `origin_exit` | exit the person came from, or `frame_edge` / `appeared_mid_frame` |
| `first_ground_norm`, `first_bh_px` | where and how big the person was when first seen |
| `n_concurrent_tracks` | reserved (`null`) until multi-track output is connected |

`data.track_summary.hindsight` – uses the whole track, **not for online use**:
entry/exit edge, duration, detection rate, path length, displacement,
straightness, mean/max speed, last position, and `camera_geom` — a linear fit of
bbox height vs. ground y whose zero crossing estimates the horizon (a proxy for
the unknown camera tilt).

## 5. Parameters

All thresholds are in `PARAMS` at the top of `extract_features_o.py` (e.g.
`min_gap_px` for the narrowest edge exit, `n_competitors`, `smooth_window`).
They are copied into `info.params` of every output, so each file records how it
was made.
