# privacy/ – dropping identity-bearing features (F0–F3)

> **Status: proposal, not yet discussed with the group.** Branch
> `privacy-feature-audit`. Nothing in the existing pipeline is changed: this
> folder only *adds* a post-processing script, its tests and this document.
> Line numbers refer to commit `757ad53`.

## Contents

1. [What this is](#1-what-this-is)
2. [Why: the evidence](#2-why-the-evidence)
3. [What is dropped: tiers F0–F3](#3-what-is-dropped-tiers-f0f3)
4. [What stays](#4-what-stays)
5. [Dependency map](#5-dependency-map)
6. [Other files that hold raw skeletons](#6-other-files-that-hold-raw-skeletons)
7. [Using the temporary script](#7-using-the-temporary-script)
8. [Proposed permanent solution](#8-proposed-permanent-solution)
9. [Limits and open questions](#9-limits-and-open-questions)

## 1. What this is

The final `<clip>_features.json` contains a few fields that let someone infer
who walked through the door: body height, body proportions and gait cues. A
camera in an office sees the same few people every day, so "a short person, a
woman, a long stride" narrows a small group down quickly. The plan is to run
the model on versions of the features **with these fields removed** and see
how much accuracy that costs:

| Set | Meaning |
|---|---|
| **F0** | all features, as today (baseline) |
| **F1** | F0 without absolute body size |
| **F2** | F1 without body proportions |
| **F3** | F2 without gait cues |

The tiers are cumulative, so the accuracy drop can be attributed to one group
at a time (the same idea as Jesper's progressive pose ablation, Group B).

## 2. Why: the evidence

We cannot run a re-identification attack (no identity labels, no time), so
the choice is argued from the literature as **data minimisation** (the
principle in GDPR Art. 5(1)(c), already cited in Jesper's thesis, ref. [16]).

| Source | What it shows | Supports |
|---|---|---|
| Moon et al., *Anonymization for Skeleton Action Recognition*, AAAI 2023 ([arXiv:2111.15129](https://arxiv.org/abs/2111.15129)) | Skeleton sequences leak private attributes: about 87 % gender and about 80 % top-1 identity accuracy on unanonymized skeletons. Adding random noise removes identity only by also destroying action accuracy. | Skeletons are not anonymous by themselves; **drop, don't add noise** |
| Cătrună, Cosma, Rădoi, *The Paradox of Motion: Evidence for Spurious Correlations in Skeleton-based Gait Recognition Models* ([arXiv:2402.08320](https://arxiv.org/pdf/2402.08320)) | Skeleton gait models rely on height and limb proportions. Removing height by skeleton-level scaling lowered mean CASIA-B accuracy from 65.22 % to 55.14 %; a single-pose model with no motion reached 57.49 %. | **F1 (size), F2 (proportions)** |
| Hanisch et al., *Understanding Person Identification through Gait* ([arXiv:2203.04179](https://arxiv.org/abs/2203.04179)) | Tests body parts and feature categories one at a time. Sex depends more on static than on dynamic features; identity stays high after removing single categories (an average pose alone still identified people at 91–94 %). Gait is highly redundant. | **F2 (static shape), F3 (gait)**, and the limit of this approach |

Weak spots, to state openly in the report:

- The studies use gait benchmarks and a small lab study (57 young adults), not
  ceiling-mounted office video and not our derived scalar features.
- Hanisch et al. show that removing single features leaves residual risk. F1–F3
  **reduce** the risk; they do not make the data anonymous.
- No paper I found isolates a hip/shoulder ratio → gender for skeletons; F2
  rests on the static-shape results above.
- Not yet agreed as sources: Xu et al., WACV 2021 (smaller stride and arm
  swing for females; would back F3 directly) and de Montjoye et al., *Unique in
  the Crowd*, Sci. Rep. 2013 (route and time make traces unique; backs the
  metadata rule).

Suggested wording for the report: *"We removed features that the literature
associates with identity and attribute leakage. Residual risk remains and was
not empirically tested."*

## 3. What is dropped: tiers F0–F3

### Per-frame fields (12 distinct names, 7 + 3 + 2)

| Tier | Field (in `data.frames[i]`) | Why it is identity-bearing |
|---|---|---|
| F1 | `pos.bbox_h_px` | Body height in pixels. At a given ground position, bbox height is proportional to real height. |
| F1 | `pos.bh_ref_px` | The body-height reference used for every `_bh` value. Same information. |
| F1 | `pose.body_height_px` | Nose to mid-ankle in pixels. |
| F1 | `kin.speed_px_s` | Twin of `speed_bh_s`: `px/s ÷ bh/s = bh_ref_px`. |
| F1 | `kin.curvature_per_px` | = turn rate ÷ pixel speed, so `turn_rate ÷ curvature ÷ speed_bh_s = bh_ref_px`. |
| F1 | `scene_rel.dist_block_px` | Twin of `dist_block_bh`. |
| F1 | `scene_rel.target.dist_px` (and the copies in `competitors[]` and `exits[]`) | Twin of `dist_bh`. |
| F2 | `pose.shoulder_width_bh` | Static body proportion. |
| F2 | `pose.hip_width_bh` | Static body proportion (with shoulders: a body-shape ratio). |
| F2 | `pose.leg_length_bh` | Static body proportion. |
| F3 | `pose.ankle_sep_bh` | Step length / stride proxy (gait). |
| F3 | `pose.wrist_spread_bh` | Arm-swing proxy (gait). |

Why the "twins" are in F1: dropping only `bbox_h_px` would be pointless while
`speed_px_s` and `speed_bh_s` sit side by side. I checked on a synthetic walk
that `speed_px_s ÷ speed_bh_s` and `dist_px ÷ dist_bh` reproduce `bh_ref_px`
within rounding (≤ 0.2 %). The curvature route also works (median error 3 %, 80
% of frames within 10 %; it is blurred where the 3-decimal rounding of small
curvature values bites).

### Track level (in F1)

| Field | Reason |
|---|---|
| `track_summary.causal.first_bh_px` | Body height in pixels at first detection. |
| `track_summary.hindsight.camera_geom.bh_vs_y_slope`, `bh_vs_y_intercept` | The fit of bbox height against ground y; the slope scales with person height. `horizon_y_px` (= −intercept/slope) is a camera property and **stays**, as do `bh_vs_y_r2`, `n_points`, `source`. |

### 3D extension (only with `--with-3d`; removed for F1–F3 unless `--keep-3d-metres`)

| Field | Reason |
|---|---|
| `geometry_3d.point_ground_estimated_m`, `distances_estimated_m`, `closure_rates_estimated_m_per_s`, and `info.depth.metric_calibration` | **Hypothesis, not verified on real 3D data.** Ground speed in metres per second (from consecutive `point_ground_estimated_m`) divided by `speed_bh_s` gives metres per body height, i.e. body height in metres. The metric scale comes from an assumed door width and is shared across clips, so it is comparable between people. The `_model` fields use an arbitrary per-clip scale and are kept. |

### Metadata (`info.*`; pseudonymized for F1–F3, `--metadata off` to disable)

| Field | Action | Reason |
|---|---|---|
| `clip`, `clip_video` | removed | File names (may contain date/time). |
| `clip_video_sha256`, `depth.source_video_sha256` | removed | Links the features to the raw video file. |
| `trajectory_meta.session_id` | removed | = recording date + clock time (`video_processor.py:644`). |
| `sources.*` | file names replaced by `true`/`false` | Names are derived from the clip name. |
| `clip_id` | added only with `--key-file` | `HMAC(secret, clip name)`, to join with a private lookup. |

## 4. What stays

29 per-frame fields, among them: `pos.ground_px/ground_norm`, `kin.speed_bh_s`,
`accel_bh_s2`, `heading_deg`, `turn_rate_deg_s`, `looming_rate_per_s`, `tau_s`,
`pose.shoulder_angle_deg`, `torso_vs_motion_deg`, `head_yaw_proxy`,
`ear_vis_asym`, all `scene_rel` `_bh`/alignment/`ttr_s` fields, and the whole
`scene_summary`. These are relative, angular or describe the room.

Worth a second look by the group (kept for now): `first_ground_norm`,
`origin_exit`, `hindsight` speed and duration statistics and the entry/exit
edges. Alone they do not identify anyone, but route plus time of day can.

## 5. Dependency map

Rule of thumb: **all removed fields can still be needed *inside* the
extractor.** Removing them from the output is easy; removing the computation
is not.

### 5.1 Inside `processing/feature_extractor.py`

| Field | Computed at | Also feeds | Safe to omit from output? |
|---|---|---|---|
| `pos.bh_ref_px` | 767–773 | Every `_bh` value (836–837, 861, 869, 900); the "moving" gate (822) → `heading_deg` (823) → `turn_rate`, `curvature`, all alignment/`ttr_s`, `torso_vs_motion_deg`; origin check (917–922); path and speed statistics (937–938) | **Yes, but keep computing it.** |
| `pos.bbox_h_px` | 764 | `bh_ref` (770); looming (796–802) → `looming_rate`, `tau_s`; truncation flag; `camera_geom` fit (943–947) | Yes, keep computing. |
| `kin.speed_px_s` | 821, 835 | `sp` drives `accel`, the moving gate, `speed_bh_s`, `curvature`, `turn_rate` | Yes (emit only). |
| `*.dist_px` | `rel()` 869 | `dist` also gives `ttr_s` (877–879). Competitor ranking uses only the alignment angles (883–884). | Yes (emit only). |
| `scene_rel.dist_block_px` | 899 | same call produces `dist_block_bh` | Yes (emit only). |
| `pose.shoulder_width_bh` | 685–689 | `sw` also feeds `head_yaw_proxy` (707–710) | Yes, but keep computing `sw`. |
| `pose.hip_width_bh` | 701–703 | `hm` is also the guard for `leg_length_bh` (720) and `wrist_spread_bh` (725) | Yes. Keep `hm` while `wrist_spread_bh` is still emitted (F0–F2). |
| `camera_geom` slope/intercept | 941–952 | `horizon_y_px` = −b/a, which stays | Yes, keep computing. |
| `pose.body_height_px` | 679–683 | nothing | Yes, nothing depends on it. |
| `kin.curvature_per_px` | 833 | nothing | Yes. |
| `pose.leg_length_bh`, `ankle_sep_bh`, `wrist_spread_bh` | 715–728 | nothing | Yes. |
| `track_summary.causal.first_bh_px` | 958 | nothing (the origin check uses a separate `bh0`, 917) | Yes. |
| `NO_POSE` | 733–735 | built from the keys of `pose_features()`, so it follows automatically | – |
| `CATALOG` | 1019–1064 | copied into the run-level `feature_catalog.json` (`pipeline.py` 550–555, `write_catalog` 1118) | 9 entries must go: `pos.bh_ref_px`, `pose.body_height_px`, `pose.shoulder_width_bh`, `pose.hip_width_bh`, `pose.leg_length_bh`, `pose.ankle_sep_bh`, `pose.wrist_spread_bh`, `kin.curvature_per_px`, `track_summary.causal.first_bh_px` |
| `draw_debug` | 1068–1109 | reads only `pos.ground_px` | unaffected |

### 5.2 Outside the extractor

| Where | What happens there | Consequence |
|---|---|---|
| `pipeline.py:397–400, 417` | `run_clip` writes the **unfiltered** file (extractor 1140–1143); the pipeline reloads it and writes it again after the 3D join | A filter at line 417 would leave the unfiltered file on disk until then. A permanent fix must filter before the first write. |
| `pipeline.py:401–402` | Adds `clip_video` and `clip_video_sha256` to `info` | Remove after the join (see next row). |
| `geometry3d/pipeline_geometry.py:312` | `join_geometry` checks `info.clip_video_sha256` against the 3D summary; the lookup has a default, so a missing key **silently skips the check** | Filter after `join_geometry`, or pass the expected hash in as an argument. |
| `geometry3d/pipeline_geometry.py:318–323` | Writes the `geometry_3d` block including `_estimated_m*` | Metres-based leak, see §3. |
| `geometry3d/pipeline_geometry.py:324–328` | Writes `info.depth` with `source_video_sha256` and `metric_calibration` | Same. |
| `geometry3d/pipeline_geometry.py:342–353` | `catalog_entries()` lists the `_estimated_m` features | Filter with the catalog. |
| `geometry3d/pipeline_geometry.py:87, 133–137` | Reads and writes `clip_video_sha256` in the **trajectory** metadata to verify the clip | Works on the intermediate trajectory, not on `_features.json`. Unaffected. |
| `geometry3d/door_distance_3d.py:190–203` | `contact()` takes ankle keypoints and bbox from the tracking folder | Independent of the features JSON, but the raw keypoints must exist until the 3D step has run. |
| `processing/video_processor.py:644, 701, 1024, 1040` | `session_id` = date + clock time, also in the `session_<date>_<time>.json` filename; wall-clock times in the trajectory meta (590, 645, 647) | The extractor copies only `session_id` (feature_extractor.py 998–999). Stop copying it. |
| `tests/test_pipeline_geometry.py:151–153, 164` | Fixture and assertion use `scene_rel.target.dist_px` | Synthetic dict, so it passes now. Switch to `dist_bh` when the field is removed at the source. |
| `FEATURES.md` lines 51, 123, 133, 136, 148–154, 160, 162, 181, 184–188 | Document the dropped fields | Update per tier. |
| `README.md` §8, `geometry3d/README.md:422–425` | State that the pipeline keeps identifying material and does not guarantee anonymity | Still true; extend (see §6). |

Scan coverage: every field name above was searched in all `*.py`, `*.js`,
`*.md`, `*.json`, `*.txt` files of the repo. `feature_extractor.py`,
`pipeline.py` and `pipeline_geometry.py` were read in full; `video_processor.py`,
`door_distance_3d.py` and `track_single_video.py` only around the matches.
`fisheye_correction.py`, `background_extraction.py`, `background_segmentation.py`
and `lingbot_*` contain no references to these fields.

## 6. Other files that hold raw skeletons

Stripping the final JSON does **not** remove the identifying inputs next to it:

| File | Content | Deleted by `pipeline.py`? |
|---|---|---|
| `<clip>.mp4` | the video | no |
| `<clip>.json` (`video_processor.py:680`) | per frame: bbox and **17 keypoints** (incl. nose, eyes) in pixels, wall-clock times | no; kept in `output/tracks` (with `--with-3d` it lives in a temporary folder that is cleaned up at the end) |
| `<clip>_pose.json` (`:684–690`) | the same skeleton as named landmarks | no |
| `<clip>/geometry_3d/distances/tracking/detections.json`, `normalized_trajectory.json` (`pipeline_geometry.py:133–137`) | bbox and keypoints again | no |
| `*_background.png`, `*_features_debug.png` | scene image, overlays | only without `--debug` (`delete_debug_files`, `pipeline.py:422–436`) |
| `pipeline_run.json` | video path and SHA-256 per clip | no |

The privacy argument is only as strong as the weakest of these. This needs a
group decision (see §8, step 5).

## 7. Using the temporary script

```bash
# one file, tier F2
python privacy/strip_features.py --tier F2 --input data/pipeline/<clip>_features.json

# all clips, tier F3, separate folder, plus a filtered catalog
python privacy/strip_features.py --tier F3 --input-dir data/pipeline --out-dir data/privacy_F3 \
       --catalog data/pipeline/feature_catalog.json

# dry run: only report what would be removed
python privacy/strip_features.py --tier F1 --input-dir data/pipeline --dry-run

# add a pseudonymous clip_id (keep the key file private and out of git)
python privacy/strip_features.py --tier F2 --input <clip>_features.json --key-file secret.key
```

- The input is never modified. Output: `<clip>_features_<tier>.json`, and
  `feature_catalog_<tier>.json` with `--catalog`. Keys are **deleted**, not set
  to `null`, so a loader that expects fixed keys must read the catalog.
- Each output gets `info.privacy` (tier, removed fields, the "not verified
  anonymisation" note).
- A file is written only if an audit finds none of the removed names left
  anywhere in it. `dist_px` under `track_summary.causal.origin_exit` is
  allowed: it has no `_bh` twin in the output.
- For F1–F3 the script also drops the 3D metre fields and pseudonymizes the
  metadata (flags above).
- Tests: `python -m unittest tests.test_strip_features -v` (10 tests, no
  recordings needed; the features are produced by the real extractor from a
  synthetic walk, so they follow the extractor's schema).
- Checked only on synthetic data. No real recordings or 3D outputs were
  available.

## 8. Proposed permanent solution

Goal: the pipeline emits a privacy-reduced file **by construction**, without
writing a full JSON first. The idea is "compute what the model needs, emit only
what the policy allows".

1. **One policy, one place.** Move the field lists from `strip_features.py`
   into `privacy/policy.py` (tiers, paths, metadata rules). Both the extractor
   and the script import it, so they cannot drift apart.
2. **`processing/feature_extractor.py`** (the main change)
   - Add a `tier` argument to `extract()`, `run_clip()` and
     `extract_features()`; the default comes from the policy.
   - Keep computing `bh_ref`, `bbox_h`, `sp`, `dist` and `sw` (§5.1). Filter
     only where the dicts are assembled: the `pos`/`kin`/`pose`/`scene_rel`
     blocks (lines 834–909) and the final rounding loop (979–986). That is a
     small diff, and since `run_clip` serializes the result it never writes the
     removed fields.
   - `pose_features()`: return everything internally, filter at the end;
     `NO_POSE` then follows the filtered keys.
   - `track_summary` and `info` (954–1007): leave out `first_bh_px`, the
     `camera_geom` slope and intercept, `session_id` (998–999), the clip name
     (990) and the source file names (1133–1138).
   - `CATALOG` and `write_catalog`: filter by the same policy.
3. **`pipeline.py`**
   - Add `--privacy-tier` to `main()` and `run_pipeline()` and pass it to
     `run_clip`.
   - Remove lines 401–402. `result` already carries the video hash (403–404),
     so nothing is lost; the hash then lives only in `pipeline_run.json`, which
     is a private file (step 5).
   - Filter the catalog (550–555) by tier.
4. **`geometry3d/pipeline_geometry.py`**
   - `join_geometry` (303–339): emit only the `_model` fields unless the tier is
     F0; no `metric_calibration` or `source_video_sha256` in `info.depth`;
     compare hashes through an explicit argument instead of
     `features['info']` (line 312).
   - `catalog_entries()`: drop the `_estimated_m` rows by tier.
5. **Private zone** (needs a group decision). Add a cleanup next to
   `delete_debug_files` (`pipeline.py:422`) that removes `<clip>.json`,
   `<clip>_pose.json` copies in `output/tracks`, the
   `geometry_3d/distances/tracking/*.json` files and the per-clip hash record
   once features and 3D are done, unless `--debug` is set. Without this the
   stripped JSON sits next to the full skeleton.
6. **`processing/video_processor.py`**: no change needed for the features. Optional:
   stop writing wall-clock times and the date-time `session_id` into the
   trajectory metadata (590, 645, 647), or keep them and rely on step 5.
7. **Tests and docs.** Switch the fixture in
   `tests/test_pipeline_geometry.py` (151–164) to `dist_bh`; add an extractor
   test asserting that the emitted keys equal the catalog for each tier, and one
   asserting that `extract(tier=T)` equals `strip(extract(tier=F0), T)`. Update
   `FEATURES.md` per tier and `README.md` §8.
8. **Ablation stays possible.** Generate F0 once in a private location and
   derive F1–F3 with the script for the experiments; the shipped dataset is
   produced at the chosen tier. Step 7's equivalence test guarantees that both
   routes give the same result.

Decisions for the group: the default tier (F2 or F3?), keeping or dropping the
estimated metres, the retention rule for raw tracks (step 5), who holds the
pseudonym key, and whether position plus time features (`first_ground_norm`,
`origin_exit`, hindsight statistics) are acceptable.

## 9. Limits and open questions

- **Not verified:** nothing here measures re-identification. The tiers rest on
  the literature in §2 and on the arithmetic in §3 (checked on synthetic data).
- **3D leak:** the metres-over-body-height argument is derived from the schema,
  not tested on real LingBot output.
- **Utility cost unknown:** `shoulder_width_bh` and `hip_width_bh` shrink when
  the body turns, which is an intent cue. The kept angles (`torso_vs_motion_deg`,
  `shoulder_angle_deg`) may cover it. The F0–F3 comparison will tell.
- **Residual risk:** gait is redundant; position, time and route features stay.
  The result is "reduced risk", not "anonymous".
