#!/usr/bin/env python3
"""Validate the supervisors' pipeline output (feat-0.3) without needing their log.

Input: one or more SOURCE folders, each holding one clip folder per clip:

    <source>/<clip_folder>/{*_background.png, *_background_overlay.png, *_features_debug.png,
                            *_background.json, *_features.json}

A source is one pipeline run / one input folder of videos (e.g. masterdata_enter, masterdata_pass).
Its name is only a tag: the real class of each clip is info.label (the tracker's label), and the
report shows the source-by-label table. Sub-folders called logs, tracks or [TEMP]* are skipped.

If <source>/tracks exists (the tracker output: clips.csv, <clip>.json, <clip>_pose.json) every clip is
also cross-checked against it: clips.csv row, label (also recomputed from the trajectory with the
tracker's own door-ellipse rule), SHA chain video -> trajectory -> features -> folder name, frame counts,
detection flags, pose file. Use --tracks NAME=PATH to point elsewhere or --no-tracks to skip.

If <source>/logs exists (pipeline_<time>.log files) the run behind the data is checked too: the log
must end "complete", its clip count and label counts must match the folders (and clips.csv), every
folder name the log announced must exist (and no unknown folder), no stage may be incomplete, warnings
and errors are listed, each clip's session must belong to that run, and the fps the tracker logged for
the source video must equal the fps in the features. Earlier failed runs are listed as context.
Use --logs NAME=PATH to point elsewhere or --no-logs to skip.

Usage:
    python validate_dataset.py data/masterdata_enter data/masterdata_pass --out report/
    python validate_dataset.py enter_run=data/masterdata_enter pass_run=data/masterdata_pass \
        --expect enter_run=829 --expect pass_run=533

Writes to --out:
    clip_report.csv    one row per clip (quality numbers + flags) - sort/filter this
    issues.csv         one row per problem: clip, severity (ERROR/WARN/INFO), code, message
    feature_nulls.csv  per feature: how often it is null, per label, in how many clips
    summary.md         the overview: what to look at first and exactly what to check (start here)

Only the standard library is used. Thresholds are command-line options (see --help).
Severity: ERROR = file/schema broken or inconsistent, WARN = look at it, INFO = context.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

FILES = {  # role -> required filename suffix
    "features": "_features.json",
    "scene": "_background.json",
    "background_img": "_background.png",
    "overlay_img": "_background_overlay.png",
    "debug_img": "_features_debug.png",
}
LABELS = {"enter", "pass", "exit"}
SKIP_DIRS = {"logs", "tracks"}  # the pipeline writes these next to the clip folders
EXPECTED_SCHEMA = {"feat-0.3"}
FRAME_BLOCKS = ("pos", "kin", "pose", "scene_rel", "geometry_3d")

# Nulls that FEATURES.md says are by design, or that are structural in the sample files
# (edge is only set for edge-type exits). Matched against the normalised feature path.
DOCUMENTED_NULLABLE = [
    r"^kin\.heading_deg$",                  # null below 0.1 bh/s (FEATURES.md 4.3)
    r"^scene_rel\.competitors\[\]$",        # null-padded to n_competitors (4.5)
    r"\.edge$",                             # only edge exits have an edge (inferred from samples)
    r"n_concurrent_tracks",                 # reserved null (4.6)
]

UNIT_INTERVAL = {"kp_conf_mean", "confidence", "brightness_mean", "brightness_contrast",
                 "detection_rate", "path_straightness"}
NON_NEGATIVE = {"bbox_h_px", "body_height_px", "leg_length_bh", "shoulder_width_bh", "hip_width_bh",
                "ankle_sep_bh", "wrist_spread_bh", "ttr_s", "tau_s", "bh_ref_px"}
OUTLIER_STATS = ["n_frames", "detection_rate", "duration_s", "floor_frac", "wall_frac", "door_frac",
                 "brightness_mean", "n_exits", "path_length_bh", "mean_speed_bh_s"]


# ----------------------------------------------------------------------------- helpers

def png_size(path):
    """(width, height) from the PNG header, or None if it is not a PNG."""
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def leaves(obj, path=""):
    """Yield (path, value) for every leaf. List indices are collapsed to '[]'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v, path + "[]")
    else:
        yield path, obj


def number_rule(path):
    """(low, high) a numeric leaf must stay within, or None. Tolerance is added by the caller."""
    name = path.replace("[]", "").split(".")[-1]
    if name.endswith(("_frac", "_norm")) or name in UNIT_INTERVAL:
        return 0.0, 1.0
    if name == "n_kp_visible":
        return 0, 17
    if name.startswith(("speed_", "dist_")) or name in NON_NEGATIVE:
        return 0.0, math.inf
    if name.endswith("_deg"):
        return -360.0, 360.0
    return None


def shoelace(poly):
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def robust_z(values):
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values)
    return med, mad


# What to look for, per issue code (shown in summary.md and issues.csv)
HINTS = {
    "FILE_MISSING": "Open the clip folder: one of the 5 expected files is absent. Ask the supervisors to re-send that clip.",
    "FILE_DUPLICATE": "Two files match the same role (e.g. two *_features.json). Remove or ask which one is the real one.",
    "FILE_EMPTY": "A 0-byte file: broken copy or crashed write. Re-copy it from the source; if still empty, re-run the clip.",
    "FILE_EXTRA": "Unexpected extra files in the clip folder. Usually harmless (e.g. a pose file); check they are not leftovers.",
    "PNG_INVALID": "The image is not a valid PNG (truncated or renamed). Open it; re-copy or re-send.",
    "IMAGE_SIZE_MISMATCH": "The three images of this clip have different sizes. Open them side by side; the clip was probably mixed up.",
    "IMAGE_SIZE_VS_JSON": "Image size differs from frame_size in the JSON, so polygons would not line up with the image.",
    "JSON_UNREADABLE": "The JSON is truncated or corrupt. Open the named file; re-copy it, else the clip must be re-run.",
    "SCHEMA_KEYS": "features.json lacks an expected top-level key. Compare with a good clip; different schema version?",
    "SCHEMA_VERSION": "Not feat-0.3. The clip came from another pipeline version; do not mix it into the same dataset.",
    "LABEL_INVALID": "info.label is not enter/pass/exit. Open features.json and check info.label.",
    "LABEL_DIFFERS_FROM_SOURCE": "Normal when a bystander walks through a recording of the other class; it is only context.",
    "LABEL_FILENAME_MISMATCH": "The label in the file name differs from info.label. Check that files from two clips were not swapped.",
    "CLIP_NAME_MISMATCH": "info.clip does not match the file name. Check that the right files sit in this folder.",
    "SOURCE_NAME_MISMATCH": "info.sources names a different file than the one in the folder: files may have been renamed or swapped.",
    "SHA_MISMATCH": "features.json and background.json were made from different videos. The background belongs to another clip: re-run or swap in the right file.",
    "FRAME_SIZE_MISMATCH": "frame size in features.json differs from background.json. They come from different videos or settings.",
    "NO_FLOOR": "The segmentation found no floor, so exits and floor features are meaningless. Open *_background_overlay.png.",
    "NO_DOOR": "No door segmented. Open *_background_overlay.png; the target and exit features may be wrong.",
    "POLYGON_DEGENERATE": "A polygon with fewer than 3 points. Minor; check the overlay image.",
    "POLYGON_OUT_OF_FRAME": "A polygon leaves the image. Check *_background_overlay.png for a bad mask.",
    "CONFIDENCE_RANGE": "A segmentation confidence is outside 0..1. Corrupt background.json.",
    "FLOOR_AREA_MISMATCH": "Floor area in the polygons differs from floor_frac in features.json. The two files may come from different runs.",
    "POLYGON_COUNT_MISMATCH": "used + dropped polygons does not equal the number of objects. Check that background.json was not edited.",
    "AREA_SUM": "Area shares sum to >1: wall and door masks overlap (common) or the masks are broken. Look at the overlay image.",
    "EXIT_COUNT_MISMATCH": "Exit counts disagree with the exits list. features.json is internally inconsistent: re-run or drop the clip.",
    "NO_EXITS": "No exit found, so competitor features are empty. Look at the debug image (exits are drawn with numbers).",
    "TARGET_OUT_OF_FRAME": "Target door centre lies outside the frame. Check door_center in the trajectory and the debug image.",
    "FRAME_INDEX": "Frame indices are not 0..n-1. Frames were dropped or reordered; inspect data.frames.",
    "TIMESTAMP": "t does not equal frame_index / fps. Check effective_fps and the frame list.",
    "TRACK_SUMMARY_MISMATCH": "The track summary counts disagree with the frame list. features.json is inconsistent.",
    "SHORT_TRACK": "Few detected frames: little signal. Consider excluding the clip from training.",
    "DETECTION_GAPS": "The person is lost and found again inside the track (ID switch or occlusion). Watch the clip or look at the debug image.",
    "FRAME_BLOCK_MISSING": "A detected frame lacks pos/kin/pose/scene_rel. features.json is incomplete.",
    "VALUE_RANGE": "A value is outside its allowed range (message gives the field and frame). Open features.json at that frame.",
    "TELEPORT": "The ground point jumps in one frame: track switched to another person or a bad foot detection. Check the debug trajectory.",
    "OFF_FLOOR": "The person is mostly outside the floor mask: bad segmentation. Look at the overlay and debug image.",
    "ENTER_NEVER_REACHES": "Labelled enter but never gets close to the target door. Check the debug image; the label is likely wrong.",
    "PASS_REACHES_TARGET": "Labelled pass but gets very close to the target door. Check the debug image; the label may be wrong.",
    "TRACKS_DUPLICATE_ROW": "The same clip_id is in several clips.csv files; the later one is used. Check which run is right.",
    "TRACKS_NO_ROW": "No clips.csv row for this clip: the folder does not belong to the tracker output you have.",
    "CSV_NOT_KEEP": "clips.csv marks this clip as removed, yet features exist. Check the removal reason and the run.",
    "LABEL_CSV_MISMATCH": "clips.csv and features.json disagree on the label. Open both; one of them was changed or is from another run.",
    "MANUAL_LABEL_NOT_APPLIED": "Someone set manual_label in clips.csv but the pipeline ignores it. Decide which label to trust.",
    "SESSION_CSV_MISMATCH": "The clip belongs to a different tracking session than clips.csv says.",
    "FRAMES_CSV_MISMATCH": "Frame count differs between clips.csv and features.json. Different runs or edited files.",
    "CSV_FIELD": "A clips.csv field could not be read. Open clips.csv at that row (Excel may have changed it).",
    "FOLDER_NAME_PATTERN": "The folder was renamed (not <clip>-<sha10>-<sha8>), so SHA checks on the name were skipped.",
    "FOLDER_NAME_CLIP": "The folder name names another clip than the files inside. Check for a mix-up.",
    "FOLDER_SHA_VIDEO": "The folder name's video SHA differs from the features: the folder was renamed or the files replaced.",
    "TRAJ_MISSING": "Trajectory JSON is not in tracks/. Ask for the complete tracks folder.",
    "FOLDER_SHA_TRAJECTORY": "The trajectory file differs from the one the features were made from. If every clip shows this, the files were altered in transfer (line endings): copy again without conversion.",
    "LABEL_TRAJ_MISMATCH": "The trajectory JSON and features.json disagree on the label: files from different runs.",
    "SIZE_TRAJ_MISMATCH": "Frame size in the trajectory differs from features.json.",
    "SHA_TRAJ_MISMATCH": "Trajectory and features refer to different clip videos: files from different runs.",
    "META_TRAJ_MISMATCH": "Session, commit, fps or tracker differ between trajectory and features (message says which).",
    "FRAMES_TRAJ_MISMATCH": "Different frame counts in trajectory and features: not made from this trajectory.",
    "POSE_MISSING": "clips.csv lists a pose file that is not in tracks/. Pose features may rest on missing input.",
    "POSE_FRAME_INDEX": "Pose frames do not line up with trajectory frames.",
    "POSE_VALUE_RANGE": "Pose landmarks outside 0..1. Check the pose file.",
    "POSE_NOT_IN_CSV": "features list a pose file but clips.csv has none. Check the tracker output.",
    "DETECTED_FLAGS_MISMATCH": "Features and trajectory disagree on which frames have a person. features.json was not made from this trajectory.",
    "TRAJ_VALUE": "Malformed bbox, keypoints or centre in the trajectory (message says which). Inspect those frames.",
    "TRAJ_TIME": "Trajectory time stamps are not increasing. Inspect t in the trajectory JSON.",
    "DETECTED_COUNT_MISMATCH": "Detected-frame counts differ between clips.csv, trajectory and frames with a centre.",
    "LABEL_RECOMPUTE": "The tracker's own rule (last centre inside the door ellipse = enter) gives another label. Label was edited or the files are mixed up.",
    "LABEL_BORDERLINE": "The person ends right at the enter/pass boundary: the label is a coin flip. Look at the debug image; consider excluding it.",
    "TRACKS_EMPTY": "No rows found in clips.csv. Check the tracks folder path (--tracks).",
    "NO_TRACKS": "No tracks folder found, so label and trajectory cross-checks were skipped.",
    "CLIP_FOLDER_MISSING": "clips.csv says keep but there is no clip folder. The clip was lost in transfer or never delivered.",
    "CSV_STALE_ROWS": "Rows from a failed or earlier run; ignored. Nothing to do.",
    "FILES_NOT_DELIVERED": "Images/background.json are in none of the folders (pipeline run without --debug). Only features.json is checked; the visual checks (overlay, debug image) are not possible.",
    "SHA_MISSING": "features.json has no video SHA, so the chain video -> background -> trajectory cannot be checked.",
    "SOURCE_EMPTY": "No clip folders in this source. Check the path you gave.",
    "CLIP_COUNT": "Number of clip folders differs from what clips.csv / --expect says. See log_vs_folders.csv for which ones.",
    "NO_LOGS": "No log found, so the checks against the run were skipped.",
    "LOG_OTHER_RUN": "An earlier or failed run. Context only.",
    "NO_COMPLETE_RUN": "No log ends with 'Pipeline complete'. Ask the supervisors for the log of the successful run.",
    "LATER_RUN_FAILED": "A later run failed. Make sure the data comes from the complete run.",
    "SEVERAL_COMPLETE_RUNS": "More than one complete run in the logs; the latest is used. Check it is the one behind the data.",
    "LOG_FAILED_CLIPS": "The run reports failed clips; they have no features. See the log lines with 'failed'.",
    "LOG_STAGE_INCOMPLETE": "A stage did not finish all items (e.g. 400/410). Open the log at that stage.",
    "LOG_WARNINGS": "The run logged warnings or errors. Open the named log and search the quoted text.",
    "LOG_CLIP_WARNING": "The log has a warning about this clip: open the log and search the clip name.",
    "LOG_CLIP_COUNT": "The log's clip count differs from your folders. See log_vs_folders.csv for the difference.",
    "LOG_FOLDER_MISSING": "The log wrote features for these clips but the folders are not in your data. Ask for them (list in log_vs_folders.csv).",
    "FOLDER_NOT_IN_LOG": "Folder not announced by the run: from another run or renamed. Check where it came from.",
    "LOG_NO_FEATURES": "The log announced the clip but never wrote its features.",
    "LOG_NOT_SEGMENTED": "No 'Segmenting' line: an existing scene file was reused. Fine if intended.",
    "LOG_LABEL_COUNTS": "Label counts in the log differ from your folders: clips are missing or mixed in.",
    "LOG_CSV_COUNTS": "Label or removed counts in the log differ from clips.csv: clips.csv is from another run or was edited.",
    "CLIP_SESSION_NOT_IN_LOG": "The clip was made in a session that the run's log does not contain: it comes from another run.",
    "LOG_FPS_MISMATCH": "The fps the tracker logged for the source video differs from the fps in the features.",
    "DUPLICATE_VIDEO": "Two clips come from the same video file. Check whether the same data was delivered twice.",
    "VARIANT_PARAMS_KEY": "Some clips were made with other parameters than most. Do not mix them without checking info.params.",
    "VARIANT_GIT_COMMIT": "Some clips were made with another code version. Check info.trajectory_meta.git_commit.",
    "VARIANT_SCHEMA": "Mixed schema versions in one dataset.",
    "MIXED_FPS": "Videos were recorded at different fps. Smoothing windows are counted in frames, so consider resampling to one fps.",
    "MIXED_SESSION": "Several tracking sessions (normally one per run). Check they match the logs.",
    "MIXED_WIDTH": "Frames have different widths. Pixel features are not comparable without scaling.",
    "MIXED_HEIGHT": "Frames have different heights. Pixel features are not comparable without scaling.",
    "FEATURE_ALWAYS_NULL": "A feature that is never filled. Either by design (check FEATURES.md) or broken: do not use it, ask the supervisors.",
    "FEATURE_MOSTLY_NULL": "Null in most clips: a property of the feature, not of single clips. Decide whether it is usable.",
    "UNUSUAL_NULLS": "Features that are normally filled are empty in this clip (listed in the message). Open the clip's debug image: usually no pose or poor detection.",
    "OUTLIER": "A number far from the rest of the dataset. Open the clip and check whether it is real or a pipeline problem.",
}


def hint(code):
    return HINTS.get(code, "")


class Issues(list):
    def add(self, clip, severity, code, message):
        self.append({"clip": clip, "severity": severity, "code": code, "message": message})


def load_json(path, clip, issues):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # truncated file, bad syntax, ...
        issues.add(clip, "ERROR", "JSON_UNREADABLE", f"{path.name}: {type(e).__name__}: {e}")
        return None


# ----------------------------------------------------------------------------- logs/ (pipeline runs)

LOG_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| (\w+)\s*\| ([\w.]+) \| (.*)$")
RE_SUMMARY = re.compile(r"^Pipeline (complete|complete_with_errors|failed) - (\d+) clip\(s\) ok, (\d+) failed \(([\d.]+) s\)$")
RE_STAGE = re.compile(r"^(.+?) - (\d+)(?:/(\d+))?(?: \((\d+) failed\))? done \(([\d.]+) s\)$")
RE_WORKSPACE = re.compile(r"^(.+)\.mp4: (\d+)x(\d+), workspace (.+)$")
RE_WRITTEN = re.compile(r"^(.+)\.mp4: features written to (.+)$")
RE_BACKGROUND = re.compile(r"^(.+)\.mp4: extracting median background$")
RE_CLIP_MSG = re.compile(r"^([^ :]+)\.mp4: ")
RE_SESSION = re.compile(r"\[DONE\] session (\d+): (\{.*?\})\s+reasons=(\{.*\})\s*$")
RE_INIT = re.compile(r"\[INIT\] video source: (.+)$")
RE_GEOM = re.compile(r"\[GEOM\] frame=(\d+)x(\d+) record_fps=([\d.]+)")
RE_SEGMENT = re.compile(r"^Segmenting (.+)_background\.png\.\.\.$")


def basename(path_text):
    return re.split(r"[\\/]", path_text.strip().rstrip("\\/"))[-1]


def parse_log(path):
    """What one pipeline_<time>.log says about its run (only the lines this validator needs)."""
    run = {"file": path.name, "started": None, "status": "incomplete", "n_ok": None, "n_failed": None,
           "duration_s": None, "stages": [], "problems": [], "workspaces": {}, "written": set(),
           "background": set(), "segmented": set(), "sessions": {}, "fps": {}, "stage_failed": None}
    last_video = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = LOG_LINE.match(raw.rstrip("\n"))
            if not m:
                continue  # traceback continuation lines
            ts, level, logger, msg = m.groups()
            run["started"] = run["started"] or ts
            if level in ("WARNING", "ERROR", "CRITICAL"):
                run["problems"].append((level, msg))
            if logger == "pipeline":
                if m2 := RE_SUMMARY.match(msg):
                    run["status"], run["n_ok"], run["n_failed"], run["duration_s"] = m2[1], int(m2[2]), int(m2[3]), float(m2[4])
                elif m2 := RE_STAGE.match(msg):
                    run["stages"].append({"stage": m2[1], "done": int(m2[2]), "total": int(m2[3]) if m2[3] else None,
                                          "failed": int(m2[4] or 0), "seconds": float(m2[5])})
                elif " FAILED after " in msg:
                    run["stage_failed"] = msg
                elif m2 := RE_WORKSPACE.match(msg):
                    run["workspaces"][m2[1]] = basename(m2[4])
                elif m2 := RE_WRITTEN.match(msg):
                    run["written"].add(m2[1])
                elif m2 := RE_BACKGROUND.match(msg):
                    run["background"].add(m2[1])
            elif logger == "pipeline.stdout":
                if m2 := RE_SESSION.search(msg):
                    try:
                        run["sessions"][m2[1]] = {"counts": ast.literal_eval(m2[2]), "reasons": ast.literal_eval(m2[3])}
                    except (ValueError, SyntaxError):
                        run["problems"].append(("WARNING", f"unreadable session line: {msg[:120]}"))
                elif m2 := RE_INIT.search(msg):
                    last_video = Path(basename(m2[1])).stem
                elif m2 := RE_GEOM.search(msg):
                    if last_video:
                        run["fps"][last_video] = float(m2[3])
                        last_video = None
                elif m2 := RE_SEGMENT.match(msg):
                    run["segmented"].add(m2[1])
    return run


def chosen_run(logs):
    done = [l for l in (logs or []) if l["status"] in ("complete", "complete_with_errors")]
    return sorted(done, key=lambda l: l["started"] or "")[-1] if done else None


def report_names(issues, source, severity, code, names, message, diff_rows=None, limit=20):
    """One issue per clip when there are few, one aggregate issue when there are many."""
    names = sorted(names)
    for n in names:
        if diff_rows is not None:
            diff_rows.append({"source": source, "folder": n, "status": code})
    if len(names) <= limit:
        for n in names:
            issues.add(f"{source}/{n}", severity, code, message)
    elif names:
        issues.add(source, severity, code, f"{len(names)} folders: {message}; first: {', '.join(names[:5])} (all in log_vs_folders.csv)")


def check_logs(source, logs, rows, csv_stat, issues, diff_rows):
    """Cross-check one source's folders against its pipeline logs. Returns the run summary (or None)."""
    if not logs:
        issues.add(source, "INFO", "NO_LOGS", "no pipeline_*.log found: log cross-checks skipped")
        return None
    logs = sorted(logs, key=lambda l: l["started"] or "")
    done = [l for l in logs if l["status"] in ("complete", "complete_with_errors")]
    for l in logs:
        if l not in done:
            issues.add(source, "INFO", "LOG_OTHER_RUN", f"{l['file']}: {l['status']}"
                       + (f" ({l['stage_failed']})" if l["stage_failed"] else "") + " - not the run behind the data")
    if not done:
        issues.add(source, "ERROR", "NO_COMPLETE_RUN", "none of the logs ends with 'Pipeline complete'")
        return None
    run = done[-1]
    if any((l["started"] or "") > (run["started"] or "") for l in logs if l not in done):
        issues.add(source, "WARN", "LATER_RUN_FAILED", f"a run started after {run['file']} and did not complete; make sure the data comes from {run['file']}")
    if len(done) > 1:
        issues.add(source, "WARN", "SEVERAL_COMPLETE_RUNS", f"{[l['file'] for l in done]}; the latest ({run['file']}) is used")
    mine = [r for r in rows if r["source"] == source]
    folders = {r["clip"].split("/", 1)[1] for r in mine}
    if run["status"] != "complete" or run["n_failed"]:
        issues.add(source, "ERROR", "LOG_FAILED_CLIPS", f"{run['n_failed']} clip(s) failed in {run['file']} (status {run['status']}); they have no features")
    for st in run["stages"]:
        if st["failed"] or (st["total"] is not None and st["done"] != st["total"]):
            issues.add(source, "ERROR", "LOG_STAGE_INCOMPLETE", f"{st['stage']}: {st['done']}/{st['total']} done, {st['failed']} failed")
    if run["problems"]:
        kinds = Counter((lvl, re.sub(r"\d+(\.\d+)?", "N", RE_CLIP_MSG.sub("<clip>: ", txt))[:150]) for lvl, txt in run["problems"])
        issues.add(source, "WARN", "LOG_WARNINGS", f"{len(run['problems'])} warning/error line(s) in {run['file']}: "
                   + "; ".join(f"{c}x [{lvl}] {t}" for (lvl, t), c in kinds.most_common(5)))
        for lvl, txt in run["problems"]:
            m = RE_CLIP_MSG.match(txt)
            if m and m[1] in run["workspaces"]:
                issues.add(f"{source}/{run['workspaces'][m[1]]}", "WARN", "LOG_CLIP_WARNING", f"[{lvl}] {txt[:200]}")
    if run["n_ok"] != len(mine):
        issues.add(source, "ERROR", "LOG_CLIP_COUNT", f"{run['file']} reports {run['n_ok']} clips ok, {len(mine)} clip folders found")

    expected = {run["workspaces"][c] for c in run["written"] if c in run["workspaces"]}
    report_names(issues, source, "ERROR", "LOG_FOLDER_MISSING", expected - folders, "the log wrote features for this clip but the folder is not here", diff_rows)
    report_names(issues, source, "WARN", "FOLDER_NOT_IN_LOG", folders - set(run["workspaces"].values()), "folder is not in the log of the run", diff_rows)
    no_features = [run["workspaces"][c] for c in run["workspaces"] if c not in run["written"]]
    report_names(issues, source, "ERROR", "LOG_NO_FEATURES", no_features, "announced in the log but no 'features written' line", diff_rows)
    not_segmented = [run["workspaces"][c] for c in run["written"] if c in run["workspaces"] and c not in run["segmented"]]
    if not_segmented:
        issues.add(source, "INFO", "LOG_NOT_SEGMENTED", f"{len(not_segmented)} clip(s) have no 'Segmenting' line (existing scene reused?)")

    counts = Counter()
    for sess in run["sessions"].values():
        counts.update(sess["counts"])
    have = Counter(r.get("label") for r in mine)
    if any(counts[k] != have[k] for k in ("enter", "pass", "exit")):
        issues.add(source, "ERROR", "LOG_LABEL_COUNTS", f"log {dict((k, counts[k]) for k in ('enter', 'pass', 'exit'))} vs folders {dict((k, have[k]) for k in ('enter', 'pass', 'exit'))}")
    if csv_stat and (any(csv_stat["by_label"][(k, "keep")] != counts[k] for k in ("enter", "pass", "exit")) or csv_stat["removed"] != counts["removed"]):
        issues.add(source, "ERROR", "LOG_CSV_COUNTS", f"log {dict(counts)} vs clips.csv keep {dict((k, csv_stat['by_label'][(k, 'keep')]) for k in ('enter', 'pass', 'exit'))}, removed {csv_stat['removed']}")
    sessions = set(run["sessions"])
    for r in mine:
        sess = str(r.get("session") or "").rsplit("_", 1)[-1]
        if sessions and sess not in sessions:
            issues.add(r["clip"], "ERROR", "CLIP_SESSION_NOT_IN_LOG", f"session {r.get('session')} is not a session of {run['file']} {sorted(sessions)}")
        video = re.sub(r"_\d+$", "", r.get("clip_key") or "")
        if video in run["fps"] and r.get("record_fps") is not None and abs(run["fps"][video] - r["record_fps"]) > 0.01:
            issues.add(r["clip"], "ERROR", "LOG_FPS_MISMATCH", f"log says the source video was recorded at {run['fps'][video]} fps, features say {r['record_fps']}")
    return {"run": run, "counts": counts, "folders": len(mine), "labels": have, "others": [l["file"] + ": " + l["status"] for l in logs if l is not run]}


# ----------------------------------------------------------------------------- tracks/ (tracker output)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ellipse_value(cx, cy, door, axes):
    """The tracker's enter test: <= 1 means inside the half-ellipse above the door centre."""
    dx, dy = cx - door[0], cy - door[1]
    if dy > 0:
        return math.inf
    return dx * dx / (axes[0] ** 2) + dy * dy / (axes[1] ** 2)


def load_tracks(tracks_dir, source, issues):
    """clip_id -> clips.csv row (+ '_dir'), from every clips.csv below tracks_dir."""
    rows = {}
    for csv_path in sorted(tracks_dir.rglob("clips.csv")):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel may have added a BOM
            head = f.readline()
            f.seek(0)
            delimiter = max(",;\t", key=head.count)                   # ... or switched the delimiter to ;
            for r in csv.DictReader(f, delimiter=delimiter):
                r["_dir"] = csv_path.parent
                old = rows.get(r["clip_id"])
                if old:
                    issues.add(f"{source}/{r['clip_id']}", "WARN", "TRACKS_DUPLICATE_ROW",
                               f"in clips.csv of {old['_dir'].name} and {csv_path.parent.name}; the later one is used")
                rows[r["clip_id"]] = r
    return rows


def check_tracks(clip_dir, clip_id, prefix, info, label, frames, W, H, sha_f, tracks, row, issues, args):
    r = tracks.get(prefix)
    if r is None:
        issues.add(clip_id, "ERROR", "TRACKS_NO_ROW", f"no clips.csv row for clip_id {prefix}")
        return
    meta = info.get("trajectory_meta") or {}
    manual = (r.get("manual_label") or "").strip()
    row.update(csv_label=r.get("label"), csv_manual_label=manual, csv_result=r.get("cleaning_result"),
               csv_reason=r.get("cleaning_reason"))
    if r.get("cleaning_result") != "keep":
        issues.add(clip_id, "ERROR", "CSV_NOT_KEEP", f"clips.csv says {r.get('cleaning_result')} ({r.get('cleaning_reason')}) but the clip has features")
    if r.get("label") != label:
        issues.add(clip_id, "ERROR", "LABEL_CSV_MISMATCH", f"features label {label}, clips.csv label {r.get('label')}")
    if manual and manual != label:
        issues.add(clip_id, "WARN", "MANUAL_LABEL_NOT_APPLIED",
                   f"clips.csv manual_label={manual} but features carry the tracker label {label} (the pipeline ignores manual_label)")
    if r.get("session_id") != meta.get("session_id"):
        issues.add(clip_id, "ERROR", "SESSION_CSV_MISMATCH", f"clips.csv {r.get('session_id')} vs features {meta.get('session_id')}")
    try:
        if int(r["n_frames"]) != len(frames):
            issues.add(clip_id, "ERROR", "FRAMES_CSV_MISMATCH", f"clips.csv n_frames {r['n_frames']} vs features {len(frames)}")
    except (KeyError, ValueError):
        issues.add(clip_id, "WARN", "CSV_FIELD", f"n_frames not readable: {r.get('n_frames')}")
    sources = info.get("sources") or {}
    for key, name in (("trajectory", prefix + ".json"), ("pose", prefix + "_pose.json")):
        if sources.get(key) not in (name, None):
            issues.add(clip_id, "WARN", "SOURCE_NAME_MISMATCH", f"info.sources.{key}={sources.get(key)} expected {name}")

    # folder name = <clip>-<video sha[:10]>-<trajectory file sha[:8]> (pipeline.py, Loading clip records)
    m = re.fullmatch(r"(.+)-([0-9a-f]{10})-([0-9a-f]{8})", clip_dir.name)
    if not m:
        issues.add(clip_id, "INFO", "FOLDER_NAME_PATTERN", "folder name is not <clip>-<sha10>-<sha8>; folder SHA checks skipped")
    else:
        if m.group(1) != prefix:
            issues.add(clip_id, "WARN", "FOLDER_NAME_CLIP", f"folder names clip {m.group(1)}, files are {prefix}")
        if sha_f and m.group(2) != sha_f[:10]:
            issues.add(clip_id, "ERROR", "FOLDER_SHA_VIDEO", f"folder says video {m.group(2)}, features say {sha_f[:10]}")

    tp = r["_dir"] / (r.get("trajectory") or prefix + ".json")
    if not tp.is_file():
        issues.add(clip_id, "ERROR", "TRAJ_MISSING", f"{tp.name} not found in {r['_dir']}")
        return
    if m and sha256_file(tp)[:8] != m.group(3):
        issues.add(clip_id, "ERROR", "FOLDER_SHA_TRAJECTORY",
                   f"{tp.name} differs from the file the features were made from (folder says {m.group(3)}); "
                   "if every clip shows this, the files were changed in transfer (line endings, re-saving)")
    traj = load_json(tp, clip_id, issues)
    if traj is None:
        return
    tmeta = traj.get("meta") or {}
    tf = traj.get("frames") or []
    if traj.get("label") != label:
        issues.add(clip_id, "ERROR", "LABEL_TRAJ_MISMATCH", f"trajectory label {traj.get('label')} vs features {label}")
    if [traj.get("frame_width"), traj.get("frame_height")] != [W, H]:
        issues.add(clip_id, "ERROR", "SIZE_TRAJ_MISMATCH", f"trajectory {[traj.get('frame_width'), traj.get('frame_height')]} vs features {[W, H]}")
    if sha_f and tmeta.get("clip_video_sha256") != sha_f:
        issues.add(clip_id, "ERROR", "SHA_TRAJ_MISMATCH", f"trajectory video SHA {str(tmeta.get('clip_video_sha256'))[:10]} vs features {sha_f[:10]}")
    for key, mine in (("session_id", meta.get("session_id")), ("git_commit", meta.get("git_commit")),
                      ("effective_fps", meta.get("effective_fps")), ("tracker", meta.get("tracker"))):
        if tmeta.get(key) != mine:
            issues.add(clip_id, "ERROR", "META_TRAJ_MISMATCH", f"{key}: trajectory {tmeta.get(key)} vs features {mine}")
    if len(tf) != len(frames):
        issues.add(clip_id, "ERROR", "FRAMES_TRAJ_MISMATCH", f"trajectory has {len(tf)} frames, features {len(frames)}")
        return

    # pose sidecar
    pose_det = {}
    if r.get("pose"):
        pp = r["_dir"] / r["pose"]
        if not pp.is_file():
            issues.add(clip_id, "ERROR", "POSE_MISSING", f"{r['pose']} listed in clips.csv but not found")
        else:
            pose = load_json(pp, clip_id, issues)
            if pose:
                pf = pose.get("frames") or []
                if [x.get("frame_index") for x in pf] != [x.get("frame_index") for x in tf]:
                    issues.add(clip_id, "ERROR", "POSE_FRAME_INDEX", "pose frame indices differ from the trajectory")
                bad = 0
                for x in pf:
                    if x.get("detected"):
                        pose_det[x["frame_index"]] = True
                        for v in (x.get("landmarks") or {}).values():
                            if not (-0.01 <= v.get("x", 0) <= 1.01 and -0.01 <= v.get("y", 0) <= 1.01 and -0.01 <= v.get("visibility", 0) <= 1.01):
                                bad += 1
                if bad:
                    issues.add(clip_id, "WARN", "POSE_VALUE_RANGE", f"{bad} landmark value(s) outside [0,1]")
    elif sources.get("pose"):
        issues.add(clip_id, "WARN", "POSE_NOT_IN_CSV", "features list a pose file but clips.csv has none")

    # detection flags: features count a frame as detected if keypoints, bbox or centre exist (pose is a fallback)
    flags_mismatch = bad_center = bad_bbox = bad_kp = 0
    for i, (fr, x) in enumerate(zip(tf, frames)):
        c, bb, kp = fr.get("center"), fr.get("bbox"), fr.get("keypoints")
        det = c is not None or bb is not None or kp is not None or bool(pose_det.get(fr.get("frame_index")))
        flags_mismatch += det != bool(x.get("detected"))
        if c is not None and not (-2 <= c[0] <= W + 2 and -2 <= c[1] <= H + 2):
            bad_center += 1
        if bb is not None and not (len(bb) == 4 and bb[0] < bb[2] and bb[1] < bb[3] and bb[0] >= -2 and bb[1] >= -2
                                   and bb[2] <= W + 2 and bb[3] <= H + 2):
            bad_bbox += 1
        if kp is not None and (len(kp) != 17 or any(not -0.01 <= p[2] <= 1.01 for p in kp)):
            bad_kp += 1
    if flags_mismatch:
        issues.add(clip_id, "ERROR", "DETECTED_FLAGS_MISMATCH", f"{flags_mismatch} frame(s) where features and trajectory disagree on detected")
    for n, what in ((bad_center, "centre outside the frame"), (bad_bbox, "bbox malformed or outside the frame"),
                    (bad_kp, "keypoints not 17 x (x, y, conf in 0..1)")):
        if n:
            issues.add(clip_id, "WARN", "TRAJ_VALUE", f"{n} frame(s) with {what}")
    ts = [fr.get("t") for fr in tf]
    if any(a is None or b is None or b <= a for a, b in zip(ts, ts[1:])):
        issues.add(clip_id, "ERROR", "TRAJ_TIME", "t is not strictly increasing")
    n_center = sum(fr.get("center") is not None for fr in tf)
    try:
        if int(r["n_detected"]) != n_center or traj.get("n_detected") != n_center:
            issues.add(clip_id, "ERROR", "DETECTED_COUNT_MISMATCH",
                       f"frames with a centre: {n_center}, clips.csv {r['n_detected']}, trajectory {traj.get('n_detected')}")
    except (KeyError, ValueError):
        pass

    # recompute the label with the tracker's rule: enter <=> last detected centre inside the door half-ellipse
    centers = [fr["center"] for fr in tf if isinstance(fr.get("center"), (list, tuple)) and len(fr["center"]) == 2]
    door, axes = traj.get("door_center"), traj.get("ellipse_axes")
    if centers and door and axes:
        value = ellipse_value(centers[-1][0], centers[-1][1], door, axes)
        row["end_ellipse_value"] = None if math.isinf(value) else round(value, 3)
        if label in ("enter", "pass") and ("enter" if value <= 1 else "pass") != label:
            issues.add(clip_id, "ERROR", "LABEL_RECOMPUTE",
                       f"label {label} but the last centre {centers[-1]} gives ellipse value {value:.2f} (<=1 means enter)")
        if label in ("enter", "pass") and not math.isinf(value) and abs(value - 1) <= args.borderline:
            issues.add(clip_id, "INFO", "LABEL_BORDERLINE", f"ellipse value {value:.2f}: ends close to the enter/pass boundary")


# ----------------------------------------------------------------------------- one clip

def validate_clip(clip_dir, source, args, issues, tracks=None, optional=frozenset()):
    clip_id = f"{source}/{clip_dir.name}"
    row = {"clip": clip_id, "source": source}
    found = {}
    for role, suffix in FILES.items():
        match = [p for p in clip_dir.iterdir() if p.is_file() and p.name.endswith(suffix)]
        if len(match) != 1:
            if not match and role in optional:
                continue  # this role is in none of the clips of the source (e.g. a run without --debug)
            issues.add(clip_id, "ERROR", "FILE_MISSING" if not match else "FILE_DUPLICATE",
                       f"{role} ({suffix}): found {len(match)}")
            continue
        found[role] = match[0]
        if match[0].stat().st_size == 0:
            issues.add(clip_id, "ERROR", "FILE_EMPTY", match[0].name)
    extra = {p.name for p in clip_dir.iterdir()} - {p.name for p in found.values()}
    if extra:
        issues.add(clip_id, "INFO", "FILE_EXTRA", ", ".join(sorted(extra)))
    row["n_files_ok"] = len(found)

    sizes = {}
    for role in ("background_img", "overlay_img", "debug_img"):
        if role in found and found[role].stat().st_size:
            size = png_size(found[role])
            if size is None:
                issues.add(clip_id, "ERROR", "PNG_INVALID", f"{found[role].name} is not a valid PNG header")
            sizes[role] = size
    if len({s for s in sizes.values() if s}) > 1:
        issues.add(clip_id, "ERROR", "IMAGE_SIZE_MISMATCH", f"image sizes differ: {sizes}")

    if "features" not in found:
        return row, None, None, None
    feat = load_json(found["features"], clip_id, issues)
    scene = load_json(found["scene"], clip_id, issues) if "scene" in found else None
    if feat is None or ("scene" in found and scene is None):
        return row, None, None, None
    try:
        info, data = feat["info"], feat["data"]
        frames, ss, ts = data["frames"], data["scene_summary"], data["track_summary"]
    except KeyError as e:
        issues.add(clip_id, "ERROR", "SCHEMA_KEYS", f"features.json lacks key {e}")
        return row, None, None, None

    prefix = found["features"].name[: -len(FILES["features"])]
    meta = info.get("trajectory_meta") or {}
    W, H = (info.get("frame_size") or [None, None])[:2]
    fps = meta.get("effective_fps")
    row.update(label=info.get("label"), schema=info.get("schema_version"), fps=fps, record_fps=meta.get("record_fps"),
               session=meta.get("session_id"), git_commit=meta.get("git_commit"),
               width=W, height=H)

    # --- provenance / identity
    if info.get("schema_version") not in EXPECTED_SCHEMA:
        issues.add(clip_id, "ERROR", "SCHEMA_VERSION", f"{info.get('schema_version')} not in {sorted(EXPECTED_SCHEMA)}")
    label = info.get("label")
    if label not in LABELS:
        issues.add(clip_id, "ERROR", "LABEL_INVALID", f"info.label={label}")
    # The folder is the source-video class; info.label is the tracker's per-person label. They can differ
    # legitimately (a passer-by in an 'enter' recording, or a person leaving), so this is a statistic.
    if source in LABELS:  # only meaningful when the source is named after a class
        row["label_matches_source"] = label == source
        if label != source:
            issues.add(clip_id, "INFO", "LABEL_DIFFERS_FROM_SOURCE", f"tracker label '{label}' in source '{source}'")
    m = re.search(r"_(enter|pass|exit)(?:_\d+)?$", prefix)
    if not m or m.group(1) != label:
        issues.add(clip_id, "WARN", "LABEL_FILENAME_MISMATCH", f"file prefix {prefix} does not end in _{label}[_N]")
    if info.get("clip") != re.sub(r"_\d+$", "", prefix):
        issues.add(clip_id, "WARN", "CLIP_NAME_MISMATCH", f"info.clip={info.get('clip')} vs file prefix {prefix}")
    sources = info.get("sources") or {}
    for key, name in (("scene", prefix + "_background.json"), ("background", prefix + "_background.png")):
        if sources.get(key) != name:
            issues.add(clip_id, "WARN", "SOURCE_NAME_MISMATCH", f"info.sources.{key}={sources.get(key)} expected {name}")
    sha_f = info.get("clip_video_sha256")
    row["sha256"] = sha_f
    if not sha_f:
        issues.add(clip_id, "ERROR", "SHA_MISSING", "features.json has no clip_video_sha256")
    if scene is not None:
        sha_s = scene.get("source_video_sha256")
        if sha_f != sha_s:
            issues.add(clip_id, "ERROR", "SHA_MISMATCH", f"features {str(sha_f)[:10]} vs background.json {str(sha_s)[:10]}")
        if [W, H] != scene.get("image_size"):
            issues.add(clip_id, "ERROR", "FRAME_SIZE_MISMATCH", f"features {[W, H]} vs background.json {scene.get('image_size')}")
    for role, size in sizes.items():
        if size and list(size) != [W, H]:
            issues.add(clip_id, "ERROR", "IMAGE_SIZE_VS_JSON", f"{role} is {size}, JSON says {[W, H]}")
    row["params_key"] = json.dumps(info.get("params"), sort_keys=True)
    row["clip_key"] = prefix
    if tracks is not None:
        check_tracks(clip_dir, clip_id, prefix, info, label, frames, W, H, sha_f, tracks, row, issues, args)

    # --- scene (background.json) and its agreement with scene_summary
    if scene is not None:
        objs = scene.get("objects") or []
        labels = Counter(o.get("label") for o in objs)
        row.update(n_floor_polys=labels["floor"], n_wall_polys=labels["wall"], n_door_polys=labels["door"])
        if not labels["floor"]:
            issues.add(clip_id, "ERROR", "NO_FLOOR", "segmentation found no floor polygon")
        if not labels["door"]:
            issues.add(clip_id, "WARN", "NO_DOOR", "segmentation found no door polygon")
        floor_area = 0.0
        for o in objs:
            poly = o.get("polygon") or []
            if len(poly) < 3:
                issues.add(clip_id, "WARN", "POLYGON_DEGENERATE", f"{o.get('label')} polygon with {len(poly)} points")
                continue
            if W and H and any(not (-2 <= x <= W + 2 and -2 <= y <= H + 2) for x, y in poly):
                issues.add(clip_id, "WARN", "POLYGON_OUT_OF_FRAME", f"{o.get('label')} polygon leaves the frame")
            if not 0 <= o.get("confidence", 0) <= 1:
                issues.add(clip_id, "WARN", "CONFIDENCE_RANGE", f"{o.get('label')} confidence {o.get('confidence')}")
            if o.get("label") == "floor":
                floor_area += shoelace(poly)
        if W and H:
            row["floor_frac_polygons"] = round(floor_area / (W * H), 4)
            if abs(floor_area / (W * H) - ss["areas"]["floor_frac"]) > args.area_tol:
                issues.add(clip_id, "WARN", "FLOOR_AREA_MISMATCH",
                           f"floor polygons cover {floor_area / (W * H):.3f} of the image, scene_summary says {ss['areas']['floor_frac']}")
        if ss["n_polygons_used"] + ss["n_polygons_dropped"] != len(objs):
            issues.add(clip_id, "WARN", "POLYGON_COUNT_MISMATCH",
                       f"used {ss['n_polygons_used']} + dropped {ss['n_polygons_dropped']} != {len(objs)} objects")
    else:
        present = set((info.get("segmentation") or {}).get("labels_present") or [])
        if "floor" not in present:
            issues.add(clip_id, "ERROR", "NO_FLOOR", "segmentation found no floor polygon (from features.json: labels_present)")
        if "door" not in present:
            issues.add(clip_id, "WARN", "NO_DOOR", "segmentation found no door polygon (from features.json: labels_present)")
    area_sum = sum(ss["areas"].values())
    if not 0.95 <= area_sum <= args.area_sum_max:
        issues.add(clip_id, "WARN", "AREA_SUM", f"areas sum to {area_sum:.3f} (overlapping masks if >1)")
    exits = ss.get("exits") or []
    c = ss.get("counts", {})
    types = Counter(e.get("type") for e in exits)
    if not (c.get("n_exits") == len(exits) == sum(types.values())
            and c.get("n_door") == types["door"] and c.get("n_edge") == types["edge"]
            and c.get("n_floor_tip") == types["floor_tip"]):
        issues.add(clip_id, "ERROR", "EXIT_COUNT_MISMATCH", f"counts {c} vs exits {dict(types)}")
    if not exits:
        issues.add(clip_id, "WARN", "NO_EXITS", "no exit found, competitor features will be empty")
    if W and H:
        cx, cy = ss["target"]["center_px"]
        if not (0 <= cx <= W and 0 <= cy <= H):
            issues.add(clip_id, "WARN", "TARGET_OUT_OF_FRAME", f"target centre {ss['target']['center_px']}")
    row.update(n_exits=len(exits), n_door=types["door"], n_edge=types["edge"], n_floor_tip=types["floor_tip"],
               floor_top_y_px=ss.get("floor_top_y_px"), n_polygons_used=ss["n_polygons_used"],
               n_polygons_dropped=ss["n_polygons_dropped"], brightness_mean=ss["brightness_mean"],
               brightness_contrast=ss["brightness_contrast"], **{k: v for k, v in ss["areas"].items()})

    # --- frames
    det = [f for f in frames if f.get("detected")]
    row.update(n_frames=len(frames), n_detected=len(det))
    hs = ts.get("hindsight", {})
    row.update(detection_rate=round(len(det) / len(frames), 4) if frames else 0,
               duration_s=hs.get("duration_s"), path_length_bh=hs.get("path_length_bh"),
               mean_speed_bh_s=hs.get("mean_speed_bh_s"), displacement_bh=hs.get("displacement_bh"))
    if [f.get("frame_index") for f in frames] != list(range(len(frames))):
        issues.add(clip_id, "ERROR", "FRAME_INDEX", "frame_index is not 0..n-1 continuous")
    if fps and any(abs(f.get("t", math.nan) - f.get("frame_index", 0) / fps) > 1e-3 for f in frames):
        issues.add(clip_id, "ERROR", "TIMESTAMP", f"t != frame_index/{fps}")
    if hs.get("n_frames") != len(frames) or hs.get("n_detected") != len(det):
        issues.add(clip_id, "ERROR", "TRACK_SUMMARY_MISMATCH",
                   f"hindsight n_frames/n_detected {hs.get('n_frames')}/{hs.get('n_detected')} vs {len(frames)}/{len(det)}")
    if len(det) < args.min_detected:
        issues.add(clip_id, "WARN", "SHORT_TRACK", f"only {len(det)} detected frames")
    mask = [bool(f.get("detected")) for f in frames]
    first = mask.index(True) if True in mask else len(mask)
    last = len(mask) - 1 - mask[::-1].index(True) if True in mask else -1
    gaps = sum(1 for i in range(first, last + 1) if not mask[i])
    row.update(lead_undetected=first, trail_undetected=len(mask) - 1 - last, gap_frames=gaps)
    if gaps > args.max_gap_frames:
        issues.add(clip_id, "WARN", "DETECTION_GAPS", f"{gaps} undetected frames inside the track (fragmented)")

    nulls, totals, bad_numbers = Counter(), Counter(), {}
    for f in det:
        missing = [b for b in FRAME_BLOCKS[:4] if b not in f]
        if missing:
            issues.add(clip_id, "ERROR", "FRAME_BLOCK_MISSING", f"frame {f.get('frame_index')} lacks {missing}")
            break
        for block in FRAME_BLOCKS:
            if block in f:
                for path, v in leaves(f[block], block):
                    totals[path] += 1
                    if v is None:
                        nulls[path] += 1
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        if not math.isfinite(v):
                            bad_numbers.setdefault((path, "non-finite"), []).append(f["frame_index"])
                        else:
                            rule = number_rule(path)
                            tol = 0.02 if rule == (0.0, 1.0) else 0
                            if rule and not rule[0] - tol <= v <= rule[1] + tol:
                                bad_numbers.setdefault((path, f"outside {rule}"), []).append(f["frame_index"])
    for section in ("scene_summary", "track_summary"):
        for path, v in leaves(data[section], section):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                rule = number_rule(path)
                tol = 0.02 if rule == (0.0, 1.0) else 0
                if not math.isfinite(v) or (rule and not rule[0] - tol <= v <= rule[1] + tol):
                    bad_numbers.setdefault((path, "non-finite" if not math.isfinite(v) else f"outside {rule}"), []).append("-")
    for (path, why), where in bad_numbers.items():
        issues.add(clip_id, "ERROR" if why == "non-finite" else "WARN", "VALUE_RANGE",
                   f"{path} {why} in {len(where)} place(s), e.g. frame {where[0]}")

    # trajectory plausibility and quality rates
    max_jump, prev = 0.0, None
    for f in det:
        gp, bh = f["pos"].get("ground_px"), f["pos"].get("bh_ref_px")
        if prev and f["frame_index"] == prev[0] + 1 and gp and prev[1] and bh:
            max_jump = max(max_jump, math.hypot(gp[0] - prev[1][0], gp[1] - prev[1][1]) / bh)
        prev = (f["frame_index"], gp)
    row["max_jump_bh_per_frame"] = round(max_jump, 3)
    if max_jump > args.max_jump_bh:
        issues.add(clip_id, "WARN", "TELEPORT", f"ground point jumps {max_jump:.2f} body heights in one frame")
    n = max(len(det), 1)
    row["truncated_rate"] = round(sum(bool(f["pos"].get("bbox_truncated")) for f in det) / n, 3)
    row["unreliable_rate"] = round(sum(f["pos"].get("reliable") is False for f in det) / n, 3)
    row["off_floor_rate"] = round(sum(f["pos"].get("on_floor") is False for f in det) / n, 3)
    row["ankle_rate"] = round(sum(f["pos"].get("ground_src") == "ankles" for f in det) / n, 3)
    pose_t = sum(v for p, v in totals.items() if p.startswith("pose."))
    row["pose_null_rate"] = round(sum(v for p, v in nulls.items() if p.startswith("pose.")) / pose_t, 3) if pose_t else None
    if row["off_floor_rate"] > args.max_off_floor:
        issues.add(clip_id, "WARN", "OFF_FLOOR", f"{row['off_floor_rate']:.0%} of frames have the person off the floor mask")

    # label vs behaviour: does the person reach the target door?
    dists = [f["scene_rel"]["target"]["dist_bh"] for f in det
             if isinstance((f["scene_rel"].get("target") or {}).get("dist_bh"), (int, float))]
    row["min_target_dist_bh"] = round(min(dists), 3) if dists else None
    if dists:
        if label == "enter" and min(dists) > args.reach_bh:
            issues.add(clip_id, "WARN", "ENTER_NEVER_REACHES", f"closest approach to target {min(dists):.2f} bh")
        if label == "pass" and min(dists) <= args.reach_bh / 2.5:
            issues.add(clip_id, "WARN", "PASS_REACHES_TARGET", f"closest approach to target {min(dists):.2f} bh")

    vec = (ss["areas"]["floor_frac"], ss["areas"]["door_frac"], (ss.get("floor_top_y_px") or 0) / (H or 480))
    null_stats = {p: (nulls[p], totals[p]) for p in totals}
    return row, null_stats, vec, sha_f


# ----------------------------------------------------------------------------- dataset

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", metavar="SOURCE",
                    help="folder with one sub-folder per clip, optionally NAME=PATH to choose the tag shown in reports "
                         "(default tag: the folder name). Give one per input folder, e.g. masterdata_enter masterdata_pass.")
    ap.add_argument("--out", type=Path, default=Path("validation_report"))
    ap.add_argument("--null-threshold", type=float, default=0.5,
                    help="warn when a feature is null in more than this share of a clip's detected frames")
    ap.add_argument("--show-null-above", type=float, default=0.30,
                    help="the null table in summary.md lists features whose pooled null fraction is above this")
    ap.add_argument("--min-frames-for-nulls", type=int, default=5, help="skip null checks on clips with fewer detected frames")
    ap.add_argument("--ignore-null", action="append", default=[], metavar="REGEX",
                    help="feature paths (e.g. 'kin.tau_s') whose nulls are expected; repeatable")
    ap.add_argument("--expect", action="append", default=[], metavar="NAME=N",
                    help="expected number of clip folders in a source (the 'clip(s) ok' number of its log); repeatable")
    ap.add_argument("--tracks", action="append", default=[], metavar="NAME=PATH",
                    help="tracker output of a source (default: <source>/tracks if it exists); repeatable")
    ap.add_argument("--no-tracks", action="store_true", help="skip the cross-check against tracks/")
    ap.add_argument("--logs", action="append", default=[], metavar="NAME=PATH",
                    help="pipeline logs of a source (default: <source>/logs if it exists); repeatable")
    ap.add_argument("--no-logs", action="store_true", help="skip the cross-check against the pipeline logs")
    ap.add_argument("--borderline", type=float, default=0.15,
                    help="flag enter/pass clips whose last point is within this distance of the ellipse boundary (value 1)")
    ap.add_argument("--min-detected", type=int, default=10)
    ap.add_argument("--max-gap-frames", type=int, default=5)
    ap.add_argument("--max-jump-bh", type=float, default=0.5, help="max ground-point move per frame, in body heights")
    ap.add_argument("--max-off-floor", type=float, default=0.5)
    ap.add_argument("--reach-bh", type=float, default=0.5,
                    help="enter clips should come within this target distance (body heights); pass clips not within 0.4x of it")
    ap.add_argument("--area-tol", type=float, default=0.03)
    ap.add_argument("--area-sum-max", type=float, default=1.15)
    ap.add_argument("--cluster-tol", type=float, default=0.04, help="scene fingerprint distance (floor/door share, floor top) for the same-location grouping")
    args = ap.parse_args()

    sources = {}
    for spec in args.sources:
        name, _, path = spec.partition("=") if "=" in spec else (None, "", spec)
        path = Path(path)
        name = name or path.resolve().name
        if not path.is_dir():
            sys.exit(f"Not a folder: {path}")
        if name in sources:
            sys.exit(f"Two sources are called '{name}'; rename one with NAME=PATH.")
        sources[name] = path
    expect = {}
    for spec in args.expect:
        key, _, num = spec.partition("=")
        if key not in sources or not num.isdigit():
            sys.exit(f"--expect needs NAME=N with NAME in {sorted(sources)}, got '{spec}'")
        expect[key] = int(num)
    args.out.mkdir(parents=True, exist_ok=True)
    allow = [re.compile(p) for p in DOCUMENTED_NULLABLE + args.ignore_null]

    issues, rows, null_by_clip, vecs, shas = Issues(), [], {}, {}, defaultdict(list)
    track_dirs = {}
    for spec in args.tracks:
        key, _, p = spec.partition("=")
        if key not in sources or not p:
            sys.exit(f"--tracks needs NAME=PATH with NAME in {sorted(sources)}, got '{spec}'")
        track_dirs[key] = Path(p)
    tracks_by_source = {}
    for name, path in sources.items():
        tdir = track_dirs.get(name, path / "tracks")
        if args.no_tracks or not tdir.is_dir():
            tracks_by_source[name] = None
            if not args.no_tracks:
                issues.add(name, "INFO", "NO_TRACKS", f"{tdir} not found: cross-checks against clips.csv/trajectories skipped")
            continue
        tracks_by_source[name] = load_tracks(tdir, name, issues)
        if not tracks_by_source[name]:
            issues.add(name, "ERROR", "TRACKS_EMPTY", f"no clips.csv rows found under {tdir}")
    log_dirs = {}
    for spec in args.logs:
        key, _, p = spec.partition("=")
        if key not in sources or not p:
            sys.exit(f"--logs needs NAME=PATH with NAME in {sorted(sources)}, got '{spec}'")
        log_dirs[key] = Path(p)
    logs_by_source = {}
    for name, path in sources.items():
        ldir = log_dirs.get(name, path / "logs")
        logs_by_source[name] = None
        if not args.no_logs and ldir.is_dir():
            logs_by_source[name] = [parse_log(p) for p in sorted(ldir.glob("*.log"))]
    for name, path in sources.items():
        clip_dirs = [p for p in sorted(path.iterdir()) if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("[TEMP]")]
        if not clip_dirs:
            issues.add(name, "ERROR", "SOURCE_EMPTY", f"no clip folders in {path}")
        present = Counter()
        for d in clip_dirs:
            names = [p.name for p in d.iterdir() if p.is_file()]
            for role, suffix in FILES.items():
                present[role] += any(n.endswith(suffix) for n in names)
        optional = frozenset(r for r in FILES if r != "features" and clip_dirs and present[r] == 0)
        if optional:
            issues.add(name, "INFO", "FILES_NOT_DELIVERED",
                       f"{sorted(optional)} are in none of the {len(clip_dirs)} clip folders (run without --debug?); checks that need them are skipped")
        for clip_dir in clip_dirs:
            row, nstats, vec, sha = validate_clip(clip_dir, name, args, issues, tracks_by_source[name], optional)
            rows.append(row)
            if nstats is not None:
                null_by_clip[row["clip"]] = nstats
                vecs[row["clip"]] = vec
                shas[sha].append(row["clip"])
    if not rows:
        sys.exit("No clip folders found (expected ROOT/enter/<clip>/ and ROOT/pass/<clip>/).")

    # --- clips.csv rows marked keep that have no folder (and default expected counts)
    csv_stats = {}
    for name, idx in tracks_by_source.items():
        if not idx:
            continue
        run = chosen_run(logs_by_source.get(name))
        run_sessions = set(run["sessions"]) if run else set()
        # tracks/ can still hold rows of an earlier or failed run (e.g. the pass run that crashed after 43 videos)
        mine = {k: r for k, r in idx.items() if not run_sessions or str(r.get("session_id")).rsplit("_", 1)[-1] in run_sessions}
        if len(mine) != len(idx):
            issues.add(name, "INFO", "CSV_STALE_ROWS", f"{len(idx) - len(mine)} clips.csv row(s) belong to sessions that are not in {run['file']}; ignored")
        keep = {k for k, r in mine.items() if r.get("cleaning_result") == "keep"}
        present = {r.get("clip_key") for r in rows if r["source"] == name}
        for k in sorted(keep - present):
            issues.add(f"{name}/{k}", "ERROR", "CLIP_FOLDER_MISSING", "clips.csv marks it keep but no clip folder was found")
        csv_stats[name] = {"keep": len(keep), "removed": len(mine) - len(keep), "stale": len(idx) - len(mine),
                           "by_label": Counter((r.get("label"), r.get("cleaning_result")) for r in mine.values()),
                           "reasons": Counter(r.get("cleaning_reason") for r in mine.values() if r.get("cleaning_result") != "keep"),
                           "manual": sum(bool((r.get("manual_label") or "").strip()) for r in mine.values())}
        expect.setdefault(name, len(keep))

    # --- pipeline logs
    log_summaries, diff_rows = {}, []
    for name, logs in logs_by_source.items():
        if logs is not None:
            log_summaries[name] = check_logs(name, logs, rows, csv_stats.get(name), issues, diff_rows)
        elif not args.no_logs:
            issues.add(name, "INFO", "NO_LOGS", f"{log_dirs.get(name, sources[name] / 'logs')} not found: log cross-checks skipped")

    # --- completeness against the supervisors' log
    for name, expected in expect.items():
        have = sum(r["source"] == name for r in rows)
        if have != expected:
            issues.add("DATASET", "ERROR", "CLIP_COUNT", f"{name}: found {have} clip folders, expected {expected} (--expect or clips.csv keep rows)")

    # --- duplicates
    for sha, clips in shas.items():
        if sha and len(clips) > 1:
            for c in clips:
                issues.add(c, "WARN", "DUPLICATE_VIDEO", f"same video SHA as {len(clips) - 1} other clip(s): {clips[:3]}")

    # --- run-to-run consistency (params, schema, commit, fps, session)
    ok_rows = [r for r in rows if r["clip"] in null_by_clip]
    for field, sev in (("params_key", "ERROR"), ("git_commit", "WARN"), ("schema", "ERROR")):
        counts = Counter(r.get(field) for r in ok_rows)
        if len(counts) > 1:
            main_val = counts.most_common(1)[0][0]
            for r in ok_rows:
                if r.get(field) != main_val:
                    issues.add(r["clip"], sev, f"VARIANT_{field.upper()}", f"differs from the most common value ({counts.most_common(1)[0][1]} clips)")
    for field in ("fps", "session", "width", "height"):
        counts = Counter(r.get(field) for r in ok_rows)
        if len(counts) > 1:
            issues.add("DATASET", "INFO", f"MIXED_{field.upper()}", f"{dict(counts)}")

    # --- feature nulls (per clip against the dataset's typical behaviour)
    per_path = defaultdict(list)  # path -> [(clip, null_frac, label)]
    label_of = {r["clip"]: r.get("label") for r in rows}
    for clip, stats in null_by_clip.items():
        for path, (n, t) in stats.items():
            if t >= args.min_frames_for_nulls:
                per_path[path].append((clip, n / t, label_of[clip]))
    null_rows, flagged_per_clip = [], defaultdict(list)
    for path, items in sorted(per_path.items()):
        fracs = [f for _, f, _ in items]
        typical = statistics.median(fracs)
        high = [(c, f) for c, f, _ in items if f > args.null_threshold]
        documented = any(p.search(path) for p in allow)
        structural = typical > args.null_threshold  # null in most clips: a property of the feature, not of one clip
        tot_n = sum(null_by_clip[c][path][0] for c, _, _ in items)
        tot_t = sum(null_by_clip[c][path][1] for c, _, _ in items)
        by_label = {}
        for lab in ("enter", "pass", "exit"):
            n = sum(null_by_clip[c][path][0] for c, _, l in items if l == lab)
            t = sum(null_by_clip[c][path][1] for c, _, l in items if l == lab)
            by_label[lab] = round(n / t, 3) if t else None
        status = ("ALWAYS_NULL" if tot_n == tot_t else "MOSTLY_NULL" if structural else
                  "SOMETIMES_HIGH" if high else "ok")
        null_rows.append({"feature": path, "pooled_null_rate": round(tot_n / tot_t, 3), "median_clip_null_rate": round(typical, 3),
                          "null_rate_enter": by_label["enter"], "null_rate_pass": by_label["pass"], "null_rate_exit": by_label["exit"],
                          "clips": len(items), "clips_over_threshold": len(high),
                          "documented_nullable": documented, "status": status})
        if status == "ALWAYS_NULL" and not documented:
            issues.add("DATASET", "WARN", "FEATURE_ALWAYS_NULL", f"{path} is null in every detected frame of every clip")
        elif status == "MOSTLY_NULL" and not documented:
            issues.add("DATASET", "WARN", "FEATURE_MOSTLY_NULL", f"{path}: median clip null rate {typical:.0%} (dataset-wide, not one clip)")
        if not documented and not structural:
            for c, f in high:
                flagged_per_clip[c].append((path, f))  # normally filled, but empty in this clip
    for clip, items in flagged_per_clip.items():
        items.sort(key=lambda x: -x[1])
        txt = "; ".join(f"{p} {f:.0%}" for p, f in items[:8]) + (f" (+{len(items) - 8} more)" if len(items) > 8 else "")
        issues.add(clip, "WARN", "UNUSUAL_NULLS", f"{len(items)} feature(s) null in >{args.null_threshold:.0%} of frames: {txt}")
    for r in rows:
        items = flagged_per_clip.get(r["clip"], [])
        r["n_unusual_null_features"] = len(items)
        r["unusual_null_features"] = "; ".join(p for p, _ in sorted(items, key=lambda x: -x[1])[:8])

    # --- scene fingerprint: same corridor / camera setup?
    centers, assign = [], {}
    for clip in sorted(vecs):
        v = vecs[clip]
        for i, c in enumerate(centers):
            if max(abs(a - b) for a, b in zip(v, c)) <= args.cluster_tol:
                assign[clip] = i
                break
        else:
            centers.append(v)
            assign[clip] = len(centers) - 1
    for r in rows:
        r["scene_cluster"] = assign.get(r["clip"])

    # --- outliers among all clips (robust z), context only
    if len(ok_rows) >= 20:
        for stat in OUTLIER_STATS:
            vals = [(r["clip"], r[stat]) for r in ok_rows if isinstance(r.get(stat), (int, float))]
            if len(vals) < 20:
                continue
            med, mad = robust_z([v for _, v in vals])
            if mad == 0:
                continue
            for clip, v in vals:
                z = 0.6745 * (v - med) / mad
                if abs(z) > 5:
                    issues.add(clip, "INFO", "OUTLIER", f"{stat}={v} (dataset median {med:.3g}, robust z={z:.1f})")

    # --- write
    sev_rank = {"ERROR": 0, "WARN": 1, "INFO": 2}
    issues.sort(key=lambda i: (sev_rank[i["severity"]], i["clip"], i["code"]))
    for i in issues:
        i["what_to_check"] = hint(i["code"])
        head, _, tail = i["clip"].partition("/")
        i["path"] = str(sources[head] / tail) if head in sources else ""
    by_clip = defaultdict(Counter)
    for i in issues:
        by_clip[i["clip"]][i["severity"]] += 1
    for r in rows:
        r["n_errors"], r["n_warnings"] = by_clip[r["clip"]]["ERROR"], by_clip[r["clip"]]["WARN"]
        r.pop("params_key", None)

    def write_csv(name, data):
        keys = list(dict.fromkeys(k for d in data for k in d))
        with open(args.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)

    write_csv("clip_report.csv", rows)
    write_csv("issues.csv", issues)
    if diff_rows:
        write_csv("log_vs_folders.csv", diff_rows)
    write_csv("feature_nulls.csv", sorted(null_rows, key=lambda r: -r["pooled_null_rate"]))

    sev_count = Counter(i["severity"] for i in issues)
    frac = lambda v: "-" if v is None else f"{v:.3f}"
    is_dataset_level = lambda c: c == "DATASET" or c in sources
    lines = ["# Validation overview\n",
             f"**{sev_count['ERROR']} error(s), {sev_count['WARN']} warning(s), {sev_count['INFO']} info.** "
             f"Clips found: {len(rows)} ({dict(Counter(r['source'] for r in rows))}), fully parsed: {len(ok_rows)}. "
             f"Sources: {', '.join(f'`{n}` = {p}' for n, p in sources.items())}\n",
             "ERROR = broken or inconsistent, fix or ask for a re-send. WARN = look at it. INFO = context. "
             "Every issue below says what to check; the same list with folder paths is in `issues.csv`.\n"]

    dataset_level = [i for i in issues if is_dataset_level(i["clip"]) and i["severity"] != "INFO"]
    lines += ["## 1. Dataset-level findings (look at these first)\n"]
    for i in dataset_level:
        lines += [f"- **{i['severity']} {i['code']}** ({i['clip']}): {i['message']}", f"  - *What to check:* {i['what_to_check']}"]
    if not dataset_level:
        lines += ["- none"]

    per_clip = defaultdict(list)
    for i in issues:
        if not is_dataset_level(i["clip"]) and i["severity"] != "INFO":
            per_clip[i["clip"]].append(i)
    ranked = sorted(per_clip.items(), key=lambda kv: (-sum(x["severity"] == "ERROR" for x in kv[1]), -len(kv[1]), kv[0]))
    esc = lambda t: str(t).replace("|", "\\|").replace("\n", " ")
    lines += [f"\n## 2. Clips to look at (errors first; {len(ranked)} clip(s) flagged, top 20 shown)\n",
              "Clip = `<source>/<folder>`; the full folder path is in the `path` column of `issues.csv`.\n",
              "| clip | severity | code | message | what to check |", "|---|---|---|---|---|"]
    for clip, items in ranked[:20]:
        for n, i in enumerate(items[:6]):
            lines.append(f"| {'`' + clip + '`' if n == 0 else ''} | {i['severity']} | {i['code']} | {esc(i['message'])} | {esc(i['what_to_check'])} |")
        if len(items) > 6:
            lines.append(f"| | | ... | {len(items) - 6} more for this clip in `issues.csv` | |")
    if not ranked:
        lines.append("| none | | | | |")
    if len(ranked) > 20:
        lines += ["", f"{len(ranked) - 20} more flagged clips: filter `issues.csv` or sort `clip_report.csv` by `n_errors`."]

    codes = Counter((i["severity"], i["code"]) for i in issues)
    lines += ["\n## 3. Issues by type\n", "| severity | code | count | what to check |", "|---|---|---|---|"]
    lines += [f"| {sv} | {c} | {n} | {hint(c)} |" for (sv, c), n in sorted(codes.items(), key=lambda x: (sev_rank[x[0][0]], -x[1]))]

    cross = Counter((r["source"], r.get("label")) for r in ok_rows)
    lines += ["\n## 4. Source folder vs tracker label (clips)\n", "| source | enter | pass | exit |", "|---|---|---|---|"]
    lines += [f"| {f} | {cross[(f, 'enter')]} | {cross[(f, 'pass')]} | {cross[(f, 'exit')]} |" for f in sources]
    if csv_stats:
        lines += ["\n## 5. clips.csv (tracker) vs folders\n", "| source | keep | removed | folders found | manual_label set | enter | pass | exit | removal reasons |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, st in csv_stats.items():
            bl = st["by_label"]
            lines.append(f"| {name} | {st['keep']} | {st['removed']} | {sum(r['source'] == name for r in rows)} | {st['manual']} | "
                         f"{bl[('enter', 'keep')]} | {bl[('pass', 'keep')]} | {bl[('exit', 'keep')]} | {dict(st['reasons'])} |")
    shown = {n: v for n, v in log_summaries.items() if v}
    if shown:
        lines += ["\n## 6. Pipeline logs (the run behind the data)\n"]
        for name, v in shown.items():
            run = v["run"]
            stages = ", ".join(f"{st['stage']} {st['done']}/{st['total']} ({st['seconds']:.0f} s)" for st in run["stages"])
            lines += [f"**{name}**: `{run['file']}` {run['status']}, {run['n_ok']} ok / {run['n_failed']} failed, "
                      f"{len(run['problems'])} warning/error line(s). Stages: {stages}.",
                      f"- log labels {dict((k, v['counts'][k]) for k in ('enter', 'pass', 'exit'))}, removed {v['counts']['removed']}; "
                      f"folders {dict((k, v['labels'][k]) for k in ('enter', 'pass', 'exit'))}",
                      f"- other logs: {v['others'] or 'none'}\n"]

    lines += [f"\n## 7. Feature null rates (features with a pooled null fraction above {args.show_null_above}; all features in `feature_nulls.csv`)\n",
              "| feature | pooled null | enter | pass | exit | clips over threshold | status |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['feature']} | {frac(r['pooled_null_rate'])} | {frac(r['null_rate_enter'])} | {frac(r['null_rate_pass'])} | {frac(r['null_rate_exit'])} | "
              f"{r['clips_over_threshold']}/{r['clips']} | {r['status']}{' (documented)' if r['documented_nullable'] else ''} |"
              for r in sorted(null_rows, key=lambda r: -r["pooled_null_rate"]) if r["pooled_null_rate"] > args.show_null_above][:40] or ["| none | | | | | | |"]
    lines += ["",
              "How to read it:",
              "- **pooled null**: fraction (0-1) of all frames with a detected person, over all clips, where the feature is null.",
              "- **enter / pass / exit**: the same share counted only over clips whose tracker label (`info.label`) is that class, "
              "not the source folder. `-` means no clip of that label had data for the feature.",
              f"- **clips over threshold**: in how many clips the feature is null in more than {args.null_threshold} of that clip's frames (the per-clip warning level, `--null-threshold`).",
              "- **status**: `ALWAYS_NULL` never filled; `MOSTLY_NULL` null in most clips (a property of the feature); "
              "`SOMETIMES_HIGH` normally filled but empty in some clips (those clips get an `UNUSUAL_NULLS` issue). "
              "`documented` means FEATURES.md or the sample files show that nulls are by design.",
              "- Large differences between enter and pass can be real behaviour (e.g. time-to-target only exists when approaching), "
              "not an error."]

    lines += ["\n## 8. Scene fingerprint groups (proxy for distinct locations / camera setups)\n",
              "| group | clips | enter | pass | fps values | sessions |", "|---|---|---|---|---|---|"]
    for g in range(len(centers)):
        members = [r for r in rows if r.get("scene_cluster") == g]
        lab = Counter(r.get("label") for r in members)
        lines.append(f"| {g} | {len(members)} | {lab['enter']} | {lab['pass']} | "
                     f"{sorted({r.get('fps') for r in members if r.get('fps')})} | {len({r.get('session') for r in members})} |")
    lines.append("\nGroups use floor and door area shares and floor height only (wall share is too noisy). Lighting changes do not split a group; "
                 "two different rooms with similar layout could merge. Check a few clips per group by eye.")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Validation finished: {len(rows)} clips, {sev_count['ERROR']} error(s), {sev_count['WARN']} warning(s), {sev_count['INFO']} info.")
    print(f"Overview with what to check: {args.out / 'summary.md'}")
    print("Details: issues.csv, clip_report.csv, feature_nulls.csv" + (", log_vs_folders.csv" if diff_rows else ""))
    sys.exit(1 if sev_count["ERROR"] else 0)

if __name__ == "__main__":
    main()