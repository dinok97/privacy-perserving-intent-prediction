#!/usr/bin/env python3
"""
video_processor.py

Turn clips (video files) into labeled clips of people entering, passing or exiting a doorway.

Read clips from `../source` and put the results into `../data`.

Each clip is processed to detect and track people, then produce 4 outputs:
  * a per-frame trajectory: center point, bounding box and the 17 keypoints;
  * the matching clip file itself (.mp4);
  * a JSON file with the trajectory and some metadata;
  * a "<name>_pose.json" sidecar with the skeleton as normalized, named landmarks.

Each finished clip is labelled enter / pass / exit and sorted into its folder.
Clips that look unusable are not deleted, only moved into removed/<reason>/, so
nothing is lost. Recordings made on the same day are appended into one dated
folder.

How to run
----------
  Set the options in main() at the bottom of this file, then just run the script
    (press Run in your editor, or `python video_processor.py`) -- no
  command-line arguments needed.

  Useful settings in main():
    save_clips=True    -> record and save clips (default)
    save_clips=False   -> track + label + show, but write nothing (monitor / setup)
    print_labels=True  -> print a line each time a person is labeled enter/pass/exit
    headless=True      -> no window (headless processing; stop with Ctrl+C)
        model="..."        -> a local YOLO pose .pt model
        source_dir="..."  -> process every supported video in this directory
        source_video="..."-> process only this specific video

Output is written under ../data:
  data/2026-05-29/
    enter/  pass/  exit/
    removed/  ( too_short/  too_few_points/  partial_bottom/  corrupt/ )
    _sessions/   one metadata file per recording session

Output clips and JSON files use the original source video filename stem. If a
source video contains multiple tracked people, later clips receive _2, _3, ...
to avoid overwriting the first clip.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

# cv2, numpy and ultralytics are imported inside LivePoseCollector so the file
# can be imported (e.g. to reuse the JSON helpers) without those heavy packages.

SCHEMA_VERSION = "2.0"

# COCO-17 keypoint order produced by Ultralytics YOLO pose models.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges (index pairs) used only for drawing the live overlay.
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
]

# The pose sidecar stores a named, normalized subset of the COCO-17 keypoints.
# This maps each landmark name to its index in the 17-point keypoint array.
SIDECAR_LANDMARK_TO_COCO = {
    "nose":            0,
    "left_ear":        3,
    "right_ear":       4,
    "left_shoulder":   5,
    "right_shoulder":  6,
    "left_elbow":      7,
    "right_elbow":     8,
    "left_wrist":      9,
    "right_wrist":     10,
    "left_hip":        11,
    "right_hip":       12,
    "left_knee":       13,
    "right_knee":      14,
    "left_ankle":      15,
    "right_ankle":     16,
}

# Jetson camera intrinsics (this pipeline runs on the Jetson only).
CAMERA_MTX = [[307.8047385, 0.0, 355.19676862],
              [0.0, 302.44366762, 233.22849986],
              [0.0, 0.0, 1.0]]
CAMERA_DIST = [-0.290496, 0.07539763, -0.00075077, -0.00159761, -0.00811828]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CollectorConfig:
    # Perception
    model: str = "yolo11n-pose.pt"   # local PyTorch pose model
    device: str = "cpu"              # use "cuda" only on a machine with CUDA PyTorch
    tracker: str = "bytetrack.yaml"  # lighter than botsort.yaml; BoT-SORT's per-frame motion compensation is wasted on a fixed camera
    conf: float = 0.3
    imgsz: int = 640

    # Camera / geometry
    camera_index: int = 0
    source_dir: str = "source"       # video files are read from this directory
    source_video: Optional[str] = None  # optional specific video; otherwise first file in source_dir
    distort: bool = False            # False -> no calibration; True -> use CAMERA_MTX/CAMERA_DIST
    ellipse_axes: tuple[int, int] = (480, 130)   # door entrance region

    # Recording
    save_clips: bool = True              # False -> track/label/show but write nothing (monitor mode)
    output_root: str = "data"
    record_fps: Optional[float] = None   # mp4 encode fps; None -> use the source's reported fps
    write_pose_sidecar: bool = True      # also write the <name>_pose.json sidecar
    session_note: str = ""               # free text saved in metadata (lighting, location, ...)

    # Track lifecycle
    grace_frames: int = 15           # frames a track may be missing before it counts as gone
    tail_seconds: float = 1.0        # extra seconds recorded after a track is gone (clip tail)
    reentry_jump_frac: float = 0.45  # a recycled id reappearing farther than this -> new clip
    min_duration_s: float = 0.5      # clips shorter than this go to removed/too_short

    # Thresholds for sorting each finished clip into enter / pass / exit / removed.
    exit_start_y_min_norm: float = 0.68
    exit_min_delta_dist_norm: float = 0.10
    exit_min_end_farther_ratio: float = 1.10
    bottom_presence_y_min_norm: float = 0.72
    bottom_presence_start_y_min_norm: float = 0.82
    bottom_presence_max_net_disp_norm: float = 0.18
    bottom_presence_max_y_span_norm: float = 0.26
    bottom_presence_min_bottom_touch_ratio: float = 0.45

    # Camera capture (USB). 640x480 matches the calibration and the training data.
    cap_width: int = 640
    cap_height: int = 480
    mjpg: bool = False               # request MJPG from the camera (often unlocks 30 fps)

    # UI
    fullscreen: bool = False
    headless: bool = False           # record without a window (screenless Jetson)
    print_labels: bool = True        # print a line each time a person is labeled enter/pass/exit
    status_every_s: float = 10.0     # print a one-line status heartbeat this often
    debug: bool = False


# ---------------------------------------------------------------------------
# Geometry and labelling helpers (plain Python; no cv2 / YOLO needed here)
# ---------------------------------------------------------------------------
def is_inside_ellipse(cx: float, cy: float,
                      door: tuple[float, float],
                      axes: tuple[float, float]) -> bool:
    """True if the point is inside the half-ellipse above the door center."""
    dx = cx - door[0]
    dy = cy - door[1]
    if dy > 0:        # below the door line -> never inside
        return False
    a, b = axes
    return (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1.0


def valid_centers(frames: list[dict]) -> list[tuple[float, float]]:
    """All center points from frames where the person was actually detected."""
    pts = []
    for fr in frames:
        c = fr.get("center")
        if isinstance(c, (list, tuple)) and len(c) == 2:
            pts.append((float(c[0]), float(c[1])))
    return pts


def normalize_distance(x: float, y: float, door_x: float, door_y: float,
                       width: int, height: int) -> float:
    """Distance to the door, with x scaled by width and y by height."""
    dx = (x - door_x) / float(max(1, width))
    dy = (y - door_y) / float(max(1, height))
    return math.sqrt(dx * dx + dy * dy)


def bottom_touch_ratio(frames: list[dict], frame_height: int, margin_px: int = 2) -> float:
    """Fraction of detected frames whose bounding box touches the bottom edge."""
    if frame_height <= 0:
        return 0.0
    total = touches = 0
    h1 = frame_height - 1
    for fr in frames:
        bb = fr.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        total += 1
        if float(bb[3]) >= (h1 - margin_px):
            touches += 1
    return (touches / total) if total else 0.0


def nearest_edge(cx: float, cy: float, width: int, height: int) -> str:
    """Which image border a point is closest to (saved as trajectory metadata)."""
    d = {"left": cx, "right": (width - 1) - cx,
         "top": cy, "bottom": (height - 1) - cy}
    return min(d, key=d.get)


def classify_trajectory(frames: list[dict],
                        door: tuple[float, float],
                        axes: tuple[float, float],
                        width: int, height: int,
                        min_frames: int,
                        cfg: CollectorConfig) -> tuple[str, str, str]:
    """
    Decide what a finished trajectory is and where it should go.

    Returns (label, result, reason):
      label  in {"enter", "pass", "exit"}     - the class
      result in {"keep", "exit", "removed"}   - the folder bucket
      reason - a short string for the metadata / subfolder name

    Checks run in order: length, then too-few-points, then exit, then a
    near-camera "partial person" case, otherwise keep.
    """
    n_frames = len(frames)
    if n_frames < min_frames:
        return "pass", "removed", "clip_shorter_than_min_duration"

    pts = valid_centers(frames)
    if len(pts) < 2:
        return "pass", "removed", "too_few_valid_points"

    door_x, door_y = float(door[0]), float(door[1])
    ys_n = [p[1] / float(max(1, height)) for p in pts]
    start, end = pts[0], pts[-1]
    start_y_n = start[1] / float(max(1, height))
    min_y_n, max_y_n = min(ys_n), max(ys_n)
    y_span_n = max_y_n - min_y_n

    dists = [normalize_distance(x, y, door_x, door_y, width, height) for x, y in pts]
    d_start, d_end = dists[0], dists[-1]
    delta_d = d_end - d_start
    net_disp = normalize_distance(end[0], end[1], start[0], start[1], width, height)

    # Base label comes from the last detected position (inside the door ellipse
    # -> enter, otherwise pass), exactly like the original script.
    base_label = "enter" if is_inside_ellipse(end[0], end[1], door, axes) else "pass"

    # exit: starts low (near the door) and ends farther away from it.
    if (start_y_n >= cfg.exit_start_y_min_norm
            and delta_d >= cfg.exit_min_delta_dist_norm
            and d_end >= max(1e-6, d_start) * cfg.exit_min_end_farther_ratio):
        return "exit", "exit", "starts_low_and_moves_away_from_door"

    # near-camera "partial person": stays low, barely moves, sits on the bottom edge.
    b_touch = bottom_touch_ratio(frames, height)
    if (start_y_n >= cfg.bottom_presence_start_y_min_norm
            and min_y_n >= cfg.bottom_presence_y_min_norm
            and net_disp <= cfg.bottom_presence_max_net_disp_norm
            and y_span_n <= cfg.bottom_presence_max_y_span_norm
            and b_touch >= cfg.bottom_presence_min_bottom_touch_ratio):
        return base_label, "removed", "partial_person_bottom_presence"

    return base_label, "keep", "accepted"


REASON_TO_SUBDIR = {
    "clip_shorter_than_min_duration": "too_short",
    "too_few_valid_points": "too_few_points",
    "partial_person_bottom_presence": "partial_bottom",
    "mp4_unreadable_or_corrupt": "corrupt",
}


def destination_dir(date_dir: Path, label: str, result: str, reason: str) -> Path:
    """Folder a finished clip should be written to."""
    if result == "keep":
        return date_dir / label                       # enter/ or pass/
    if result == "exit":
        return date_dir / "exit"
    sub = REASON_TO_SUBDIR.get(reason, "other")
    return date_dir / "removed" / sub


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------
def session_stamp(dt: datetime.datetime) -> str:
    """HHMMSS of the session start (keeps several same-day sessions apart)."""
    return dt.strftime("%H%M%S")


def date_stamp(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def clip_basename(source_stem: str, occurrence: int) -> str:
    """Use the source filename, adding a suffix only when needed."""
    return source_stem if occurrence == 1 else f"{source_stem}_{occurrence}"


# ---------------------------------------------------------------------------
# JSON assembly
# ---------------------------------------------------------------------------
def build_sample_json(*, track_id: int, label: str, result: str, reason: str,
                      fps: Optional[float], width: int, height: int,
                      door: tuple[float, float], axes: tuple[int, int],
                      frames: list[dict], meta: dict[str, Any]) -> dict[str, Any]:
    """
    Build the per-clip JSON. The top-level keys (label, fps, frame_width,
    frame_height, door_center, frames) describe the trajectory; the rest of the
    detail sits under "meta".
    """
    n_detected = sum(1 for f in frames if f.get("center") is not None)
    return {
        "id": int(track_id),
        "label": label,
        "fps": float(fps) if fps is not None else None,
        "frame_width": int(width),
        "frame_height": int(height),
        "door_center": [int(door[0]), int(door[1])],
        "ellipse_axes": [int(axes[0]), int(axes[1])],
        "cleaning_result": result,
        "cleaning_reason": reason,
        "n_frames": len(frames),
        "n_detected": n_detected,
        "detection_rate": round(n_detected / max(1, len(frames)), 4),
        "meta": meta,
        "frames": frames,
    }


def build_pose_sidecar(*, frames: list[dict], source_name: str, label: str,
                       fps: Optional[float], width: int, height: int) -> dict[str, Any]:
    """
    Write the skeleton as a compact "<name>_pose.json".

    Each frame is either {"frame_index": i, "detected": false} or, when the
    person was detected, a "landmarks" dict mapping each landmark name to
    {x, y, visibility}, with x/y normalised to [0, 1] over the full frame and
    visibility taken from the YOLO keypoint confidence. The frame indices match
    the trajectory JSON one-to-one.
    """
    w = float(max(1, width))
    h = float(max(1, height))
    out_frames: list[dict[str, Any]] = []
    n_detected = 0

    for fr in frames:
        fi = fr.get("frame_index")
        kpts = fr.get("keypoints")
        center = fr.get("center")
        if center is None or not kpts:
            out_frames.append({"frame_index": fi, "detected": False})
            continue

        landmarks: dict[str, dict[str, float]] = {}
        for name, idx in SIDECAR_LANDMARK_TO_COCO.items():
            if idx >= len(kpts):
                continue
            kx, ky, kc = kpts[idx][0], kpts[idx][1], kpts[idx][2]
            landmarks[name] = {
                "x": round(min(1.0, max(0.0, kx / w)), 5),
                "y": round(min(1.0, max(0.0, ky / h)), 5),
                "visibility": round(float(kc), 4),
            }
        out_frames.append({"frame_index": fi, "detected": True, "landmarks": landmarks})
        n_detected += 1

    return {
        "source": source_name,
        "label": label,
        "fps": float(fps) if fps is not None else None,
        "keypoint_source": "yolo-pose-coco17",
        "n_detected": n_detected,
        "n_frames": len(out_frames),
        "frames": out_frames,
    }


def _git_commit(cwd: Path) -> Optional[str]:
    """Short git commit of the repo this script lives in, if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd), stderr=subprocess.DEVNULL, timeout=2)
        return out.decode().strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live collector
# ---------------------------------------------------------------------------
class LivePoseCollector:
    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg

        # Heavy imports kept local to the collector.
        import cv2  # noqa
        import numpy as np  # noqa
        import ultralytics  # noqa
        from ultralytics import YOLO  # noqa
        self.cv2 = cv2
        self.np = np

        self.session_start = datetime.datetime.now()
        self.date_str = date_stamp(self.session_start)
        self.sess_str = session_stamp(self.session_start)

        root = Path(cfg.output_root)
        self.date_dir = root / self.date_str
        self.temp_dir = self.date_dir / "_temp"
        self.sessions_dir = self.date_dir / "_sessions"
        if cfg.save_clips:
            for d in (self.temp_dir, self.sessions_dir,
                      self.date_dir / "enter", self.date_dir / "pass",
                      self.date_dir / "exit",
                      self.date_dir / "removed" / "too_short",
                      self.date_dir / "removed" / "too_few_points",
                      self.date_dir / "removed" / "partial_bottom",
                      self.date_dir / "removed" / "corrupt"):
                d.mkdir(parents=True, exist_ok=True)

        # Calibration
        self.mtx = np.array(CAMERA_MTX)
        self.dist = np.array(CAMERA_DIST)

        # Software versions / commit, recorded in the metadata.
        self.versions = {
            "python": platform.python_version(),
            "opencv": getattr(cv2, "__version__", "?"),
            "numpy": getattr(np, "__version__", "?"),
            "ultralytics": getattr(ultralytics, "__version__", "?"),
        }
        self.git_commit = _git_commit(Path(__file__).resolve().parent)

        print(f"[INIT] Loading model: {cfg.model}")
        self.model = YOLO(cfg.model)

        # Runtime state (geometry is set per source by _init_geometry).
        self.cap = None
        self.apply_undistort = True
        self.frame_width = self.frame_height = 0
        self.clean_width = self.clean_height = 0
        self.door = (0, 0)
        self.axes = cfg.ellipse_axes
        self.record_fps = float(cfg.record_fps) if cfg.record_fps else 0.0
        self.min_frames = 2
        self.tail_frames = 0

        self.active: dict[int, dict] = {}   # track_id -> recording state
        self.current_source_stem = "source"
        self.source_clip_counts: dict[str, int] = {}
        self.seq = 0                        # clip counter for the whole session
        self.counts = {"enter": 0, "pass": 0, "exit": 0, "removed": 0}
        self.reason_counts: dict[str, int] = {}

        self._fps = 0
        self._fps_frames = 0
        self._fps_t0 = time.time()
        self._status_t0 = time.time()
        self._stage_ms = {"read": 0.0, "undistort": 0.0, "track+proc": 0.0, "draw": 0.0}
        self._stage_n = 0
        self._first_shape = None

        if cfg.save_clips:
            print(f"[INIT] output: {self.date_dir}  session: {self.sess_str}  "
                  f"sidecar: {'on' if cfg.write_pose_sidecar else 'off'}")
        else:
            print("[INIT] MONITOR mode (save_clips=False) - tracking/labeling only, nothing written.")

    def _init_geometry(self, width: int, height: int, reported_fps: float,
                       apply_undistort: bool = True, for_writing: bool = True):
        """Set the clean frame size, the door position and the recording fps."""
        cv2 = self.cv2
        self.frame_width, self.frame_height = int(width), int(height)
        self.apply_undistort = apply_undistort

        if apply_undistort and self.cfg.distort:
            self.newmtx, self.roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (self.frame_width, self.frame_height), 1,
                (self.frame_width, self.frame_height))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.newmtx,
                (self.frame_width, self.frame_height), cv2.CV_16SC2)  # fixed-point maps: faster remap
            self.clean_width, self.clean_height = self.roi[2], self.roi[3]
            mode = "remap+crop"
        elif apply_undistort:
            # Precompute the undistort maps once (same result as cv2.undistort,
            # but remap per frame is far cheaper than recomputing the map each call).
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.mtx, self.dist, None, self.mtx,
                (self.frame_width, self.frame_height), cv2.CV_16SC2)  # fixed-point maps: faster remap
            self.clean_width, self.clean_height = self.frame_width, self.frame_height
            mode = "undistort (remap)"
        else:
            self.clean_width, self.clean_height = self.frame_width, self.frame_height
            mode = "none"

        # Door at the bottom-center of the frame (h-1 = bottom row).
        self.door = (self.clean_width // 2, self.clean_height - 1)
        self.axes = self.cfg.ellipse_axes

        # Encode fps. Required when saving (to encode the mp4); when not saving it
        # is only informational. No guessed fallback: use record_fps if set, else
        # the rate the camera reports.
        if self.cfg.record_fps and self.cfg.record_fps > 0:
            self.record_fps = float(self.cfg.record_fps)
            fps_origin = "record_fps"
        elif reported_fps and reported_fps > 1:
            self.record_fps = float(reported_fps)
            fps_origin = "camera-reported"
        elif not for_writing:
            self.record_fps = 0.0
            fps_origin = "unknown (not saving)"
        else:
            raise RuntimeError(
                "The camera reported no usable FPS and record_fps was not set. "
                "Set record_fps in CollectorConfig so the recorder encodes at the "
                "real capture rate (this script never falls back to a guessed fps).")

        if self.record_fps > 0:
            self.min_frames = max(2, int(round(self.cfg.min_duration_s * self.record_fps)))
            self.tail_frames = max(0, int(round(self.cfg.tail_seconds * self.record_fps)))
        else:
            self.min_frames = 2
            self.tail_frames = 0

        note = "" if for_writing else "  (not saving)"
        print(f"[GEOM] frame={self.clean_width}x{self.clean_height} "
              f"record_fps={self.record_fps:.3f} ({fps_origin}) min_frames={self.min_frames} "
              f"grace={self.cfg.grace_frames} tail={self.tail_frames} undistort={mode}{note}")

    # -- per-track lifecycle -------------------------------------------------
    def _start_track(self, track_id: int, t_clip0: float):
        cv2 = self.cv2
        self.source_clip_counts[self.current_source_stem] += 1
        source_occurrence = self.source_clip_counts[self.current_source_stem]
        if self.cfg.save_clips:
            temp_path = self.temp_dir / f"track_{track_id}_{int(t_clip0 * 1000)}.mp4"
            writer = cv2.VideoWriter(
                str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.record_fps, (self.clean_width, self.clean_height))
        else:
            temp_path = None
            writer = None
        self.active[track_id] = {
            "writer": writer,
            "temp_path": temp_path,
            "frames": [],
            "vanish": 0,
            "last_center": None,
            "t0": t_clip0,
            "wall_start": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_stem": self.current_source_stem,
            "source_occurrence": source_occurrence,
        }
        if self.cfg.debug:
            print(f"[START] id={track_id} -> {temp_path.name}")

    def _append_frame(self, track_id: int, t_now: float, center, bbox, keypoints):
        rec = self.active[track_id]
        rec["frames"].append({
            "frame_index": len(rec["frames"]),
            "t": round(t_now - rec["t0"], 4),
            "center": center,
            "bbox": bbox,
            "keypoints": keypoints,
        })
        if center is not None:
            rec["last_center"] = center

    def _finalize_track(self, track_id: int):
        cfg = self.cfg
        rec = self.active.pop(track_id)
        if rec["writer"] is not None:
            rec["writer"].release()
        frames = rec["frames"]

        # Effective fps measured from the per-frame timestamps (the real achieved
        # rate, written as the top-level "fps"). No fallback: a one-frame clip that
        # cannot be measured (and is removed anyway) gets null.
        span = frames[-1]["t"] if frames else 0.0
        eff_fps = (len(frames) - 1) / span if span > 1e-6 else None

        # When saving, a clip whose mp4 was never written is treated as corrupt.
        corrupt = False
        if cfg.save_clips and rec["temp_path"] is not None:
            corrupt = (not rec["temp_path"].exists()) or rec["temp_path"].stat().st_size < 1024
        if corrupt:
            label, result, reason = "pass", "removed", "mp4_unreadable_or_corrupt"
        else:
            label, result, reason = classify_trajectory(
                frames, self.door, self.axes,
                self.clean_width, self.clean_height,
                self.min_frames, cfg)

        self.seq += 1
        n_kpts = sum(1 for f in frames if f.get("keypoints"))

        if cfg.save_clips:
            pts = valid_centers(frames)
            entry_edge = nearest_edge(*pts[0], self.clean_width, self.clean_height) if pts else None
            exit_edge = nearest_edge(*pts[-1], self.clean_width, self.clean_height) if pts else None
            meta = {
                "schema_version": SCHEMA_VERSION,
                "session_id": f"{self.date_str}_{self.sess_str}",
                "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "wall_start": rec["wall_start"],
                "wall_end": datetime.datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(span, 3),
                "effective_fps": round(eff_fps, 3) if eff_fps is not None else None,
                "record_fps": round(self.record_fps, 3),   # fps the mp4 is encoded at
                "n_gap_frames": sum(1 for f in frames if f.get("center") is None),
                "n_pose_frames": n_kpts,
                "entry_edge": entry_edge,
                "exit_edge": exit_edge,
                "model": cfg.model,
                "source_video": rec["source_stem"],
                "tracker": cfg.tracker,
                "conf_threshold": cfg.conf,
                "imgsz": cfg.imgsz,
                "camera_index": cfg.camera_index,
                "capture_resolution": [int(cfg.cap_width), int(cfg.cap_height)],
                "distort": cfg.distort,
                "session_note": cfg.session_note,
                "software": self.versions,
                "git_commit": self.git_commit,
                "keypoint_format": "COCO-17",
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "keypoint_coords": "pixels (divide by frame_width/height to normalise)",
                "pose_sidecar": cfg.write_pose_sidecar,
            }
            sample = build_sample_json(
                track_id=track_id, label=label, result=result, reason=reason,
                fps=eff_fps, width=self.clean_width, height=self.clean_height,
                door=self.door, axes=self.axes, frames=frames, meta=meta)
            base = clip_basename(rec["source_stem"], rec["source_occurrence"])
            dst = destination_dir(self.date_dir, label, result, reason)
            dst.mkdir(parents=True, exist_ok=True)
            with (dst / f"{base}.json").open("w", encoding="utf-8") as f:
                json.dump(sample, f, indent=2)
            if cfg.write_pose_sidecar and result == "keep" and n_kpts > 0:
                sidecar = build_pose_sidecar(
                    frames=frames, source_name=f"{base}.mp4", label=label,
                    fps=eff_fps, width=self.clean_width, height=self.clean_height)
                with (dst / f"{base}_pose.json").open("w", encoding="utf-8") as f:
                    json.dump(sidecar, f)
            if rec["temp_path"] is not None and rec["temp_path"].exists():
                os.replace(rec["temp_path"], dst / f"{base}.mp4")

        # Tally + one-line console update (printed only if print_labels).
        bucket = "removed" if result == "removed" else label
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        if cfg.print_labels:
            c = self.counts
            totals = f"enter={c['enter']} pass={c['pass']} exit={c['exit']} removed={c['removed']}"
            if result == "removed":
                verb = "removed" if cfg.save_clips else "would remove"
                print(f"[REMOVED:{REASON_TO_SUBDIR.get(reason, reason)}] id{track_id} "
                      f"({span:.1f}s) {verb}   so far: {totals}")
            else:
                verb = "saved" if cfg.save_clips else "labeled"
                print(f"[{label.upper()}] {verb} id{track_id} ({span:.1f}s, {len(frames)} frames)"
                      f"   so far: {totals}")

    # -- perception ----------------------------------------------------------
    def _undistort(self, frame):
        if not self.apply_undistort:
            return frame
        cv2 = self.cv2
        frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        if self.cfg.distort:
            x, y, w, h = self.roi
            return frame[y:y + h, x:x + w]
        return frame

    def _extract_detections(self, result):
        """Return [(track_id, bbox[4], center[2], keypoints[17] or None), ...]."""
        out = []
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return out
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()

        kxy = kconf = None
        kpts = getattr(result, "keypoints", None)
        if kpts is not None and kpts.xy is not None:
            kxy = kpts.xy.cpu().numpy()                       # (N, 17, 2)
            kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else None

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            keypoints = None
            if kxy is not None and i < len(kxy):
                pts17 = []
                for j in range(kxy.shape[1]):
                    kx, ky = float(kxy[i, j, 0]), float(kxy[i, j, 1])
                    kc = float(kconf[i, j]) if kconf is not None else 0.0
                    pts17.append([round(kx, 2), round(ky, 2), round(kc, 4)])
                keypoints = pts17
            out.append((int(tid), bbox, center, keypoints))
        return out

    def _draw_overlay(self, frame, dets):
        cv2 = self.cv2
        for tid, bbox, center, kpts in dets:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if kpts:
                for (kx, ky, kc) in kpts:
                    if kc > 0.3:
                        cv2.circle(frame, (int(kx), int(ky)), 2, (0, 200, 255), -1)
                for a, b in COCO_SKELETON:
                    if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                        cv2.line(frame, (int(kpts[a][0]), int(kpts[a][1])),
                                 (int(kpts[b][0]), int(kpts[b][1])), (255, 180, 0), 1)
        cv2.ellipse(frame, self.door, self.axes, 0, 180, 360, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {self._fps}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        c = self.counts
        tag = "" if self.cfg.save_clips else "NOT SAVING  "
        cv2.putText(frame, f"{tag}enter:{c['enter']} pass:{c['pass']} "
                            f"exit:{c['exit']} removed:{c['removed']}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

    def _process_frame(self, clean, t_now: float):
        """Run detection + tracking and update the per-track recordings."""
        result = self.model.track(
            clean, persist=True, classes=[0],
            tracker=self.cfg.tracker, conf=self.cfg.conf,
            imgsz=self.cfg.imgsz, device=self.cfg.device, verbose=False)[0]
        dets = self._extract_detections(result)

        seen = set()
        for tid, bbox, center, kpts in dets:
            seen.add(tid)
            if tid in self.active:
                rec = self.active[tid]
                # A recycled id that pops back far from where it disappeared is a
                # different person -> close the old clip and start a new one.
                if rec["vanish"] > 0 and rec["last_center"] is not None:
                    jump = normalize_distance(
                        center[0], center[1],
                        rec["last_center"][0], rec["last_center"][1],
                        self.clean_width, self.clean_height)
                    if jump > self.cfg.reentry_jump_frac:
                        self._finalize_track(tid)
                        self._start_track(tid, t_now)
                if tid in self.active:
                    self.active[tid]["vanish"] = 0
            else:
                self._start_track(tid, t_now)

            if self.active[tid]["writer"] is not None:
                self.active[tid]["writer"].write(clean)
            self._append_frame(tid, t_now, center, bbox, kpts)

        # Tracks not seen this frame: keep them alive within the grace window
        # (writing null frames so the indices stay aligned), then finalise.
        for tid in list(self.active.keys()):
            if tid in seen:
                continue
            rec = self.active[tid]
            rec["vanish"] += 1
            if rec["writer"] is not None:
                rec["writer"].write(clean)
            self._append_frame(tid, t_now, None, None, None)
            if rec["vanish"] >= self.cfg.grace_frames + self.tail_frames:
                self._finalize_track(tid)

        return dets

    def _source_paths(self) -> list[Path]:
        if self.cfg.source_video:
            sources = [Path(self.cfg.source_video)]
        else:
            source_dir = Path(self.cfg.source_dir)
            sources = sorted(
                path for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
            ) if source_dir.is_dir() else []
            if not sources:
                raise RuntimeError(
                    f"No video files found in {source_dir}. "
                    "Place an .mp4, .avi, .mov, .mkv, or .m4v file there."
                )
        for source in sources:
            if not source.exists():
                raise RuntimeError(f"Video source does not exist: {source}")
        return sources

    def _open_source(self, source: Path):
        cv2 = self.cv2
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        print(f"[INIT] video source: {source}")
        return cap

    def _reset_tracker(self):
        """Clear Ultralytics tracker state before processing another video."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if reset is not None:
                    reset()

    def _run_source(self, source: Path, display: bool) -> bool:
        """Process one video and return False when the user requested a stop."""
        cv2 = self.cv2
        self.current_source_stem = source.stem
        self.source_clip_counts.setdefault(self.current_source_stem, 0)
        self.cap = self._open_source(source)
        self.active.clear()
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._init_geometry(w, h, reported,
                            apply_undistort=self.cfg.distort,
                            for_writing=self.cfg.save_clips)
        try:
            while True:
                t0 = time.perf_counter()
                ret, frame = self.cap.read()
                t_now = time.perf_counter()          # read done (also the per-frame timestamp)
                if not ret:
                    print(f"[RUN] Finished video: {source.name}")
                    return True
                clean = self._undistort(frame)
                t_und = time.perf_counter()
                dets = self._process_frame(clean, t_now)
                t_proc = time.perf_counter()

                if self.cfg.debug:
                    self._stage_ms["read"] += (t_now - t0) * 1000.0
                    self._stage_ms["undistort"] += (t_und - t_now) * 1000.0
                    self._stage_ms["track+proc"] += (t_proc - t_und) * 1000.0
                    self._stage_n += 1
                    if self._first_shape is None:
                        self._first_shape = clean.shape
                        print(f"[DEBUG] actual captured frame shape: {clean.shape}")

                self._fps_frames += 1
                now = time.time()
                if now - self._fps_t0 >= 1.0:
                    self._fps = self._fps_frames
                    self._fps_frames = 0
                    self._fps_t0 = now
                if now - self._status_t0 >= self.cfg.status_every_s:
                    self._status_t0 = now
                    c = self.counts
                    word = "recording" if self.cfg.save_clips else "monitoring"
                    msg = (f"[..] {word}  fps={self._fps}  tracking={len(self.active)}  "
                           f"|  enter={c['enter']} pass={c['pass']} exit={c['exit']} removed={c['removed']}")
                    if self.cfg.debug and self._stage_n:
                        n = self._stage_n
                        msg += ("  ||  per-frame ms:"
                                f" read={self._stage_ms['read'] / n:.1f}"
                                f" undistort={self._stage_ms['undistort'] / n:.1f}"
                                f" track+proc={self._stage_ms['track+proc'] / n:.1f}"
                                f" draw={self._stage_ms['draw'] / n:.1f}")
                        for k in self._stage_ms:
                            self._stage_ms[k] = 0.0
                        self._stage_n = 0
                    print(msg)

                if display:
                    t_d = time.perf_counter()
                    disp = clean.copy()
                    self._draw_overlay(disp, dets)
                    if not self.cfg.fullscreen:
                        disp = cv2.resize(disp, (960, 540))
                    cv2.imshow("Pose Data Collection", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if self.cfg.debug:
                        self._stage_ms["draw"] += (time.perf_counter() - t_d) * 1000.0
                    if key == ord("q"):
                        return False
        except KeyboardInterrupt:
            print("\n[RUN] Stopped by user (Ctrl+C).")
            return False
        finally:
            for tid in list(self.active.keys()):
                self._finalize_track(tid)
            self.active.clear()
            self.cap.release()
            self.cap = None

    def run(self, display: bool = True):
        """Process all configured videos. display=False runs headlessly."""
        cv2 = self.cv2
        action = "Recording" if self.cfg.save_clips else "Monitoring (not saving)"
        sources = self._source_paths()
        print(f"[RUN] {action}: {len(sources)} video(s)")
        if display:
            print("[RUN] Press 'q' in the window to stop.")
            cv2.namedWindow("Pose Data Collection", cv2.WINDOW_NORMAL)
            if self.cfg.fullscreen:
                cv2.setWindowProperty("Pose Data Collection",
                                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            print(f"[RUN] {action} (headless). Press Ctrl+C to stop.")

        try:
            for index, source in enumerate(sources, start=1):
                print(f"[RUN] video {index}/{len(sources)}")
                self._reset_tracker()
                if not self._run_source(source, display):
                    break
        except KeyboardInterrupt:
            print("\n[RUN] Stopped by user (Ctrl+C).")
        finally:
            self.cleanup()

    def cleanup(self):
        for tid in list(self.active.keys()):
            self._finalize_track(tid)
        if self.cap is not None:
            self.cap.release()
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass
        if self.cfg.save_clips:
            self._write_session_metadata()
        print(f"[DONE] session {self.sess_str}: {self.counts}  reasons={self.reason_counts}")

    def _write_session_metadata(self):
        meta = {
            "schema_version": SCHEMA_VERSION,
            "session_id": f"{self.date_str}_{self.sess_str}",
            "session_start": self.session_start.isoformat(timespec="seconds"),
            "session_end": datetime.datetime.now().isoformat(timespec="seconds"),
            "session_note": self.cfg.session_note,
            "software": self.versions,
            "git_commit": self.git_commit,
            "config": _config_to_jsonable(self.cfg),
            "frame_width": self.clean_width,
            "frame_height": self.clean_height,
            "record_fps": self.record_fps,
            "door_center": [int(self.door[0]), int(self.door[1])],
            "ellipse_axes": [int(self.axes[0]), int(self.axes[1])],
            "n_clips_total": self.seq,
            "counts": self.counts,
            "reason_counts": self.reason_counts,
        }
        path = self.sessions_dir / f"session_{self.date_str}_{self.sess_str}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[DONE] session metadata: {path}")


def _config_to_jsonable(cfg: CollectorConfig) -> dict:
    d = asdict(cfg)
    d["ellipse_axes"] = list(cfg.ellipse_axes)
    return d


# ---------------------------------------------------------------------------
# Run the collector
# ---------------------------------------------------------------------------
def main() -> int:
    # ----------------------------------------------------------------------- #
    # Settings -- edit these, then just run the script (Run in your editor, or #
    # `python video_processor.py`). No command-line arguments are needed. #
    # ----------------------------------------------------------------------- #
    cfg = CollectorConfig(
        # --- perception / speed knobs ---
        model=str(Path(__file__).resolve().parents[1] / "yolo11n-pose.pt"),
        device="cpu",
        tracker="bytetrack.yaml",   # lighter than botsort.yaml for CPU processing
        imgsz=640,                   # lower (e.g. 480) trades accuracy for CPU speed
        # --- video source / geometry ---
        source_dir=str(Path(__file__).resolve().parents[1] / "source"),
        source_video=None,           # set a path to override source_dir selection
        ellipse_axes=(480, 130),     # door entrance region (half-ellipse)
        distort=False,               # new video source: no Jetson calibration by default
        # --- run mode ---
        fullscreen=False,
        save_clips=True,             # False -> track + label + show, but write nothing (monitor)
        print_labels=True,            # print a line each time a person is labeled enter/pass/exit
        headless=True,                # no preview window; set False to watch the live window
        debug=False,                  # prints the per-frame ms breakdown; set False for normal runs
        session_note="",             # optional free-text note saved in each clip's metadata
    )

    # The dataset is written next to this script, in ../data.
    cfg.output_root = str(Path(__file__).resolve().parents[1] / "data")

    LivePoseCollector(cfg).run(display=not cfg.headless)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


