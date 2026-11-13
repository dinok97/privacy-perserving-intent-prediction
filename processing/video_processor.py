#!/usr/bin/env python3
"""
video_processor.py

Track the people in each input video and label each of them enter / pass / exit.

Nothing is clipped and no video is written. For every tracked person the input
video produces two files:
  * <name>.json       the trajectory: per frame the center, bounding box and the
                      17 keypoints, plus the label and some metadata;
  * <name>_pose.json  the skeleton as normalized, named landmarks.

Each JSON covers the WHOLE input video: frame_index 0 is the first frame of the
input video and t = frame_index / fps. Frames where the person is not visible
have center/bbox/keypoints = null. The video the frames belong to is the input
video itself; its sha256 is stored as meta.clip_video_sha256.

<name> is the input video's filename stem, with _2, _3, ... when one video has
several people.

Tracks that look unusable (too short, too few points, a partial person at the
bottom edge) are not deleted, only marked cleaning_result="removed"; they stay
out of the processed manifest that pipeline.py reads.

Output is written under ../data:
  data/2026-05-29/
    <name>.json  <name>_pose.json      one pair per tracked person
    clips.csv                          one row per person: label, cleaning result, files
    session_<date>_<time>.json         one metadata file per run

Set manual_label in clips.csv to correct a label by hand - it is kept when the
same video is processed again.

How to run
----------
  Set the options in run() at the bottom of this file, then just run the script
  (`python video_processor.py`). No command-line arguments are needed.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

# cv2 and ultralytics are imported inside LivePoseCollector so the file can be
# imported (e.g. to reuse the JSON helpers) without those heavy packages.

SCHEMA_VERSION = "3.0"  # 3.0: frames cover the whole input video, no clip video
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# COCO-17 keypoint order produced by Ultralytics YOLO pose models.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
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

    # Input videos
    source_dir: str = "source"       # video files are read from this directory
    source_video: Optional[str] = None  # optional specific video; otherwise every video in source_dir
    source_videos: Optional[list] = None  # explicit list of videos to process (overrides source_dir scanning)
    processed_manifest: Optional[str] = None  # explicit manifest path for pipeline orchestration
    ellipse_axes: tuple[int, int] = (480, 130)   # door entrance region

    # Output
    output_root: str = "data"
    write_pose_sidecar: bool = True      # also write the <name>_pose.json sidecar
    session_note: str = ""               # free text saved in metadata (lighting, location, ...)

    # Track lifecycle
    grace_frames: int = 15           # frames a track may be missing before it counts as gone
    reentry_jump_frac: float = 0.45  # a recycled id reappearing farther than this -> new person
    min_duration_s: float = 0.5      # tracks shorter than this are removed (too short)

    # Thresholds for labelling each track enter / pass / exit, or removing it.
    exit_start_y_min_norm: float = 0.68
    exit_min_delta_dist_norm: float = 0.10
    exit_min_end_farther_ratio: float = 1.10
    bottom_presence_y_min_norm: float = 0.72
    bottom_presence_start_y_min_norm: float = 0.82
    bottom_presence_max_net_disp_norm: float = 0.18
    bottom_presence_max_y_span_norm: float = 0.26
    bottom_presence_min_bottom_touch_ratio: float = 0.45

    print_labels: bool = True        # print a line each time a person is labeled enter/pass/exit


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
    Decide what a track is: its label, and whether it is usable.

    frames are the track's own frames, from its first to its last detection.

    Returns (label, result, reason):
      label  in {"enter", "pass", "exit"}     - the class
      result in {"keep", "removed"}           - usable or not; the class is only in label
      reason - a short string for the metadata

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
        return "exit", "keep", "starts_low_and_moves_away_from_door"

    # near-camera "partial person": stays low, barely moves, sits on the bottom edge.
    b_touch = bottom_touch_ratio(frames, height)
    if (start_y_n >= cfg.bottom_presence_start_y_min_norm
            and min_y_n >= cfg.bottom_presence_y_min_norm
            and net_disp <= cfg.bottom_presence_max_net_disp_norm
            and y_span_n <= cfg.bottom_presence_max_y_span_norm
            and b_touch >= cfg.bottom_presence_min_bottom_touch_ratio):
        return base_label, "removed", "partial_person_bottom_presence"

    return base_label, "keep", "accepted"


CLIPS_CSV_COLUMNS = ["clip_id", "label", "manual_label", "cleaning_result", "cleaning_reason",
                     "session_id", "n_frames", "n_detected", "duration_s",
                     "video", "trajectory", "pose"]


def update_clips_csv(path: Path, rows: list[dict]) -> None:
    """Merge rows into clips.csv by clip_id. Rows of reprocessed clips are replaced,
    but a manual_label set by hand is kept."""
    existing: dict[str, dict] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["clip_id"]] = row
    for row in rows:
        old = existing.get(row["clip_id"])
        if old and old.get("manual_label"):
            row["manual_label"] = old["manual_label"]
        existing[row["clip_id"]] = row
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIPS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing.values())
    os.replace(tmp, path)


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
    Build the per-person JSON. The top-level keys (label, fps, frame_width,
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
# Tracker
# ---------------------------------------------------------------------------
class LivePoseCollector:
    """Tracks the people in each input video and writes their JSON files.

    (The name is kept from the old live recorder: geometry3d/track_single_video.py
    imports this class for _extract_detections.)"""

    def __init__(self, cfg: CollectorConfig):
        self.cfg = cfg

        # Heavy imports kept local to the collector.
        import cv2  # noqa
        import numpy as np  # noqa
        import ultralytics  # noqa
        from ultralytics import YOLO  # noqa
        self.cv2 = cv2

        self.session_start = datetime.datetime.now()
        self.date_str = date_stamp(self.session_start)
        self.sess_str = session_stamp(self.session_start)
        self.session_id = f"{self.date_str}_{self.sess_str}"

        self.date_dir = Path(cfg.output_root) / self.date_str
        self.date_dir.mkdir(parents=True, exist_ok=True)

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

        # Per-video geometry (set by _run_source).
        self.width = self.height = 0
        self.fps = 0.0
        self.door = (0, 0)
        self.axes = cfg.ellipse_axes
        self.min_frames = 2

        self.active: dict[int, dict] = {}   # track_id -> track being followed
        self.finished: list[dict] = []      # tracks of the current video that have ended
        self.source_clip_counts: dict[str, int] = {}
        self.current_source_stem = "source"
        self.seq = 0                        # person counter for the whole session
        self.processed_records: list[dict] = []   # kept tracks only -> pipeline manifest
        self.clip_rows: list[dict] = []           # all tracks -> clips.csv
        self.counts = {"enter": 0, "pass": 0, "exit": 0, "removed": 0}
        self.reason_counts: dict[str, int] = {}

        print(f"[INIT] output: {self.date_dir}  session: {self.sess_str}  "
              f"sidecar: {'on' if cfg.write_pose_sidecar else 'off'}")

    # -- per-track lifecycle -------------------------------------------------
    def _start_track(self, track_id: int):
        self.source_clip_counts[self.current_source_stem] += 1
        occurrence = self.source_clip_counts[self.current_source_stem]
        self.active[track_id] = {
            "track_id": track_id,
            "base": clip_basename(self.current_source_stem, occurrence),
            "detections": {},     # frame_index -> (center, bbox, keypoints)
            "vanish": 0,
            "last_center": None,
        }

    def _end_track(self, track_id: int):
        self.finished.append(self.active.pop(track_id))

    def _write_track(self, rec: dict, source: Path, n_frames: int, video_sha256: str):
        """Label one finished track and write its JSON files over the whole video."""
        cfg = self.cfg
        detections = rec["detections"]
        first, last = min(detections), max(detections)

        def frame(i):
            center, bbox, kpts = detections.get(i, (None, None, None))
            return {"frame_index": i, "t": round(i / self.fps, 4),
                    "center": center, "bbox": bbox, "keypoints": kpts}

        # Label from the track's own span; the JSON covers the whole input video.
        label, result, reason = classify_trajectory(
            [frame(i) for i in range(first, last + 1)], self.door, self.axes,
            self.width, self.height, self.min_frames, cfg)
        frames = [frame(i) for i in range(n_frames)]

        self.seq += 1
        n_kpts = sum(1 for f in frames if f.get("keypoints"))
        pts = valid_centers(frames)
        base = rec["base"]
        meta = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "track_start_frame": first,
            "track_end_frame": last,
            "duration_s": round((last - first) / self.fps, 3),
            "effective_fps": round(self.fps, 3),
            "record_fps": round(self.fps, 3),
            "n_gap_frames": (last - first + 1) - len(detections),
            "n_pose_frames": n_kpts,
            "entry_edge": nearest_edge(*pts[0], self.width, self.height) if pts else None,
            "exit_edge": nearest_edge(*pts[-1], self.width, self.height) if pts else None,
            "model": cfg.model,
            "source_video": source.name,
            # sha256 of the input video these frames index into (key name kept for
            # geometry3d.adapt_trajectory and validation/validate_dataset.py)
            "clip_video_sha256": video_sha256,
            "time_basis": "source_video_frames",
            "tracker": cfg.tracker,
            "conf_threshold": cfg.conf,
            "imgsz": cfg.imgsz,
            "session_note": cfg.session_note,
            "software": self.versions,
            "git_commit": self.git_commit,
            "keypoint_format": "COCO-17",
            "keypoint_names": COCO_KEYPOINT_NAMES,
            "keypoint_coords": "pixels (divide by frame_width/height to normalise)",
            "pose_sidecar": cfg.write_pose_sidecar,
        }
        sample = build_sample_json(
            track_id=rec["track_id"], label=label, result=result, reason=reason,
            fps=self.fps, width=self.width, height=self.height,
            door=self.door, axes=self.axes, frames=frames, meta=meta)
        trajectory_path = self.date_dir / f"{base}.json"
        with trajectory_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2)

        # pose for every track, so a removed one that is restored by hand still has it
        pose_path = None
        if cfg.write_pose_sidecar and n_kpts > 0:
            sidecar = build_pose_sidecar(
                frames=frames, source_name=source.name, label=label,
                fps=self.fps, width=self.width, height=self.height)
            pose_path = self.date_dir / f"{base}_pose.json"
            with pose_path.open("w", encoding="utf-8") as f:
                json.dump(sidecar, f)

        if result == "keep":          # removed tracks never reach the pipeline
            record = {"video": str(source.resolve()), "trajectory": str(trajectory_path.resolve())}
            if pose_path is not None:
                record["pose"] = str(pose_path.resolve())
            self.processed_records.append(record)
        self.clip_rows.append({
            "clip_id": base, "label": label, "manual_label": "",
            "cleaning_result": result, "cleaning_reason": reason,
            "session_id": self.session_id, "n_frames": len(frames),
            "n_detected": sample["n_detected"], "duration_s": meta["duration_s"],
            "video": source.name, "trajectory": trajectory_path.name,
            "pose": pose_path.name if pose_path else None,
        })

        bucket = "removed" if result == "removed" else label
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        if cfg.print_labels:
            c = self.counts
            totals = f"enter={c['enter']} pass={c['pass']} exit={c['exit']} removed={c['removed']}"
            if result == "removed":
                print(f"[REMOVED:{reason}] {base} ({meta['duration_s']:.1f}s)   so far: {totals}")
            else:
                print(f"[{label.upper()}] {base} ({meta['duration_s']:.1f}s, "
                      f"{sample['n_detected']} detected frames)   so far: {totals}")

    # -- perception ----------------------------------------------------------
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

    def _process_frame(self, frame, frame_index: int):
        """Run detection + tracking on one frame and update the tracks."""
        result = self.model.track(
            frame, persist=True, classes=[0],
            tracker=self.cfg.tracker, conf=self.cfg.conf,
            imgsz=self.cfg.imgsz, device=self.cfg.device, verbose=False)[0]

        seen = set()
        for tid, bbox, center, kpts in self._extract_detections(result):
            seen.add(tid)
            if tid in self.active:
                rec = self.active[tid]
                # A recycled id that pops back far from where it disappeared is a
                # different person -> end the old track and start a new one.
                if rec["vanish"] > 0 and rec["last_center"] is not None:
                    jump = normalize_distance(
                        center[0], center[1],
                        rec["last_center"][0], rec["last_center"][1],
                        self.width, self.height)
                    if jump > self.cfg.reentry_jump_frac:
                        self._end_track(tid)
                        self._start_track(tid)
                self.active[tid]["vanish"] = 0
            else:
                self._start_track(tid)
            rec = self.active[tid]
            rec["detections"][frame_index] = (center, bbox, kpts)
            rec["last_center"] = center

        # Tracks not seen this frame end once they are missing for grace_frames.
        for tid in list(self.active.keys()):
            if tid in seen:
                continue
            self.active[tid]["vanish"] += 1
            if self.active[tid]["vanish"] >= self.cfg.grace_frames:
                self._end_track(tid)

    def _source_paths(self) -> list[Path]:
        if self.cfg.source_video:
            sources = [Path(self.cfg.source_video)]
        elif self.cfg.source_videos:
            sources = [Path(video) for video in self.cfg.source_videos]
        else:
            source_dir = Path(self.cfg.source_dir)
            sources = sorted(
                path for path in source_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
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

    def _reset_tracker(self):
        """Clear Ultralytics tracker state before processing another video."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if reset is not None:
                    reset()

    def _run_source(self, source: Path):
        """Track one whole video, then write one JSON pair per tracked person."""
        cv2 = self.cv2
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        print(f"[INIT] video source: {source}")
        self.current_source_stem = source.stem
        self.source_clip_counts.setdefault(self.current_source_stem, 0)
        self.active.clear()
        self.finished.clear()
        try:
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.fps <= 0:
                raise RuntimeError(f"Video reports no usable FPS: {source}")
            self.door = (self.width // 2, self.height - 1)   # bottom-center of the frame
            self.min_frames = max(2, int(round(self.cfg.min_duration_s * self.fps)))

            n_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self._process_frame(frame, n_frames)
                n_frames += 1
        finally:
            cap.release()

        # Written only after the whole video was read, so every JSON has one
        # frame per video frame (a video that fails midway writes nothing).
        for tid in list(self.active.keys()):
            self._end_track(tid)
        if n_frames != declared:
            print(f"[WARN] {source.name}: read {n_frames} frames, the video declares {declared}")
        if self.finished:
            video_sha256 = file_sha256(source)
            for rec in self.finished:
                self._write_track(rec, source, n_frames, video_sha256)
        self.finished.clear()
        print(f"[RUN] Finished video: {source.name}")

    def run(self, on_progress=None, failures=None, fail_fast: bool = False):
        """Process all configured videos.

        on_progress, if given, is called once after each finished video
        (on_progress(ok=False) for a video that failed).
        A video that raises is skipped and added to failures as
        {'video', 'stage', 'error'}; with fail_fast=True the error is raised instead."""
        sources = self._source_paths()
        print(f"[RUN] Tracking {len(sources)} video(s). Press Ctrl+C to stop.")
        try:
            for index, source in enumerate(sources, start=1):
                print(f"[RUN] video {index}/{len(sources)}")
                self._reset_tracker()
                try:
                    self._run_source(source)
                except Exception as error:
                    if fail_fast:
                        raise
                    print(f"[RUN] {Path(source).name}: FAILED ({type(error).__name__}: {error})")
                    if failures is not None:
                        failures.append({"video": str(source), "stage": "tracking",
                                         "error": f"{type(error).__name__}: {error}"})
                    if on_progress:
                        on_progress(ok=False)
                    continue
                if on_progress:
                    on_progress()
        except KeyboardInterrupt:
            print("\n[RUN] Stopped by user (Ctrl+C).")
        finally:
            self.cleanup()

    def cleanup(self):
        self._write_session_metadata()
        self._write_processed_manifest()
        if self.clip_rows:
            update_clips_csv(self.date_dir / "clips.csv", self.clip_rows)
            print(f"[DONE] clip index: {self.date_dir / 'clips.csv'}")
        print(f"[DONE] session {self.sess_str}: {self.counts}  reasons={self.reason_counts}")

    def _write_session_metadata(self):
        meta = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(timespec="seconds"),
            "session_end": datetime.datetime.now().isoformat(timespec="seconds"),
            "session_note": self.cfg.session_note,
            "software": self.versions,
            "git_commit": self.git_commit,
            "config": _config_to_jsonable(self.cfg),
            "n_clips_total": self.seq,
            "counts": self.counts,
            "reason_counts": self.reason_counts,
        }
        path = self.date_dir / f"session_{self.session_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[DONE] session metadata: {path}")

    def _write_processed_manifest(self):
        """{video, trajectory, pose} per kept track; video is the input video itself."""
        path = (Path(self.cfg.processed_manifest) if self.cfg.processed_manifest else
                Path(self.cfg.source_dir).parent / "processed_video_paths.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.processed_records, f, indent=2)


def _config_to_jsonable(cfg: CollectorConfig) -> dict:
    d = asdict(cfg)
    d["ellipse_axes"] = list(cfg.ellipse_axes)
    return d


# ---------------------------------------------------------------------------
# Run the tracker
# ---------------------------------------------------------------------------
def run(source_dir=None, output_root=None, processed_manifest=None, device="cpu",
        on_progress=None, failures=None, fail_fast=False, videos=None) -> list[dict]:
    # ----------------------------------------------------------------------- #
    # Settings -- edit these, then just run the script                         #
    # (`python video_processor.py`). No command-line arguments are needed.     #
    # ----------------------------------------------------------------------- #
    if source_dir is None:
        source_dir = Path(__file__).resolve().parents[1] / "source"
        undistorted_source_dir = source_dir / "[TEMP]_undistorted_results"
        if undistorted_source_dir.is_dir():
            source_dir = undistorted_source_dir

    cfg = CollectorConfig(
        # --- perception / speed knobs ---
        model=str(Path(__file__).resolve().parents[1] / "yolo11n-pose.pt"),
        device=device,
        tracker="bytetrack.yaml",   # lighter than botsort.yaml for CPU processing
        imgsz=640,                   # lower (e.g. 480) trades accuracy for CPU speed
        # --- input videos ---
        source_dir=str(source_dir),
        processed_manifest=str(processed_manifest) if processed_manifest else None,
        source_video=None,           # set a path to override source_dir selection
        source_videos=[str(v) for v in videos] if videos else None,  # exact list from the pipeline; else scan source_dir
        ellipse_axes=(480, 130),     # door entrance region (half-ellipse)
        # --- output ---
        print_labels=True,            # print a line each time a person is labeled enter/pass/exit
        session_note="",             # optional free-text note saved in each person's metadata
    )

    # The dataset is written next to this script, in ../data.
    cfg.output_root = str(output_root or Path(__file__).resolve().parents[1] / "data")

    collector = LivePoseCollector(cfg)
    collector.run(on_progress=on_progress, failures=failures, fail_fast=fail_fast)
    return collector.processed_records


if __name__ == "__main__":
    run()