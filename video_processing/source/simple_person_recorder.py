#!/usr/bin/env python3
"""
simple_person_recorder.py

Minimal person-triggered video recorder, adapted from SeamAI's
pose_data_collection.py for a plain PC (no Jetson, no .engine model).

What it does
------------
Watches a USB camera. Whenever YOLO detects and tracks a person, it records
an .mp4 clip covering the time they're in frame (plus a short "tail" after
they leave, so the clip doesn't cut off abruptly). No pose/skeleton data,
no JSON metadata, no enter/pass/exit labeling -- just video clips.

Clips are saved directly next to this script, as:
    video_<YYYY-MM-DD>_<HHMMSS>_id<track_id>.mp4

How to run
----------
1. Edit the settings in main() at the bottom if needed (camera index,
   model, output folder).
2. python simple_person_recorder.py
   A window opens showing the live feed with boxes around detected people.
   Press 'q' in that window to quit, or Ctrl+C in the terminal if headless.

Notes
-----
- model="yolo11n.pt" is a standard (non-Jetson) Ultralytics detection model.
  It auto-downloads the first time you run this, no manual setup needed.
- If you don't want a preview window (e.g. running unattended), set
  headless=True in main().
- Clips are recorded at a fixed target_fps (config, default 15). Frames are
  duplicated whenever the detection loop falls behind real time, so clip
  duration always tracks how long the person was actually in frame -- this
  avoids sped-up/slowed-down playback that would otherwise depend on how
  fast your particular PC happens to run the model each session. Raise
  target_fps for smoother-looking video (bigger files), lower it for
  smaller files (choppier motion).
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RecorderConfig:
    # Perception
    model: str = "yolo11n.pt"          # standard detection model (not pose, not Jetson .engine)
    tracker: str = "bytetrack.yaml"
    conf: float = 0.3
    imgsz: int = 640

    # Camera
    camera_index: int = 0
    cap_width: int = 640
    cap_height: int = 480
    mjpg: bool = False                 # request MJPG from the camera (often unlocks higher fps)

    # Recording
    output_root: str = "data"
    target_fps: float = 15.0            # fixed encoding fps for every clip; frames are duplicated when the
                                         # processing loop falls behind so playback always matches real time,
                                         # regardless of how fast the camera+model actually run that session

    # Track lifecycle
    grace_frames: int = 15              # frames a person may briefly vanish (occlusion) before ending the clip
    tail_seconds: float = 1.0           # extra seconds recorded after the person leaves frame
    min_duration_s: float = 1.0         # clips shorter than this (by *detected* time, not counting the tail) are discarded

    # UI
    headless: bool = False
    fullscreen: bool = False


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class SimplePersonRecorder:
    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg

        import cv2  # noqa
        from ultralytics import YOLO  # noqa

        self.cv2 = cv2

        self.session_start = datetime.datetime.now()
        self.date_dir = Path(cfg.output_root)  # save directly here, no per-date subfolder
        self.date_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INIT] Loading model: {cfg.model}")
        self.model = YOLO(cfg.model)

        self.cap = None
        self.width = self.height = 0
        self.record_fps = float(cfg.target_fps)
        self.grace_frames = cfg.grace_frames
        self.tail_frames = 0
        self.min_frames = 2

        self.active: dict[int, dict] = {}  # track_id -> recording state
        self.saved_count = 0
        self.discarded_count = 0

        self._fps = 0
        self._fps_frames = 0
        self._fps_t0 = time.time()

        print(f"[INIT] output: {self.date_dir}")

    # -- camera setup ---------------------------------------------------

    def _open_camera(self):
        cv2 = self.cv2
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.cfg.camera_index}")
        if self.cfg.mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cap_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _init_geometry(self, width: int, height: int, reported_fps: float):
        self.width, self.height = int(width), int(height)
        self.record_fps = float(self.cfg.target_fps)

        self.min_frames = max(2, int(round(self.cfg.min_duration_s * self.record_fps)))
        self.tail_frames = max(0, int(round(self.cfg.tail_seconds * self.record_fps)))

        print(f"[GEOM] frame={self.width}x{self.height} target_fps={self.record_fps:.1f} "
              f"(camera reports {reported_fps:.1f}) min_frames={self.min_frames} tail_frames={self.tail_frames}")

    # -- per-track lifecycle ---------------------------------------------

    def _start_track(self, track_id: int):
        cv2 = self.cv2
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        temp_path = self.date_dir / f"_temp_id{track_id}_{ts}.mp4"
        writer = cv2.VideoWriter(
            str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"),
            self.record_fps, (self.width, self.height))
        self.active[track_id] = {
            "writer": writer,
            "temp_path": temp_path,
            "n_frames": 0,      # total frames written, including tail padding and pacing duplicates
            "n_detected": 0,    # frames where the person was actually detected (excludes tail)
            "vanish": 0,
            "start_ts": ts,
            "t0": time.perf_counter(),  # real wall-clock reference used to pace writes to real time
        }
        print(f"[START] tracking id={track_id}")

    def _sync_frame(self, track_id: int, frame, detected: bool):
        """Write frame(s) so the clip's timeline tracks real elapsed time.

        The processing loop's actual frame rate varies run to run (model
        warm-up, CPU load, etc.), so instead of trusting any single measured
        fps, we write frames against a fixed target_fps and duplicate the
        current frame whenever real elapsed time has moved further ahead
        than what's been written so far. This keeps clip duration matching
        real time regardless of how fast/slow processing happens to be.
        """
        rec = self.active[track_id]
        elapsed = time.perf_counter() - rec["t0"]
        expected = int(elapsed * self.record_fps) + 1
        while rec["n_frames"] < expected:
            rec["writer"].write(frame)
            rec["n_frames"] += 1
        if detected:
            rec["n_detected"] += 1

    def _finalize_track(self, track_id: int):
        rec = self.active.pop(track_id)
        rec["writer"].release()

        # Judge length by frames the person was actually detected in, not the
        # total including the tail padding recorded after they left frame --
        # otherwise a near-instant false detection can pad its way past the
        # minimum just by sitting in the grace/tail window.
        too_short = rec["n_detected"] < self.min_frames
        corrupt = (not rec["temp_path"].exists()) or rec["temp_path"].stat().st_size < 1024

        if too_short or corrupt:
            rec["temp_path"].unlink(missing_ok=True)
            self.discarded_count += 1
            reason = "too_short" if too_short else "corrupt"
            print(f"[DISCARD:{reason}] id={track_id} "
                  f"({rec['n_detected']} detected / {rec['n_frames']} total frames)")
            return

        final_name = f"video_{rec['start_ts']}_id{track_id}.mp4"
        final_path = self.date_dir / final_name
        rec["temp_path"].rename(final_path)
        self.saved_count += 1
        print(f"[SAVED] {final_path} ({rec['n_frames']} frames, "
              f"total saved so far: {self.saved_count})")

    # -- perception --------------------------------------------------------

    def _process_frame(self, frame):
        result = self.model.track(
            frame, persist=True, classes=[0],  # class 0 = person
            tracker=self.cfg.tracker, conf=self.cfg.conf,
            imgsz=self.cfg.imgsz, verbose=False)[0]

        boxes = result.boxes
        seen = set()
        dets = []
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            for i, tid in enumerate(ids):
                seen.add(tid)
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                dets.append((tid, x1, y1, x2, y2))

                if tid not in self.active:
                    self._start_track(tid)
                else:
                    self.active[tid]["vanish"] = 0

                if tid in self.active:
                    self._sync_frame(tid, frame, detected=True)

        # Tracks not seen this frame: keep recording through the grace/tail
        # window (so a clip doesn't cut off the instant someone is briefly
        # occluded), then finalize.
        for tid in list(self.active.keys()):
            if tid in seen:
                continue
            rec = self.active[tid]
            rec["vanish"] += 1
            self._sync_frame(tid, frame, detected=False)
            if rec["vanish"] >= self.grace_frames + self.tail_frames:
                self._finalize_track(tid)

        return dets

    def _draw_overlay(self, frame, dets):
        cv2 = self.cv2
        for tid, x1, y1, x2, y2 in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"FPS: {self._fps}  saved: {self.saved_count}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -- main loop -----------------------------------------------------

    def run(self):
        cv2 = self.cv2
        self.cap = self._open_camera()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

        self._init_geometry(w, h, reported)

        display = not self.cfg.headless
        if display:
            print("[RUN] Recording. Press 'q' in the window to stop.")
            cv2.namedWindow("Person Recorder", cv2.WINDOW_NORMAL)
            if self.cfg.fullscreen:
                cv2.setWindowProperty("Person Recorder",
                                       cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            print("[RUN] Recording (headless). Press Ctrl+C to stop.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[RUN] Camera read failed; stopping.")
                    break

                dets = self._process_frame(frame)

                self._fps_frames += 1
                now = time.time()
                if now - self._fps_t0 >= 1.0:
                    self._fps = self._fps_frames
                    self._fps_frames = 0
                    self._fps_t0 = now

                if display:
                    disp = frame.copy()
                    self._draw_overlay(disp, dets)
                    cv2.imshow("Person Recorder", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
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
        print(f"[DONE] saved={self.saved_count} discarded={self.discarded_count}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> int:
    cfg = RecorderConfig(
        model="yolo11n.pt",       # auto-downloads on first run
        camera_index=0,           # change if the wrong camera opens
        headless=False,           # set True to run with no preview window
        fullscreen=False,
    )
    cfg.output_root = str(Path(__file__).resolve().parent)  # save directly into this script's own folder

    SimplePersonRecorder(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
