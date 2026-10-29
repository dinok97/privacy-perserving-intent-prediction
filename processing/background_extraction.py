"""Extract the static background of videos as images.

Flow: video -> every STEP-th frame -> people found with YOLO and masked out ->
per-pixel median over the frames where that pixel is free of people ->
check the result for people once more -> <video name>_background.png in the
same folder as the video.

Why the masking: a plain median keeps what a pixel shows most of the time, so a
person who stands still for more than half the clip stays in the image as a
ghost. With people masked first, each pixel's median only uses frames in which
nobody covers it.

Why the check: a person left in the background is a stored picture of a person,
which the privacy-by-design pipeline must not keep. Any person still found in
the finished image is pixelated (ON_PERSON_FOUND) and reported.
"""

from pathlib import Path
import sys
import warnings

import cv2
import numpy as np

# Videos to extract a background from, e.g. ["20250916_095917_id7_enter.mp4"]
# (paths given on the command line are used instead when present)
VIDEOS = []

STEP = 3          # use every STEP-th frame: fewer frames to hold in memory
MAX_FRAMES = 60   # long videos are sampled more sparsely so memory stays bounded

# Person detection: the same model file video_processor.py uses (it finds people
# as well as their poses), so nothing new has to be downloaded
MODEL = Path(__file__).resolve().parents[1] / "yolo11n-pose.pt"
DEVICE = "cpu"
MASK_CONF = 0.25   # low on purpose: masking a false detection only costs a few samples
CHECK_CONF = 0.40  # a person this confident in the finished background is reported
PAD = 0.15         # each box grows by this fraction per side (loose limbs, shadows) ...
PAD_PX = 8         # ... plus this many pixels

# What to do with a person still found in the finished background:
# "pixelate" (unrecognisable, the default) or "keep" (only report it)
ON_PERSON_FOUND = "pixelate"

_model = None


def _load_model():
    """Loads YOLO once, so a pipeline run over many videos pays for it once."""
    global _model
    if _model is None:
        from ultralytics import YOLO  # heavy import, only when needed
        _model = YOLO(str(MODEL) if MODEL.is_file() else MODEL.name)
    return _model


def detect_people(frame, conf):
    """Boxes [x1, y1, x2, y2] of the people YOLO finds in one frame."""
    result = _load_model().predict(frame, classes=[0], conf=conf,
                                   device=DEVICE, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    return [[float(v) for v in box] for box in result.boxes.xyxy.cpu().numpy()]


def padded(box, w, h):
    """A detection box grown by PAD and PAD_PX, clipped to the image."""
    x1, y1, x2, y2 = box
    px = PAD * (x2 - x1) + PAD_PX
    py = PAD * (y2 - y1) + PAD_PX
    return (max(0, int(x1 - px)), max(0, int(y1 - py)),
            min(w, int(np.ceil(x2 + px))), min(h, int(np.ceil(y2 + py))))


def read_frames(source, step=STEP):
    """Every step-th frame of a video (sparser for long videos, see MAX_FRAMES)."""
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(step, total // MAX_FRAMES)
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {source}")
    return frames


def masked_median(frames, boxes_per_frame):
    """Per-pixel median over the frames where no person covers that pixel.

    Returns (image, holes). holes marks pixels that a person covered in every
    frame: no clean sample exists there, so they are filled in from their
    surroundings (inpainted) rather than showing the person.
    """
    h, w = frames[0].shape[:2]
    stack = np.stack(frames).astype(np.float32)
    for k, boxes in enumerate(boxes_per_frame):
        for box in boxes:
            x1, y1, x2, y2 = padded(box, w, h)
            stack[k, y1:y2, x1:x2] = np.nan

    holes = np.isnan(stack[..., 0]).all(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the all-masked pixels are the holes
        image = np.nanmedian(stack, axis=0)
    image = np.round(np.nan_to_num(image)).astype(np.uint8)

    if holes.any():
        image = cv2.inpaint(image, holes.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    return image, holes


def pixelate(image, box):
    """Makes the area of a box unrecognisable (16 px blocks)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = padded(box, w, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    region = image[y1:y2, x1:x2]
    small = cv2.resize(region, (max(1, (x2 - x1) // 16), max(1, (y2 - y1) // 16)),
                       interpolation=cv2.INTER_AREA)
    image[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def check_background(image):
    """Looks for people left in a finished background; pixelates them if set to.

    Returns the boxes found (empty list = the background is clear).
    """
    found = detect_people(image, CHECK_CONF)
    if found and ON_PERSON_FOUND == "pixelate":
        for box in found:
            pixelate(image, box)
    return found


def masked_background(source, step=STEP):
    """Person-free background of a video: median over frames with people masked.

    Returns (image, report). The report says how many frames were used, in how
    many a person was masked, and how many pixels had no clean sample.
    """
    frames = read_frames(source, step)
    boxes = [detect_people(f, MASK_CONF) for f in frames]
    image, holes = masked_median(frames, boxes)
    report = {
        "frames": len(frames),
        "frames_with_people": sum(1 for b in boxes if b),
        "hole_pixels": int(holes.sum()),
    }
    return image, report


def median_background(source, step=STEP):
    """Person-free background of a video (same call and return as before)."""
    return masked_background(source, step)[0]


def median_background_over_a_video(source, step=STEP):
    """Builds the background image of a video and saves it next to the video.

    Returns the path of the saved image.
    """
    image, report = masked_background(source, step)
    found = check_background(image)

    out = Path(source).with_name(f"{Path(source).stem}_background.png")
    cv2.imwrite(str(out), image)

    print(f"[BACKGROUND] {Path(source).name}: {report['frames']} frames, "
          f"people masked in {report['frames_with_people']}")
    if report["hole_pixels"]:
        print(f"[BACKGROUND]   {report['hole_pixels']} pixels were covered by a person in "
              "every frame and were filled in from their surroundings")
    if found:
        action = "pixelated" if ON_PERSON_FOUND == "pixelate" else "kept (ON_PERSON_FOUND='keep')"
        print(f"[BACKGROUND]   WARNING: {len(found)} person(s) still visible in the "
              f"background, {action}: {out.name}")
    return str(out)


def main():
    videos = sys.argv[1:] or VIDEOS
    if not videos:
        raise SystemExit("Nothing to do: fill in VIDEOS or pass video paths")

    for video in videos:
        background = median_background_over_a_video(video)
        print(f"Background saved to: {background}")


if __name__ == "__main__":
    main()