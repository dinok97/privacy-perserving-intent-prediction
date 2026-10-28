"""Extract the static background of videos as images.

Flow: video -> every STEP-th frame -> per-pixel median over those frames ->
<video name>_background.png in the same folder as the video. People move, so
at each pixel the median keeps what is there most of the time: the empty room.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

# Videos to extract a background from, e.g. ["20250916_095917_id7_enter.mp4"]
# (paths given on the command line are used instead when present)
VIDEOS = []

STEP = 3  # use every STEP-th frame: fewer frames to hold in memory


def median_background(source, step=STEP):
    """Person-free background of a video as the median over its frames."""
    cap = cv2.VideoCapture(str(source))
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
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def median_background_over_a_video(source, step=STEP):
    """Builds the background image of a video and saves it next to the video.

    Returns the path of the saved image.
    """
    out = Path(source).with_name(f"{Path(source).stem}_background.png")
    cv2.imwrite(str(out), median_background(source, step))
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