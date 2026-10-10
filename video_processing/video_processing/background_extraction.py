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
VIDEOS = ["../source/motion_20260605_131200_4s.mp4"]

STEP = 3  # use every STEP-th frame: fewer frames to hold in memory


def median_background_over_a_video(source, step=STEP):
    """Builds a person-free background image as the median over frames.

    The image is saved in the same folder as the video; its path is returned.
    """
    cap = cv2.VideoCapture(source)
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

    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    out = Path(source).with_name(f"{Path(source).stem}_background.png")
    cv2.imwrite(str(out), background)
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