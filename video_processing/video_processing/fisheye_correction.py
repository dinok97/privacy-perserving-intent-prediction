"""Detect fisheye / wide-angle distortion in a video and undistort the video.

Flow: <video> -> median background (background_extraction.py, kept in
memory) -> long edge chains -> the focal length that makes those chains
straightest (straight lines in a room must be straight in a distortion-free
image) -> if that bends the image noticeably: every frame is undistorted
into <video>_undistorted.mp4. <video>_lens.json records the decision and the
lens: K and D for OpenCV's cv2.fisheye functions, and new_K, the camera
matrix of the undistorted video (needed for depth / 3D mapping).
Everything is saved next to the video.

The lens is detected on the median background rather than on single frames:
people are removed there, so the only edges left belong to the room. The
camera does not move during a video, so one lens estimate fits every frame.

No checkerboard is needed, but the scene has to contain straight edges
(walls, door frames, floor lines), which a room does.
"""

from pathlib import Path
import json
import sys

import cv2
import numpy as np

from background_extraction import median_background

# Videos to check, e.g. ["20250916_095917_id7_enter.mp4"]
# (paths given on the command line are used instead when present)
VIDEOS = []

MIN_DISTORTION = 0.03  # corner pixels must sit >= 3 % too close to the centre
MIN_GAIN = 0.15        # share of the edge length that must get straighter
MIN_STRAIGHT = 0.3     # share of the edge length that must end up straight
MIN_CHAINS = 8         # fewer usable edge chains than this -> no decision
BALANCE = 0.0          # 0 = crop to valid pixels ... 1 = keep every source pixel
CODEC = "mp4v"         # fourcc of the undistorted video

# Edge detection, in pixels of an image resized to WORK_SIZE
WORK_SIZE = 1280       # longer image side
BLUR = 1.5             # Gaussian sigma applied before edge detection
EDGE_PERCENTILE = 90   # gradient strength above which a pixel starts an edge
EDGE_LOW = 0.4         # ...and the fraction of it at which an edge continues
MIN_LENGTH = 0.08      # shortest edge chain used, as a fraction of the diagonal
CORNER_STEP = 7        # pixels before/after a point used to measure its turn
CORNER_ANGLE = 30.0    # a chain is cut where it turns more than this (degrees)
SMOOTH_TOL = 1.0       # max RMS wobble around a smooth curve
STRAIGHT = 0.75        # a chain bending less than this (RMS) is straight
CHANGE = 0.3           # smallest change in bend (RMS) that counts
MAX_THETA = 1.45       # largest half field of view searched (radians, ~83 deg)


def edge_chains(gray):
    """Returns the long, smooth edge chains of an image as (N, 2) pixel arrays.

    Chains are cut at corners, so each one follows a single edge: a straight
    line in the room, drawn straight or bent depending on the lens.
    """
    gray = cv2.GaussianBlur(gray, (0, 0), BLUR)
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    high = float(np.percentile(np.hypot(dx, dy), EDGE_PERCENTILE))
    edges = cv2.Canny(dx, dy, EDGE_LOW * high, high, L2gradient=True)

    min_len = MIN_LENGTH * np.hypot(*gray.shape)
    pieces = []
    for contour in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
        points = contour[:, 0, :].astype(np.float64)
        if len(points) >= min_len:
            pieces += split_at_corners(points, min_len)

    # A contour walks along a thin edge and back again, so every piece shows
    # up twice: keep the longest and drop pieces that reuse its pixels
    chains, used = [], np.zeros(gray.shape, bool)
    for points in sorted(pieces, key=len, reverse=True):
        x, y = points.astype(int).T
        if used[y, x].mean() < 0.5 and is_smooth(points):
            chains.append(points)
        used[y, x] = True
    return chains


def split_at_corners(points, min_len):
    """Cuts a closed pixel contour where it turns sharply."""
    before = points - np.roll(points, CORNER_STEP, axis=0)
    after = np.roll(points, -CORNER_STEP, axis=0) - points
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    turn = np.degrees(np.abs(np.arctan2(cross, (before * after).sum(axis=1))))

    corner = turn > CORNER_ANGLE
    if not corner.any():
        return [points]
    start = int(np.argmax(corner))  # begin at a corner so no piece wraps around
    points, corner = np.roll(points, -start, axis=0), np.roll(corner, -start)

    cuts = np.flatnonzero(np.diff(corner.astype(int)))
    pieces = np.split(points, cuts + 1)[1::2]  # odd pieces lie between corners
    return [p for p in pieces if len(p) >= min_len]


def is_smooth(points):
    """True for a line or gentle arc, False for a wiggly outline."""
    centred = points - points.mean(axis=0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    along, across = centred @ axes[0], centred @ axes[1]
    fit = np.polyval(np.polyfit(along, across, 3), along)
    return np.sqrt(np.mean((across - fit) ** 2)) < SMOOTH_TOL


def camera_matrix(focal, size):
    """K for a lens whose optical axis hits the middle of the image."""
    w, h = size
    return np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1.0]])


def bends(chains, focal, size):
    """Bend of every chain in pixels: its RMS distance from a straight line.

    focal=None tests the chains as they are. Otherwise the line is fitted
    after undistorting with that focal length and then drawn back into the
    image, so the bend is still measured in image pixels.
    """
    K, D = camera_matrix(focal or 1.0, size), np.zeros(4)
    out = []
    for chain in chains:
        flat = chain
        if focal is not None:
            flat = cv2.fisheye.undistortPoints(chain.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        centre = flat.mean(axis=0)
        direction = np.linalg.eigh(np.cov(flat.T))[1][:, 1]  # main axis
        line = centre + np.outer((flat - centre) @ direction, direction)
        if focal is not None:
            line = cv2.fisheye.distortPoints(line.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        out.append(np.sqrt(np.mean(np.sum((chain - line) ** 2, axis=1))))
    return np.array(out)


def estimate_focal(chains, size):
    """Finds the fisheye focal length (pixels) that straightens the chains.

    Model: OpenCV fisheye with D = 0, where a ray at angle theta from the
    optical axis lands focal * theta pixels from the image centre. A large
    focal length means almost no distortion, a small one a strong fisheye.
    Returns None when the chains are straightest without any correction.
    """
    w, h = size
    lengths = np.array([len(c) for c in chains], dtype=np.float64)
    reach = np.linalg.norm(
        np.concatenate(chains) - [(w - 1) / 2, (h - 1) / 2], axis=1).max()

    def loss(inverse_focal):
        b = bends(chains, 1 / inverse_focal if inverse_focal else None, size)
        # Robust: really curved objects (chairs, lamps) cannot outvote lines
        return np.sum(lengths * b**2 / (b**2 + STRAIGHT**2))

    low, high = 0.0, MAX_THETA / reach
    for _ in range(3):  # coarse search, then zoom in twice
        grid = np.linspace(low, high, 41)
        best = grid[int(np.argmin([loss(g) for g in grid]))]
        step = grid[1] - grid[0]
        low, high = max(best - step, 0.0), min(best + step, high)
    return 1 / best if best else None


def detect_distortion(image):
    """Decides whether an image is fisheye-distorted and estimates the lens.

    Returns a dict that also serves as the content of the lens JSON.
    """
    h, w = image.shape[:2]
    # Work on a standard size so the pixel thresholds mean the same everywhere
    # (small images are enlarged, which also gives finer edge positions)
    scale = WORK_SIZE / max(h, w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    size = (gray.shape[1], gray.shape[0])
    chains = edge_chains(gray)
    lens = {
        "image_size": [w, h],
        "distorted": False,
        "model": "opencv_fisheye",
        "n_chains": len(chains),
    }
    if len(chains) < MIN_CHAINS:
        lens["reason"] = "too few long edges to decide"
        return lens

    focal = estimate_focal(chains, size)
    if focal is None:
        lens["reason"] = "edges are straightest without any correction"
        return lens

    # Shares of the total edge length, compared before and after correcting
    weights = np.array([len(c) for c in chains], dtype=np.float64)
    weights /= weights.sum()
    before, after = bends(chains, None, size), bends(chains, focal, size)
    straighter = float(weights[before - after > CHANGE].sum())
    more_bent = float(weights[after - before > CHANGE].sum())
    straight = float(weights[after < STRAIGHT].sum())

    # How much closer to the centre the image corner sits than it should
    theta = min(np.hypot(*size) / 2 / focal, np.pi / 2)
    distortion = float(1 - theta / np.tan(theta))

    lens["K"] = camera_matrix(focal / scale, (w, h)).round(3).tolist()
    lens["D"] = [0.0, 0.0, 0.0, 0.0]
    lens["corner_distortion"] = round(distortion, 3)
    lens["edges_straighter"] = round(straighter, 3)
    lens["edges_more_bent"] = round(more_bent, 3)
    lens["edges_straight_after"] = round(straight, 3)

    if distortion < MIN_DISTORTION:
        lens["reason"] = "distortion too small to matter"
    elif straighter - more_bent < MIN_GAIN:
        lens["reason"] = "correcting does not straighten enough edges"
    elif straight < MIN_STRAIGHT:
        lens["reason"] = "edges stay curved whatever the lens (not a scene of straight lines)"
    else:
        lens["distorted"] = True
        lens["reason"] = "edges become straight under a fisheye lens"
    return lens


def undistort_maps(lens, balance=BALANCE):
    """Pixel maps that undistort a frame; computed once, used for every frame.

    Adds new_K to the lens: the camera matrix of the undistorted frames.
    """
    size = tuple(lens["image_size"])
    K, D = np.array(lens["K"]), np.array(lens["D"])
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, size, np.eye(3), balance=balance)
    lens["new_K"] = new_K.round(3).tolist()
    lens["balance"] = balance
    return cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, size, cv2.CV_16SC2)


def undistort_image(image, lens, balance=BALANCE):
    """Undistorts a single image with the lens found by detect_distortion."""
    map1, map2 = undistort_maps(lens, balance)
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)


def undistort_video(video_path, out_path, lens, balance=BALANCE):
    """Writes an undistorted copy of a video, frame by frame.

    Same size and frame rate as the input. Returns the number of frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"{video_path} reports no frame rate")

    map1, map2 = undistort_maps(lens, balance)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*CODEC),
                             fps, tuple(lens["image_size"]))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open {out_path} for writing")

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR))
        n += 1
    cap.release()
    writer.release()
    return n


def correct_video(video_path, balance=BALANCE):
    """Checks a video for fisheye distortion and undistorts it when needed.

    Returns the path of the video to use from here on: <video>_undistorted.mp4,
    or the original when no distortion was found. <video>_lens.json is
    written in both cases, so the decision is recorded next to the video.
    """
    video_path = Path(video_path)
    lens = detect_distortion(median_background(video_path))
    lens["video"] = video_path.name
    result = video_path

    if lens["distorted"]:
        result = video_path.with_name(f"{video_path.stem}_undistorted.mp4")
        lens["n_frames"] = undistort_video(video_path, result, lens, balance)
        lens["undistorted_video"] = result.name

    with open(video_path.with_name(f"{video_path.stem}_lens.json"), "w") as f:
        json.dump(lens, f, indent=2)
    return str(result)


def main():
    videos = sys.argv[1:] or VIDEOS
    if not videos:
        raise SystemExit("Nothing to do: fill in VIDEOS or pass video paths")

    for video in videos:
        result = correct_video(video)
        with open(Path(video).with_name(f"{Path(video).stem}_lens.json")) as f:
            lens = json.load(f)
        verdict = "DISTORTED" if lens["distorted"] else "not distorted"
        print(f"{video}: {verdict} ({lens['reason']}, {lens['n_chains']} edge chains)")
        if lens["distorted"]:
            print(f"  focal length {lens['K'][0][0]:.0f} px, corner distortion "
                  f"{100 * lens['corner_distortion']:.0f} %, saved {result}")
        else:
            print(f"  use the original video: {result}")


if __name__ == "__main__":
    main()