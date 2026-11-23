"""Render tracked positions over a plain canvas or a reviewed empty-room depth map."""
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from processing.feature_extractor import K, PARAMS


SIZE = (640, 480)
PLOT_SIZE = np.array([616, 376])
ORIGIN = np.array([12, 44])
BOX_SIZE = np.array([42, 66])
BACKGROUND = (244, 247, 249)
NEUTRAL = (120, 120, 120)
DEPTH_COLOURS = ((108, 51, 30), (154, 96, 29), (143, 137, 29),
                 (93, 175, 70), (30, 194, 217))
TORSO = [K[name] for name in ('left_shoulder', 'right_shoulder', 'right_hip', 'left_hip')]


def subject_depth(detection, depth, source_size):
    """Get one depth value from the inner torso using confident pose points."""
    keypoints = np.asarray(detection.get('keypoints'), dtype=float)
    if keypoints.shape != (len(K), 3):
        return None
    torso = keypoints[TORSO]
    if not np.isfinite(torso).all() or np.any(torso[:, 2] < PARAMS['kp_conf_min']):
        return None
    bbox = np.asarray(detection['bbox'], dtype=float)
    points = torso[:, :2]
    if (np.any(points < bbox[:2]) or np.any(points > bbox[2:])
            or np.any(points < 0) or np.any(points >= source_size)):
        return None
    centre = points.mean(axis=0)
    inner = centre + 0.6 * (points - centre)
    scale = np.array(depth.shape[::-1]) / source_size
    polygon = np.rint((inner + 0.5) * scale - 0.5).astype(np.int32)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 12:
        return None
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    torso_depth = depth[mask > 0]
    valid = torso_depth[np.isfinite(torso_depth) & (torso_depth > 0)]
    if valid.size < max(12, torso_depth.size / 2):
        return None
    return float(np.median(valid))


def preview_canvas(heatmap=None, palette=DEPTH_COLOURS, origin=ORIGIN):
    canvas = np.full((SIZE[1], SIZE[0], 3), (255, 255, 255), dtype=np.uint8)
    x, y = ORIGIN
    w, h = PLOT_SIZE
    canvas[y:y + h, x:x + w] = BACKGROUND
    if heatmap is not None:
        x, y = origin
        h, w = heatmap.shape[:2]
        canvas[y:y + h, x:x + w] = heatmap
    text = (54, 43, 30)
    title = ('Room depth and tracked position' if heatmap is not None
             else 'Tracked position and relative depth')
    cv2.putText(canvas, title, (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text, 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Near', (12, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text, 1, cv2.LINE_AA)
    for i, colour in enumerate(palette):
        cv2.rectangle(canvas, (58 + i * 24, 438), (80 + i * 24, 456), colour, -1)
    cv2.putText(canvas, 'Far', (188, 454), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (286, 438), (308, 456), NEUTRAL, -1)
    cv2.putText(canvas, 'Depth unavailable', (318, 454),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text, 1, cv2.LINE_AA)
    return canvas


def read_depth(predictions, index, shape):
    with np.load(predictions / f'frame_{index:06d}.npz', allow_pickle=False) as archive:
        depth = archive['depth']
    if depth.shape != (*shape, 1):
        raise ValueError('Shareable depth preview depth shape does not match the aligned grid')
    return depth[..., 0]


def depth_style(room, subject_depths, plot_size):
    """Map the room and subject to one clip-wide scale."""
    if room is None:
        low, high = (min(subject_depths), max(subject_depths)) if subject_depths else (0.0, 0.0)
        return None, DEPTH_COLOURS, low, high

    valid = np.isfinite(room) & (room > 0)
    if not valid.any():
        raise ValueError('The reviewed background frame has no valid depth')
    low, high = np.percentile(room[valid], [1, 99])
    if subject_depths:
        low = min(low, min(subject_depths))
        high = max(high, max(subject_depths))
    levels = np.full(room.shape, 128, dtype=np.uint8)
    if high > low:
        levels[valid] = np.rint(255 * np.clip((room[valid] - low) / (high - low), 0, 1))
    heatmap = cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)
    heatmap[~valid] = BACKGROUND
    ramp = np.rint(np.linspace(0, 255, len(DEPTH_COLOURS))).astype(np.uint8)
    colours = tuple(tuple(map(int, row[0])) for row in cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO))
    return cv2.resize(heatmap, tuple(plot_size)), colours, low, high


def render_preview(predictions, tracking, alignment, fps, destination, *, background_frame=None):
    source_size = np.array([tracking['metadata']['frame_width'], tracking['metadata']['frame_height']])
    if (not np.array_equal(source_size, alignment['source_image_size'])
            or alignment['crop_offset'] != [0, 0] or alignment['padding'] != [0, 0]):
        raise ValueError('Shareable depth preview requires aligned, uncropped predictions')
    depth_shape = tuple(reversed(alignment['image_size']))
    frames = tracking['frames']
    if [frame['frame_index'] for frame in frames] != list(range(tracking['metadata']['frame_count'])):
        raise ValueError('Shareable depth preview requires tracking for the complete clip timeline')

    room = None
    plot_size, origin = PLOT_SIZE, ORIGIN
    if background_frame is not None:
        if type(background_frame) is not int or not 0 <= background_frame < len(frames):
            raise ValueError('Shareable depth background frame must be a valid zero-based frame index')
        if frames[background_frame]['detections']:
            raise ValueError('The selected shareable depth background frame contains a tracked person')
        room = read_depth(predictions, background_frame, depth_shape)
        plot_size = np.rint(source_size * min(PLOT_SIZE / source_size)).astype(int)
        if np.any(plot_size < BOX_SIZE):
            raise ValueError('Source aspect ratio leaves insufficient room for a fixed-size preview marker')
        origin = ORIGIN + (PLOT_SIZE - plot_size) // 2

    timeline = []
    depths = []
    for frame in frames:
        detections = frame['detections']
        markers = []
        if detections:
            field = read_depth(predictions, frame['frame_index'], depth_shape)
            for detection in detections:
                bbox = np.asarray(detection.get('bbox'), dtype=float)
                if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
                    continue
                centre = (bbox[:2] + bbox[2:]) / 2
                top_left = np.rint(centre * plot_size / source_size - BOX_SIZE / 2).astype(int)
                top_left = origin + np.clip(top_left, 0, plot_size - BOX_SIZE)
                depth = subject_depth(detection, field, source_size)
                markers.append((top_left, depth))
                if depth is not None:
                    depths.append(depth)
        timeline.append(markers)

    # A single scale applies throughout the clip. Only one of five colours is published.
    heatmap, colours, low, high = depth_style(room, depths, plot_size)
    canvas = preview_canvas(heatmap, colours, origin)
    destination = Path(destination)
    with TemporaryDirectory(prefix='.preview-', dir=destination.parent) as directory:
        staged = Path(directory) / destination.name
        writer = cv2.VideoWriter(str(staged), cv2.VideoWriter_fourcc(*'mp4v'), fps, SIZE)
        try:
            if not writer.isOpened():
                raise RuntimeError('Could not open the shareable MP4 writer')
            for markers in timeline:
                image = canvas.copy()
                for top_left, depth in markers:
                    if depth is None:
                        colour = NEUTRAL
                    else:
                        fraction = (depth - low) / (high - low) if high > low else 0.5
                        colour = colours[round(fraction * (len(colours) - 1))]
                    start, end = tuple(top_left), tuple(top_left + BOX_SIZE - 1)
                    cv2.rectangle(image, start, end, colour, -1)
                    if room is not None:
                        cv2.rectangle(image, start, end, (36, 36, 36), 2)
                writer.write(image)
        finally:
            writer.release()
        video = cv2.VideoCapture(str(staged))
        try:
            if not video.isOpened() or round(video.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frames):
                raise RuntimeError('Shareable preview did not encode the complete clip')
        finally:
            video.release()
        staged.replace(destination)
    return {'path': str(destination), 'frames': len(frames), 'background_frame': background_frame,
            'frames_with_marker': sum(bool(markers) for markers in timeline),
            'frames_with_depth': sum(any(depth is not None for _, depth in markers) for markers in timeline)}
