"""Track one original video without trimming, undistorting or changing frame indices.

Uses YOLO11n-pose/ByteTrack and reuses the repository's detection/JSON helpers.
The exported detections can be reused without rerunning inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import sys

import cv2
import torch
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent / 'video_processing'))
from video_processor import LivePoseCollector, build_pose_sidecar, build_sample_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default='yolo11n-pose.pt')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'detections.json').exists():
        parser.error('detections.json already exists; reuse it rather than overwrite inference.')
    frames_dir = output / 'yolo_frames'
    frames_dir.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = (int(capture.get(code)) for code in
                     [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT])
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or min(width, height) <= 0:
        raise ValueError('Cannot read source video geometry/timing')
    model = YOLO(args.model)
    writer = cv2.VideoWriter(str(output / 'yolo_tracking.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Cannot open annotated video writer')
    frames, seen_ids = [], set()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            index = len(frames)
            result = model.track(image, persist=True, tracker='bytetrack.yaml',
                                 conf=0.3, imgsz=640, device=args.device,
                                 classes=[0], verbose=False)[0]
            extracted = LivePoseCollector._extract_detections(None, result)
            confidences = dict(zip(result.boxes.id.int().cpu().tolist(),
                                   result.boxes.conf.cpu().tolist())) if extracted else {}
            detections = []
            for track_id, bbox, centre, keypoints in extracted:
                seen_ids.add(track_id)
                detections.append({'track_id': track_id, 'confidence': confidences[track_id],
                                   'bbox': bbox, 'center': centre, 'keypoints': keypoints})
            frames.append({'frame_index': index, 't': index / fps, 'detections': detections})
            annotated = result.plot()
            writer.write(annotated)
            cv2.imwrite(str(frames_dir / f'frame_{index:06d}.jpg'), annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f'Frame {index + 1}/{expected_frames}: track IDs '
                  f'{[d["track_id"] for d in detections]}', flush=True)
    finally:
        capture.release()
        writer.release()
    if len(frames) != expected_frames:
        raise ValueError(f'Decoded {len(frames)} frames, expected {expected_frames}')
    metadata = {
        'source_video': args.video.name, 'source_video_sha256': sha256(args.video),
        'model': Path(args.model).name, 'model_sha256': sha256(args.model),
        'tracker': 'bytetrack.yaml', 'confidence_threshold': 0.3, 'imgsz': 640,
        'device': args.device, 'fps': fps, 'frame_width': width, 'frame_height': height,
        'frame_count': len(frames), 'track_ids': sorted(seen_ids),
        'track_frame_counts': {str(tid): sum(any(d['track_id'] == tid for d in f['detections'])
                                           for f in frames) for tid in sorted(seen_ids)},
        'pixel_coordinates': 'Original decoded video; no undistortion, crop or frame resampling',
        'time_definition': 'frame_index / source FPS, not inference wall-clock time',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'ultralytics': ultralytics.__version__, 'opencv': cv2.__version__},
        'runtime_seconds': time.perf_counter() - started,
        'script_sha256': sha256(__file__),
        'reused_detection_and_json_helpers_sha256': sha256(Path(__file__).parent / 'video_processing/video_processor.py'),
    }
    # Full original timeline per track, including explicitly missing frames.
    for tid in sorted(seen_ids):
        trajectory = []
        for f in frames:
            match = next((d for d in f['detections'] if d['track_id'] == tid), None)
            trajectory.append({'frame_index': f['frame_index'], 't': f['t'],
                               'center': match['center'] if match else None,
                               'bbox': match['bbox'] if match else None,
                               'keypoints': match['keypoints'] if match else None})
        sample = build_sample_json(track_id=tid, label='unlabelled', result='unfiltered',
                                   reason='single_video_tracking_only', fps=fps, width=width,
                                   height=height, door=(width // 2, height - 1),
                                   axes=(480, 130), frames=trajectory, meta=metadata)
        name = f'{args.video.stem}_track{tid}'
        (output / f'{name}.json').write_text(json.dumps(sample, indent=2) + '\n')
        pose = build_pose_sidecar(frames=trajectory, source_name=args.video.name,
                                  label='unlabelled', fps=fps, width=width, height=height)
        (output / f'{name}_pose.json').write_text(json.dumps(pose, indent=2) + '\n')
    result = {'metadata': metadata, 'frames': frames}
    (output / 'detections.json').write_text(json.dumps(result, indent=2) + '\n')
    validation_capture = cv2.VideoCapture(str(output / 'yolo_tracking.mp4'))
    output_count = int(validation_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    validation_capture.release()
    assert output_count == len(frames)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
