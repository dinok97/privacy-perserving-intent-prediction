"""Move duplicate, very short, and low-FPS videos out of the pipeline input folder."""
import argparse
import hashlib
import logging
import math
from pathlib import Path
import shutil

import cv2

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
MIN_DURATION_SECONDS = 0.5
MIN_FRAME_RATE = 5.0

logger = logging.getLogger(__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _video_info(path):
    """Return duration and FPS from video metadata, or raise if unreadable."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if (not math.isfinite(frame_count) or not math.isfinite(fps)
                or frame_count <= 0 or fps <= 0):
            raise RuntimeError(f'Video has invalid frame count or FPS: {path}')
        return frame_count / fps, fps
    finally:
        capture.release()


def video_duration_seconds(path):
    """Return duration from the video frame count and FPS, or raise if unreadable."""
    duration, _ = _video_info(path)
    return duration


def _move_to_category(video, category_dir):
    category_dir.mkdir(parents=True, exist_ok=True)
    destination = category_dir / video.name
    if destination.exists():
        raise FileExistsError(f'Cannot move {video}: destination already exists: {destination}')
    return Path(shutil.move(str(video), str(destination)))


def clean_videos(source_dir, *, min_duration=MIN_DURATION_SECONDS):
    """Move duplicate, short, and low-FPS videos from the source root.

    Nested directories are not scanned. For identical files, the first filename
    in sorted order is kept in the source directory. Videos below 5 FPS are
    moved to ``deleted/low_fps``.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f'Source directory does not exist: {source_dir}')
    if not math.isfinite(min_duration) or min_duration < 0:
        raise ValueError(f'Minimum duration must be a non-negative number: {min_duration}')

    deleted_dir = source_dir / 'deleted'
    seen_hashes = set()
    duplicates = []
    too_short = []
    low_fps = []

    videos = sorted(
        (path for path in source_dir.iterdir()
         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name,
    )
    for video in videos:
        digest = file_sha256(video)
        
        # Check for duplicates by comparing SHA-256 hashes
        if digest in seen_hashes:
            moved = _move_to_category(video, deleted_dir / 'duplicated')
            duplicates.append(moved)
            print(f'Moved duplicate video: {moved}')
            continue
        seen_hashes.add(digest)

        duration, fps = _video_info(video)
        # Check for low FPS and short duration
        if fps < MIN_FRAME_RATE:
            moved = _move_to_category(video, deleted_dir / 'low_fps')
            low_fps.append(moved)
            print(f'Moved low-FPS video ({fps:.3f} FPS): {moved}')
            continue
        
        # Check for short duration
        if duration < min_duration:
            moved = _move_to_category(video, deleted_dir / 'too_short')
            too_short.append(moved)
            print(f'Moved short video ({duration:.3f} s): {moved}')

    print(f'Video cleanup complete: {len(duplicates)} duplicate(s), '
          f'{len(too_short)} too-short video(s), {len(low_fps)} low-FPS video(s)')
    return {'duplicates': duplicates, 'too_short': too_short, 'low_fps': low_fps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'source',
                        help='Directory whose root-level videos will be cleaned')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    clean_videos(args.source_dir)


if __name__ == '__main__':
    main()
