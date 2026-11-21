"""One background, one lens and one segmentation per camera placement.

A run uses a single background for every clip, because a background made per
clip is another stored picture of the room taken at another moment. The
background is made once, a person approves it, SAM 3 segments it once, and the
same lens correction is applied to every video.

Flow (prepare):
  1. For the videos in file-name order: median background of the video, the lens
     estimated from it, the background undistorted with that lens
     -> <directory>/candidate.png, shown to the person.
  2. Approved -> candidate.png becomes background.png, the caller segments it
     (background.json) and placement.json is written. Not approved -> the next
     video is tried, until the person approves one or the videos run out.
  3. If <directory> already holds an approved placement from an earlier run, the
     person is asked about that background first. A different image size always
     stops the run (a different camera setup).

<directory> holds, after a run:
  background.png          the approved background (people masked, undistorted)
  background.json         SAM 3 segmentation of it
  background_overlay.png  the segmentation drawn on it
  placement.json          image size, lens, which video it came from

The person has to answer in a terminal; without one the run stops before any
long work, because nothing else can approve a background.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import sys

import cv2

if __package__:
    from . import background_extraction, fisheye_correction
else:
    import background_extraction
    import fisheye_correction

logger = logging.getLogger(__name__)

FOLDER = 'placement'
BACKGROUND = 'background.png'
SCENE = 'background.json'
OVERLAY = 'background_overlay.png'
CANDIDATE = 'candidate.png'
META = 'placement.json'


class PlacementError(RuntimeError):
    """The background cannot be used or was not approved; the run stops."""


def video_size(video):
    """[width, height] of a video."""
    cap = cv2.VideoCapture(str(video))
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cap.release()
    if not all(size):
        raise PlacementError(f'Could not read the frame size of {video}')
    return size


def check_sizes(videos, size):
    """Stops the run if any video has another frame size than the placement."""
    for video in videos:
        found = video_size(video)
        if found != list(size):
            raise PlacementError(
                f'{Path(video).name} is {found[0]}x{found[1]} but the placement background is '
                f'{size[0]}x{size[1]}: this is another camera setup. Use another --output directory.')


def interactive():
    return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())


def ask(question):
    """y/n question in the terminal; returns True for yes."""
    while True:
        answer = input(f'{question} [y/n] ').strip().lower()
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False


def lens_text(lens):
    if lens is None:
        return 'lens: not checked (undistortion skipped)'
    verdict = 'DISTORTED, undistorted' if lens['distorted'] else 'not distorted'
    return f'lens: {verdict} ({lens["reason"]})'


def candidate(video, undistort):
    """Background of one video as the clips will see it.

    Returns (image, lens, report). With undistort, the lens is estimated on the
    background of this video and applied to the background; every other video
    will later get this same lens. lens is None when undistort is False."""
    image, report = background_extraction.median_background(video)
    lens = None
    if undistort:
        lens = fisheye_correction.detect_distortion(image)
        lens['video'] = Path(video).name
        if lens['distorted']:
            image = fisheye_correction.undistort_image(image, lens)  # also adds new_K to lens
    return image, lens, report


def load(directory):
    """The approved placement saved in directory, or None."""
    directory = Path(directory)
    meta_path = directory / META
    if not (meta_path.is_file() and (directory / BACKGROUND).is_file() and (directory / SCENE).is_file()):
        return None
    with open(meta_path, encoding='utf-8') as f:
        return json.load(f)


def result(directory, meta, reused):
    directory = Path(directory)
    return {'background': directory / BACKGROUND, 'scene': directory / SCENE, 'lens': meta['lens'],
            'image_size': meta['image_size'], 'undistort': meta['undistort'], 'reused': reused}


def prepare(videos, directory, undistort, segment):
    """Returns the approved placement as {'background', 'scene', 'lens', 'image_size', 'undistort', 'reused'}.

    videos    paths in the order candidates are tried (file name order)
    directory where the placement is kept (normally <output>/placement)
    undistort True: estimate the lens and undistort the background; False: use
              the videos as they are; None: use the videos as they are and accept
              an existing placement whichever way it was made (videos already tracked)
    segment   function(background_path) that writes background.json next to it
    """
    videos = [Path(v) for v in videos]
    directory = Path(directory)
    if not videos:
        raise PlacementError('No videos to make a background from')
    if not interactive():
        raise PlacementError('The background has to be approved by a person: run the pipeline in a terminal '
                             '(a redirected or non-interactive run cannot answer the question).')

    existing = load(directory)
    if existing is not None:
        check_sizes(videos, existing['image_size'])
        if undistort is not None and undistort != existing['undistort']:
            raise PlacementError(
                f'The placement in {directory} was made {"with" if existing["undistort"] else "without"} '
                f'undistortion but this run is {"with" if undistort else "without"} it. '
                f'Remove that folder or use another --output directory.')
        print(f'\nA background from an earlier run exists: {directory / BACKGROUND}')
        print(f'  {lens_text(existing["lens"])}, made from {existing.get("source_video")}')
        if ask('Use this background?'):
            return result(directory, existing, reused=True)

    directory.mkdir(parents=True, exist_ok=True)
    candidate_path = directory / CANDIDATE
    for video in videos:
        image, lens, report = candidate(video, bool(undistort))
        size = [image.shape[1], image.shape[0]]
        check_sizes(videos, size)
        if not cv2.imwrite(str(candidate_path), image):
            raise PlacementError(f'Could not save {candidate_path}')
        print(f'\nBackground made from {video.name}: {candidate_path}')
        print(f'  {lens_text(lens)}; people still found and pixelated: {report["people_left"]}')
        if ask('Use this background for all videos?'):
            break
        candidate_path.unlink(missing_ok=True)
        print('  Trying the next video.')
    else:
        raise PlacementError('No video gave a background that was approved')

    for name in (BACKGROUND, SCENE, OVERLAY, META):
        (directory / name).unlink(missing_ok=True)
    shutil.move(str(candidate_path), str(directory / BACKGROUND))
    segment(directory / BACKGROUND)
    if not (directory / SCENE).is_file():
        raise PlacementError(f'Segmentation did not write {directory / SCENE}')
    meta = {'image_size': size, 'undistort': bool(undistort), 'lens': lens, 'source_video': video.name,
            'approved': True}
    with open(directory / META, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return result(directory, meta, reused=False)