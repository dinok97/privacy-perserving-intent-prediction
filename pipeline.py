"""Video -> tracking -> segmentation -> 2D features, optionally extended with 3D.

From the repository root:
  python pipeline.py --source-dir INPUT --with-3d --lingbot-runs RUNS
  python pipeline.py --processed-manifest CLIPS.json --with-3d --lingbot-runs RUNS

Processed manifests accept Dino's clip-path list or {video, trajectory, scene,
pose, background} records. Paths are relative to the manifest. This mode reuses
YOLO detections; no tracking model is loaded. Optional scene paths reuse SAM3.

For missing depth, --lingbot-config accepts a local or ssh-docker runtime JSON
(see geometry3d/lingbot_runtime.py). LingBot stays in its own checkout/environment. The
legacy --lingbot-command-config alias also supports custom command argv/cwd.

External 3D dependency: https://github.com/Robbyant/lingbot-map
Clone/install LingBot in a separate Python environment following its upstream
instructions, and download lingbot-map-long.pt. Neither is included by cloning
this project. Copy geometry3d/lingbot.local.example.json, replace its
python/repo/checkpoint paths, and pass it with --with-3d --lingbot-config FILE.
Cached depth exports can be used without installing LingBot.

Logging: the console shows one line per pipeline step, "TIME INFO step - x/n",
updated in place while the step runs and kept when it finishes. Warnings and
errors get their own line and the progress line continues below them.
OUTPUT/logs/pipeline_<timestamp>.log always gets everything at DEBUG: full
tracebacks plus the print()/progress-bar output of the submodules. Pass
--show-output to also see that raw output on the console (this turns off the
in-place line). When stdout is not a terminal (e.g. redirected to a file),
only the final line of each step is printed.

Progress inside submodules: processing.fisheye_correction.correct_videos,
processing.video_processor.run and
processing.background_segmentation.segment_images may accept an
optional `on_progress` argument and call on_progress() after each item, or
on_progress(ok=False) after a failed one. Without it the step shows 0/n until
it finishes.

A clip that fails is logged, recorded in pipeline_run.json under
"failed_clips" and skipped; the other clips continue. Use --fail-fast to stop
at the first failing clip instead. Exit codes: 0 complete, 2 complete with
failed clips, 1 pipeline failed. Other modules join the same log through
logging.getLogger(__name__).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import inspect
import logging
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parent
from geometry3d import pipeline_geometry as geometry

SOURCE_DIR = PROJECT / 'source'
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
NOISY_LIBRARIES = ('PIL', 'matplotlib', 'urllib3', 'ultralytics', 'h5py', 'numba',
                   'httpx', 'httpcore', 'huggingface_hub', 'transformers', 'filelock')
MIN_VALID_3D_RATIO = .5  # below this share of valid 3D frames a warning is logged
CAPTURE_OUTPUT = True     # send print()/progress bars of submodules to the log file only

logger = logging.getLogger('pipeline')
CONSOLE = None  # the ConsoleHandler, set by setup_logging


# ----------------------------------------------------------------------------- logging helpers

class StreamToLogger:
    """File-like object: print() and progress-bar output go to a logger at DEBUG (log file only)."""

    def __init__(self, name, original):
        self.log, self.original, self.buffer = logging.getLogger(name), original, ''

    def write(self, text):
        self.buffer += text
        *lines, self.buffer = self.buffer.split('\n')
        for line in lines:
            self._emit(line)
        return len(text)

    def _emit(self, line):
        # tqdm redraws with '\r'; keep only the last state of the bar
        parts = [p.rstrip() for p in line.split('\r') if p.strip()]
        if parts:
            self.log.debug(parts[-1])

    def flush(self):
        pass

    def close_buffer(self):
        if self.buffer:
            self._emit(self.buffer)
            self.buffer = ''

    def isatty(self):
        return False

    def fileno(self):
        return self.original.fileno()


class ConsoleFormatter(logging.Formatter):
    """Short console format without tracebacks (those go to the log file)."""

    def __init__(self):
        super().__init__('%(asctime)s %(levelname)-7s %(message)s', '%H:%M:%S')

    def format(self, record):
        saved = record.exc_info, record.exc_text, record.stack_info
        record.exc_info = record.exc_text = record.stack_info = None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text, record.stack_info = saved


class ConsoleHandler(logging.StreamHandler):
    """Console handler that keeps one live progress line below the log lines."""

    def __init__(self, stream):
        super().__init__(stream)
        isatty = getattr(stream, 'isatty', None)
        self.live = bool(isatty and isatty())  # no '\r' tricks when output goes to a file
        self.progress = None
        self.shown = 0  # length of the progress text currently on screen

    def clear(self):
        if self.shown:
            self.stream.write('\r' + ' ' * self.shown + '\r')
            self.stream.flush()
            self.shown = 0

    def show(self, text):
        if not self.live:
            return
        # a line wider than the terminal wraps, and '\r' can only go back to the start of the last row
        text = text[:shutil.get_terminal_size().columns - 1]
        self.clear()
        self.stream.write(text)
        self.stream.flush()
        self.shown = len(text)

    def emit(self, record):
        self.clear()                # the message takes the progress line's place ...
        super().emit(record)
        if self.progress is not None:
            self.progress.render()  # ... and the progress line continues below it


class Progress:
    """x/n counter of one step. Pass it to a submodule as on_progress."""

    def __init__(self, name, total=None):
        self.name, self.total, self.done, self.failed = name, total, 0, 0
        self.t0 = time.perf_counter()

    def __call__(self, ok=True):
        self.done += 1
        if not ok:
            self.failed += 1
        self.render()

    def elapsed(self):
        return time.perf_counter() - self.t0

    def text(self, suffix=''):
        count = str(self.done) if self.total is None else f'{self.done}/{self.total}'
        failed = f' ({self.failed} failed)' if self.failed else ''
        return f'{self.name} - {count}{failed}{suffix}'

    def render(self):
        if CONSOLE is not None:
            CONSOLE.show(f'{datetime.now():%H:%M:%S} {"INFO":<7} {self.text(" ...")}')


def _stop_progress():
    if CONSOLE is not None:
        CONSOLE.progress = None
        CONSOLE.clear()


def setup_logging(log_dir):
    """Console at INFO (short format, live progress), file at DEBUG (full format). Returns the log file path."""
    global CONSOLE
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'
    CONSOLE = ConsoleHandler(sys.stdout)
    CONSOLE.setFormatter(ConsoleFormatter())
    CONSOLE.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                                                '%Y-%m-%d %H:%M:%S'))
    file_handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = [CONSOLE, file_handler]
    for name in NOISY_LIBRARIES:  # keep third-party debug spam out of the file
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file


@contextmanager
def stage(name, total=None):
    """One pipeline step: live x/n line on the console, captured submodule output, timing."""
    progress = Progress(name, total)
    logger.debug('%s - started', name)
    if CONSOLE is not None:
        CONSOLE.progress = progress
    progress.render()
    out = StreamToLogger('pipeline.stdout', sys.stdout)
    err = StreamToLogger('pipeline.stderr', sys.stderr)
    try:
        if CAPTURE_OUTPUT:
            with redirect_stdout(out), redirect_stderr(err):
                yield progress
        else:
            yield progress
    except Exception:
        _stop_progress()
        # traceback is logged once, by run_pipeline, to avoid duplicates
        logger.error('%s FAILED after %.1f s', progress.text(), progress.elapsed())
        raise
    finally:
        out.close_buffer()
        err.close_buffer()
        _stop_progress()
    level = logging.WARNING if progress.failed else logging.INFO
    logger.log(level, '%s', progress.text(f' done ({progress.elapsed():.1f} s)'))


def call_with_progress(func, progress, *args, **kwargs):
    """Give func the progress counter as on_progress if it accepts one.

    Otherwise the step shows 0/n while func runs and n/n once it returns."""
    try:
        accepts = 'on_progress' in inspect.signature(func).parameters
    except (TypeError, ValueError):
        accepts = False
    if accepts:
        return func(*args, on_progress=progress, **kwargs)
    logger.debug('%s has no on_progress argument; counting all items at the end', func.__qualname__)
    result = func(*args, **kwargs)
    if progress.total is not None:
        progress.done = progress.total
    return result


def mark_failed(record, stage_name, error):
    record['error'] = f'{type(error).__name__}: {error}'
    record['failed_stage'] = stage_name


def log_summary(summary):
    failed = summary.get('failed_clips', [])
    level = {'complete': logging.INFO, 'failed': logging.ERROR}.get(summary['status'], logging.WARNING)
    logger.log(level, 'Pipeline %s - %d clip(s) ok, %d failed (%.1f s)', summary['status'],
               len(summary['clips']), len(failed), summary['duration_s'])
    for item in failed:
        logger.warning('  %s [%s]: %s', Path(item['video']).name, item['stage'], item['error'])
    if summary.get('log_file'):
        logger.info('Log file: %s', short_path(summary['log_file']))


def short_path(path):
    """Path relative to the current folder for display; absolute if impossible (e.g. another drive on Windows)."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return str(path)


def list_videos(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


# ----------------------------------------------------------------------------- stages

def run_fisheye_corrector(videos, source_dir, progress):
    from processing import fisheye_correction
    for video in videos:
        logger.debug('  input: %s', video)
    call_with_progress(fisheye_correction.correct_videos, progress, videos)
    return Path(source_dir) / '[TEMP]_undistorted_results'


def run_trajectory_tracker(source_dir, output, manifest, device, progress):
    from processing import video_processor
    logger.debug('Tracking videos from %s on device=%s', source_dir, device)
    result = call_with_progress(video_processor.run, progress, source_dir=source_dir, output_root=output,
                                processed_manifest=manifest, device=device)
    if not Path(manifest).is_file():
        raise FileNotFoundError(f'Tracker finished but wrote no manifest: {manifest}')
    if not geometry.load(manifest):
        raise RuntimeError('Tracker saved no clips; see the [REMOVED] lines in the log file for the reasons')
    logger.debug('Tracker wrote manifest %s', manifest)
    return result


def run_background_extractor(records, progress, fail_fast=False):
    """Median background per clip and a check of existing scene files.

    Returns the background images that still need segmentation. Failing records get an 'error' key."""
    from processing import background_extraction
    pending = []
    for record in records:
        video, work = record['video'], record['workspace']
        try:
            scene = record.get('scene')
            sibling_scene = video.with_name(video.stem + '_background.json')
            if scene is None:
                scene = sibling_scene if sibling_scene.is_file() else work / (video.stem + '_background.json')
            record['scene'] = scene
            background = record.get('background') or work / (video.stem + '_background.png')
            record['background'] = background
            if background.is_file():
                logger.debug('%s: reusing background %s', video.name, background)
            else:
                import cv2
                logger.debug('%s: extracting median background', video.name)
                background.parent.mkdir(parents=True, exist_ok=True)
                image = background_extraction.median_background(video)
                if not cv2.imwrite(str(background), image):
                    raise OSError(f'Could not save background {background}')
            if not scene.is_file():
                if scene != background.with_suffix('.json'):
                    raise FileNotFoundError(f'Explicit segmentation is missing: {scene}')
                pending.append(str(background))
            else:
                parsed = geometry.load(scene)
                size = [record['video_info']['width'], record['video_info']['height']]
                if parsed['image_size'] != size:
                    raise ValueError(f'Scene image size {parsed["image_size"]} != video size {size}: {scene}')
                if parsed.get('source_video_sha256') not in (None, record['video_info']['sha256']):
                    raise ValueError(f'Scene segmentation was made from a different video: {scene}')
                logger.debug('%s: reusing scene segmentation %s', video.name, scene)
        except Exception as error:
            if fail_fast:
                raise
            logger.exception('%s: background failed: %s', video.name, error)
            mark_failed(record, 'background', error)
            progress(ok=False)
            continue
        progress()
    return pending


def run_segmentation(records, pending, progress, fail_fast=False):
    from processing import background_segmentation
    call_with_progress(background_segmentation.segment_images, progress, pending)
    for record in records:
        if 'error' in record or str(record['background']) not in pending:
            continue
        try:
            scene = geometry.load(record['scene'])
            scene['source_video_sha256'] = record['video_info']['sha256']
            geometry.save(record['scene'], scene)
        except Exception as error:
            if fail_fast:
                raise
            logger.exception('%s: segmentation output missing or unreadable: %s', record['video'].name, error)
            mark_failed(record, 'segmentation', error)
            progress.failed += 1
            progress.render()


def run_feature_extractor(record, with_3d=False, runs_root=None, command_config=None, assumed_width=.825):
    from processing import feature_extractor
    name = record['video'].name
    trajectory = record['trajectory']
    if with_3d:
        record['tracking'] = record['workspace'] / 'geometry_3d/distances/tracking'
        logger.debug('%s: adapting trajectory for 3D', name)
        trajectory = geometry.adapt_trajectory(record['video'], trajectory, record['tracking'],
                                               feature_extractor.load_keypoints, record.get('pose'))
    if record.get('pose') is None:
        logger.debug('%s: no pose file, pose features skipped', name)
    out = record['workspace'] / f'features_{record["video"].stem}.json'
    feature_extractor.run_clip(trajectory, record['scene'], record.get('pose'),
                              record['background'], out, debug=True)
    features = geometry.load(out)
    features['info']['clip_video'] = record['video'].name
    features['info']['clip_video_sha256'] = record['video_info']['sha256']
    result = {'video': str(record['video']), 'features': str(out),
              'source_video_sha256': record['video_info']['sha256'], 'geometry_3d': None}
    if with_3d:
        logger.debug('%s: running 3D geometry', name)
        directory, summary = geometry.run_geometry(record, runs_root, command_config, assumed_width)
        features = geometry.join_geometry(features, geometry.load(trajectory),
                                           geometry.load(directory / 'trajectory_3d.json'), summary)
        features['info']['sources']['depth'] = str(directory / 'ground_geometry.json')
        valid, total = summary['valid_3d_primary_frames'], summary['total_frames']
        logger.debug('%s: 3D valid frames %d/%d', name, valid, total)
        if total and valid / total < MIN_VALID_3D_RATIO:
            logger.warning('%s: only %.0f%% of frames have valid 3D', name, 100 * valid / total)
        result['geometry_3d'] = {'directory': str(directory), 'viewer': str(directory / 'viewer.html'),
                                  'valid_frames': valid, 'total_frames': total, 'yolo_rerun': False}
    geometry.save(out, features)
    logger.debug('%s: features written to %s', name, out)
    return result


# ----------------------------------------------------------------------------- orchestration

def run_pipeline(*, source_dir=SOURCE_DIR, output=PROJECT / 'data/pipeline', processed_manifest=None,
                 with_3d=False, lingbot_runs=PROJECT / 'data/lingbot', lingbot_command_config=None,
                 assumed_width=.825, skip_undistort=False, device='cpu', fail_fast=False, log_file=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir).resolve()
    reused_tracking = processed_manifest is not None
    summary = {'status': 'running', 'with_3d': with_3d, 'reused_tracking': reused_tracking,
               'processed_manifest': None, 'log_file': str(log_file) if log_file else None,
               'clips': [], 'failed_clips': []}
    logger.debug('Pipeline started | output=%s | with_3d=%s | reused_tracking=%s | device=%s | fail_fast=%s',
                 output, with_3d, reused_tracking, device, fail_fast)
    t_start = time.perf_counter()
    try:
        if processed_manifest is None:
            if skip_undistort:
                logger.info('Fisheye correction - skipped (--skip-undistort)')
                input_dir = source_dir
            else:
                videos = list_videos(source_dir)
                if not videos:
                    raise ValueError(f'No input videos found in {source_dir}')
                with stage('Fisheye correction', len(videos)) as progress:
                    input_dir = run_fisheye_corrector(videos, source_dir, progress)
            processed_manifest = output / 'processed_video_paths.json'
            with stage('Trajectory tracking', len(list_videos(input_dir))) as progress:
                run_trajectory_tracker(input_dir, output / 'tracks', processed_manifest, device, progress)
        summary['processed_manifest'] = str(Path(processed_manifest).resolve())

        with stage('Loading clip records') as progress:
            records = geometry.records_from_manifest(processed_manifest)
            progress.total = len(records)
            logger.debug('Loaded %d clip record(s) from %s', len(records), processed_manifest)
            seen = set()
            for record in records:
                info = geometry.video_info(record['video'])
                key = (str(record['video']), str(record['trajectory']))
                if key in seen:
                    raise ValueError(f'Duplicate video/trajectory record in processed manifest: {key}')
                seen.add(key)
                record['workspace'] = output / (record['video'].stem + '-' + info['sha256'][:10] + '-' + geometry.sha(record['trajectory'])[:8])
                record['workspace'].mkdir(parents=True, exist_ok=True)
                record['video_info'] = info
                logger.debug('%s: %sx%s, workspace %s', record['video'].name,
                             info['width'], info['height'], record['workspace'])
                progress()

        with stage('Background extraction', len(records)) as progress:
            pending = run_background_extractor(records, progress, fail_fast)
        if pending:
            with stage('Segmentation', len(pending)) as progress:
                run_segmentation(records, pending, progress, fail_fast)
        else:
            logger.info('Segmentation - nothing to segment (existing scenes reused)')

        ready = [r for r in records if 'error' not in r]
        with stage('Feature extraction', len(ready)) as progress:
            for record in ready:
                name = record['video'].name
                logger.debug('Feature extraction: %s', name)
                try:
                    summary['clips'].append(run_feature_extractor(record, with_3d, lingbot_runs,
                                                                  lingbot_command_config, assumed_width))
                except Exception as error:
                    if fail_fast:
                        raise
                    logger.exception('%s: feature extraction failed: %s', name, error)
                    mark_failed(record, '3d' if with_3d and 'tracking' in record else 'features', error)
                    progress(ok=False)
                    continue
                progress()

        summary['failed_clips'] = [{'video': str(r['video']), 'stage': r['failed_stage'], 'error': r['error']}
                                   for r in records if 'error' in r]
        if not summary['clips']:
            raise RuntimeError('No clip was processed successfully; see the log for each clip')

        from processing import feature_extractor
        catalog = list(feature_extractor.CATALOG)
        if with_3d:
            catalog.extend(geometry.catalog_entries())
        geometry.save(output / 'feature_catalog.json', {'schema_version': feature_extractor.SCHEMA_VERSION,
            'extensions': {'geometry_3d': '1.0'} if with_3d else {}, 'features': catalog})
        logger.info('Feature catalog - %d feature(s) written', len(catalog))
        summary['status'] = 'complete_with_errors' if summary['failed_clips'] else 'complete'
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        logger.exception('Pipeline failed: %s', summary['error'])
        raise
    finally:
        summary['duration_s'] = round(time.perf_counter() - t_start, 1)
        geometry.save(output / 'pipeline_run.json', summary)
        logger.debug('Run summary saved to %s', output / 'pipeline_run.json')
        log_summary(summary)
    return summary


def main():
    global CAPTURE_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--output', type=Path, default=PROJECT / 'data/pipeline')
    parser.add_argument('--processed-manifest', type=Path, help='Reuse saved clips/trajectory JSON; skip fisheye and YOLO stages')
    parser.add_argument('--with-3d', action='store_true', help='Append LingBot distances and closure rates to 2D features')
    parser.add_argument('--lingbot-runs', type=Path, default=PROJECT / 'data/lingbot')
    parser.add_argument('--lingbot-config', '--lingbot-command-config', dest='lingbot_command_config',
                        type=Path, help='LingBot local/ssh-docker runtime JSON (or custom command argv/cwd)')
    parser.add_argument('--door-width-m', type=float, default=.825)
    parser.add_argument('--skip-undistort', action='store_true')
    parser.add_argument('--device', default='cpu', help='YOLO device for new tracking runs')
    parser.add_argument('--fail-fast', action='store_true', help='Stop at the first failing clip instead of skipping it')
    parser.add_argument('--show-output', action='store_true',
                        help='Also print the raw output of the submodules (YOLO, SAM3, undistort) to the console')
    args = parser.parse_args()
    CAPTURE_OUTPUT = not args.show_output
    log_file = setup_logging(args.output.resolve() / 'logs')
    if args.show_output:
        CONSOLE.live = False  # raw prints would land in the middle of the in-place line
    logger.debug('Log file: %s', log_file)
    try:
        result = run_pipeline(source_dir=args.source_dir, output=args.output, processed_manifest=args.processed_manifest,
            with_3d=args.with_3d, lingbot_runs=args.lingbot_runs, lingbot_command_config=args.lingbot_command_config,
            assumed_width=args.door_width_m, skip_undistort=args.skip_undistort, device=args.device,
            fail_fast=args.fail_fast, log_file=log_file)
    except Exception:
        sys.exit(1)  # already logged with traceback
    sys.exit(2 if result['failed_clips'] else 0)


if __name__ == '__main__':
    main()