"""Video -> tracking -> segmentation -> 2D features, optionally extended with 3D.

From the repository root:
  python pipeline.py --source-dir INPUT --with-3d --lingbot-runs RUNS
  python pipeline.py --processed-manifest CLIPS.json --with-3d --lingbot-runs RUNS

Processed manifests accept Dino's clip-path list or {video, trajectory, pose}
records. Paths are relative to the manifest. This mode reuses YOLO detections;
no tracking model is loaded.

One background per run: before anything long, the first video (file name order)
gives a candidate background in OUTPUT/placement/candidate.png, with the lens
estimated from it. A person approves it in the terminal (n = try the next video).
SAM 3 then segments it once and the same lens is applied to every video. An
earlier OUTPUT/placement is offered for reuse and still needs a yes. A video of
another size stops the run. See processing/placement.py. The run needs a
terminal for this question.

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

Privacy tiers: after the features, each clip also gets <clip>_features_F1..F4.json
(cumulative tiers, privacy/strip_features.py) next to <clip>_features.json, plus one
feature_catalog_private.json. pipeline_run.json lists the tiers written under
"privacy_tiers". --no-privacy-tiers skips this; the script also runs alone.

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
from tempfile import TemporaryDirectory
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


def count_people(records, videos_found=None, videos_tracked=None):
    """How many videos went in and how many people (clips) came out of the tracker.

    A video with several people gives one clip per person, so clips >= videos with people.
    videos_found / videos_tracked are None when the tracking was done in an earlier run."""
    per_video = {}
    for record in records:
        per_video[str(record['video'])] = per_video.get(str(record['video']), 0) + 1
    return {'videos_found': videos_found, 'videos_tracked': videos_tracked,
            'videos_with_people': len(per_video), 'people_tracked': len(records),
            'videos_with_several_people': sum(n > 1 for n in per_video.values()),
            'most_people_in_one_video': max(per_video.values(), default=0)}


def log_summary(summary):
    failed = summary.get('failed_clips', [])
    level = {'complete': logging.INFO, 'failed': logging.ERROR}.get(summary['status'], logging.WARNING)
    logger.log(level, 'Pipeline %s - %d clip(s) ok, %d failed (%.1f s)', summary['status'],
               len(summary['clips']), len(failed), summary['duration_s'])
    counts = summary.get('counts')
    if counts:
        logger.info('Videos: %s found, %s tracked, %d with people -> %d people (clips); %d video(s) with several people '
                    '(most in one video: %d)', counts['videos_found'] if counts['videos_found'] is not None else '-',
                    counts['videos_tracked'] if counts['videos_tracked'] is not None else '-',
                    counts['videos_with_people'], counts['people_tracked'],
                    counts['videos_with_several_people'], counts['most_people_in_one_video'])
    if summary.get('privacy_tiers'):
        logger.info('Privacy tiers written: %s', ', '.join(summary['privacy_tiers']))
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

def run_fisheye_corrector(videos, progress, failures, fail_fast=False, lens=None):
    """Undistorts each video into [TEMP]_undistorted_results (emptied first). Returns the videos to track.

    lens is the one lens of the approved placement; every video is corrected with it.

    A failing video is added to failures and skipped (unless fail_fast)."""
    from processing import fisheye_correction
    for video in videos:
        logger.debug('  input: %s', video)
    return call_with_progress(fisheye_correction.correct_videos, progress, videos, failures=failures, fail_fast=fail_fast, lens=lens)


def run_trajectory_tracker(videos, source_dir, output, manifest, device, progress, failures, fail_fast=False):
    """Tracks exactly the given videos. A failing video is added to failures and skipped (unless fail_fast)."""
    from processing import video_processor
    logger.debug('Tracking %d video(s) on device=%s', len(videos), device)
    result = call_with_progress(video_processor.run, progress, source_dir=source_dir, output_root=output,
                                processed_manifest=manifest, device=device, failures=failures, fail_fast=fail_fast,
                                videos=videos)
    if not Path(manifest).is_file():
        raise FileNotFoundError(f'Tracker finished but wrote no manifest: {manifest}')
    if not geometry.load(manifest):
        raise RuntimeError('Tracker saved no clips; see the [REMOVED] lines in the log file for the reasons')
    logger.debug('Tracker wrote manifest %s', manifest)
    return result


def clean_source_videos(source_dir):
    """Remove duplicate, too-short, and low-FPS videos before raw input discovery."""
    from processing.clean_datasource import clean_videos

    result = clean_videos(source_dir)
    logger.info('Video cleanup - moved %d duplicate(s), %d too-short video(s), and %d low-FPS video(s)',
                len(result['duplicates']), len(result['too_short']), len(result['low_fps']))


def run_placement(videos, directory, undistort):
    """One approved background, lens and segmentation for the whole run (processing/placement.py).

    The person is asked in the terminal, so this runs outside any stage (a stage captures the output).
    undistort: True/False, or None for videos that are already tracked (an existing placement is accepted as it is)."""
    from processing import background_segmentation, placement

    def segment(background):
        with stage('Segmentation', 1) as progress:
            call_with_progress(background_segmentation.segment_images, progress, [str(background)])

    logger.info('Background - making a candidate from the first video')
    result = placement.prepare(videos, directory, undistort, segment)
    logger.info('Background - %s: %s', 'reused' if result['reused'] else 'approved', result['background'])
    return result


def run_feature_extractor(record, with_3d=False, runs_root=None, command_config=None, assumed_width=.825,
                          *, debug=True, shareable_depth_preview=None):
    from processing import feature_extractor
    name = record['video'].name
    trajectory = record['trajectory']
    if with_3d:
        workspace = record['workspace'] / 'geometry_3d'
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / '.depth-only').touch(exist_ok=True)
        record['tracking'] = record['workspace'] / 'geometry_3d/distances/tracking'
        logger.debug('%s: adapting trajectory for 3D', name)
        trajectory = geometry.adapt_trajectory(record['video'], trajectory, record['tracking'],
                                               feature_extractor.load_keypoints, record.get('pose'))
    if record.get('pose') is None:
        logger.debug('%s: no pose file, pose features skipped', name)
    out = record['workspace'] / f'{record["video"].stem}_features.json'
    feature_extractor.run_clip(trajectory, record['scene'], record.get('pose'),
                              record['background'], out, debug=debug, video_p=record['video'])
    features = geometry.load(out)
    features['info']['clip_video'] = record['video'].name
    features['info']['clip_video_sha256'] = record['video_info']['sha256']
    result = {'video': str(record['video']), 'features': str(out),
              'source_video_sha256': record['video_info']['sha256'], 'geometry_3d': None}
    if with_3d:
        logger.debug('%s: running 3D geometry', name)
        directory, summary = geometry.run_geometry(record, runs_root, command_config, assumed_width,
                                                   shareable_depth_preview=shareable_depth_preview)
        features = geometry.join_geometry(features, geometry.load(trajectory),
                                           geometry.load(directory / 'trajectory_3d.json'), summary)
        features['info']['sources']['depth'] = str(directory / 'ground_geometry.json')
        valid, total = summary['valid_3d_primary_frames'], summary['total_frames']
        logger.debug('%s: 3D valid frames %d/%d', name, valid, total)
        if total and valid / total < MIN_VALID_3D_RATIO:
            logger.warning('%s: only %.0f%% of frames have valid 3D', name, 100 * valid / total)
        result['geometry_3d'] = {'directory': str(directory), 'viewer': str(directory / 'viewer.html'),
                                  'valid_frames': valid, 'total_frames': total, 'yolo_rerun': False}
        if shareable_depth_preview is not None:
            result['geometry_3d']['shareable_depth_preview'] = summary['shareable_depth_preview']
    geometry.save(out, features)
    logger.debug('%s: features written to %s', name, out)
    return result


def run_privacy_tiers(clips):
    """Write <clip>_features_F1/F2/F3/F4.json next to each <clip>_features.json.

    clips: the per-clip results of run_feature_extractor. Returns [{'video','stage','error'}] for failures."""
    from privacy import strip_features
    failures = []
    for clip in clips:
        try:
            for tier, path, _ in strip_features.write_tiers(clip['features']):
                logger.debug('%s: %s written to %s', Path(clip['video']).name, tier, path)
        except Exception as error:
            logger.exception('%s: privacy tiers failed: %s', Path(clip['video']).name, error)
            failures.append({'video': clip['video'], 'stage': 'privacy', 'error': f'{type(error).__name__}: {error}'})
    return failures


def delete_debug_files(records):
    """Remove the feature-debug image of each clip (and background files left by older versions)."""
    for record in records:
        video_path = Path(record['video'])
        video_filename = video_path.stem
        workspace = Path(record['workspace'])
        filenames = (
            f'{video_filename}_background.json',
            f'{video_filename}_background_overlay.png',
            f'{video_filename}_background.png',
            f'{video_filename}_features_debug.png',
            f'features_{video_filename}_debug.png',
        )
        for filename in filenames:
            (workspace / filename).unlink(missing_ok=True)


def delete_undistorted_results(source_dir):
    """Remove the temporary undistorted-video directory created during correction (always, also with --debug)."""
    undistorted_results = Path(source_dir) / '[TEMP]_undistorted_results'
    if undistorted_results.exists():
        shutil.rmtree(undistorted_results)


# ----------------------------------------------------------------------------- orchestration

def run_pipeline(*, source_dir=SOURCE_DIR, 
                 output=PROJECT / 'data/pipeline', 
                 processed_manifest=None,
                 with_3d=False, 
                 lingbot_runs=PROJECT / 'data/lingbot', 
                 lingbot_command_config=None,
                 assumed_width=.825, 
                 skip_undistort=False, 
                 device='cpu', 
                 fail_fast=False, 
                 log_file=None,
                 debug=False,
                 privacy_tiers=True,
                 shareable_depth_preview_dir=None):
    output = Path(output).resolve()
    if shareable_depth_preview_dir is not None:
        if not with_3d:
            raise ValueError('--shareable-depth-preview-dir requires --with-3d')
        shareable_depth_preview_dir = Path(shareable_depth_preview_dir).resolve()
        if shareable_depth_preview_dir.is_relative_to(output) or output.is_relative_to(shareable_depth_preview_dir):
            raise ValueError('Choose a shareable preview directory separate from --output')
        if shareable_depth_preview_dir.exists() and (
                not shareable_depth_preview_dir.is_dir() or any(shareable_depth_preview_dir.iterdir())):
            raise ValueError('Choose a new or empty shareable preview directory')
    if with_3d and ((output / 'tracks').exists() or any(
            not (workspace / '.depth-only').is_file() for workspace in output.glob('*/geometry_3d'))):
        message = 'This output directory may contain video from an older run; use a fresh --output directory.'
        logger.error(message)
        raise ValueError(message)
    output.mkdir(parents=True, exist_ok=True)
    if shareable_depth_preview_dir is not None:
        shareable_depth_preview_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir).resolve()
    reused_tracking = processed_manifest is not None
    summary = {'status': 'running', 'with_3d': with_3d, 'reused_tracking': reused_tracking,
               'processed_manifest': None, 'log_file': str(log_file) if log_file else None,
               'placement': None, 'counts': None, 'privacy_tiers': [], 'clips': [], 'failed_clips': []}
    logger.debug('Pipeline started | output=%s | with_3d=%s | reused_tracking=%s | device=%s | fail_fast=%s',
                 output, with_3d, reused_tracking, device, fail_fast)
    t_start = time.perf_counter()
    private_tracking = None
    created_manifest = None  # the manifest this run wrote itself (not one given by the user)
    records = []
    videos = tracked = None
    try:
        if processed_manifest is None:
            clean_source_videos(source_dir)
            videos = list_videos(source_dir)
            if not videos:
                raise ValueError(f'No input videos found in {source_dir}')
            # the background (and with it the lens) is approved first, so the long steps below run unattended
            place = run_placement(videos, output / 'placement', undistort=not skip_undistort)
            if skip_undistort:
                logger.info('Fisheye correction - skipped (--skip-undistort)')
                tracked = videos  # tracked straight from the input folder; no temp folder
            else:
                with stage('Fisheye correction', len(videos)) as progress:
                    tracked = run_fisheye_corrector(videos, progress, summary['failed_clips'], fail_fast, place['lens'])
                if not tracked:
                    raise RuntimeError('Fisheye correction failed for every input video')
            tracking_root = output
            if with_3d:
                private_tracking = TemporaryDirectory(prefix='pipeline-tracking-')
                tracking_root = Path(private_tracking.name)
            processed_manifest = tracking_root / 'processed_video_paths.json'
            if private_tracking is None:
                created_manifest = processed_manifest
            with stage('Trajectory tracking', len(tracked)) as progress:
                run_trajectory_tracker(tracked, source_dir, tracking_root / 'tracks', processed_manifest, device, progress,
                                       summary['failed_clips'], fail_fast)
        if private_tracking is None:
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

        summary['counts'] = count_people(records, len(videos) if videos else None, len(tracked) if tracked else None)

        if reused_tracking:
            # tracking was done in an earlier run: the placement is made from these videos as they are
            place = run_placement(sorted(str(r['video']) for r in records), output / 'placement', undistort=None)
        for record in records:
            record['background'], record['scene'] = place['background'], place['scene']
        summary['placement'] = {'directory': str(place['background'].parent), 'reused': place['reused']}

        ready = [r for r in records if 'error' not in r]
        with stage('Feature extraction', len(ready)) as progress:
            for clip_number, record in enumerate(ready, start=1):
                name = record['video'].name
                preview = (shareable_depth_preview_dir / f'clip_{clip_number:04d}.mp4'
                           if shareable_depth_preview_dir is not None else None)
                logger.debug('Feature extraction: %s', name)
                try:
                    summary['clips'].append(run_feature_extractor(
                        record, with_3d, lingbot_runs, lingbot_command_config, assumed_width,
                        debug=debug or not with_3d, shareable_depth_preview=preview))
                except Exception as error:
                    if fail_fast:
                        raise
                    logger.exception('%s: feature extraction failed: %s', name, error)
                    mark_failed(record, '3d' if with_3d and 'tracking' in record else 'features', error)
                    progress(ok=False)
                    continue
                progress()

        summary['failed_clips'] += [{'video': str(r['video']), 'stage': r['failed_stage'], 'error': r['error']}
                                    for r in records if 'error' in r]
        if not summary['clips']:
            raise RuntimeError('No clip was processed successfully; see the log for each clip')

        from processing import feature_extractor
        catalog = list(feature_extractor.CATALOG)
        if with_3d:
            catalog.extend(geometry.catalog_entries())
        geometry.save(output / 'feature_catalog.json', {'schema_version': feature_extractor.SCHEMA_VERSION,
            'extensions': {'geometry_3d': '1.0'} if with_3d else {}, 'features': catalog})
        
        if privacy_tiers:
            from privacy import strip_features
            with stage('Privacy tiers F1-F4', len(summary['clips'])) as progress:
                for clip in summary['clips']:
                    summary['failed_clips'] += run_privacy_tiers([clip])
                    progress()
            strip_features.write_private_catalog(output / 'feature_catalog.json')
            summary['privacy_tiers'] = list(strip_features.PIPELINE_TIERS)

        if not debug:
            delete_debug_files(records=records)
        
        logger.info('Feature catalog - %d feature(s) written', len(catalog))
        summary['status'] = 'complete_with_errors' if summary['failed_clips'] else 'complete'
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        logger.exception('Pipeline failed: %s', summary['error'])
        raise
    finally:
        if private_tracking is not None:
            private_tracking.cleanup()
        if not reused_tracking:
            delete_undistorted_results(source_dir)  # no video is kept, whatever the flags or the outcome
        if created_manifest is not None and not debug:
            Path(created_manifest).unlink(missing_ok=True)  # pipeline_run.json is the record of the run
            summary['processed_manifest'] = None
        if with_3d and not debug and summary['status'] == 'failed':
            delete_debug_files([record for record in records if 'workspace' in record])
        summary['duration_s'] = round(time.perf_counter() - t_start, 1)
        geometry.save(output / 'pipeline_run.json', summary)
        logger.debug('Run summary saved to %s', output / 'pipeline_run.json')
        log_summary(summary)
    return summary


def main():
    global CAPTURE_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-dir', type=Path, default=SOURCE_DIR)
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: data/pipeline/<name of the input folder>, so every input folder '
                             'gets its own background and run summary). Use a fresh directory for 3D after upgrading from an older version.')
    parser.add_argument('--processed-manifest', type=Path, help='Reuse saved clips/trajectory JSON; skip fisheye and YOLO stages')
    parser.add_argument('--with-3d', action='store_true', help='Append LingBot distances and closure rates to 2D features')
    parser.add_argument('--shareable-depth-preview-dir', type=Path,
                        help='With --with-3d, write depth-coloured box videos to a separate new or empty directory')
    parser.add_argument('--lingbot-runs', type=Path, default=PROJECT / 'data/lingbot')
    parser.add_argument('--lingbot-config', '--lingbot-command-config', dest='lingbot_command_config',
                        type=Path, help='LingBot local/ssh-docker runtime JSON (or custom command argv/cwd)')
    parser.add_argument('--door-width-m', type=float, default=.825)
    parser.add_argument('--skip-undistort', action='store_true')
    parser.add_argument('--device', default='cpu', help='YOLO device for new tracking runs')
    parser.add_argument('--fail-fast', action='store_true', help='Stop at the first failing clip instead of skipping it')
    parser.add_argument('--show-output', action='store_true',
                        help='Also print the raw output of the submodules (YOLO, SAM3, undistort) to the console')
    parser.add_argument('--debug', action='store_true',
                        help='Keep the per-clip feature debug images and the processed_video_paths.json manifest '
                             '(no video is ever kept)')
    parser.add_argument('--no-privacy-tiers', action='store_true',
                        help='Do not write the <clip>_features_F1/F2/F3/F4.json files (privacy/strip_features.py can make them later)')
    args = parser.parse_args()
    if args.shareable_depth_preview_dir is not None and not args.with_3d:
        parser.error('--shareable-depth-preview-dir requires --with-3d')
    
    CAPTURE_OUTPUT = not args.show_output
    if args.output is None:  # one output folder per input folder
        name = args.processed_manifest.resolve().parent.name if args.processed_manifest else args.source_dir.resolve().name
        args.output = PROJECT / 'data/pipeline' / name
    log_file = setup_logging(args.output.resolve() / 'logs')
    if args.show_output:
        CONSOLE.live = False  # raw prints would land in the middle of the in-place line
    logger.debug('Log file: %s', log_file)
    try:
        result = run_pipeline(
                source_dir=args.source_dir, 
                output=args.output, 
                processed_manifest=args.processed_manifest,
                with_3d=args.with_3d, 
                lingbot_runs=args.lingbot_runs, 
                lingbot_command_config=args.lingbot_command_config,
                assumed_width=args.door_width_m, 
                skip_undistort=args.skip_undistort, 
                device=args.device,
                fail_fast=args.fail_fast, 
                log_file=log_file,
                debug=args.debug,
                privacy_tiers=not args.no_privacy_tiers,
                shareable_depth_preview_dir=args.shareable_depth_preview_dir,
            )
    except Exception:
        sys.exit(1)  # already logged with traceback
    sys.exit(2 if result['failed_clips'] else 0)


if __name__ == '__main__':
    main()
