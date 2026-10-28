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

Logging: every run writes to the console (--log-level, default INFO) and to
OUTPUT/logs/pipeline_<timestamp>.log (always DEBUG, with full tracebacks).
A clip that fails is logged, recorded in pipeline_run.json under
"failed_clips" and skipped; the other clips continue. Use --fail-fast to stop
at the first failing clip instead. Exit codes: 0 complete, 2 complete with
failed clips, 1 pipeline failed. Other modules join the same log through
logging.getLogger(__name__).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import logging
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parent
from geometry3d import pipeline_geometry as geometry

SOURCE_DIR = PROJECT / 'source'
NOISY_LIBRARIES = ('PIL', 'matplotlib', 'urllib3', 'ultralytics', 'h5py', 'numba')
MIN_VALID_3D_RATIO = .5  # below this share of valid 3D frames a warning is logged

logger = logging.getLogger('pipeline')


# ----------------------------------------------------------------------------- logging helpers

def setup_logging(log_dir, level=logging.INFO):
    """Console at `level`, file at DEBUG. Returns the log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'pipeline_{datetime.now():%Y%m%d_%H%M%S}.log'
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(level)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers = [console, file_handler]
    for name in NOISY_LIBRARIES:  # keep third-party debug spam out of the file
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file


@contextmanager
def stage(name):
    """Log start, duration and failure of a pipeline stage."""
    logger.info('%s - started', name)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        # traceback is logged once, by run_pipeline, to avoid duplicates
        logger.error('%s - FAILED after %.1f s', name, time.perf_counter() - t0)
        raise
    logger.info('%s - done in %.1f s', name, time.perf_counter() - t0)


def mark_failed(record, stage_name, error):
    record['error'] = f'{type(error).__name__}: {error}'
    record['failed_stage'] = stage_name


def log_summary(summary):
    failed = summary.get('failed_clips', [])
    logger.info('=' * 70)
    logger.info('Status: %s | clips ok: %d | clips failed: %d',
                summary['status'], len(summary['clips']), len(failed))
    for item in failed:
        logger.warning('  failed [%s] %s: %s', item['stage'], Path(item['video']).name, item['error'])
    if summary.get('error'):
        logger.error('  pipeline error: %s', summary['error'])
    logger.info('=' * 70)


# ----------------------------------------------------------------------------- stages

def run_fisheye_corrector(source_dir):
    from processing import fisheye_correction
    videos = sorted(str(p) for p in Path(source_dir).iterdir()
                    if p.is_file() and p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.m4v'})
    if not videos:
        raise ValueError(f'No input videos found in {source_dir}')
    logger.info('Found %d input video(s) in %s', len(videos), source_dir)
    for video in videos:
        logger.debug('  input: %s', video)
    fisheye_correction.correct_videos(videos)
    return Path(source_dir) / '[TEMP]_undistorted_results'


def run_trajectory_tracker(source_dir, output, manifest, device):
    from processing import video_processor
    logger.info('Tracking videos from %s on device=%s', source_dir, device)
    result = video_processor.run(source_dir=source_dir, output_root=output,
                                 processed_manifest=manifest, device=device)
    if not Path(manifest).is_file():
        raise FileNotFoundError(f'Tracker finished but wrote no manifest: {manifest}')
    logger.info('Tracker wrote manifest %s', manifest)
    return result


def run_background_extractor_and_segmentation(records, fail_fast=False):
    """Prepare background + scene per record. Failing records get an 'error' key."""
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
                logger.info('%s: extracting median background', video.name)
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
            logger.exception('%s: background/scene preparation failed', video.name)
            mark_failed(record, 'background', error)
    if pending:
        from processing import background_segmentation
        logger.info('Segmenting %d background image(s)', len(pending))
        background_segmentation.segment_images(pending)
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
                logger.exception('%s: segmentation output missing or unreadable', record['video'].name)
                mark_failed(record, 'segmentation', error)


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
        logger.info('%s: running 3D geometry', name)
        directory, summary = geometry.run_geometry(record, runs_root, command_config, assumed_width)
        features = geometry.join_geometry(features, geometry.load(trajectory),
                                           geometry.load(directory / 'trajectory_3d.json'), summary)
        features['info']['sources']['depth'] = str(directory / 'ground_geometry.json')
        valid, total = summary['valid_3d_primary_frames'], summary['total_frames']
        logger.info('%s: 3D valid frames %d/%d', name, valid, total)
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
    logger.info('Pipeline started | output=%s | with_3d=%s | reused_tracking=%s | device=%s | fail_fast=%s',
                output, with_3d, reused_tracking, device, fail_fast)
    t_start = time.perf_counter()
    try:
        if processed_manifest is None:
            if skip_undistort:
                logger.info('Fisheye correction skipped (--skip-undistort)')
                input_dir = source_dir
            else:
                with stage('Fisheye correction'):
                    input_dir = run_fisheye_corrector(source_dir)
            processed_manifest = output / 'processed_video_paths.json'
            with stage('Trajectory tracking'):
                run_trajectory_tracker(input_dir, output / 'tracks', processed_manifest, device)
        summary['processed_manifest'] = str(Path(processed_manifest).resolve())

        with stage('Loading clip records'):
            records = geometry.records_from_manifest(processed_manifest)
            logger.info('Loaded %d clip record(s) from %s', len(records), processed_manifest)
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

        with stage('Background extraction + segmentation'):
            run_background_extractor_and_segmentation(records, fail_fast)

        ready = [r for r in records if 'error' not in r]
        with stage(f'Feature extraction ({len(ready)} clip(s))'):
            for i, record in enumerate(ready, 1):
                name, t0 = record['video'].name, time.perf_counter()
                logger.info('[%d/%d] %s', i, len(ready), name)
                try:
                    summary['clips'].append(run_feature_extractor(record, with_3d, lingbot_runs,
                                                                  lingbot_command_config, assumed_width))
                except Exception as error:
                    if fail_fast:
                        raise
                    logger.exception('[%d/%d] %s: feature extraction failed', i, len(ready), name)
                    mark_failed(record, '3d' if with_3d and 'tracking' in record else 'features', error)
                    continue
                logger.info('[%d/%d] %s done in %.1f s', i, len(ready), name, time.perf_counter() - t0)

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
        logger.info('Feature catalog written with %d feature(s)', len(catalog))
        summary['status'] = 'complete_with_errors' if summary['failed_clips'] else 'complete'
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        logger.exception('Pipeline failed')
        raise
    finally:
        summary['duration_s'] = round(time.perf_counter() - t_start, 1)
        geometry.save(output / 'pipeline_run.json', summary)
        log_summary(summary)
        logger.info('Run summary saved to %s (%.1f s total)', output / 'pipeline_run.json', summary['duration_s'])
    return summary


def main():
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
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Console log level (the log file always records DEBUG)')
    args = parser.parse_args()
    log_file = setup_logging(args.output.resolve() / 'logs', getattr(logging, args.log_level))
    logger.info('Log file: %s', log_file)
    try:
        result = run_pipeline(source_dir=args.source_dir, output=args.output, processed_manifest=args.processed_manifest,
            with_3d=args.with_3d, lingbot_runs=args.lingbot_runs, lingbot_command_config=args.lingbot_command_config,
            assumed_width=args.door_width_m, skip_undistort=args.skip_undistort, device=args.device,
            fail_fast=args.fail_fast, log_file=log_file)
    except Exception:
        sys.exit(1)  # already logged with traceback
    logger.info('Pipeline %s: %d clip(s) ok, %d failed; outputs: %s', result['status'],
                len(result['clips']), len(result['failed_clips']), args.output.resolve())
    sys.exit(2 if result['failed_clips'] else 0)


if __name__ == '__main__':
    main()
