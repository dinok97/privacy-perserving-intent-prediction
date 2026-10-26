"""Video -> tracking -> segmentation -> 2D features, optionally extended with 3D.

From the repository root:
  python video_processing/video_processing/pipeline.py --source-dir INPUT --with-3d --lingbot-runs RUNS
  python video_processing/video_processing/pipeline.py --processed-manifest CLIPS.json --with-3d --lingbot-runs RUNS

Processed manifests accept Dino's clip-path list or {video, trajectory, scene,
pose, background} records. Paths are relative to the manifest. This mode reuses
YOLO detections; no tracking model is loaded. Optional scene paths reuse SAM3.

For missing depth, --lingbot-config accepts a local or ssh-docker runtime JSON
(see lingbot_runtime.py). LingBot stays in its own checkout/environment. The
legacy --lingbot-command-config alias also supports custom command argv/cwd.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
import pipeline_geometry as geometry

SOURCE_DIR = PROJECT / 'source'


def run_fisheye_corrector(source_dir):
    import fisheye_correction
    videos = sorted(str(p) for p in Path(source_dir).iterdir()
                    if p.is_file() and p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.m4v'})
    if not videos:
        raise ValueError(f'No input videos found in {source_dir}')
    fisheye_correction.correct_videos(videos)
    return Path(source_dir) / '[TEMP]_undistorted_results'


def run_trajectory_tracker(source_dir, output, manifest, device):
    import video_processor
    return video_processor.run(source_dir=source_dir, output_root=output,
                               processed_manifest=manifest, device=device)


def run_background_extractor_and_segmentation(records):
    import background_extraction
    pending = []
    for record in records:
        video, work = record['video'], record['workspace']
        scene = record.get('scene')
        sibling_scene = video.with_name(video.stem + '_background.json')
        if scene is None:
            scene = sibling_scene if sibling_scene.is_file() else work / (video.stem + '_background.json')
        record['scene'] = scene
        background = record.get('background') or work / (video.stem + '_background.png')
        record['background'] = background
        if not background.is_file():
            import cv2
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
            if parsed['image_size'] != size or parsed.get('source_video_sha256') not in (None, record['video_info']['sha256']):
                raise ValueError(f'Scene segmentation does not match this video geometry: {scene}')
    if pending:
        import background_segmentation
        background_segmentation.segment_images(pending)
        for record in records:
            if str(record['background']) in pending:
                scene = geometry.load(record['scene'])
                scene['source_video_sha256'] = record['video_info']['sha256']
                geometry.save(record['scene'], scene)


def run_feature_extractor(record, with_3d=False, runs_root=None, command_config=None, assumed_width=.825):
    import feature_extractor
    trajectory = record['trajectory']
    if with_3d:
        record['tracking'] = record['workspace'] / 'geometry_3d/distances/tracking'
        trajectory = geometry.adapt_trajectory(record['video'], trajectory, record['tracking'],
                                               feature_extractor.load_keypoints, record.get('pose'))
    out = record['workspace'] / f'features_{record["video"].stem}.json'
    feature_extractor.run_clip(trajectory, record['scene'], record.get('pose'),
                              record['background'], out, debug=True)
    features = geometry.load(out)
    features['info']['clip_video'] = record['video'].name
    features['info']['clip_video_sha256'] = record['video_info']['sha256']
    result = {'video': str(record['video']), 'features': str(out),
              'source_video_sha256': record['video_info']['sha256'], 'geometry_3d': None}
    if with_3d:
        directory, summary = geometry.run_geometry(record, runs_root, command_config, assumed_width)
        features = geometry.join_geometry(features, geometry.load(trajectory),
                                           geometry.load(directory / 'trajectory_3d.json'), summary)
        features['info']['sources']['depth'] = str(directory / 'ground_geometry.json')
        result['geometry_3d'] = {'directory': str(directory), 'viewer': str(directory / 'viewer.html'),
                                  'valid_frames': summary['valid_3d_primary_frames'],
                                  'total_frames': summary['total_frames'], 'yolo_rerun': False}
    geometry.save(out, features)
    return result


def run_pipeline(*, source_dir=SOURCE_DIR, output=PROJECT / 'data/pipeline', processed_manifest=None,
                 with_3d=False, lingbot_runs=PROJECT / 'data/lingbot', lingbot_command_config=None,
                 assumed_width=.825, skip_undistort=False, device='cpu'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir).resolve()
    reused_tracking = processed_manifest is not None
    if processed_manifest is None:
        input_dir = source_dir if skip_undistort else run_fisheye_corrector(source_dir)
        processed_manifest = output / 'processed_video_paths.json'
        run_trajectory_tracker(input_dir, output / 'tracks', processed_manifest, device)
    records = geometry.records_from_manifest(processed_manifest)
    seen = set()
    for record in records:
        info = geometry.video_info(record['video'])
        key = (str(record['video']), str(record['trajectory']))
        if key in seen:
            raise ValueError('Duplicate video/trajectory record in processed manifest')
        seen.add(key)
        record['workspace'] = output / (record['video'].stem + '-' + info['sha256'][:10] + '-' + geometry.sha(record['trajectory'])[:8])
        record['workspace'].mkdir(parents=True, exist_ok=True)
        record['video_info'] = info
    summary = {'status': 'running', 'with_3d': with_3d, 'reused_tracking': reused_tracking,
               'processed_manifest': str(Path(processed_manifest).resolve()), 'clips': []}
    try:
        run_background_extractor_and_segmentation(records)
        for record in records:
            summary['clips'].append(run_feature_extractor(record, with_3d, lingbot_runs,
                                                          lingbot_command_config, assumed_width))
        import feature_extractor
        catalog = list(feature_extractor.CATALOG)
        if with_3d:
            catalog.extend(geometry.catalog_entries())
        geometry.save(output / 'feature_catalog.json', {'schema_version': feature_extractor.SCHEMA_VERSION,
            'extensions': {'geometry_3d': '1.0'} if with_3d else {}, 'features': catalog})
        summary['status'] = 'complete'
    except Exception as error:
        summary.update(status='failed', error=str(error))
        raise
    finally:
        geometry.save(output / 'pipeline_run.json', summary)
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
    args = parser.parse_args()
    result = run_pipeline(source_dir=args.source_dir, output=args.output, processed_manifest=args.processed_manifest,
        with_3d=args.with_3d, lingbot_runs=args.lingbot_runs, lingbot_command_config=args.lingbot_command_config,
        assumed_width=args.door_width_m, skip_undistort=args.skip_undistort, device=args.device)
    print(f'Pipeline complete: {len(result["clips"])} clip(s); outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
