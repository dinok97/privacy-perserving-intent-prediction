"""Bridge the team's trajectory/feature exports to the saved LingBot 3D stages.

Depth is reused only for a byte-identical video. A configured external command
can produce missing predictions; the pipeline never silently uses depth from an
original video for an undistorted, cropped, or re-encoded clip.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from project_paths import resolve_project_path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        result = {'name': Path(path).name, 'stem': Path(path).stem,
                  'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                  'fps': float(cap.get(cv2.CAP_PROP_FPS)), 'sha256': sha(path)}
    finally:
        cap.release()
    if min(result['width'], result['height'], result['frame_count']) <= 0 or result['fps'] <= 0:
        raise ValueError(f'Cannot read video geometry/timing: {path}')
    return result


def records_from_manifest(path):
    """Accept Dino's path list or explicit {video, trajectory, scene, pose} records."""
    path = resolve_project_path(Path(path).resolve())
    items = load(path)
    if not isinstance(items, list) or not items:
        raise ValueError('Processed-video manifest must contain a nonempty list')
    records = []
    for item in items:
        record = {'video': item} if isinstance(item, str) else dict(item)
        if not record.get('video'):
            raise ValueError('Every processed record needs a video path')
        for key in ['video', 'trajectory', 'scene', 'pose', 'background']:
            if record.get(key):
                p = Path(record[key])
                record[key] = resolve_project_path(p, base=path.parent)
        record.setdefault('trajectory', record['video'].with_suffix('.json'))
        record.setdefault('pose', record['trajectory'].with_name(record['trajectory'].stem + '_pose.json'))
        if not record['video'].is_file() or not record['trajectory'].is_file():
            raise FileNotFoundError(f'Video/trajectory pair is missing: {record}')
        records.append(record)
    return records


def adapt_trajectory(video, trajectory, output, keypoint_loader, pose=None):
    """Reuse detections, preserve frame IDs and use encoded-video time for both feature paths."""
    import cv2
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    source = load(trajectory)
    meta = source.get('meta', {})
    recorded_sha = meta.get('clip_video_sha256') or meta.get('source_video_sha256')
    if recorded_sha and recorded_sha != info['sha256']:
        raise ValueError('Trajectory provenance hash does not match the supplied clip')
    if [source['frame_width'], source['frame_height']] != [info['width'], info['height']]:
        raise ValueError('Trajectory and video use different image coordinates')
    original_frames = source['frames']
    if len(original_frames) != info['frame_count']:
        raise ValueError('Trajectory and video have different frame counts; clipping needs an explicit mapping')
    if [f['frame_index'] for f in original_frames] != list(range(info['frame_count'])):
        raise ValueError('Trajectory frame indices must exactly match the decoded clip timeline')
    cache_key = {'video_sha256': info['sha256'], 'trajectory_sha256': sha(trajectory),
                 'pose_sha256': sha(pose) if pose and Path(pose).is_file() else None,
                 'adapter_version': 2}
    if (output / 'adapter_provenance.json').is_file() and load(output / 'adapter_provenance.json') == cache_key:
        expected = [output / 'detections.json', output / 'normalized_trajectory.json', output / 'yolo_tracking.mp4']
        expected += [output / 'yolo_frames' / f'frame_{i:06d}.jpg' for i in range(info['frame_count'])]
        if all(p.is_file() for p in expected):
            return output / 'normalized_trajectory.json'
    keypoints = keypoint_loader(source, load(pose) if pose and Path(pose).is_file() else None,
                                info['width'], info['height'])
    tid = int(source.get('id', 1))
    frames, normalized = [], []
    for i, old in enumerate(original_frames):
        timestamp = i / info['fps']
        normalized.append({**old, 'original_t': old.get('t'), 't': timestamp})
        bbox = old.get('bbox')
        centre = old.get('center')
        if centre is None and bbox is not None:
            centre = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        detections = []
        if bbox is not None and centre is not None:
            kp = keypoints[i]
            detections.append({'track_id': tid, 'bbox': bbox, 'center': centre,
                               'keypoints': np.asarray(kp).tolist() if kp is not None else None,
                               'confidence': old.get('confidence')})
        frames.append({'frame_index': i, 't': timestamp, 'detections': detections})
    metadata = {'source_video': info['name'], 'source_video_sha256': info['sha256'],
        'fps': info['fps'], 'frame_width': info['width'], 'frame_height': info['height'],
        'frame_count': info['frame_count'], 'track_ids': [tid],
        'track_frame_counts': {str(tid): sum(bool(f['detections']) for f in frames)},
        'tracker': meta.get('tracker'), 'model': meta.get('model'), 'yolo_rerun': False,
        'time_definition': 'frame_index / encoded-video FPS; original trajectory times retained separately',
        'trajectory_provenance': 'source_hash_verified' if recorded_sha else 'legacy_dimensions_and_frame_count_only',
        'pixel_coordinates': 'Supplied processed clip, unchanged; no additional lens correction',
        'adapter_provenance': cache_key}
    save(output / 'detections.json', {'metadata': metadata, 'frames': frames})
    save(output / 'normalized_trajectory.json', {**source, 'fps': info['fps'], 'frames': normalized,
        'meta': {**meta, 'clip_video_sha256': info['sha256'], 'time_basis': 'encoded_video',
                 'original_fps': source.get('fps'), 'original_effective_fps': meta.get('effective_fps'),
                 'effective_fps': info['fps'], 'record_fps': info['fps']}})
    images = output / 'yolo_frames'
    images.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(output / 'yolo_tracking.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             info['fps'], (info['width'], info['height']))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Could not create the reused-tracking preview')
    try:
        for frame in frames:
            ok, image = cap.read()
            if not ok:
                raise ValueError('Decoded video ended before the saved trajectory')
            for detection in frame['detections']:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(image, f'Reused track {tid}', (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 100, 0), 1)
                for x, y, confidence in detection['keypoints'] or []:
                    if confidence >= .5:
                        cv2.circle(image, (int(x), int(y)), 2, (0, 220, 220), -1)
            writer.write(image)
            cv2.imwrite(str(images / f'frame_{frame["frame_index"]:06d}.jpg'), image)
    finally:
        cap.release()
        writer.release()
    save(output / 'adapter_provenance.json', cache_key)
    return output / 'normalized_trajectory.json'


def select_depth_run(runs_root, info):
    from .background_depth import source_info
    root = Path(runs_root)
    manifests = [root] if root.is_file() else sorted(root.rglob('processing_manifest.json'))
    for path in manifests:
        manifest = load(path)
        source = source_info(manifest)
        if source.get('sha256') != info['sha256']:
            continue
        if manifest.get('extrinsic_convention', 'world_to_camera') != 'world_to_camera':
            raise ValueError('LingBot exports must use world-to-camera extrinsics')
        if source['frame_count'] != info['frame_count'] or not math.isclose(source['fps'], info['fps'], rel_tol=1e-4):
            raise ValueError('LingBot provenance matches bytes but declares different video timing')
        if source.get('width', info['width']) != info['width'] or source.get('height', info['height']) != info['height']:
            raise ValueError('LingBot provenance declares different image dimensions')
        return path
    return None


def ensure_depth_run(video, info, runs_root, workspace, command_config=None):
    found = select_depth_run(runs_root, info)
    local_manifest = Path(workspace) / 'lingbot_run/processing_manifest.json'
    if found is None and local_manifest.is_file():
        found = select_depth_run(local_manifest, info)
    if found:
        return found
    if command_config is None:
        raise ValueError(f'No LingBot predictions match the exact clip {video}. '
                         'Use --lingbot-runs with matching exports or --lingbot-config. '
                         'Original-video depth cannot be substituted for a processed/undistorted clip.')
    config_path = Path(command_config).resolve()
    config = load(config_path)
    command = config.get('command')
    if 'runtime' not in config and (not isinstance(command, list) or not command or not all(isinstance(a, str) for a in command)):
        raise ValueError('LingBot command config needs a nonempty command array; shell strings are not accepted')
    run = Path(workspace) / 'lingbot_run'
    run.mkdir(parents=True, exist_ok=True)
    values = {'video': str(Path(video).resolve()), 'output': str(run.resolve()),
              'stem': info['stem'], 'predictions': str((run / info['stem']).resolve())}
    # Replace only our tokens; literal braces in Python/JSON arguments are not templates.
    argv = list(command or [])
    for key, value in values.items():
        argv = [arg.replace('{' + key + '}', value) for arg in argv]
    cwd = config.get('cwd')
    if cwd is not None:
        cwd = (config_path.parent / cwd).resolve()
    with (run / 'inference.log').open('w', encoding='utf-8') as log:
        try:
            if 'runtime' in config:
                from . import lingbot_runtime
                lingbot_runtime.run(config, config_path, video, run, log)
            else:
                subprocess.run(argv, cwd=cwd, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            marker = run / 'processing_manifest.json'
            if marker.is_file():
                marker.replace(run / 'provider_manifest.json')
            raise RuntimeError(f'LingBot inference failed; see {run / "inference.log"}') from error
    # A provider completion marker becomes a cache entry only after our checks.
    manifest = run / 'processing_manifest.json'
    existing = load(manifest) if manifest.is_file() else {}
    if manifest.is_file():
        manifest.replace(run / 'provider_manifest.json')
    predictions = run / info['stem']
    expected = [f'frame_{i:06d}.npz' for i in range(info['frame_count'])]
    if sorted(p.name for p in predictions.glob('frame_*.npz')) != expected:
        raise ValueError('Configured LingBot command must export every frame into {predictions}/frame_XXXXXX.npz')
    import numpy as np
    for name in expected:
        with np.load(predictions / name, allow_pickle=False) as data:
            required = {'depth', 'depth_conf', 'images', 'intrinsic', 'extrinsic'}
            if not required.issubset(data.files) or not all(np.isfinite(data[k]).all() for k in required):
                raise ValueError(f'Incomplete or nonfinite prediction arrays: {name}')
            h, w, channels = data['depth'].shape
            if (channels != 1 or data['depth_conf'].shape != (h, w) or data['images'].shape != (3, h, w)
                    or data['intrinsic'].shape != (3, 3) or data['extrinsic'].shape != (3, 4)
                    or not (data['depth'] > 0).all()):
                raise ValueError(f'Invalid prediction geometry: {name}')
    # The orchestrator records the exact command input, not guessed filenames from another run.
    provider_source = existing.get('source', {})
    for field in ('sha256', 'width', 'height', 'frame_count'):
        if field in provider_source and provider_source[field] != info[field]:
            raise ValueError(f'Provider wrote provenance for a different source video ({field})')
    if 'fps' in provider_source and not math.isclose(provider_source['fps'], info['fps'], rel_tol=1e-4):
        raise ValueError('Provider wrote provenance for different video timing')
    if existing.get('extrinsic_convention', 'world_to_camera') != 'world_to_camera':
        raise ValueError('LingBot exports must use world-to-camera extrinsics')
    save(manifest, {**existing, 'source': info, 'provider_config_sha256': sha(config_path),
                    'all_source_frames_processed': True})
    return run / 'processing_manifest.json'


def run_geometry(record, runs_root, command_config=None, assumed_width=.825):
    from . import background_depth, segment_background_3d, door_distance_3d
    video = record['video']
    info = record['video_info']
    workspace = record['workspace'] / 'geometry_3d'
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = ensure_depth_run(video, info, runs_root, workspace, command_config)
    depth_source = background_depth.source_info(load(manifest))
    scene = load(record['scene'])
    if scene['image_size'] != [info['width'], info['height']]:
        raise ValueError('Scene polygons and processed video use different image coordinates')
    if scene.get('source_video_sha256') not in (None, info['sha256']):
        raise ValueError('Scene segmentation belongs to a different video')
    args = SimpleNamespace(relative_tolerance=.10, min_inlier_fraction=.80, min_valid_fraction=.90)
    with (workspace / 'postprocess.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        background_depth.process_run(manifest, workspace / 'depth', args)
        segments = workspace / 'segments'
        segment_background_3d.process_scene(record['scene'], workspace / 'depth', segments,
            video_stem=depth_source['stem'], allow_undistorted=True)
        output = workspace / 'distances'
        summary = door_distance_3d.main(['--background', str(segments / depth_source['stem']),
            '--tracking', str(record['tracking']), '--video', str(video), '--output', str(output),
            '--door-width-m', str(assumed_width)])
    return output, summary


def join_geometry(features, trajectory, geometry, summary):
    """Join by frame ID, retaining unavailable rows and keeping 2D exit IDs separate."""
    tid = int(trajectory.get('id', geometry['primary_track_id']))
    rows = [r for r in geometry['frames'] if r['track_id'] == tid]
    by_frame = {r['frame_index']: r for r in rows}
    target = features['data']['frames']
    if (len(by_frame) != len(rows) or len(target) != len(rows)
            or {r['frame_index'] for r in target} != set(by_frame)):
        raise ValueError('2D and 3D exports have different or duplicate frame IDs')
    if features['info'].get('clip_video_sha256', summary['source_video_sha256']) != summary['source_video_sha256']:
        raise ValueError('2D and 3D features belong to different video bytes')
    for frame in target:
        row = by_frame[frame['frame_index']]
        if not math.isclose(frame['t'], row['t'], abs_tol=.001):
            raise ValueError('2D and 3D exports have different video times')
        frame['geometry_3d'] = {'available': row['valid_3d'], 'contact_source': row['contact_source'],
            'contact_confidence': row['contact_confidence'], 'point_ground_model': row['point_ground'],
            'point_ground_estimated_m': row['point_ground_estimated_m'],
            'distances_model': row['distances'], 'distances_estimated_m': row['distances_estimated_m'],
            'closure_rates_model_per_s': row['closure_rates'],
            'closure_rates_estimated_m_per_s': row['closure_rates_estimated_m_per_s']}
    features['info']['depth'] = {'provider': 'LingBot', 'source_video_sha256': summary['source_video_sha256'],
        'geometry_extension_version': '1.0', 'metric_calibration': summary['estimated_metric_calibration'],
        'causal': False, 'scene_geometry_uses_whole_clip': True,
        'door_id_scope': '1-based object instance IDs in the input segmentation; separate from 2D exit order',
        'main_entrance': 'bottom-centre floor reference, not a measured doorway threshold'}
    features['info'].setdefault('units', {}).update({'_model': 'LingBot model units',
        '_estimated_m': 'metres estimated from assumed door-leaf width',
        '_estimated_m_per_s': 'estimated metres per source-video second'})
    features['info']['notes'] = ('2D motion features use current/past tracks given a fixed scene calibration. '
        'geometry_3d uses whole-clip depth, background and camera estimates; it is NOT causal '
        'and needs a prior independent scene calibration before online/TTE benchmarking. '
        'Track hindsight summaries remain unsuitable for online prediction.')
    features['data']['scene_summary']['geometry_3d'] = {'door_ids': list(rows[0]['distances']) if rows else [],
        'valid_frames': sum(r['valid_3d'] for r in rows), 'total_frames': len(rows),
        'metric_scale_is_assumed': True}
    return features


def catalog_entries():
    return [{'name': 'geometry_3d.' + name, 'level': 'frame', 'group': 'geometry_3d',
        'unit': unit, 'needs': ['trajectory', 'pose', 'scene', 'LingBot depth'], 'causal': False,
        'description': description + '; scene/camera estimates use the whole clip.'}
        for name, unit, description in [
            ('available', 'boolean', 'Whether an estimated ground contact is available'),
            ('point_ground_model', 'model', 'Fitted-floor person position'),
            ('point_ground_estimated_m', 'estimated_m', 'Assumption-scaled person position'),
            ('distances_model.*', 'model', 'Distance to each segmented panel reference and main entrance reference'),
            ('distances_estimated_m.*', 'estimated_m', 'Assumption-scaled per-door distances'),
            ('closure_rates_model_per_s.*', 'model/s', 'Negative means approaching; missing contacts remain unavailable'),
            ('closure_rates_estimated_m_per_s.*', 'estimated_m/s', 'Assumption-scaled per-door closure rates')]]
