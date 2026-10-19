"""Lift saved YOLO floor contacts into LingBot geometry and measure door distances.

No inference is performed here. All coordinates and distances are model units.
The bottom-centre entrance is an image-reference point, not a measured threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEMANTIC_COLOURS = np.array([[140, 148, 158], [75, 166, 235], [235, 183, 59],
                             [231, 83, 91], [150, 108, 225]], dtype=np.uint8)
DOOR_COLOURS = ['#c62e78', '#7e4cc7', '#178750', '#166ac0']
DOOR_WIDTH_SOURCE = 'https://www.swedoor.se/tips-och-rad/mattabeller-innerdorrar'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def robust_floor(points, seed=19):
    """RANSAC followed by an SVD fit; deterministic, with residuals retained."""
    if len(points) < 3:
        raise ValueError('At least three stable, floor-labelled 3D points are required')
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(5000, len(points)), replace=False)]
    tolerance = float(0.02 * np.linalg.norm(np.ptp(sample, axis=0)))
    best_count, normal, offset = 0, None, None
    for _ in range(300):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        candidate = np.cross(b - a, c - a)
        length = np.linalg.norm(candidate)
        if length < 1e-10:
            continue
        candidate /= length
        d = -candidate @ a
        count = int((np.abs(sample @ candidate + d) < tolerance).sum())
        if count > best_count:
            best_count, normal, offset = count, candidate, d
    if normal is None:
        raise ValueError('Insufficient non-collinear floor points')
    for _ in range(5):
        inliers = points[np.abs(points @ normal + offset) < tolerance]
        centre = inliers.mean(axis=0)
        normal = np.linalg.svd(inliers - centre, full_matrices=False)[2][-1]
        offset = float(-normal @ centre)
    if offset < 0:
        normal, offset = -normal, -offset
    residuals = np.abs(points @ normal + offset)
    return normal, offset, {'normal_camera': normal.tolist(), 'offset_camera': offset,
        'equation': 'normal dot X_camera + offset = 0', 'seed': seed, 'ransac_trials': 300,
        'inlier_tolerance_model_units': tolerance, 'input_points': len(points),
        'inlier_fraction': float((residuals < tolerance).mean()),
        'absolute_residual_model_units_p50_p90_p99': np.quantile(residuals, [.5, .9, .99]).tolist()}


def lift(uv, intrinsic, normal, offset):
    ray = np.linalg.solve(intrinsic, np.r_[uv, 1.0])
    denominator = float(normal @ ray)
    depth = -offset / denominator if abs(denominator) > 1e-9 else -1
    return ray * depth if np.isfinite(depth) and depth > 0 else None


def image_point(point, intrinsic):
    p = intrinsic @ point
    return p[:2] / p[2]


def nearest_on_segment(point, segment):
    a, b = np.asarray(segment)
    ab = b - a
    t = np.clip((point - a) @ ab / max(float(ab @ ab), 1e-12), 0, 1)
    return a + t * ab


def metric_calibration(doors, assumed_width):
    """One global scale from assumed door-leaf widths, with cropped support explicit."""
    if not np.isfinite(assumed_width) or assumed_width <= 0:
        raise ValueError('Assumed door-leaf width must be finite and positive')
    candidates = []
    for door in doors:
        if door['primary']:
            continue
        segment = np.asarray(door['segment_camera'])
        width = float(np.linalg.norm(segment[1] - segment[0]))
        if not np.isfinite(width) or width <= 1e-8:
            raise ValueError(f'{door["id"]}: invalid reconstructed width')
        candidates.append({'door_id': door['id'], 'name': door['name'],
            'width_model_units': width, 'boundary_cropped': door['boundary_cropped'],
            'included_by_default': not door['boundary_cropped'],
            'implied_metres_per_model_unit': float(assumed_width / width)})
    usable = [c for c in candidates if c['included_by_default']]
    if not usable:
        raise ValueError('No uncropped door is available for the default metric scale')
    factors = [c['implied_metres_per_model_unit'] for c in usable]
    factor = float(np.median(factors))
    return {'status': 'assumption_based_estimated_metres_not_measured_calibration',
        'assumed_door_leaf_width_m': float(assumed_width),
        'metres_per_model_unit': factor, 'default_reference': 'uncropped_median',
        'reference_door_ids': [c['door_id'] for c in usable], 'candidates': candidates,
        'method': 'Median of assumed physical leaf width / reconstructed floor-level panel span '
                  'for uncropped doors; apply this one factor to every 3D coordinate, distance and rate.',
        'width_estimate': 'The existing door segment spans the 1st–99th percentiles of the '
                          'visible panel cloud along its fitted floor-intersection direction.',
        'reference_factor_range': [float(min(factors)), float(max(factors))],
        'reference_factor_spread_fraction_of_median': float((max(factors) - min(factors)) / factor),
        'factor_range_interpretation': 'Disagreement between door-based scales; not a confidence interval.',
        'width_reference': {'source': DOOR_WIDTH_SOURCE, 'accessed_date': '2026-10-19',
            'door_leaf_widths_m': {'M8': .725, 'M9': .825, 'M10': .925},
            'note': 'Module, frame, clear opening and door-leaf width are different measurements. '
                    'The leaf width is used because segmentation identifies door panels. '
                    'The actual sizes in this video are unknown.'},
        'limitations': ['A standard-size assumption does not establish the true door width.',
            'Uniform scale does not correct reconstruction shape, perspective, pose or door-reference errors.',
            'A cropped panel gives a partial-width estimate and is excluded from the default reference set.']}


def build_doors(scene, xyz, stable, instance_ids, intrinsic, normal, offset,
                source_width, source_height):
    height, width = stable.shape
    # Exactly the same pixel-centre mapping as the existing segmentation export.
    main_source = np.array([source_width // 2, source_height - 1], dtype=float)
    main_uv = (main_source + .5) * [width / source_width, height / source_height] - .5
    main_point = lift(main_uv, intrinsic, normal, offset)
    if main_point is None:
        raise ValueError('Bottom-centre reference ray does not meet the floor in front of camera')
    doors = [{'id': 'main_bottom_centre', 'name': 'Main entrance (bottom-centre reference)',
        'colour': '#d26900', 'kind': 'reference_point', 'primary': True,
        'source_pixel': main_source.tolist(), 'depth_grid_pixel': main_uv.tolist(),
        'point_camera': main_point.tolist(), 'segment_camera': [main_point.tolist()] * 2,
        'status': 'image_reference_only_not_a_measured_doorway',
        'interpretation': 'Distance to the floor hit of the original bottom-centre pixel. '
                          'Physical doorway position, width and crossing time are not known.'}]
    for obj in scene['objects']:
        if obj['label'] != 'door':
            continue
        instance = obj['instance_id']
        points = xyz[(instance_ids == instance) & stable]
        if len(points) < 20:
            raise ValueError(f'Door {instance}: insufficient stable 3D support')
        centre = points.mean(axis=0)
        door_normal = np.linalg.svd(points - centre, full_matrices=False)[2][-1]
        door_offset = float(-door_normal @ centre)
        direction = np.cross(normal, door_normal)
        if np.linalg.norm(direction) < .1:
            raise ValueError(f'Door {instance}: door and floor planes nearly parallel')
        direction /= np.linalg.norm(direction)
        origin = np.linalg.lstsq(np.vstack([normal, door_normal]),
                                [-offset, -door_offset], rcond=None)[0]
        low, high = np.quantile((points - origin) @ direction, [.01, .99])
        segment = origin + np.array([low, high])[:, None] * direction
        vertices = np.asarray(obj['polygon'])
        cropped = bool(np.any(vertices[:, 0] <= .5) or np.any(vertices[:, 0] >= width - 1.5)
                       or np.any(vertices[:, 1] <= .5) or np.any(vertices[:, 1] >= height - 1.5))
        name = f'Door {instance}' + (' (cropped)' if cropped else '')
        doors.append({'id': f'door_{instance}', 'name': name, 'primary': False,
            'colour': DOOR_COLOURS[(len(doors) - 1) % len(DOOR_COLOURS)],
            'kind': 'estimated_floor_segment', 'segmentation_instance_id': instance,
            'source_polygon': obj['polygon'], 'segment_camera': segment.tolist(),
            'segment_projected_depth_pixels': [image_point(p, intrinsic).tolist() for p in segment],
            'point_camera': segment.mean(axis=0).tolist(), 'boundary_cropped': cropped,
            'door_plane_normal_camera': door_normal.tolist(), 'door_plane_offset_camera': door_offset,
            'door_plane_residual_p50_p90': np.quantile(abs((points - centre) @ door_normal), [.5, .9]).tolist(),
            'stable_support_points': len(points),
            'status': 'proxy_from_visible_door_panel_not_surveyed_opening_threshold',
            'interpretation': 'Intersection of fitted door-panel and floor planes; endpoints '
                              'span the 1st–99th percentiles of visible support along the line. '
                              'Open leaves and cropped doors can differ from the real threshold.'})
    return doors, main_point


def contact(det, width, height):
    """Use observed ankles, and avoid silently treating cropped legs as ground contacts."""
    keypoints = det.get('keypoints') or []
    ankles = [p for p in keypoints[15:17] if p[2] >= .5 and
              2 <= p[0] < width - 2 and 2 <= p[1] < height - 3]
    if len(ankles) == 2:
        return np.mean(np.asarray(ankles)[:, :2], axis=0), 'ankle_midpoint', float(min(p[2] for p in ankles))
    if len(ankles) == 1:
        return np.asarray(ankles[0][:2]), 'single_visible_ankle', float(ankles[0][2])
    x1, y1, x2, y2 = det['bbox']
    if y2 >= height - 3:
        return None, 'feet_cropped_or_unobserved_at_bottom', None
    return np.array([(x1 + x2) / 2, y2]), 'bbox_bottom_proxy', float(det['confidence'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background', type=Path, required=True)
    parser.add_argument('--tracking', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path,
                        help='Original video for hash verification; defaults to videos/<source_video>')
    parser.add_argument('--depth-metadata', type=Path,
                        help='Original background-depth metadata.json for older segmentation exports without a source hash')
    parser.add_argument('--door-width-m', type=float, default=.825,
                        help='Assumed physical door-leaf width for estimated metres (default: M9, 0.825 m)')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tracking = json.loads((args.tracking / 'detections.json').read_text())
    meta = tracking['metadata']
    sw, sh = meta['frame_width'], meta['frame_height']
    bgmeta = json.loads((args.background / 'metadata.json').read_text())
    assert meta['source_video'] == bgmeta['video']
    video = args.video or HERE / 'videos' / meta['source_video']
    if sha(video) != meta['source_video_sha256']:
        raise ValueError('Original video hash does not match the saved YOLO detections')
    background_video_hash = bgmeta.get('source_video_sha256')
    if background_video_hash is None:
        if args.depth_metadata is None:
            parser.error('Older background export: provide --depth-metadata from its background-depth run')
        depth_meta = json.loads(args.depth_metadata.read_text())
        background_video_hash = depth_meta['source_video_sha256']
        if sha(args.depth_metadata.parent / 'background_depth.npy') != bgmeta['background_depth_sha256']:
            raise ValueError('Background-depth metadata does not refer to this segmentation depth')
    if meta['source_video_sha256'] != background_video_hash:
        raise ValueError('YOLO and depth estimates were produced from different source videos')
    if not meta['track_ids']:
        raise ValueError('The saved YOLO run contains no tracked people')
    scene = json.loads((args.background / 'aligned_scene.json').read_text())
    with np.load(args.background / 'segmented_background.npz', allow_pickle=False) as a:
        arrays = {name: a[name].copy() for name in a.files}
    depth, labels, xyz = arrays['depth'], arrays['class_id'], arrays['xyz_camera'].astype(float)
    stable = arrays['stable_mask'].astype(bool) & np.isfinite(xyz).all(axis=-1)
    intrinsic = arrays['intrinsic']
    normal, offset, floor = robust_floor(xyz[(labels == 1) & stable])
    doors, origin = build_doors(scene, xyz, stable, arrays['instance_id'], intrinsic,
                                normal, offset, sw, sh)
    right = np.array([1., 0, 0]) - normal[0] * normal
    right /= np.linalg.norm(right)
    forward = np.cross(normal, right)
    basis = np.stack([right, forward, normal], axis=1)
    for door in doors:
        door['segment_ground'] = ((np.asarray(door['segment_camera']) - origin) @ basis).tolist()
    ids = meta['track_ids']
    primary_id = max(ids, key=lambda tid: meta['track_frame_counts'][str(tid)])
    rows, image_errors = [], []
    h, w = depth.shape
    for frame in tracking['frames']:
        for tid in ids:
            det = next((d for d in frame['detections'] if d['track_id'] == tid), None)
            row = {'frame_index': frame['frame_index'], 't': frame['t'], 'track_id': tid,
                   'detected': det is not None, 'valid_3d': False, 'contact_source': 'not_detected',
                   'contact_confidence': None, 'contact_pixel': None, 'contact_depth_pixel': None,
                   'point_camera': None, 'point_world': None, 'point_ground': None,
                   'contact_on_floor_label': None, 'contact_background_stable': None,
                   'distances': {}, 'closure_rates': {}, 'distance_2d_main_bbox_norm': None}
            if det:
                cx, cy = det['center']
                row['distance_2d_main_bbox_norm'] = float(np.linalg.norm(
                    [(cx - sw // 2) / sw, (cy - (sh - 1)) / sh]))
                uv, source, confidence = contact(det, sw, sh)
                row.update(contact_source=source, contact_confidence=confidence)
                if uv is not None:
                    matched_uv = (uv + .5) * [w / sw, h / sh] - .5
                    point = lift(matched_uv, intrinsic, normal, offset)
                    row.update(contact_pixel=uv.tolist(), contact_depth_pixel=matched_uv.tolist())
                    if point is not None:
                        pixel = np.clip(np.rint(matched_uv).astype(int), [0, 0], [w - 1, h - 1])
                        rotation = arrays['reference_extrinsic'][:, :3]
                        translation = arrays['reference_extrinsic'][:, 3]
                        row.update(valid_3d=True, point_camera=point.tolist(),
                                   point_world=((point - translation) @ rotation).tolist(),
                                   point_ground=((point - origin) @ basis).tolist(),
                                   contact_on_floor_label=bool(labels[pixel[1], pixel[0]] == 1),
                                   contact_background_stable=bool(stable[pixel[1], pixel[0]]))
                        image_errors.append(float(np.linalg.norm(image_point(point, intrinsic) - matched_uv)))
                        for door in doors:
                            closest = nearest_on_segment(point, door['segment_camera'])
                            row['distances'][door['id']] = float(np.linalg.norm(point - closest))
                    else:
                        row['contact_source'] += '_ray_failed'
            for door in doors:
                row['distances'].setdefault(door['id'], None)
            previous = next((p for p in reversed(rows) if p['track_id'] == tid), None)
            for door in doors:
                rate = None
                if previous and row['valid_3d'] and previous['valid_3d']:
                    rate = ((row['distances'][door['id']] - previous['distances'][door['id']]) /
                            (row['t'] - previous['t']))
                row['closure_rates'][door['id']] = rate
            rows.append(row)
    calibration = metric_calibration(doors, args.door_width_m)
    factor = calibration['metres_per_model_unit']
    for row in rows:
        row['distances_estimated_m'] = {k: v * factor if v is not None else None
                                        for k, v in row['distances'].items()}
        row['closure_rates_estimated_m_per_s'] = {k: v * factor if v is not None else None
                                                  for k, v in row['closure_rates'].items()}
        row['point_ground_estimated_m'] = ([v * factor for v in row['point_ground']]
                                            if row['point_ground'] is not None else None)
    dump(out / 'metric_calibration.json', calibration)
    geometry = {'units': 'LingBot model units, not pixels or calibrated metres',
        'estimated_metric_calibration_file': 'metric_calibration.json',
        'floor': floor, 'ground_origin_camera': origin.tolist(), 'ground_basis_camera_columns': basis.tolist(),
        'ground_axes': ['right along fitted floor', 'forward along fitted floor', 'fitted floor normal'],
        'source_to_depth_pixel_transform': '(uv+0.5)*[depth_width/source_width,depth_height/source_height]-0.5',
        'intrinsic': intrinsic.tolist(), 'reference_extrinsic': arrays['reference_extrinsic'].tolist(),
        'doors': doors}
    dump(out / 'ground_geometry.json', geometry)
    dump(out / 'trajectory_3d.json', {'source': meta, 'primary_track_id': primary_id,
                                    'geometry_file': 'ground_geometry.json',
                                    'estimated_metric_calibration': calibration, 'frames': rows})
    csv_file = export_csv(out, rows, doors, calibration)
    render_frames(out, args.tracking, rows, doors, sw, sh, w, h)
    render_figures(out, rows, doors, xyz, labels, stable, origin, basis, normal, offset, primary_id)
    render_figures(out, rows, doors, xyz, labels, stable, origin, basis, normal, offset, primary_id,
                   scale=factor, unit='estimated m', suffix='_estimated_metres',
                   assumed_width=args.door_width_m)
    write_viewer(out, rows, doors, xyz, labels, stable, origin, basis, meta, primary_id, calibration, csv_file)
    valid = [r for r in rows if r['valid_3d'] and r['track_id'] == primary_id]
    if not valid:
        raise ValueError('No valid ground contacts')
    # Verify floor constraints, ray reprojection and actual saved distances independently.
    for r in valid:
        assert abs(np.dot(normal, r['point_camera']) + offset) < 1e-8
        assert abs(r['point_ground'][2]) < 1e-8
        for door in doors:
            q = np.asarray(door['segment_camera'])
            p = np.asarray(r['point_camera'])
            # Independent endpoint/interior candidate solution for the segment minimum.
            length = np.linalg.norm(q[1] - q[0])
            expected = min(np.linalg.norm(p - q[0]), np.linalg.norm(p - q[1]))
            if length > 1e-10:
                unit = (q[1] - q[0]) / length
                along = (p - q[0]) @ unit
                if 0 <= along <= length:
                    expected = np.linalg.norm(np.cross(p - q[0], unit))
            assert abs(expected - r['distances'][door['id']]) < 1e-8
    assert len(rows) == meta['frame_count'] * len(ids)
    assert max(image_errors) < 1e-6
    summaries = {d['id']: {'name': d['name'], 'first': valid[0]['distances'][d['id']],
        'last': valid[-1]['distances'][d['id']],
        'minimum': min(r['distances'][d['id']] for r in valid),
        'maximum': max(r['distances'][d['id']] for r in valid)} for d in doors}
    limitations = [
        'Depth and camera estimates are not physically calibrated; model scale is clip-specific.',
        'Estimated metres assume the selected door-leaf width; one global scale factor does not establish ground truth.',
        'Main entrance is the floor hit of the bottom-centre image reference, not a measured physical doorway.',
        'Visible-door segments come from door-panel/floor plane intersections, not surveyed opening thresholds.',
        'Door leaves can be open and one visible door is cropped; target locations are approximations.',
        'A single fitted floor plane approximates the curved LingBot floor geometry.',
        'Ankle keypoints approximate ground contact; ankle height, pose errors and fisheye distortion remain.',
        'No ground point or distance is fabricated when the feet are cropped or the person is absent.',
        'Distance is straight-line floor proximity, not navigable route length or validated intent.',
    ]
    summary = {'video': meta['source_video'], 'primary_track_id': primary_id,
        'total_frames': meta['frame_count'], 'detected_primary_frames': meta['track_frame_counts'][str(primary_id)],
        'valid_3d_primary_frames': len(valid), 'first_valid_frame': valid[0]['frame_index'],
        'last_valid_frame': valid[-1]['frame_index'], 'fps': meta['fps'],
        'contact_source_counts': {s: sum(r['contact_source'] == s for r in rows)
                                  for s in sorted({r['contact_source'] for r in rows})},
        'door_count_segmented': len(doors) - 1, 'main_entrance_is_reference': True,
        'distance_summary_model_units': summaries, 'floor_fit': floor,
        'estimated_metric_calibration': calibration,
        'distance_csv_file': csv_file,
        'distance_summary_estimated_metres': {key: {k: v * factor if isinstance(v, (int, float)) else v
                                                    for k, v in value.items()} for key, value in summaries.items()},
        'maximum_ground_point_reprojection_error_pixels': max(image_errors),
        'source_video_sha256': meta['source_video_sha256'],
        'tracking_json_sha256': sha(args.tracking / 'detections.json'),
        'background_npz_sha256': sha(args.background / 'segmented_background.npz'),
        'script_sha256': sha(__file__),
        'viewer_controls_sha256': sha(HERE / 'door_distance_viewer.js'), 'limitations': limitations,
        'checks': {'full_source_timeline_preserved': True, 'source_video_hash_matches_depth_run': True,
                   'ground_plane_constraint_and_reprojection': True, 'independent_segment_distance_checks': True}}
    (out / 'README.txt').write_text(
        'One-video YOLO tracking and estimated 3D floor distances\n\n'
        'Open viewer.html to synchronize the annotated video frames, 3D person position and distance curves.\n'
        'tracking/ retains detections.json, per-track trajectory/pose JSON, YOLO video and annotated frames.\n'
        'trajectory_3d.json retains full frame timing, ground contacts, 3D coordinates and all door distances.\n'
        f'{csv_file} is a wide table, one row per original frame and track; empty cells mean unavailable.\n'
        'ground_geometry.json records the plane, camera, coordinate mapping and each door-reference definition.\n'
        'metric_calibration.json records the assumed door-leaf width, reference doors and uniform metric scale.\n'
        'Viewer unit switch rescales the 3D axes, distances, curves and rates together; image pixels and time are unchanged.\n'
        'Default estimated metres assume 0.825 m leaves and use the median scale of uncropped door panels.\n'
        'Edit the assumed width/reference in the viewer; Download current CSV exports the current settings.\n'
        'The saved CSV/JSON use the default CLI calibration, with explicit estimated-metre columns.\n'
        'closure_rate is change in distance per source-video second; negative means approaching.\n'
        'No temporal smoothing, interpolation, automatic relabelling or event-time estimate is applied.\n\n'
        + '\n'.join(limitations) + '\n\n'
        'Geometry can be regenerated from the saved detections, without rerunning YOLO:\n'
        'python video_processing/door_distance_3d.py --background BACKGROUND_FOLDER '
        '--tracking TRACKING_FOLDER --output OUTPUT_FOLDER\n', encoding='utf-8')
    summary['output_sha256'] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
                                if p.is_file() and p.name != 'summary.json'}
    dump(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def export_csv(out, rows, doors, calibration):
    fields = ['frame_index', 't', 'track_id', 'detected', 'valid_3d', 'contact_source',
              'contact_confidence', 'contact_x', 'contact_y', 'ground_right', 'ground_forward',
              'ground_height', 'distance_2d_main_bbox_norm']
    fields += [f'distance_3d_{d["id"]}' for d in doors]
    fields += [f'closure_rate_3d_{d["id"]}' for d in doors]
    fields += ['assumed_door_leaf_width_m', 'metres_per_model_unit']
    fields += [f'distance_3d_{d["id"]}_estimated_m' for d in doors]
    fields += [f'closure_rate_3d_{d["id"]}_estimated_m_per_s' for d in doors]
    path = out / 'distances.csv'
    try:
        stream = path.open('w', newline='', encoding='utf-8')
    except PermissionError:
        # An open Excel workbook can lock the existing CSV. Leave that file intact.
        path = out / 'distances_with_estimated_metres.csv'
        stream = path.open('w', newline='', encoding='utf-8')
    with stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            r = {key: row[key] for key in fields if key in row}
            r['contact_x'], r['contact_y'] = row['contact_pixel'] or [None, None]
            r['ground_right'], r['ground_forward'], r['ground_height'] = row['point_ground'] or [None] * 3
            r.update({f'distance_3d_{key}': value for key, value in row['distances'].items()})
            r.update({f'closure_rate_3d_{key}': value for key, value in row['closure_rates'].items()})
            r['assumed_door_leaf_width_m'] = calibration['assumed_door_leaf_width_m']
            r['metres_per_model_unit'] = calibration['metres_per_model_unit']
            r.update({f'distance_3d_{key}_estimated_m': value for key, value in row['distances_estimated_m'].items()})
            r.update({f'closure_rate_3d_{key}_estimated_m_per_s': value
                      for key, value in row['closure_rates_estimated_m_per_s'].items()})
            writer.writerow(r)
    return path.name


def render_frames(out, tracking, rows, doors, sw, sh, dw, dh):
    folder = out / 'frames'
    folder.mkdir(exist_ok=True)
    scale = np.array([sw / dw, sh / dh])
    for index in sorted({r['frame_index'] for r in rows}):
        image = Image.open(tracking / 'yolo_frames' / f'frame_{index:06d}.jpg').convert('RGB')
        draw = ImageDraw.Draw(image)
        for door in doors[1:]:
            polygon = (np.asarray(door['source_polygon']) + .5) * scale - .5
            draw.line([tuple(p) for p in polygon] + [tuple(polygon[0])], fill=door['colour'], width=2)
            x, y = polygon.mean(axis=0)
            draw.text((int(x), int(y)), f'D{door["segmentation_instance_id"]}', fill=door['colour'],
                      stroke_width=1, stroke_fill='white')
        x, y = doors[0]['source_pixel']
        draw.line([(x - 15, y - 3), (x + 15, y - 3)], fill=doors[0]['colour'], width=4)
        draw.line([(x, y - 18), (x, y)], fill=doors[0]['colour'], width=4)
        draw.text((x + 20, y - 18), 'MAIN reference', fill=doors[0]['colour'], stroke_width=1, stroke_fill='white')
        for r in [r for r in rows if r['frame_index'] == index]:
            if r['contact_pixel']:
                x, y = r['contact_pixel']
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='#00dddd', outline='black', width=1)
        image.save(folder / f'frame_{index:06d}.jpg', quality=92)


def render_figures(out, rows, doors, xyz, labels, stable, origin, basis, normal, offset, tid,
                   scale=1., unit='model units', suffix='', assumed_width=None):
    points = ((xyz - origin) @ basis) * scale
    sampling = np.zeros_like(stable)
    sampling[::4, ::4] = True
    keep = stable & sampling
    valid = [r for r in rows if r['valid_3d'] and r['track_id'] == tid]
    track = np.array([r['point_ground'] for r in valid]) * scale
    fig = plt.figure(figsize=(14, 6.2), layout='constrained')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for label in np.unique(labels[keep]):
        p = points[keep & (labels == label)]
        ax.scatter(*p.T, s=.6, color=SEMANTIC_COLOURS[label] / 255,
                   alpha=.18 if label == 0 else .4, depthshade=False, rasterized=True)
    ax.plot(*track.T, color='#007b8a', lw=3, label='Person ground track')
    ax.scatter(*track[0], color='#007b8a', marker='o', s=30)
    ax.scatter(*track[-1], color='#007b8a', marker='X', s=60)
    for door in doors:
        seg = np.array(door['segment_ground']) * scale
        ax.plot(*seg.T, color=door['colour'], lw=4)
        ax.scatter(*seg.mean(axis=0), color=door['colour'], marker='*' if door['primary'] else 's', s=75)
    ax.view_init(elev=32, azim=-65)
    ax.set_xlabel(f'Right ({unit})')
    ax.set_ylabel('Forward')
    ax.set_zlabel('Height above fitted floor')
    ax.set_title('Estimated 3D floor trajectory; main reference at origin')
    ax = fig.add_subplot(1, 2, 2)
    for door in doors:
        ax.plot([r['t'] for r in valid], [r['distances'][door['id']] * scale for r in valid],
                color=door['colour'], label=door['name'], lw=2.7 if door['primary'] else 1.5)
    ax.set(xlabel='Source-video time (s)', ylabel=f'Estimated distance ({unit})',
           title='Distances to floor-level door references', xlim=(0, max(r['t'] for r in rows)))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc='best')
    last = valid[-1]['t']
    ax.axvspan(last + 1e-3, max(r['t'] for r in rows), alpha=.08, color='grey')
    ax.text(.02, .02, 'Shaded: feet unobserved or person absent; no extrapolated distances.',
            transform=ax.transAxes, fontsize=8, va='bottom')
    fig.suptitle('YOLO floor contacts projected into existing LingBot geometry', fontsize=13)
    if suffix:
        fig.supxlabel(f'Assumed door-leaf width: {assumed_width:.3f} m; global scale from uncropped doors. '
                      'Estimated units do not establish physical accuracy.', fontsize=9)
    fig.savefig(out / f'trajectory_3d_preview{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout='constrained')
    for door in doors:
        seg = np.asarray(door['segment_ground']) * scale
        axes[0].plot(seg[:, 0], seg[:, 1], color=door['colour'], lw=4, label=door['name'])
        axes[0].scatter(*seg.mean(axis=0)[:2], color=door['colour'], marker='*' if door['primary'] else 's', s=70)
    axes[0].plot(track[:, 0], track[:, 1], '-o', color='#007b8a', ms=2)
    axes[0].set(xlabel=f'Right ({unit})', ylabel=f'Forward ({unit})', title='Top-down fitted-floor coordinates')
    axes[0].axis('equal')
    axes[0].legend(fontsize=8)
    residual = np.where((labels == 1) & stable, np.abs(xyz @ normal + offset) * scale, np.nan)
    plot = axes[1].imshow(residual, cmap='magma', vmin=0, vmax=float(np.nanpercentile(residual, 99)))
    axes[1].set_title('Floor reconstruction residual from fitted plane')
    axes[1].set(xlabel='Depth-grid pixel x', ylabel='Depth-grid pixel y')
    fig.colorbar(plot, ax=axes[1], label=unit)
    fig.savefig(out / f'floor_fit_validation{suffix}.png', dpi=140)
    plt.close(fig)


def write_viewer(out, rows, doors, xyz, labels, stable, origin, basis, meta, tid, calibration,
                 csv_file='distances.csv'):
    points = (xyz - origin) @ basis
    sampling = np.zeros_like(stable)
    sampling[::4, ::4] = True
    keep = stable & sampling
    scene = go.Figure()
    for label in np.unique(labels[keep]):
        p = points[keep & (labels == label)]
        colour = 'rgb(' + ','.join(str(c) for c in SEMANTIC_COLOURS[label]) + ')'
        scene.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
            marker={'size': 1.5, 'color': colour, 'opacity': .2 if label == 0 else .4},
            name=['Unlabelled background', 'Floor', 'Wall', 'Door panel', 'Window'][label], hoverinfo='skip'))
    for door in doors:
        seg = np.asarray(door['segment_ground'])
        scene.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode='lines+markers',
            marker={'size': 7 if door['primary'] else 4, 'color': door['colour']},
            line={'color': door['colour'], 'width': 7}, name=door['name']))
    track_rows = [r for r in rows if r['track_id'] == tid]
    coords = [r['point_ground'] or [None] * 3 for r in track_rows]
    scene.add_trace(go.Scatter3d(x=[p[0] for p in coords], y=[p[1] for p in coords], z=[p[2] for p in coords],
        mode='lines', line={'color': '#007b8a', 'width': 5}, name=f'Track {tid} (floor contacts)', connectgaps=False))
    marker_index = len(scene.data)
    scene.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers', marker={'size': 8, 'color': '#00dddd',
        'line': {'color': '#004455', 'width': 2}}, name='Current person floor position', hoverinfo='skip'))
    scene.update_layout(template='plotly_white', height=600, margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
        scene={'aspectmode': 'data', 'xaxis_title': 'Right', 'yaxis_title': 'Forward',
               'zaxis_title': 'Height above fitted floor',
               'camera': {'eye': {'x': 1.5, 'y': -1.7, 'z': 1.1}, 'up': {'x': 0, 'y': 0, 'z': 1}}},
        legend={'orientation': 'h', 'font': {'size': 10}}, uirevision='keep-camera')
    graph = go.Figure()
    for door in doors:
        graph.add_trace(go.Scatter(x=[r['t'] for r in track_rows],
            y=[r['distances'][door['id']] for r in track_rows], mode='lines', name=door['name'],
            line={'color': door['colour'], 'width': 3 if door['primary'] else 1.7}, connectgaps=False,
            hovertemplate='%{x:.2f}s: %{y:.4f} model units<extra>%{fullData.name}</extra>'))
    graph.add_vline(x=0, line_color='#007b8a', line_dash='dot')
    graph.update_layout(template='plotly_white', height=360, margin={'l': 60, 'r': 15, 'b': 55, 't': 25},
                        xaxis={'title': 'Source-video time (seconds)', 'range': [0, track_rows[-1]['t']]},
                        yaxis_title='Distance (model units)', legend={'orientation': 'h', 'font': {'size': 11}})
    config = {'displaylogo': False, 'responsive': True, 'scrollZoom': True}
    scene_html = scene.to_html(full_html=False, include_plotlyjs='plotly.min.js', div_id='scene', config=config)
    graph_html = graph.to_html(full_html=False, include_plotlyjs=False, div_id='curves', config=config)
    scene_html = scene_html.replace('Plotly.newPlot(', 'window.sceneReady = Plotly.newPlot(', 1)
    graph_html = graph_html.replace('Plotly.newPlot(', 'window.curvesReady = Plotly.newPlot(', 1)
    base_coordinates = [{'x': np.asarray(t.x).tolist(), 'y': np.asarray(t.y).tolist(),
                         'z': np.asarray(t.z).tolist()} for t in scene.data]
    data = json.dumps({'rows': track_rows, 'doors': [{'id': d['id'], 'name': d['name'], 'colour': d['colour']}
                       for d in doors], 'fps': meta['fps'], 'markerIndex': marker_index,
                       'videoStem': Path(meta['source_video']).stem,
                       'calibration': calibration, 'sceneCoordinates': base_coordinates}, allow_nan=False)
    reference_options = ('<option value="uncropped_median">Median of uncropped doors (recommended)</option>'
                         '<option value="all_median">Median of all segmented doors (includes cropped)</option>')
    reference_options += ''.join(f'<option value="{html.escape(c["door_id"])}">'
                                  f'{html.escape(c["name"])} only</option>' for c in calibration['candidates'])
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{{VIDEO_NAME}} — estimated 3D door distances</title>
<style>body{font:15px system-ui;margin:0;background:#f3f5f7;color:#17232e}main{max-width:1420px;margin:auto;padding:20px}
h1{font-size:24px;margin-bottom:8px}p{line-height:1.5}.notice{background:#fff4df;padding:12px;border-left:4px solid #d26900}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}.card{background:white;padding:12px;border-radius:8px;margin:16px 0}
h2{font-size:17px}img{width:100%;height:auto;display:block}button{padding:8px 18px;font:inherit;border:0;border-radius:5px;background:#245482;color:white;cursor:pointer}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}input{flex:1;min-width:180px}table{border-collapse:collapse;width:100%}td,th{padding:6px;text-align:left;border-bottom:1px solid #e5e8eb}
.metric{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.metric label{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.metric input{flex:0;width:95px;min-width:75px}.metric select,.metric input{font:inherit;padding:7px;border:1px solid #acbac7;border-radius:4px;max-width:100%}
#scale-error{color:#a12828}details{margin-top:12px}summary{cursor:pointer}.table-scroll{overflow:auto}button:disabled{opacity:.5}
.muted{color:#536577;font-size:13px}a{color:#245482}.links{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:1000px){.grid{grid-template-columns:1fr}}
</style></head><body><main><h1>{{VIDEO_NAME}}: person-to-door distances in estimated 3D</h1>
<p>YOLO11n-pose + ByteTrack, original video timing, existing LingBot background.</p>
<p class="notice"><strong>Main entrance:</strong> the bottom-centre image point projected onto the fitted floor.
It is a reference, not a measured doorway threshold. Other targets approximate the floor-level positions of segmented door panels.</p>
<div class="card"><div class="metric">
<label for="measurement-units">Measurements <select id="measurement-units"><option value="model">Model units (original)</option><option value="metres">Estimated metres</option></select></label>
<label for="door-width">Assumed door-leaf width (m) <input id="door-width" type="number" min="0.001" step="0.001" value="{{ASSUMED_WIDTH}}" list="door-width-presets"></label>
<datalist id="door-width-presets"><option value="0.725">M8: 72.5 cm</option><option value="0.760">Custom: 76 cm</option><option value="0.825">M9: 82.5 cm</option><option value="0.925">M10: 92.5 cm</option></datalist>
<label for="scale-reference">Scale reference <select id="scale-reference">{{REFERENCE_OPTIONS}}</select></label>
</div><p id="unit-description">Original LingBot model units. These 3D units are not image pixels.</p>
<p id="scale-note"></p><p id="scale-error" role="alert"></p>
<p class="muted">Assuming a door width sets one scale for the entire scene; it does not correct shape errors.
<a href="https://www.swedoor.se/tips-och-rad/mattabeller-innerdorrar" target="_blank" rel="noopener">Swedoor's table</a>
lists door-leaf widths of 72.5 cm (M8), 82.5 cm (M9) and 92.5 cm (M10). Actual widths here are unknown.</p>
<details><summary>Door-width evidence and scale disagreement</summary><div class="table-scroll"><table><thead><tr><th>Segmented panel</th><th>Width (model units)</th><th>Implied m/unit</th><th>Width under selected scale (estimated m)</th><th>Reference</th></tr></thead><tbody id="scale-evidence"></tbody></table></div><p id="scale-disagreement" class="muted"></p></details>
</div>
<div class="card controls"><button id="play">Play</button><label for="frame">Frame</label><input id="frame" type="range" min="0" max="MAX_FRAME" value="0" step="1">
<output id="time"></output></div><div class="grid"><div class="card"><h2>Original image: YOLO pose and floor contact</h2>
<img id="image" src="frames/frame_000000.jpg" alt="YOLO pose, door outlines and estimated floor-contact pixel">
<p class="muted">Cyan: floor-contact pixel. Coloured outlines: segmented door instances. Orange: main entrance reference.</p>
<p id="status"></p><table><thead><tr><th>Door reference</th><th id="distance-heading">3D distance (model units)</th><th id="rate-heading">Change (model units/s)</th></tr></thead><tbody id="values"></tbody></table>
</div><div class="card"><h2>3D background and ground trajectory</h2>SCENE_HTML
<p class="muted">Rotate with the mouse. The main entrance reference is the floor-coordinate origin. Current location is cyan.</p></div></div>
<div class="card"><h2>Distance over time</h2>GRAPH_HTML
<p class="muted">Negative change means approaching. Gaps indicate unavailable floor contact or no detection; no positions are invented.</p></div>
<div class="card links"><button id="download-current">Download current calibration CSV</button><a href="{{DISTANCE_CSV}}">Saved CSV (default calibration)</a><a href="trajectory_3d.json">3D trajectory JSON</a>
<a href="metric_calibration.json">Metric scale assumptions</a>
<a href="ground_geometry.json">Door/floor geometry</a><a href="tracking/detections.json">Saved YOLO detections</a>
<a href="tracking/yolo_tracking.mp4">YOLO video</a><a href="floor_fit_validation.png">Floor-fit validation</a><a href="summary.json">Summary and limitations</a></div>
</main><script id="measurement-data" type="application/json">DATA_JSON</script>
<script src="measurement-controls.js"></script></body></html>'''
    page = (page.replace('MAX_FRAME', str(len(track_rows) - 1)).replace('SCENE_HTML', scene_html)
            .replace('GRAPH_HTML', graph_html).replace('DATA_JSON', data)
            .replace('{{ASSUMED_WIDTH}}', str(calibration['assumed_door_leaf_width_m']))
            .replace('{{DISTANCE_CSV}}', html.escape(csv_file))
            .replace('{{VIDEO_NAME}}', html.escape(Path(meta['source_video']).stem))
            .replace('{{REFERENCE_OPTIONS}}', reference_options))
    (out / 'viewer.html').write_text(page, encoding='utf-8')
    (out / 'measurement-controls.js').write_text((HERE / 'door_distance_viewer.js').read_text(encoding='utf-8'),
                                                encoding='utf-8')
    (out / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')


if __name__ == '__main__':
    main()
