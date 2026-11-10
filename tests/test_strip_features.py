"""Tests for privacy/strip_features.py on a real-schema features file.

The features come from processing.feature_extractor.extract() run on a short
synthetic walk, so the tests follow the extractor's schema when it changes.
Run: python -m unittest discover -s tests -v
No recordings or models are required.
"""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from privacy import strip_features as sf
from processing import feature_extractor as fe

CLIP = 'office_2026-11-03_141502_enter.mp4'
SHA = 'deadbeef' * 8


def synthetic_features():
    W, H, N, FPS = 640, 480, 30, 12
    frames = []
    for i in range(N):
        u = i / (N - 1)
        cx, feet_y, h = 520 - 220 * u + 40 * math.sin(6 * u), 200 + 250 * u, 60 + 150 * u
        top, w = feet_y - h, h * .35
        pt = lambda dx, fy: [cx + dx * w, top + fy * h, .9]
        kp = [pt(0, .08), pt(-.05, .06), pt(.05, .06), pt(-.1, .08), pt(.1, .08), pt(-.4, .2), pt(.4, .2),
              pt(-.45, .38), pt(.45, .38), pt(-.45, .52), pt(.45, .52), pt(-.25, .52), pt(.25, .52),
              pt(-.25, .75), pt(.25, .75), pt(-.2, 1.), pt(.2, 1.)]
        frames.append({'frame_index': i, 't': i / FPS, 'center': [cx, top + h / 2],
                       'bbox': [cx - w / 2, top, cx + w / 2, feet_y], 'keypoints': kp})
    traj = {'id': 1, 'label': 'enter', 'frame_width': W, 'frame_height': H, 'fps': FPS,
            'door_center': [320, 479], 'frames': frames,
            'meta': {'session_id': '2026-11-03_141502', 'effective_fps': FPS, 'record_fps': FPS,
                     'tracker': 'bytetrack', 'git_commit': 'abc', 'distort': False}}
    scene = {'image_size': [W, H], 'objects': [
        {'label': 'floor', 'polygon': [[0, 200], [640, 200], [640, 480], [0, 480]], 'source': 'sam3'},
        {'label': 'wall', 'polygon': [[0, 0], [640, 0], [640, 200], [0, 200]], 'source': 'sam3'},
        {'label': 'door', 'polygon': [[40, 60], [120, 60], [120, 200], [40, 200]], 'source': 'sam3'}]}
    result, _ = fe.extract(traj, scene, None, None, {'trajectory': 'office_enter.json', 'scene': 'office_enter_background.json',
                                                      'pose': None, 'background': None, 'depth': None}, None)
    # what pipeline.py adds after the extractor, and what the 3D join adds
    result['info']['clip_video'], result['info']['clip_video_sha256'] = CLIP, SHA
    result['info']['clip'] = CLIP
    for fr in result['data']['frames']:
        if fr['detected']:
            fr['geometry_3d'] = {'available': True, 'point_ground_model': [1, 2, 0],
                                 'point_ground_estimated_m': [3, 6, 0],
                                 'distances_model': {'main': 2}, 'distances_estimated_m': {'main': 6},
                                 'closure_rates_model_per_s': {'main': -.2},
                                 'closure_rates_estimated_m_per_s': {'main': -.6}}
    result['info']['depth'] = {'provider': 'LingBot', 'source_video_sha256': SHA, 'causal': False,
                               'metric_calibration': {'model_units_per_m': .33}}
    return result


def leaf_names(obj, found=None):
    found = set() if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(k)
            leaf_names(v, found)
    elif isinstance(obj, list):
        for v in obj:
            leaf_names(v, found)
    return found


class StripFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = synthetic_features()

    def strip(self, tier, **kw):
        before = copy.deepcopy(self.original)
        out, report = sf.strip(self.original, tier, **kw)
        self.assertEqual(before, self.original, 'input must not be modified')
        return out, report

    def test_tier_sizes_are_7_10_12_distinct_per_frame_fields(self):
        for tier, expected in (('F0', 0), ('F1', 7), ('F2', 10), ('F3', 12)):
            _, rep = self.strip(tier)
            names = {p.split('.')[-1] for p, n in rep['frame_fields'].items() if n}
            self.assertEqual(len(names), expected, tier)

    def test_removed_names_are_gone_and_everything_else_is_unchanged(self):
        out, _ = self.strip('F3')
        frame_paths, _ = sf.paths_for('F3')
        names = leaf_names(out['data']['frames'])
        for p in frame_paths:
            self.assertNotIn(p.split('.')[-1], names, p)
        kept = lambda d: {k: v for k, v in d.items() if k not in ('pos', 'kin', 'pose', 'scene_rel', 'geometry_3d')}
        for a, b in zip(self.original['data']['frames'], out['data']['frames']):
            self.assertEqual(kept(a), kept(b))
            if a['detected']:
                # features that must survive keep their exact value
                self.assertEqual(a['kin']['speed_bh_s'], b['kin']['speed_bh_s'])
                self.assertEqual(a['pose']['torso_vs_motion_deg'], b['pose']['torso_vs_motion_deg'])
                self.assertEqual(a['scene_rel']['target']['dist_bh'], b['scene_rel']['target']['dist_bh'])
                self.assertEqual(a['pos']['ground_px'], b['pos']['ground_px'])

    def test_body_height_was_recoverable_before_and_is_not_after(self):
        frames = [f for f in self.original['data']['frames'] if f['detected'] and f['kin']['speed_px_s']
                  and f['kin']['speed_bh_s']]
        self.assertTrue(frames)
        for f in frames:
            bh = f['kin']['speed_px_s'] / f['kin']['speed_bh_s']
            self.assertAlmostEqual(bh / f['pos']['bh_ref_px'], 1, delta=.02)
            self.assertAlmostEqual(f['scene_rel']['target']['dist_px'] / f['scene_rel']['target']['dist_bh']
                                   / f['pos']['bh_ref_px'], 1, delta=.02)
        out, _ = self.strip('F1')
        for f in out['data']['frames']:
            if f['detected']:
                self.assertNotIn('speed_px_s', f['kin'])
                self.assertNotIn('dist_px', f['scene_rel']['target'])
                self.assertTrue(all(c is None or 'dist_px' not in c for c in f['scene_rel']['competitors']))
                self.assertTrue(all('dist_px' not in e for e in f['scene_rel']['exits']))

    def test_origin_exit_distance_and_camera_horizon_are_kept(self):
        out, _ = self.strip('F3')
        causal = out['data']['track_summary']['causal']
        self.assertNotIn('first_bh_px', causal)
        self.assertIn('first_ground_norm', causal)
        geom = out['data']['track_summary']['hindsight']['camera_geom']
        self.assertNotIn('bh_vs_y_slope', geom)
        self.assertNotIn('bh_vs_y_intercept', geom)
        self.assertIn('horizon_y_px', geom)
        if causal['origin_exit'] and 'dist_px' in self.original['data']['track_summary']['causal']['origin_exit']:
            self.assertIn('dist_px', causal['origin_exit'])

    def test_metadata_and_3d_metres(self):
        out, _ = self.strip('F1')
        info = out['info']
        for name in ('clip', 'clip_video', 'clip_video_sha256'):
            self.assertNotIn(name, info)
        self.assertNotIn('session_id', info['trajectory_meta'])
        self.assertNotIn('source_video_sha256', info['depth'])
        self.assertNotIn('metric_calibration', info['depth'])
        self.assertTrue(all(isinstance(v, bool) for v in info['sources'].values()))
        fr = next(f for f in out['data']['frames'] if f['detected'])
        self.assertEqual(set(fr['geometry_3d']), {'available', 'point_ground_model', 'distances_model',
                                                  'closure_rates_model_per_s'})
        kept, _ = self.strip('F1', keep_3d_metres=True)
        self.assertIn('distances_estimated_m', next(f for f in kept['data']['frames'] if f['detected'])['geometry_3d'])

    def test_f0_is_a_copy_apart_from_the_privacy_tag(self):
        out, _ = self.strip('F0')
        out['info'].pop('privacy')
        self.assertEqual(out, self.original)

    def test_pseudonym_needs_a_key_and_is_stable(self):
        none, _ = self.strip('F2')
        self.assertNotIn('clip_id', none['info'])
        a, _ = self.strip('F2', key=b'secret')
        b, _ = self.strip('F2', key=b'secret')
        c, _ = self.strip('F2', key=b'other')
        self.assertEqual(a['info']['clip_id'], b['info']['clip_id'])
        self.assertNotEqual(a['info']['clip_id'], c['info']['clip_id'])
        self.assertNotIn(CLIP, json.dumps(a))

    def test_audit_catches_a_leftover_field(self):
        broken = copy.deepcopy(self.original)
        frame_paths, data_paths = sf.paths_for('F2')
        left = sf.audit(broken, sf._forbidden_names(frame_paths, data_paths), [])
        self.assertTrue(any(p.endswith('hip_width_bh') for p in left))
        out, _ = self.strip('F2')
        self.assertEqual(sf.audit(out, sf._forbidden_names(frame_paths, data_paths), list(sf.INFO_DROP_PATHS)), [])

    def test_stripping_twice_changes_nothing_more(self):
        once, _ = self.strip('F2')
        twice, _ = sf.strip(once, 'F2')
        self.assertEqual(once['data'], twice['data'])

    def test_cli_writes_files_and_filters_the_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / 'office_features.json').write_text(json.dumps(self.original))
            fe.write_catalog(temp)
            code = sf.main(['--tier', 'F3', '--input-dir', str(temp), '--out-dir', str(temp / 'out'),
                            '--catalog', str(temp / 'feature_catalog.json')])
            self.assertEqual(code, 0)
            self.assertTrue((temp / 'out' / 'office_features_F3.json').is_file())
            names = {e['name'] for e in json.loads((temp / 'out' / 'feature_catalog_F3.json').read_text())['features']}
            for gone in ('pos.bh_ref_px', 'pose.body_height_px', 'pose.hip_width_bh', 'pose.ankle_sep_bh',
                         'kin.curvature_per_px', 'track_summary.causal.first_bh_px'):
                self.assertNotIn(gone, names)
            self.assertIn('kin.speed_bh_s', names)
            # its own output is not picked up again as an input
            with self.assertRaises(SystemExit):
                sf.main(['--tier', 'F3', '--input-dir', str(temp / 'out'), '--dry-run'])


if __name__ == '__main__':
    unittest.main()
