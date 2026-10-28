"""Contract tests for joining existing tracking/2D outputs with the 3D stages.

Run: python -m unittest discover -s tests -v
Models and personal recordings are not required.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from geometry3d import pipeline_geometry as geometry
import pipeline
from processing import video_processor


class DepthContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.info = {'name': 'clip.mp4', 'stem': 'clip', 'sha256': 'correct',
                     'width': 640, 'height': 480, 'fps': 12., 'frame_count': 2}

    def manifest(self, source):
        path = self.root / 'cached/processing_manifest.json'
        geometry.save(path, {'source': source})
        return path

    def test_same_name_different_video_is_not_reused(self):
        self.manifest({**self.info, 'sha256': 'original_not_undistorted'})
        self.assertIsNone(geometry.select_depth_run(self.root, self.info))

    def test_identical_bytes_can_be_renamed(self):
        p = self.manifest({**self.info, 'name': 'another_name.mp4', 'stem': 'another_name'})
        self.assertEqual(geometry.select_depth_run(self.root, self.info), p)

    def test_bad_timing_rejected_despite_matching_hash(self):
        self.manifest({**self.info, 'fps': 30.})
        with self.assertRaisesRegex(ValueError, 'timing'):
            geometry.select_depth_run(self.root, self.info)

    def test_missing_depth_does_not_fall_back_to_original_video(self):
        self.manifest({**self.info, 'sha256': 'other'})
        with self.assertRaisesRegex(ValueError, 'exact clip'):
            geometry.ensure_depth_run('clip.mp4', self.info, self.root, self.root / 'work')

    def test_provider_is_argv_based_and_reused_after_success(self):
        config = self.root / 'command.json'
        code = ("from pathlib import Path; import sys,numpy as np; p=Path(sys.argv[1]); "
                "p.mkdir(parents=True,exist_ok=True); "
                "[np.savez_compressed(p/f'frame_{i:06d}.npz',depth=np.ones((14,14,1)),"
                "depth_conf=np.ones((14,14)),images=np.zeros((3,14,14)),"
                "intrinsic=np.eye(3),extrinsic=np.eye(4)[:3]) for i in range(2)]")
        geometry.save(config, {'command': [sys.executable, '-c', code, '{predictions}']})
        workspace = self.root / 'work with spaces'
        manifest = geometry.ensure_depth_run('clip.mp4', self.info, self.root / 'empty', workspace, config)
        self.assertEqual(geometry.load(manifest)['source'], self.info)
        with patch.object(geometry.subprocess, 'run', side_effect=AssertionError('Provider rerun')):
            again = geometry.ensure_depth_run('clip.mp4', self.info, self.root / 'empty', workspace, config)
        self.assertEqual(again, manifest)

    def test_shell_command_string_rejected(self):
        config = self.root / 'command.json'
        geometry.save(config, {'command': 'python arbitrary.py'})
        with self.assertRaisesRegex(ValueError, 'command array'):
            geometry.ensure_depth_run('clip.mp4', self.info, self.root / 'empty', self.root / 'work', config)

    def test_partial_provider_output_is_not_published_as_complete(self):
        config = self.root / 'command.json'
        geometry.save(config, {'command': [sys.executable, '-c', 'pass']})
        with self.assertRaisesRegex(ValueError, 'every frame'):
            geometry.ensure_depth_run('clip.mp4', self.info, self.root / 'empty', self.root / 'work', config)
        self.assertFalse((self.root / 'work/lingbot_run/processing_manifest.json').exists())


class JoinContracts(unittest.TestCase):
    def setUp(self):
        def row(i, valid):
            return {'frame_index': i, 't': i / 12, 'track_id': 7, 'valid_3d': valid,
                    'contact_source': 'ankle_midpoint' if valid else 'not_detected',
                    'contact_confidence': .9 if valid else None,
                    'point_ground': [1, 2, 0] if valid else None,
                    'point_ground_estimated_m': [3, 6, 0] if valid else None,
                    'distances': {'main_bottom_centre': 2 if valid else None},
                    'distances_estimated_m': {'main_bottom_centre': 6 if valid else None},
                    'closure_rates': {'main_bottom_centre': None},
                    'closure_rates_estimated_m_per_s': {'main_bottom_centre': None}}
        self.depth = {'primary_track_id': 7, 'frames': [row(1, False), row(0, True)]}
        self.features = {'info': {'clip_video_sha256': 'correct', 'notes': 'old'},
                         'data': {'scene_summary': {}, 'frames': [
                             {'frame_index': i, 't': i / 12, 'scene_rel': {'target': {'dist_px': 17}}}
                             for i in range(2)]}}
        self.summary = {'source_video_sha256': 'correct', 'estimated_metric_calibration': {'assumed': True}}

    def join(self):
        return geometry.join_geometry(self.features, {'id': 7}, self.depth, self.summary)

    def test_joins_by_frame_and_keeps_2d_fields_and_nulls(self):
        result = self.join()
        self.assertEqual(result['data']['frames'][0]['geometry_3d']['distances_estimated_m']['main_bottom_centre'], 6)
        self.assertIsNone(result['data']['frames'][1]['geometry_3d']['distances_model']['main_bottom_centre'])
        self.assertEqual(result['data']['frames'][0]['scene_rel']['target']['dist_px'], 17)

    def test_whole_clip_geometry_is_explicitly_noncausal(self):
        result = self.join()
        self.assertFalse(result['info']['depth']['causal'])
        self.assertTrue(result['info']['depth']['scene_geometry_uses_whole_clip'])
        self.assertTrue(all(not entry['causal'] for entry in geometry.catalog_entries()))

    def test_duplicate_feature_frames_rejected(self):
        self.features['data']['frames'].append(copy.deepcopy(self.features['data']['frames'][0]))
        with self.assertRaisesRegex(ValueError, 'frame IDs'):
            self.join()

    def test_wrong_video_times_rejected(self):
        self.features['data']['frames'][0]['t'] = 1.
        with self.assertRaisesRegex(ValueError, 'times'):
            self.join()

    def test_wrong_video_hash_rejected(self):
        self.features['info']['clip_video_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'video bytes'):
            self.join()


class TrackingAdapter(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'clip.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 12., (32, 24))
        for _ in range(2):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
        writer.release()
        self.trajectory = self.root / 'clip.json'
        self.source = {'id': 7, 'frame_width': 32, 'frame_height': 24, 'fps': .5,
            'meta': {'clip_video_sha256': geometry.sha(self.video)}, 'frames': [
                {'frame_index': 0, 't': 100., 'bbox': [3, 3, 15, 15], 'center': [9, 9], 'keypoints': None},
                {'frame_index': 1, 't': 200., 'bbox': None, 'center': None, 'keypoints': None}]}
        geometry.save(self.trajectory, self.source)

    def adapt(self, loader=None):
        return geometry.adapt_trajectory(self.video, self.trajectory, self.root / 'adapted',
                                         loader or (lambda *_: [None, None]))

    def test_video_timing_normalized_without_inference_and_original_times_retained(self):
        path = self.adapt()
        source = geometry.load(path)
        self.assertEqual([r['t'] for r in source['frames']], [0, 1 / 12])
        self.assertEqual([r['original_t'] for r in source['frames']], [100, 200])
        detections = geometry.load(path.parent / 'detections.json')
        self.assertFalse(detections['metadata']['yolo_rerun'])
        self.assertEqual(detections['frames'][1]['detections'], [])
        self.assertIsNone(detections['frames'][0]['detections'][0]['confidence'])

    def test_cached_adapter_does_not_decode_or_load_keypoints_again(self):
        path = self.adapt()
        timestamp = (path.parent / 'yolo_tracking.mp4').stat().st_mtime_ns
        again = self.adapt(Mock(side_effect=AssertionError('Keypoints reloaded')))
        self.assertEqual(again, path)
        self.assertEqual((path.parent / 'yolo_tracking.mp4').stat().st_mtime_ns, timestamp)

    def test_mismatched_dimensions_rejected(self):
        self.source['frame_width'] = 640
        geometry.save(self.trajectory, self.source)
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            self.adapt()

    def test_partial_track_timeline_rejected(self):
        self.source['frames'].pop()
        geometry.save(self.trajectory, self.source)
        with self.assertRaisesRegex(ValueError, 'frame counts'):
            self.adapt()

    def test_wrong_clip_hash_rejected(self):
        self.source['meta']['clip_video_sha256'] = 'original_before_undistortion'
        geometry.save(self.trajectory, self.source)
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.adapt()


class TrackerClock(unittest.TestCase):
    def test_runner_passes_paths_and_device_to_existing_collector(self):
        collector = Mock(processed_video_paths=['/saved/clip.mp4'])
        with patch.object(video_processor, 'LivePoseCollector', return_value=collector) as factory:
            result = video_processor.run(source_dir='inputs', output_root='outputs',
                processed_manifest='outputs/clips.json', device='cpu')
        cfg = factory.call_args.args[0]
        self.assertEqual(cfg.source_dir, 'inputs')
        self.assertEqual(cfg.output_root, 'outputs')
        self.assertEqual(cfg.processed_manifest, 'outputs/clips.json')
        self.assertEqual(result, ['/saved/clip.mp4'])
        collector.run.assert_called_once_with(display=False)

    def test_offline_tracking_uses_video_time_not_processing_time(self):
        import cv2
        tracker = object.__new__(video_processor.LivePoseCollector)
        tracker.cv2 = cv2
        tracker.cfg = SimpleNamespace(distort=False, save_clips=False, debug=False, status_every_s=1e9)
        tracker.source_clip_counts = {}
        tracker.active = {}
        tracker.record_fps = 12.
        tracker._fps_frames = 0
        tracker._fps_t0 = tracker._status_t0 = time.time()
        cap = Mock()
        cap.get.side_effect = lambda code: {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                                          cv2.CAP_PROP_FPS: 12}[code]
        cap.read.side_effect = [(True, object()) for _ in range(3)] + [(False, None)]
        tracker._open_source = Mock(return_value=cap)
        tracker._init_geometry = Mock()
        tracker._undistort = lambda image: image
        times = []
        tracker._process_frame = lambda image, timestamp: times.append(timestamp) or []
        self.assertTrue(tracker._run_source(Path('clip.mp4'), display=False))
        self.assertEqual(times, [0., 1 / 12, 2 / 12])
        cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
