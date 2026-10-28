"""LingBot adapter/runtime contracts; no checkpoint or GPU required."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry3d import lingbot_inference as inference
from geometry3d import lingbot_runtime as runtime
from geometry3d import pipeline_geometry as geometry


def arrays(count=2, height=14, width=14):
    extrinsic = np.tile(np.eye(4)[:3], (1, count, 1, 1))
    extrinsic[..., 0, 3] = 2  # Nonidentity pose catches accidental c2w inversion.
    return {'depth': np.ones((1, count, height, width, 1)),
            'depth_conf': np.ones((1, count, height, width)),
            'intrinsic': np.tile(np.eye(3), (1, count, 1, 1)), 'extrinsic': extrinsic}


class Tensor(np.ndarray):
    """Small CPU tensor stand-in for testing the model boundary, not predictions."""
    def float(self):
        return self

    def cpu(self):
        return self


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_unbatch_and_preserve_world_to_camera_pose(self):
        out = self.root / 'frames'
        inference.export_arrays(arrays(), np.zeros((2, 3, 14, 14)), out, 2)
        with np.load(out / 'frame_000001.npz') as frame:
            self.assertEqual(frame['depth'].shape, (14, 14, 1))
            self.assertEqual(frame['images'].dtype, np.float32)
            self.assertEqual(frame['extrinsic'][0, 3], 2)

    def test_single_frame_does_not_lose_its_frame_axis(self):
        out = self.root / 'single'
        inference.export_arrays(arrays(count=1), np.zeros((1, 3, 14, 14)), out, 1)
        self.assertEqual([p.name for p in out.iterdir()], ['frame_000000.npz'])

    def test_invalid_predictions_never_publish_frames(self):
        for mutation in ('missing', 'negative', 'nonfinite', 'partial', 'rotation', 'focal'):
            values = arrays()
            if mutation == 'missing':
                values.pop('depth_conf')
            elif mutation == 'negative':
                values['depth'][0, 0, 0, 0, 0] = -1
            elif mutation == 'nonfinite':
                values['intrinsic'][0, 0, 0, 0] = np.nan
            elif mutation == 'partial':
                values['depth_conf'] = values['depth_conf'][:, :1]
            elif mutation == 'rotation':
                values['extrinsic'][0, 0, 0, 0] = 2
            else:
                values['intrinsic'][0, 0, 0, 0] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inference.export_arrays(values, np.zeros((2, 3, 14, 14)), self.root / mutation, 2)
            self.assertFalse((self.root / mutation).exists())

    def video(self, width=32, height=24):
        import cv2
        video = self.root / 'clip with spaces.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 12, (width, height))
        self.assertTrue(writer.isOpened())
        for value in (20, 150):
            writer.write(np.full((height, width, 3), value, np.uint8))
        writer.release()
        return video

    def test_decode_preserves_every_frame_and_encoded_fps(self):
        video = self.video()
        paths, info = inference.decode_video(video, self.root)
        self.assertEqual(len(paths), 2)
        self.assertEqual(info['fps'], 12)
        self.assertEqual(info['sha256'], geometry.sha(video))
        self.assertTrue(all(Path(p).suffix == '.png' for p in paths))

    def test_portrait_cropping_is_rejected_before_geometry_misalignment(self):
        with self.assertRaisesRegex(ValueError, 'centre cropping'):
            inference.decode_video(self.video(24, 32), self.root)

    def test_actual_adapter_flow_uses_official_api_and_publishes_provenance(self):
        video = self.video()
        repo = self.root / 'separate checkout'
        model_file = repo / 'lingbot_map/models/gct_stream.py'
        model_file.parent.mkdir(parents=True)
        model_file.touch()
        checkpoint = self.root / 'model.pt'
        checkpoint.write_bytes(b'test checkpoint')
        result = arrays(height=392, width=518)
        result['pose_enc'] = np.zeros((1, 2, 9)).view(Tensor)
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.load_state_dict.return_value = ([], ['unused_point_head'])
        model.inference_streaming.return_value = result
        factory = Mock(return_value=model)
        preprocess = Mock(return_value=np.zeros((2, 3, 392, 518)))
        pose = Mock(return_value=(result['extrinsic'], result['intrinsic']))
        fake_torch = SimpleNamespace(device=lambda name: SimpleNamespace(type=name), float32='float32',
            no_grad=contextlib.nullcontext, load=Mock(return_value={'model': {'weights': 1}}), __version__='fixture')
        modules = {'torch': fake_torch, 'lingbot_map': SimpleNamespace(),
            'lingbot_map.models': SimpleNamespace(), 'lingbot_map.utils': SimpleNamespace(),
            'lingbot_map.models.gct_stream': SimpleNamespace(GCTStream=factory),
            'lingbot_map.utils.load_fn': SimpleNamespace(load_and_preprocess_images=preprocess),
            'lingbot_map.utils.pose_enc': SimpleNamespace(pose_encoding_to_extri_intri=pose)}
        args = SimpleNamespace(repo=repo, checkpoint=checkpoint, video=video, output=self.root / 'result',
            device='cpu', dtype='auto', scale_frames=8, cache_window=64, keyframe_interval=1)
        with patch.dict(sys.modules, modules), patch.object(inference.subprocess, 'check_output', side_effect=['commit\n', '']):
            manifest = inference.infer(args)
        self.assertEqual(manifest['source']['frame_count'], 2)
        self.assertEqual(manifest['provider']['checkpoint_sha256'], geometry.sha(checkpoint))
        self.assertEqual(manifest['extrinsic_convention'], 'world_to_camera')
        self.assertEqual(model.inference_streaming.call_args.kwargs['num_scale_frames'], 2)
        self.assertTrue(factory.call_args.kwargs['use_sdpa'])
        self.assertEqual(inference.sha256(args.output / video.stem / 'frame_000000.npz'),
                         geometry.sha(args.output / video.stem / 'frame_000000.npz'))
        self.assertTrue((args.output / 'processing_manifest.json').is_file())
        self.assertEqual(preprocess.call_args.kwargs, {'mode': 'crop', 'image_size': 518, 'patch_size': 14})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'result'
        self.output.mkdir()
        self.config_path = self.root / 'settings.json'
        self.video = self.root / 'clip with spaces.mp4'
        self.video.write_bytes(b'video')
        self.config = {'runtime': 'ssh-docker', 'host': 'user@fd7a::1', 'port': 2222,
            'identity_file': '~/.ssh/id_ed25519', 'repo': '/srv/lingbot',
            'checkpoint': '/srv/models/weights.pt', 'remote_workdir': '/srv/jobs', 'image': 'lingbot:gpu'}

    def test_local_environment_is_a_separate_argv_process(self):
        python = self.root / 'external environment/python.exe'
        python.parent.mkdir()
        python.touch()
        config = {'runtime': 'local', 'python': str(python), 'repo': 'separate repo', 'checkpoint': 'weights.pt'}
        with patch.object(runtime.subprocess, 'run') as run:
            runtime.run(config, self.config_path, self.video, self.output, io.StringIO())
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], str(python))
        self.assertIn(str(self.video), argv)
        self.assertEqual(run.call_args.kwargs['cwd'], self.root / 'separate repo')
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_broken_remote_gpu_stops_before_uploading_video(self):
        with patch.object(runtime.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'nvidia-smi')) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                runtime.run(self.config, self.config_path, self.video, self.output, io.StringIO())
        self.assertEqual(run.call_count, 1)
        self.assertIn('nvidia-smi -L', run.call_args.args[0][-1])

    def test_ssh_job_quotes_paths_runs_adapter_and_downloads_outputs(self):
        calls = []

        def execute(argv, **kwargs):
            calls.append(argv)
            if argv[0] == 'scp' and '-r' in argv:
                downloaded = Path(argv[-1]) / 'output'
                (downloaded / self.video.stem).mkdir(parents=True)
                (downloaded / 'processing_manifest.json').write_text(json.dumps({'source': {'sha256': 'bytes'}}))
            return SimpleNamespace(stdout='1000\n' if argv[-1] in ('id -u', 'id -g') else '')

        with patch.object(runtime.subprocess, 'run', side_effect=execute):
            runtime.run(self.config, self.config_path, self.video, self.output, io.StringIO())
        docker = next(shlex.split(argv[-1]) for argv in calls if argv[0] == 'ssh' and argv[-1].startswith('docker '))
        self.assertIn('/job/' + self.video.name, docker)
        self.assertIn('type=bind,src=/srv/lingbot,dst=/lingbot,readonly', docker)
        self.assertIn('--gpus', docker)
        uploads = [argv for argv in calls if argv[0] == 'scp' and '-r' not in argv]
        self.assertEqual(len(uploads), 2)
        self.assertTrue(all(argv[-1].startswith('user@[fd7a::1]:') for argv in uploads))
        result = json.loads((self.output / 'processing_manifest.json').read_text())
        self.assertEqual(result['remote_execution']['image'], 'lingbot:gpu')

    def test_pipeline_dispatches_builtin_runtime_and_retains_its_provenance(self):
        config = self.root / 'local.json'
        geometry.save(config, {'runtime': 'local'})
        info = {'name': self.video.name, 'stem': self.video.stem, 'sha256': 'clip',
                'fps': 12., 'frame_count': 2, 'width': 32, 'height': 24}

        def provider(config, path, video, output, log):
            inference.export_arrays(arrays(), np.zeros((2, 3, 14, 14)), output / video.stem, 2)
            geometry.save(output / 'processing_manifest.json', {'source': info, 'provider': {'revision': 'external'}})

        with patch.object(runtime, 'run', side_effect=provider) as run:
            manifest = geometry.ensure_depth_run(self.video, info, self.root / 'empty', self.output, config)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(geometry.load(manifest)['provider']['revision'], 'external')
        self.assertEqual(geometry.load(manifest)['source'], info)

    def test_cached_camera_to_world_exports_are_rejected(self):
        path = self.root / 'cache/processing_manifest.json'
        geometry.save(path, {'source': {'sha256': 'same'}, 'extrinsic_convention': 'camera_to_world'})
        with self.assertRaisesRegex(ValueError, 'world-to-camera'):
            geometry.select_depth_run(path, {'sha256': 'same'})

    def test_partial_provider_manifest_never_becomes_a_cache_entry(self):
        config = self.root / 'local.json'
        geometry.save(config, {'runtime': 'local'})
        info = {'stem': self.video.stem, 'sha256': 'clip', 'fps': 12., 'frame_count': 2}

        def incomplete(config, path, video, output, log):
            geometry.save(output / 'processing_manifest.json', {'source': info})

        with patch.object(runtime, 'run', side_effect=incomplete), self.assertRaisesRegex(ValueError, 'every frame'):
            geometry.ensure_depth_run(self.video, info, self.root / 'empty', self.output, config)
        self.assertFalse((self.output / 'lingbot_run/processing_manifest.json').exists())
        self.assertIsNone(geometry.select_depth_run(self.output, info))


if __name__ == '__main__':
    unittest.main()
