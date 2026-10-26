"""Run in the external LingBot environment; export the pipeline's depth contract.

This script imports LingBot from --repo. It neither vendors nor installs it.
Every encoded video frame is processed, in order, without temporal sampling.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def decode_video(video, directory):
    import cv2
    cap = cv2.VideoCapture(str(video))
    paths = []
    try:
        width, height = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or min(width, height, count) <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot read video geometry and timing')
        resized_height = round(height * 518 / width / 14) * 14
        if resized_height > 518 or resized_height < 14:
            raise ValueError('This geometry pipeline requires landscape/square video without centre cropping')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError('Video resolution changed during decoding')
            path = directory / f'frame_{len(paths):06d}.png'
            if not cv2.imwrite(str(path), frame):
                raise OSError(f'Could not save {path}')
            paths.append(str(path))
        if len(paths) != count:
            raise ValueError(f'Decoded {len(paths)} frames; video declares {count}')
    finally:
        cap.release()
    return paths, {'name': video.name, 'stem': video.stem, 'sha256': sha256(video),
                   'width': width, 'height': height, 'fps': fps, 'frame_count': count}


def export_arrays(predictions, images, directory, count):
    """Validate the entire result before publishing any completion manifest."""
    import numpy as np
    expected_ndim = {'depth': 5, 'depth_conf': 4, 'intrinsic': 4, 'extrinsic': 4, 'images': 5}
    arrays = {}
    for key, value in {**predictions, 'images': images}.items():
        if key not in expected_ndim:
            continue
        if hasattr(value, 'detach'):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == expected_ndim[key] and value.shape[0] == 1:
            value = value[0]
        arrays[key] = value
    if set(arrays) != set(expected_ndim):
        raise ValueError('LingBot did not produce depth, confidence and calibrated cameras')
    depth = arrays['depth']
    if depth.ndim != 4:
        raise ValueError('Invalid depth dimensions')
    n, h, w, channels = depth.shape
    shapes = {'depth': (count, h, w, 1), 'depth_conf': (count, h, w),
              'images': (count, 3, h, w), 'intrinsic': (count, 3, 3), 'extrinsic': (count, 3, 4)}
    for key, value in arrays.items():
        if value.shape != shapes[key] or not np.isfinite(value).all():
            raise ValueError(f'Invalid {key} shape or nonfinite values: {value.shape}')
    if n != count or channels != 1 or not (depth > 0).all():
        raise ValueError('Every frame must have positive depth')
    if arrays['images'].min() < 0 or arrays['images'].max() > 1:
        raise ValueError('Preprocessed RGB must be in [0, 1]')
    k = arrays['intrinsic']
    r = arrays['extrinsic'][..., :3]
    if not ((k[:, 0, 0] > 0).all() and (k[:, 1, 1] > 0).all()):
        raise ValueError('Camera focal lengths must be positive')
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=.001) or not np.allclose(np.linalg.det(r), 1, atol=.001):
        raise ValueError('Camera rotations are not rigid transforms')
    directory.mkdir()
    for i in range(count):
        np.savez_compressed(directory / f'frame_{i:06d}.npz', **{k: v[i] for k, v in arrays.items()})
    return [w, h]


def infer(args):
    import torch
    repo, checkpoint, video, output = (Path(p).resolve() for p in (args.repo, args.checkpoint, args.video, args.output))
    if not (repo / 'lingbot_map/models/gct_stream.py').is_file():
        raise FileNotFoundError(f'Not a compatible LingBot checkout: {repo}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'LingBot checkpoint missing: {checkpoint}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the LingBot environment; check the GPU driver/container runtime')
    dtype_name = args.dtype
    if dtype_name == 'auto':
        dtype_name = ('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if device.type == 'cuda' else 'float32'
    if device.type != 'cuda' and dtype_name != 'float32':
        raise ValueError('CPU inference requires float32')
    dtype = getattr(torch, dtype_name)
    sys.path.insert(0, str(repo))
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    output.mkdir(parents=True, exist_ok=True)
    if (output / 'processing_manifest.json').exists() or (output / video.stem).exists():
        raise FileExistsError('Refusing to overwrite published LingBot predictions; use a new output directory')
    checkpoint_hash = sha256(checkpoint)
    try:
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    with tempfile.TemporaryDirectory(prefix='.lingbot-', dir=output) as temp:
        temp = Path(temp)
        paths, source = decode_video(video, temp)
        images = load_and_preprocess_images(paths, mode='crop', image_size=518, patch_size=14)
        expected = (source['frame_count'], 3, round(source['height'] * 518 / source['width'] / 14) * 14, 518)
        if tuple(images.shape) != expected:
            raise ValueError(f'Unsupported LingBot preprocessing result: {tuple(images.shape)}')
        scale_frames = min(args.scale_frames, source['frame_count'])
        model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True,
                          max_frame_num=max(1024, source['frame_count']),
                          kv_cache_sliding_window=args.cache_window, kv_cache_scale_frames=scale_frames,
                          kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                          use_sdpa=True, camera_num_iterations=4)
        # Load on CPU, then cast only the trunk to save the fp32 GPU weight copy.
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = weights.get('model', weights)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f'Checkpoint is missing model weights: {missing[:10]}')
        del state, weights
        if device.type == 'cuda':
            model.aggregator = model.aggregator.to(dtype=dtype)
        gc.collect()
        model = model.to(device).eval()
        autocast = torch.amp.autocast('cuda', dtype=dtype) if device.type == 'cuda' and dtype != torch.float32 else contextlib.nullcontext()
        with torch.no_grad(), autocast:
            predictions = model.inference_streaming(images, num_scale_frames=scale_frames,
                keyframe_interval=args.keyframe_interval, output_device=torch.device('cpu'))
        # This utility returns WORLD-TO-CAMERA. Do not use demo.postprocess(),
        # which inverts poses for its viewer; our projection expects w2c.
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'].float().cpu(), images.shape[-2:])
        predictions['extrinsic'], predictions['intrinsic'] = extrinsic, intrinsic
        resolution = export_arrays(predictions, images, temp / video.stem, source['frame_count'])
        if sha256(video) != source['sha256']:
            raise ValueError('Input video changed while inference was running')
        manifest = {'source': source, 'all_source_frames_processed': True,
            'provider': {'name': 'lingbot-map', 'adapter_version': 1, 'repo': str(repo),
                         'revision': revision, 'tracked_changes': dirty, 'checkpoint': str(checkpoint),
                         'checkpoint_sha256': checkpoint_hash, 'torch': str(torch.__version__),
                         'device': str(device), 'dtype': dtype_name, 'mode': 'streaming',
                         'num_scale_frames': scale_frames, 'keyframe_interval': args.keyframe_interval,
                         'cache_window': args.cache_window, 'attention': 'sdpa',
                         'unexpected_checkpoint_keys': unexpected},
            'preprocessing': {'width': 518, 'patch_size': 14, 'resolution': resolution,
                              'crop': False, 'frame_stride': 1, 'image_encoding': 'lossless PNG'},
            'extrinsic_convention': 'world_to_camera', 'depth_units': 'model units; not metres'}
        staged_manifest = temp / 'processing_manifest.json'
        staged_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (temp / video.stem).rename(output / video.stem)
        staged_manifest.replace(output / 'processing_manifest.json')
    print(f'LingBot exported {source["frame_count"]} frames to {output}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'checkpoint', 'video', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    parser.add_argument('--scale-frames', type=int, default=8)
    parser.add_argument('--cache-window', type=int, default=64)
    parser.add_argument('--keyframe-interval', type=int, default=1)
    args = parser.parse_args()
    if min(args.scale_frames, args.cache_window, args.keyframe_interval) < 1:
        parser.error('Scale frames, cache window and keyframe interval must be positive')
    infer(args)


if __name__ == '__main__':
    main()
