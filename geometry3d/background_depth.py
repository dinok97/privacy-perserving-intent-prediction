"""Estimate fixed-camera background depth from locally saved LingBot predictions.

No person detection or new model inference is performed. Each clip is handled
separately: independently reconstructed clips need not share a physical scale.
Requires numpy, matplotlib and Pillow.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_info(manifest: dict) -> dict:
    if 'source' in manifest:
        return manifest['source']
    name = Path(manifest['source_video']).name
    return {
        'name': name, 'stem': Path(name).stem,
        'frame_count': manifest['source_frame_count'],
        'fps': manifest['source_fps'], 'sha256': manifest['source_sha256'],
    }


def render_maps(output: Path, stem: str, count: int, rgb: np.ndarray,
                depth: np.ndarray, relative_mad: np.ndarray, stable: np.ndarray,
                low: float, high: float, tolerance: float, inlier_fraction: float):
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('#777777')
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    plot = ax.imshow(depth, cmap=cmap, vmin=low, vmax=high, interpolation='nearest')
    ax.set_title(f'{stem}\nBackground depth candidate: median of {count} frames', fontsize=12)
    ax.set_xlabel('Pixel x (prediction grid)')
    ax.set_ylabel('Pixel y (prediction grid)')
    fig.colorbar(plot, ax=ax, label='Predicted depth (model units; not calibrated metres)', extend='both')
    fig.savefig(output / 'background_depth_heatmap.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Temporal-median RGB reference')
    p = axes[0, 1].imshow(depth, cmap=cmap, vmin=low, vmax=high, interpolation='nearest')
    axes[0, 1].set_title('Temporal-median depth')
    fig.colorbar(p, ax=axes[0, 1], label='Model units', shrink=0.85, extend='both')
    q = axes[1, 0].imshow(100 * relative_mad, cmap='magma', vmin=0, vmax=30,
                         interpolation='nearest')
    axes[1, 0].set_title('Temporal variation (lower is more consistent)')
    fig.colorbar(q, ax=axes[1, 0], label='Median absolute deviation / depth (%)',
                 shrink=0.85, extend='max')
    axes[1, 1].imshow(stable, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[1, 1].set_title(
        f'White: {100 * inlier_fraction:.0f}% of valid frames within '
        f'{100 * tolerance:.0f}% of median\n'
        f'{100 * stable.mean():.1f}% of pixels pass the stability criteria', fontsize=10)
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle(f'{stem} | {count} frames | estimated background, no person labels', fontsize=13)
    fig.savefig(output / 'validation_preview.png', dpi=160)
    plt.close(fig)


def process_run(manifest_path: Path, output_root: Path, args) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source = source_info(manifest)
    stem = source['stem']
    prediction_dir = manifest_path.parent / stem
    files = sorted(prediction_dir.glob('frame_*.npz'))
    count = int(source['frame_count'])
    expected_names = [f'frame_{i:06d}.npz' for i in range(count)]
    if [file.name for file in files] != expected_names:
        raise ValueError(f'{stem}: missing or unexpected prediction frames')
    depth_frames, confidence_frames, rgb_frames = [], [], []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            depth = data['depth']
            if depth.ndim != 3 or depth.shape[-1] != 1:
                raise ValueError(f'{file}: unsupported depth shape {depth.shape}')
            depth = depth[..., 0].astype(np.float32)
            confidence = data['depth_conf'].astype(np.float32)
            rgb = data['images'].transpose(1, 2, 0)
            if confidence.shape != depth.shape or rgb.shape != (*depth.shape, 3):
                raise ValueError(f'{file}: RGB/confidence are not on the depth grid')
            if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                raise ValueError(f'{file}: RGB values outside the expected [0, 1] range')
            valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
            depth_frames.append(np.where(valid, depth, np.nan))
            confidence_frames.append(np.where(valid, confidence, np.nan))
            rgb_frames.append(np.rint(255 * rgb).astype(np.uint8))

    depth_stack = np.stack(depth_frames)
    del depth_frames
    valid_count = np.isfinite(depth_stack).sum(axis=0).astype(np.uint32)
    median = np.nanmedian(depth_stack, axis=0).astype(np.float32)
    mean = np.nanmean(depth_stack, axis=0).astype(np.float32)
    absolute_difference = np.abs(depth_stack - median)
    mad = np.nanmedian(absolute_difference, axis=0).astype(np.float32)
    relative_mad = np.divide(mad, median, out=np.full_like(mad, np.nan), where=median > 0)
    inliers = np.isfinite(depth_stack) & (absolute_difference <= args.relative_tolerance * median)
    inlier_count = inliers.sum(axis=0).astype(np.uint32)
    inlier_fraction = np.divide(
        inlier_count, valid_count, out=np.zeros_like(median), where=valid_count > 0)
    minimum_valid = int(np.ceil(args.min_valid_fraction * count))
    stable = ((valid_count >= minimum_valid) & np.isfinite(median)
              & (inlier_fraction >= args.min_inlier_fraction))
    stable_depth = np.where(stable, median, np.nan).astype(np.float32)
    # A diagnostic only: a shared multiplicative drift in otherwise stable pixels.
    # No scale correction or per-frame colour normalization is applied to the data.
    frame_ratios = np.array([
        np.nanmedian(frame[stable] / median[stable]) if stable.any() else np.nan
        for frame in depth_stack
    ], dtype=np.float32)
    reference_rgb = np.median(np.stack(rgb_frames), axis=0).astype(np.uint8)
    confidence_median = np.nanmedian(np.stack(confidence_frames), axis=0).astype(np.float32)
    del depth_stack, absolute_difference, inliers, rgb_frames, confidence_frames

    output = output_root / stem
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'background_depth.npy', median, allow_pickle=False)
    np.save(output / 'stable_background_depth.npy', stable_depth, allow_pickle=False)
    np.savez_compressed(
        output / 'background_depth.npz', depth=median, depth_mean=mean,
        stable_depth=stable_depth, stable_mask=stable.astype(np.uint8),
        absolute_mad=mad, relative_mad=relative_mad,
        valid_observation_count=valid_count, inlier_observation_count=inlier_count,
        inlier_fraction=inlier_fraction, confidence_median=confidence_median,
        frame_to_median_scale_ratio=frame_ratios,
    )
    Image.fromarray(reference_rgb).save(output / 'background_rgb_reference.png')
    Image.fromarray(stable.astype(np.uint8) * 255).save(output / 'stable_mask.png')
    low, high = np.nanpercentile(median, [1, 99]).tolist()
    if high <= low:
        high = low + max(abs(low) * 0.01, 1e-6)
    render_maps(output, stem, count, reference_rgb, median, relative_mad,
                stable, low, high, args.relative_tolerance, args.min_inlier_fraction)

    # Reload the actual deliverables, ensuring array values and binary masks agree.
    saved = np.load(output / 'background_depth.npy', allow_pickle=False)
    assert np.array_equal(saved, median, equal_nan=True)
    with np.load(output / 'background_depth.npz', allow_pickle=False) as archive:
        assert np.array_equal(archive['depth'], saved, equal_nan=True)
        assert np.array_equal(archive['stable_mask'], stable.astype(np.uint8))
    binary_png = np.asarray(Image.open(output / 'stable_mask.png'))
    assert np.array_equal(binary_png, stable.astype(np.uint8) * 255)
    assert np.array_equal(np.isfinite(stable_depth), stable)

    summary = {
        'video': source['name'], 'source_video_sha256': source.get('sha256'),
        'source_run': manifest_path.parent.resolve().as_posix(),
        'source_manifest_sha256': sha256(manifest_path),
        'frames_used': count, 'prediction_width': median.shape[1],
        'prediction_height': median.shape[0], 'depth_dtype': str(median.dtype),
        'method': 'per-pixel temporal median of finite positive predicted depths',
        'confidence_filter': 'nonfinite confidence excluded; no uncalibrated score threshold applied',
        'person_detection_used': False, 'scale_correction_applied': False,
        'scale': 'model units; no physical calibration; separate estimate per clip',
        'stable_mask_parameters': {
            'relative_tolerance': args.relative_tolerance,
            'min_inlier_fraction_of_valid_frames': args.min_inlier_fraction,
            'min_valid_fraction_of_all_frames': args.min_valid_fraction,
            'minimum_valid_observations': minimum_valid,
        },
        'stable_pixel_fraction': float(stable.mean()),
        'finite_depth_pixel_fraction': float(np.isfinite(median).mean()),
        'median_depth_range': [float(np.nanmin(median)), float(np.nanmax(median))],
        'median_relative_mad': float(np.nanmedian(relative_mad)),
        'frame_scale_ratio_5th_95th_percentile': (
            np.nanpercentile(frame_ratios, [5, 95]).tolist() if stable.any() else None),
        'heatmap_colour_limits': {'percentiles': [1, 99], 'values': [low, high]},
        'limitations': [
            'Temporal consensus does not identify people or prove empty-background visibility.',
            'Persistent occluders can be retained; short clips may never reveal the background.',
            'Stable mask thresholds are exploratory settings, not calibrated guarantees.',
            'Pixel coordinates refer to the prediction grid, not original-video resolution.',
            'A consistent temporal depth estimate can still be geometrically inaccurate.',
        ],
    }
    summary['output_files_sha256'] = {
        path.name: sha256(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != 'metadata.json'
    }
    (output / 'metadata.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f"{stem}: {count} frames; {median.shape[1]}x{median.shape[0]}; "
          f"{100 * stable.mean():.1f}% stable pixels", flush=True)
    return summary


def write_index(output: Path, summaries: list[dict]):
    cards = []
    for row in summaries:
        stem = Path(row['video']).stem
        escaped = html.escape(stem)
        cards.append(
            f'<article><h2>{escaped}</h2><p>{row["frames_used"]} frames; '
            f'{row["prediction_width"]} x {row["prediction_height"]}; '
            f'{100 * row["stable_pixel_fraction"]:.1f}% of pixels pass stability criteria.</p>'
            f'<a href="{escaped}/validation_preview.png"><img loading="lazy" '
            f'src="{escaped}/validation_preview.png" alt="Depth and stability preview for {escaped}"></a>'
            f'<nav><a href="{escaped}/background_depth_heatmap.png">Depth heatmap</a>'
            f'<a href="{escaped}/background_depth.npy" download>Depth array (.npy)</a>'
            f'<a href="{escaped}/background_depth.npz" download>Depth and diagnostics (.npz)</a>'
            f'<a href="{escaped}/stable_mask.png">Binary stability mask</a>'
            f'<a href="{escaped}/metadata.json">Settings and checks</a></nav></article>')
    text = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Background depth estimates</title><style>'
            'body{font:16px system-ui;max-width:1150px;margin:auto;padding:24px;'
            'background:#f3f5f7;color:#17232e}article{background:white;border-radius:8px;'
            'padding:20px;margin:24px 0}h2{font-size:19px;overflow-wrap:anywhere}'
            'img{width:100%;height:auto}nav{display:flex;gap:18px;flex-wrap:wrap}'
            'a{color:#245482}p{line-height:1.5}</style><h1>Background depth estimates</h1>'
            f'<p>{len(summaries)} clips; {sum(x["frames_used"] for x in summaries)} frames. '
            'Per-pixel temporal median; no person detection or new model inference.</p>'
            '<p>Depth uses model units, without metric calibration. Each clip has its own '
            'colour limits and scale. White in the binary mask indicates temporal consistency '
            'under exploratory thresholds. Persistent occluders may remain.</p>'
            + ''.join(cards) + '</html>')
    (output / 'index.html').write_text(text, encoding='utf-8')
    (output / 'README.txt').write_text(
        'Background depth estimation from existing LingBot predictions\n\n'
        'background_depth.npy: float32 [height, width], depth[y, x], per-pixel temporal median.\n'
        'background_depth.npz: median, mean, stable depth, mask, MAD, counts, confidence and scale diagnostic.\n'
        'stable_background_depth.npy: same median with NaN for pixels failing stability criteria.\n'
        'stable_mask.png: original prediction-grid binary image (white=1; black=0).\n'
        'background_depth_heatmap.png: median depth with model-unit colour bar (1st-99th percentile limits).\n'
        'validation_preview.png: RGB reference, depth, temporal variability and binary mask.\n'
        'background_rgb_reference.png: temporal median of aligned saved RGB frames.\n'
        'metadata.json: source provenance, settings, limitations and output checksums.\n\n'
        'No person labels or inference. Estimates are separate per clip. No scale alignment is applied.\n'
        'Stability is not proof of background: an object present for most frames may be retained.\n'
        'No known physical distance was supplied; values are not validated metres.\n'
        'Coordinates refer to the resized prediction grid. Never sample heatmap colours as depth.\n'
        'An empty-scene reference or longer recordings can improve background visibility.\n\n'
        'Reproduce: python -m geometry3d.background_depth --output OUTPUT_FOLDER\n'
        'Dependencies: numpy, matplotlib, Pillow. Use --video STEM to process one clip.\n',
        encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=REPO / 'data/lingbot')
    parser.add_argument('--output', type=Path, default=REPO / 'data/lingbot/background_depth')
    parser.add_argument('--video', help='Only process this video stem')
    parser.add_argument('--relative-tolerance', type=float, default=0.10)
    parser.add_argument('--min-inlier-fraction', type=float, default=0.80)
    parser.add_argument('--min-valid-fraction', type=float, default=0.90)
    args = parser.parse_args()
    for name in ['relative_tolerance', 'min_inlier_fraction', 'min_valid_fraction']:
        if not 0 < getattr(args, name) <= 1:
            parser.error(f'{name} must be in (0, 1]')
    manifests = sorted(args.input.resolve().glob('*/processing_manifest.json'))
    if args.video:
        manifests = [p for p in manifests if source_info(json.loads(p.read_text()))['stem'] == args.video]
    if not manifests:
        parser.error('No matching saved LingBot runs found')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = [process_run(path, output, args) for path in manifests]
    (output / 'batch_summary.json').write_text(json.dumps({
        'videos': len(summaries), 'frames': sum(x['frames_used'] for x in summaries),
        'script_sha256': sha256(Path(__file__)), 'results': summaries,
    }, indent=2) + '\n', encoding='utf-8')
    write_index(output, summaries)
    print(output / 'index.html', flush=True)


if __name__ == '__main__':
    main()
