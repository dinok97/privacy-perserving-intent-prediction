"""Match scene polygons to saved background depth and project them into 3D.

No perception models are rerun. Outputs are derived from existing polygons,
temporal-median depth and predicted cameras. Scale is not calibrated in metres.
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
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import trimesh
from project_paths import resolve_project_path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NAMES = ['unlabelled', 'floor', 'wall', 'door', 'window']
COLOURS = np.array([[140, 148, 158], [75, 166, 235], [235, 183, 59],
                    [231, 83, 91], [150, 108, 225]], dtype=np.uint8)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def polygon_mask(obj, width, height):
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(p) for p in obj['polygon']], fill=1)
    for ring in obj.get('holes', []):
        draw.polygon([tuple(p) for p in ring], fill=0)
    return np.asarray(image, dtype=bool)


def align_labels(scene, width, height, *, allow_undistorted=False):
    source_width, source_height = scene['image_size']
    if '_undistorted' in scene['image'] and not allow_undistorted:
        raise ValueError('Undistorted polygons require a lens remapping, not only resizing.')
    original_classes = np.zeros((source_height, source_width), dtype=np.uint8)
    original_instances = np.zeros((source_height, source_width), dtype=np.int32)
    masks = [polygon_mask(obj, source_width, source_height) for obj in scene['objects']]
    # Paint weakest objects first. Manual annotations win, then confidence,
    # then smaller regions. All ties resolve deterministically by object index.
    def priority(i):
        confidence = scene['objects'][i].get('confidence')
        return (confidence is None, float(confidence or 0), -int(masks[i].sum()), i)
    overlaps = np.zeros_like(original_instances)
    for mask in masks:
        overlaps += mask
    for i in sorted(range(len(masks)), key=priority):
        label = scene['objects'][i]['label'].lower()
        if label not in NAMES:
            raise ValueError(f'Unsupported scene label: {label}')
        original_classes[masks[i]] = NAMES.index(label)
        original_instances[masks[i]] = i + 1

    # PIL nearest-neighbour uses pixel-centre correspondence and never blends IDs.
    classes = np.asarray(Image.fromarray(original_classes).resize(
        (width, height), Image.Resampling.NEAREST)).copy()
    instances = np.asarray(Image.fromarray(original_instances).resize(
        (width, height), Image.Resampling.NEAREST)).astype(np.int32)
    sx, sy = width / source_width, height / source_height
    aligned_objects = []
    for i, obj in enumerate(scene['objects']):
        def transform(ring):
            return [[(float(x) + 0.5) * sx - 0.5,
                     (float(y) + 0.5) * sy - 0.5] for x, y in ring]
        aligned_objects.append({**obj, 'instance_id': i + 1,
                                'polygon': transform(obj['polygon']),
                                'holes': [transform(ring) for ring in obj.get('holes', [])]})
    # Resize a unique source-pixel index grid to independently verify registration.
    # Preserve the library's tie-breaking at exact half-pixel boundaries.
    source_indices = np.arange(source_width * source_height, dtype=np.int32).reshape(
        source_height, source_width)
    lookup = np.asarray(Image.fromarray(source_indices).resize(
        (width, height), Image.Resampling.NEAREST))
    xs = lookup[0] % source_width
    ys = lookup[:, 0] // source_width
    expected_x = (np.arange(width) + 0.5) / sx
    expected_y = (np.arange(height) + 0.5) / sy
    assert np.all((xs == np.floor(expected_x)) | np.isclose(expected_x, xs + 1, atol=1e-9, rtol=0))
    assert np.all((ys == np.floor(expected_y)) | np.isclose(expected_y, ys + 1, atol=1e-9, rtol=0))
    assert np.array_equal(classes, original_classes[np.ix_(ys, xs)])
    assert np.array_equal(instances, original_instances[np.ix_(ys, xs)])
    return classes, instances, {
        'source_image': scene['image'], 'source_image_size': scene['image_size'],
        'image_size': [width, height], 'scale_x': sx, 'scale_y': sy,
        'crop_offset': [0, 0], 'padding': [0, 0],
        'pixel_centre_forward_transform': 'x_new=(x_old+0.5)*scale_x-0.5; same for y',
        'label_grid_transfer': 'nearest-neighbour resize of source label and instance rasters',
        'nearest_source_x_indices': xs.tolist(), 'nearest_source_y_indices': ys.tolist(),
        'raster_authority': 'aligned_label_map.npy; transformed polygons are descriptive',
        'overlap_policy': 'manual annotation, then higher confidence, then smaller region, then later object',
        'source_overlap_pixels': int((overlaps > 1).sum()),
        'class_names': dict(enumerate(NAMES)), 'objects': aligned_objects,
    }


def project(depth, intrinsic, extrinsic):
    y, x = np.indices(depth.shape)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsic).T
    camera = rays * depth[..., None]
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    world = (camera - translation) @ rotation
    # Verify against the inverse operation on all valid pixels.
    reconstructed_camera = world @ rotation.T + translation
    projected = reconstructed_camera @ intrinsic.T
    uv = projected[..., :2] / projected[..., 2:3]
    error = float(np.nanmax(np.abs(uv - np.stack([x, y], axis=-1))))
    if error > 1e-3:
        raise ValueError(f'3D reprojection error {error} pixels exceeds tolerance')
    return camera.astype(np.float32), world.astype(np.float32), error


def write_ply(path, world, colours, classes, instances, uv):
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('class_id', 'u1'), ('instance_id', '<i4'),
        ('pixel_x', '<u2'), ('pixel_y', '<u2'),
    ])
    vertices = np.empty(len(world), dtype=dtype)
    for i, axis in enumerate(['x', 'y', 'z']):
        vertices[axis] = world[:, i]
    for i, channel in enumerate(['red', 'green', 'blue']):
        vertices[channel] = colours[:, i]
    vertices['class_id'] = classes
    vertices['instance_id'] = instances
    vertices['pixel_x'], vertices['pixel_y'] = uv[:, 0], uv[:, 1]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment semantic background cloud; model units; stable pixels only\n'
              f'element vertex {len(vertices)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property uchar class_id\nproperty int instance_id\n'
              'property ushort pixel_x\nproperty ushort pixel_y\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(vertices.tobytes())


def render_previews(out, stem, rgb, classes, world, stable):
    coloured = COLOURS[classes]
    overlay = rgb.copy()
    labelled = classes > 0
    overlay[labelled] = np.rint(0.52 * rgb[labelled] + 0.48 * coloured[labelled]).astype(np.uint8)
    Image.fromarray(coloured).save(out / 'aligned_label_colours.png')
    Image.fromarray(overlay).save(out / 'aligned_segmentation_overlay.png')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    for ax, image, title in zip(axes, [rgb, overlay, coloured],
                              ['Median RGB on depth grid', 'Matched segmentation overlay', 'Semantic label map']):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f'{stem}: matched pixels ({classes.shape[1]} x {classes.shape[0]})', fontsize=13)
    fig.legend(handles=[Patch(facecolor=COLOURS[i] / 255, label=NAMES[i])
                        for i in np.unique(classes)], loc='lower center', ncol=5)
    fig.savefig(out / 'alignment_preview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    sampling = np.zeros_like(stable)
    sampling[::3, ::3] = True
    keep = sampling & stable & np.isfinite(world).all(axis=-1)
    points = world[keep]
    ids = classes[keep]
    display = points[:, [0, 2, 1]] * [1, 1, -1]
    centre = (display.min(axis=0) + display.max(axis=0)) / 2
    span = max(np.ptp(display, axis=0))
    fig = plt.figure(figsize=(13, 6.4), layout='constrained')
    for n, (elevation, azimuth) in enumerate([(22, -65), (8, -90)], start=1):
        ax = fig.add_subplot(1, 2, n, projection='3d')
        for i in np.unique(ids):
            subset = display[ids == i]
            ax.scatter(*subset.T, s=1.1 if i else 0.5,
                       color=COLOURS[i] / 255, alpha=1 if i else 0.22,
                       depthshade=False, label=NAMES[i], rasterized=True)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_box_aspect([1, 1, 1])
        for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], centre):
            setter(c - span / 2, c + span / 2)
        ax.set_xlabel('World X (model units)')
        ax.set_ylabel('World Z')
        ax.set_zlabel('-World Y')
        ax.set_title('Oblique view' if n == 1 else 'View towards scene')
    fig.suptitle(f'{stem}: 3D semantic background (temporally stable pixels)', fontsize=13)
    fig.legend(handles=[Patch(facecolor=COLOURS[i] / 255, label=NAMES[i])
                        for i in np.unique(ids)], loc='lower center', ncol=5)
    fig.savefig(out / '3d_segmentation_preview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_viewer(out, stem, world, rgb, classes, instances, stable, depth, source_scene,
                 *, allow_undistorted=False):
    # process_scene is also called directly by the integrated pipeline, without
    # the gallery CLI that normally creates this shared offline dependency.
    javascript = out.parent / 'plotly.min.js'
    if not javascript.is_file():
        javascript.write_text(get_plotlyjs(), encoding='utf-8')
    source_width, source_height = source_scene['image_size']
    original_classes, _, _ = align_labels(source_scene, source_width, source_height,
                                         allow_undistorted=allow_undistorted)
    Image.fromarray(COLOURS[original_classes]).save(out / 'original_2d_segmentation.png')
    y, x = np.indices(classes.shape)
    sampling = (x % 2 == 0) & (y % 2 == 0)
    keep = sampling & stable & np.isfinite(world).all(axis=-1)
    figure = go.Figure()
    semantic_colours, rgb_colours = [], []
    for i in np.unique(classes[keep]):
        region = keep & (classes == i)
        p = world[region]
        custom = np.column_stack([x[region], y[region], depth[region], instances[region]])
        semantic = f'rgb({COLOURS[i,0]},{COLOURS[i,1]},{COLOURS[i,2]})'
        original = [f'rgb({r},{g},{b})' for r, g, b in rgb[region]]
        semantic_colours.append(semantic)
        rgb_colours.append(original)
        figure.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 2], z=-p[:, 1], mode='markers', name=NAMES[i],
            marker={'size': 2.0, 'color': semantic, 'opacity': 1 if i else 0.25},
            customdata=custom,
            hovertemplate=(f'{NAMES[i]}<br>Matched pixel: (%{{customdata[0]:.0f}}, '
                           '%{customdata[1]:.0f})<br>Depth: %{customdata[2]:.4f} model units'
                           '<br>Object ID: %{customdata[3]:.0f}<extra></extra>'),
        ))
    scene = {
        'aspectmode': 'data', 'xaxis': {'title': 'World X (model units)'},
        'yaxis': {'title': 'World Z'}, 'zaxis': {'title': '-World Y'},
        'camera': {'eye': {'x': 1.25, 'y': -1.8, 'z': 0.8}, 'up': {'x': 0, 'y': 0, 'z': 1}},
    }
    figure.update_layout(
        title={'text': f'{stem}<br><sup>3D labels from matched background pixels; model units</sup>', 'x': 0.03},
        scene=scene, template='plotly_white', height=780,
        margin={'l': 10, 'r': 10, 'b': 95, 't': 105}, uirevision=stem,
        legend={'title': {'text': 'Click a class to hide/show'}},
        updatemenus=[{
            'type': 'buttons', 'direction': 'left', 'x': 0, 'y': 1.06,
            'buttons': [
                {'label': 'Semantic colours', 'method': 'restyle', 'args': [{'marker.color': semantic_colours}]},
                {'label': 'Original RGB', 'method': 'restyle', 'args': [{'marker.color': rgb_colours}]},
            ],
        }],
        annotations=[{'text': 'Drag to rotate; scroll to zoom; hover for pixel and depth.<br>'
                      'Only temporally stable pixels are shown.<br>Floor is not calibrated to gravity.',
                      'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': -0.07,
                      'showarrow': False, 'font': {'size': 12}}],
    )
    low, high = np.nanpercentile(depth, [1, 99])
    if high <= low:
        high = low + max(abs(low) * 0.01, 1e-6)
    details = np.stack([classes, stable.astype(np.uint8)], axis=-1)
    heatmap = go.Figure(go.Heatmap(
        z=depth, customdata=details, colorscale='Turbo', zmin=float(low), zmax=float(high),
        colorbar={'title': {'text': 'Depth<br>(model units)'}},
        hovertemplate='Pixel: (%{x}, %{y})<br>Depth: %{z:.5f} model units'
                      '<br>Class ID: %{customdata[0]}<br>Stable: %{customdata[1]} (1=yes)<extra></extra>',
    ))
    heatmap.update_layout(
        title={'text': 'Background pixel depth — temporal median', 'x': 0.03},
        template='plotly_white', height=650, margin={'l': 65, 'r': 90, 't': 70, 'b': 65},
        xaxis={'title': 'Pixel x (depth grid)', 'range': [-0.5, depth.shape[1] - 0.5],
               'constrain': 'domain'},
        yaxis={'title': 'Pixel y (depth grid)', 'range': [depth.shape[0] - 0.5, -0.5],
               'scaleanchor': 'x', 'scaleratio': 1, 'constrain': 'domain'},
    )
    config = {'displaylogo': False, 'scrollZoom': True, 'responsive': True}
    plot = figure.to_html(include_plotlyjs='../plotly.min.js', full_html=False,
                          div_id='scene-plot', config=config)
    depth_plot = heatmap.to_html(include_plotlyjs=False, full_html=False,
                                div_id='depth-plot', config=config)
    legend = ''.join(f'<span><i style="background:rgb({int(c[0])},{int(c[1])},{int(c[2])})"></i>'
                     f'{html.escape(name)} ({i})</span>' for i, (name, c) in enumerate(zip(NAMES, COLOURS)))
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Background views — STEM</title><style>
body{margin:0;background:#f3f5f7;color:#17232e;font:16px system-ui}
main{max-width:1280px;margin:auto;padding:18px}h1{font-size:22px;overflow-wrap:anywhere}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.tabs button{font:inherit;padding:10px 16px;border:1px solid #b6c6d4;border-radius:6px;
background:white;color:#245482;cursor:pointer}.tabs button[aria-selected="true"]{background:#245482;color:white}
section{background:white;border-radius:8px;padding:12px}section[hidden]{display:none}
.images{display:grid;grid-template-columns:1fr 1fr;gap:20px}.images img{width:100%;height:auto}
h2{font-size:18px}p{line-height:1.5}.legend{display:flex;gap:16px;flex-wrap:wrap;padding:12px 0}
.legend span{display:flex;align-items:center;gap:6px}.legend i{display:inline-block;width:18px;height:14px}
@media(max-width:850px){.images{grid-template-columns:1fr}}
</style></head><body><main><h1>STEM</h1>
<div class="tabs" role="tablist" aria-label="Background view">
<button id="tab-3d" role="tab" aria-controls="panel-3d" aria-selected="true" data-view="3d">3D segmentation</button>
<button id="tab-2d" role="tab" aria-controls="panel-2d" aria-selected="false" data-view="2d">Original 2D segmentation</button>
<button id="tab-depth" role="tab" aria-controls="panel-depth" aria-selected="false" data-view="depth">Pixel depth heatmap</button>
</div><section id="panel-3d" role="tabpanel" aria-labelledby="tab-3d">{{SCENE_PLOT}}</section>
<section id="panel-2d" role="tabpanel" aria-labelledby="tab-2d" hidden>
<div class="images"><div><h2>Original segmentation (SOURCE_SIZE)</h2>
<p>Existing polygon labels on their original image grid.</p>
<img src="original_2d_segmentation.png" alt="Original 2D segmentation label map"></div>
<div><h2>Matched overlay (DEPTH_SIZE)</h2><p>The labels used for the 3D projection, over the median background image.</p>
<img src="aligned_segmentation_overlay.png" alt="Segmentation aligned with the depth image"></div></div>
<div class="legend">LEGEND</div></section>
<section id="panel-depth" role="tabpanel" aria-labelledby="tab-depth" hidden>
<p>Hover to read each pixel’s depth. This shows the full median depth map; the 3D scene uses only stable pixels.
The colour range uses the 1st–99th percentiles; hover values retain the actual depth. Values are model units.</p>
{{DEPTH_PLOT}}</section></main><script>
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{
  const view=button.dataset.view;
  document.querySelectorAll('[data-view]').forEach(tab=>tab.setAttribute('aria-selected',String(tab===button)));
  document.querySelectorAll('[role="tabpanel"]').forEach(panel=>panel.hidden=panel.id!=='panel-'+view);
  if(view==='3d'||view==='depth') Plotly.Plots.resize(document.getElementById(view==='3d'?'scene-plot':'depth-plot'));
}));
</script></body></html>'''
    document = (document.replace('STEM', html.escape(stem))
                .replace('SOURCE_SIZE', f'{source_width} × {source_height}')
                .replace('DEPTH_SIZE', f'{depth.shape[1]} × {depth.shape[0]}')
                .replace('LEGEND', legend).replace('{{SCENE_PLOT}}', plot)
                .replace('{{DEPTH_PLOT}}', depth_plot))
    (out / 'viewer.html').write_text(document, encoding='utf-8')
    return int(keep.sum())


def process_scene(path, depth_root, output_root, *, video_stem=None, allow_undistorted=False):
    stem = video_stem or path.stem.removeprefix('scene_')
    background_folder = depth_root / stem
    metadata_path = background_folder / 'metadata.json'
    if not metadata_path.is_file():
        raise ValueError(f'No background depth estimate for {stem}')
    background_meta = json.loads(metadata_path.read_text())
    run = resolve_project_path(background_meta['source_run'])
    scene = json.loads(path.read_text())
    depth = np.load(background_folder / 'background_depth.npy', allow_pickle=False)
    with np.load(background_folder / 'background_depth.npz', allow_pickle=False) as data:
        stable = data['stable_mask'].astype(bool)
    rgb = np.asarray(Image.open(background_folder / 'background_rgb_reference.png').convert('RGB'))
    height, width = depth.shape
    source_width, source_height = scene['image_size']
    expected_height = round(source_height * (518 / source_width) / 14) * 14
    if (width, height) != (518, min(518, expected_height)) or expected_height > 518:
        raise ValueError('This scene requires preprocessing other than the verified uncropped resize.')
    classes, instances, aligned = align_labels(scene, width, height, allow_undistorted=allow_undistorted)
    intrinsics = []
    files = sorted((run / stem).glob('frame_*.npz'))
    for frame in files:
        with np.load(frame, allow_pickle=False) as prediction:
            intrinsics.append(prediction['intrinsic'])
    intrinsic = np.median(np.stack(intrinsics), axis=0).astype(np.float64)
    with np.load(files[0], allow_pickle=False) as prediction:
        extrinsic = prediction['extrinsic'].astype(np.float64)
    camera, world, projection_error = project(depth, intrinsic, extrinsic)
    keep = stable & np.isfinite(world).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    y, x = np.indices(depth.shape)
    out = output_root / stem
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'aligned_label_map.npy', classes, allow_pickle=False)
    np.save(out / 'aligned_instance_map.npy', instances, allow_pickle=False)
    np.savez_compressed(
        out / 'segmented_background.npz', depth=depth, class_id=classes,
        instance_id=instances, stable_mask=stable.astype(np.uint8),
        xyz_camera=camera, xyz_world=world, rgb=rgb,
        intrinsic=intrinsic, reference_extrinsic=extrinsic,
    )
    (out / 'aligned_scene.json').write_text(json.dumps(aligned, indent=2) + '\n', encoding='utf-8')
    Image.fromarray(classes).save(out / 'aligned_label_ids.png')
    world_points, ids = world[keep], classes[keep]
    write_ply(out / 'segmented_background.ply', world_points, COLOURS[ids],
              ids, instances[keep], np.column_stack([x[keep], y[keep]]))

    cloud_scene = trimesh.Scene()
    for i in np.unique(ids):
        indices = ids == i
        rgba = np.column_stack([np.broadcast_to(COLOURS[i], (indices.sum(), 3)),
                                np.full(indices.sum(), 255, dtype=np.uint8)])
        cloud_scene.add_geometry(trimesh.points.PointCloud(world_points[indices], colors=rgba),
                                 geom_name=NAMES[i], node_name=NAMES[i])
    # OpenCV world axes to a Y-up display convention. This is not a gravity calibration.
    glb_transform = np.diag([1, -1, -1, 1])
    cloud_scene.apply_transform(glb_transform)
    cloud_scene.export(out / 'segmented_background.glb')
    render_previews(out, stem, rgb, classes, world, keep)
    viewer_points = write_viewer(out, stem, world, rgb, classes, instances, keep, depth, scene,
                                  allow_undistorted=allow_undistorted)

    # Independently reload both 3D exports and verify all points survived export.
    glb = trimesh.load(out / 'segmented_background.glb', force='scene', process=False)
    glb_count = sum(len(geometry.vertices) for geometry in glb.geometry.values())
    ply = trimesh.load(out / 'segmented_background.ply', process=False)
    assert glb_count == len(world_points) == len(ply.vertices)
    assert np.allclose(ply.vertices, world_points, atol=1e-6)
    with np.load(out / 'segmented_background.npz', allow_pickle=False) as archive:
        assert archive['depth'].shape == archive['class_id'].shape == depth.shape
        assert np.array_equal(archive['class_id'], classes)
        assert np.array_equal(archive['instance_id'], instances)
    counts = {NAMES[i]: {'matched_pixels': int((classes == i).sum()),
                         'exported_3d_points': int((ids == i).sum())}
              for i in range(len(NAMES))}
    result = {
        'video': background_meta['video'], 'segmentation_source': path.resolve().as_posix(),
        'source_video_sha256': background_meta['source_video_sha256'],
        'segmentation_sha256': digest(path), 'background_depth_sha256': digest(background_folder / 'background_depth.npy'),
        'source_image_size': [source_width, source_height], 'matched_image_size': [width, height],
        'colour_legend': {NAMES[i]: COLOURS[i].tolist() for i in range(len(NAMES))},
        'class_counts': counts, 'source_objects': len(scene['objects']),
        'overlap_pixels_resolved': aligned['source_overlap_pixels'],
        'camera_model': 'pinhole, median predicted intrinsics across all frames',
        'pose_reference': 'frame_000000.npz world-to-camera extrinsic',
        'intrinsic': intrinsic.tolist(), 'reference_extrinsic': extrinsic.tolist(),
        'ply_coordinates': 'LingBot reference world coordinates',
        'glb_world_to_display_transform': glb_transform.tolist(),
        'viewer_axes': 'X=world X; Y=world Z; Z=-world Y',
        'exported_points': len(world_points), 'viewer_points': viewer_points,
        'maximum_roundtrip_reprojection_error_pixels': projection_error,
        'checks': {'discrete_pixel_correspondence_verified': True,
                   'npz_label_and_depth_shapes_match': True,
                   'ply_glb_point_counts_reloaded_and_verified': True},
        'limitations': [
            'Predicted camera parameters and depth are not physically calibrated.',
            'Temporal median and stability filtering do not guarantee empty background.',
            'Labels are transferred from existing polygons; segmentation errors are preserved.',
            'No gravity alignment, plane fitting or metric scale setting is performed.',
            'Viewer subsamples every second pixel; 3D exports keep every accepted pixel.',
        ],
    }
    result['output_files_sha256'] = {p.name: digest(p) for p in sorted(out.iterdir())
                                    if p.is_file() and p.name != 'metadata.json'}
    (out / 'metadata.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'{stem}: {len(world_points):,} points; '+', '.join(
        f'{name}={count["exported_3d_points"]:,}' for name, count in counts.items()), flush=True)
    return result


def write_index(output, results):
    cards = []
    for result in results:
        stem = Path(result['video']).stem
        safe = html.escape(stem)
        cards.append(
            f'<article><h2>{safe}</h2><p>{result["exported_points"]:,} accepted 3D points; '
            f'{result["source_image_size"][0]} x {result["source_image_size"][1]} segmentation '
            f'matched to {result["matched_image_size"][0]} x {result["matched_image_size"][1]} depth.</p>'
            f'<p><a class="button" href="{safe}/viewer.html">Open interactive 3D viewer</a></p>'
            f'<a href="{safe}/alignment_preview.png"><img src="{safe}/alignment_preview.png" '
            f'alt="Matched segmentation for {safe}" loading="lazy"></a>'
            f'<a href="{safe}/3d_segmentation_preview.png"><img src="{safe}/3d_segmentation_preview.png" '
            f'alt="3D semantic scene for {safe}" loading="lazy"></a>'
            f'<nav><a href="{safe}/segmented_background.glb">3D scene (.glb)</a>'
            f'<a href="{safe}/segmented_background.ply">Point cloud (.ply)</a>'
            f'<a href="{safe}/segmented_background.npz">Labels, depth and XYZ (.npz)</a>'
            f'<a href="{safe}/aligned_scene.json">Matched polygons (.json)</a>'
            f'<a href="{safe}/metadata.json">Settings and checks</a></nav></article>')
    (output / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Segmented 3D backgrounds</title><style>'
        'body{font:16px system-ui;max-width:1150px;margin:auto;padding:24px;background:#f3f5f7;'
        'color:#17232e}p{line-height:1.5}article{background:white;border-radius:8px;padding:20px;'
        'margin:24px 0}h2{font-size:19px;overflow-wrap:anywhere}img{width:100%;height:auto}'
        'nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#245482}'
        '.button{display:inline-block;background:#245482;color:white;padding:10px 16px;'
        'border-radius:5px;text-decoration:none}</style><h1>Segmented 3D backgrounds</h1>'
        f'<p>{len(results)} videos with existing scene segmentation. Floor is blue, walls gold, '
        'doors red and unlabelled regions grey. Labels are matched to the depth grid before projection.</p>'
        '<p>Depth and camera parameters are model estimates. Scale is not calibrated in metres. '
        'Only temporally stable pixels are exported; segmentation and persistent occlusion errors can remain.</p>'
        + ''.join(cards) + '</html>', encoding='utf-8')
    (output / 'README.txt').write_text(
        '3D background segmentation from existing 2D polygons and saved median depth\n\n'
        'aligned_label_map.npy / aligned_label_ids.png: one class ID per matched depth pixel.\n'
        'aligned_instance_map.npy: original JSON object IDs (1-based; 0=unlabelled).\n'
        'aligned_scene.json: transformed polygons and exact image-grid mapping metadata.\n'
        'segmented_background.npz: full-grid labels, median depth, RGB, camera/world XYZ and cameras.\n'
        'segmented_background.ply: full-resolution stable cloud with class/object IDs and matched pixel coordinates.\n'
        'segmented_background.glb: stable semantic-colour scene, separate named nodes per class.\n'
        'viewer.html: interactive offline 3D viewer (needs ../plotly.min.js alongside its parent index).\n'
        'Viewer tabs also show original 2D segmentation and a per-pixel median depth heatmap.\n'
        'original_2d_segmentation.png: source polygon labels at their original image resolution.\n'
        'alignment_preview.png: check 2D label/depth registration.\n'
        '3d_segmentation_preview.png: two 3D views of the same semantic scene.\n'
        'metadata.json: camera/coordinate conventions, class counts, checks and output hashes.\n\n'
        'IDs: 0 unlabelled; 1 floor; 2 wall; 3 door; 4 window. No new segmentation inference.\n'
        'Nearest-neighbour label-grid resizing preserves discrete labels. Existing overlapping polygons\n'
        'are resolved by manual status, confidence and area; no polygon is edited at source.\n'
        'Projection uses median predicted intrinsics and the first frame pose as a fixed-camera reference.\n'
        'Pinhole ray: X=(u-cx)*depth/fx; Y=(v-cy)*depth/fy; Z=depth. Invert reference pose for world XYZ.\n'
        'Viewer uses Z-up display axes (world X, world Z, -world Y); GLB flips world Y/Z for Y-up display.\n'
        'These display transforms are not gravity calibration. Values are not measured metres.\n'
        'The 3D model is a labelled point cloud, not a watertight mesh or newly fitted room layout.\n'
        'Videos without corresponding scene JSONs are not assigned invented labels.\n\n'
        'Reproduce: python -m geometry3d.segment_background_3d --output OUTPUT_FOLDER\n'
        'Install: pip install -r geometry3d/requirements-background-depth.txt\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, default=REPO / 'source')
    parser.add_argument('--depth', type=Path, default=REPO / 'data/lingbot/background_depth')
    parser.add_argument('--output', type=Path, default=REPO / 'data/lingbot/segmented_background')
    args = parser.parse_args()
    scenes = sorted(args.scenes.resolve().glob('scene_*.json'))
    if not scenes:
        parser.error('No scene segmentation JSONs found')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    results = [process_scene(path, args.depth.resolve(), output) for path in scenes]
    write_index(output, results)
    (output / 'batch_summary.json').write_text(json.dumps({
        'videos': len(results), 'script_sha256': digest(Path(__file__)),
        'results': results,
    }, indent=2) + '\n', encoding='utf-8')
    print(output / 'index.html', flush=True)


if __name__ == '__main__':
    main()
