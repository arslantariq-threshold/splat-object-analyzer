"""Capture parsing, bounded sampling and render-path checks without model downloads."""
import json
from pathlib import Path
import struct
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from camera_inputs import load_camera_input, camera_fingerprint
from config import PipelineConfig
import render_cameras as render

TEXT = '# Number of images: 2\n1 0.7071067811865476 0 0 0.7071067811865476 1 2 3 1 front/image with spaces.jpg\n0 0 -1\n2 1 0 0 0 0 0 0 2 back/image.jpg\n\n'


class PreviewRenderer:
    device = torch.device('cpu')
    name = 'fixture'

    def prepare(self, arrays):
        return arrays

    def render_rgb(self, g, poses, K, width, height):
        checker = (np.indices((height, width)).sum(axis=0) // 4 % 2 * 200 + 30).astype(np.uint8)
        return np.repeat(np.repeat(checker[None, :, :, None], 3, axis=3), len(poses), axis=0)

    def render_depth(self, g, poses, K, width, height):
        return np.full((len(poses), height, width), 2., dtype=np.float32)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    text = root/'images.txt'; text.write_text(TEXT)
    model = load_camera_input(text)
    expected = -Rotation.from_quat([0, 0, 2**-.5, 2**-.5]).as_matrix().T @ [1, 2, 3]
    assert np.allclose(model['poses'][0]['position'], expected)
    assert model['poses'][0]['name'] == 'front/image with spaces.jpg'
    normalized = root/'camera.json'; normalized.write_text(json.dumps(model))
    assert camera_fingerprint(load_camera_input(normalized)) == camera_fingerprint(model)
    text.write_text(TEXT.replace('\n', '\r\n'))
    assert load_camera_input(text) == model

    binary = root/'images.bin'
    with binary.open('wb') as stream:
        stream.write(struct.pack('<Q', 2))
        for p, q, t in [(model['poses'][0], [2**-.5,0,0,2**-.5], [1,2,3]), (model['poses'][1], [1,0,0,0], [0,0,0])]:
            stream.write(struct.pack('<i7di', p['id'], *q, *t, p['camera_id']))
            stream.write(p['name'].encode() + b'\0' + struct.pack('<Q', 1) + struct.pack('<ddq', 12, 34, -1))
    b = load_camera_input(binary)
    assert np.allclose([p['position'] for p in b['poses']], [p['position'] for p in model['poses']])
    assert load_camera_input(root)['format'] == 'colmap_binary'
    binary.write_bytes(binary.read_bytes()[:-1])
    try: load_camera_input(binary)
    except ValueError: pass
    else: raise AssertionError('Truncated binary accepted')
    for value in [TEXT.replace('2 1 0 0 0', '1 1 0 0 0'), TEXT.replace('images: 2', 'images: 3'), TEXT.replace('1 0.7071067811865476 0 0 0.7071067811865476', '1 0 0 0 0')]:
        text.write_text(value)
        try: load_camera_input(text)
        except ValueError: pass
        else: raise AssertionError('Malformed text model accepted')
    for value in [dict(space='viewer', positions=[[0,0,0]]), dict(space='source',positions=[[float('nan'),0,0]]), dict(space='source',positions=[])]:
        normalized.write_text(json.dumps(value))
        try: load_camera_input(normalized)
        except ValueError: pass
        else: raise AssertionError('Invalid camera JSON accepted')
    print('PASS text/binary/JSON parsing, independent pose inversion, provenance and malformed inputs')

    axis = np.linspace(-1, 1, 31)
    volume = np.array(np.meshgrid(axis, axis, axis)).reshape(3, -1).T
    shell = volume[np.any(np.abs(volume) == 1, axis=1)]
    # 300 capture instants, two nearly co-located lenses; adjacent frames overlap.
    positions = [[.5*np.sin(i/300*2*np.pi), 0, .5*np.cos(i/300*2*np.pi)+lens*.002] for lens in range(2) for i in range(300)]
    normalized.write_text(json.dumps(dict(space='source', positions=positions)))
    cfg = PipelineConfig(camera_input=str(normalized), camera_candidates=12, camera_limit=2, n_positions=3, preview_size=32)
    candidates, _, sampling = render._provided_camera_positions(shell, cfg)
    assert len(candidates) <= 12 < sampling['input_pose_count'] == 600
    assert len(candidates) == len(np.unique(candidates, axis=0))
    assert all(np.min(np.linalg.norm(np.array(positions)-p, axis=1)) < 1e-6 for p in candidates)
    assert np.max(np.linalg.norm(candidates[:,None]-candidates[None],axis=2)) > .9
    with patch.object(render, '_generate_camera_positions', side_effect=AssertionError('Automatic fallback used')):
        selected, _, report, _, _ = render._preview_camera_positions(shell, PreviewRenderer(), {}, cfg, root/'camera_diagnostics.json')
    assert len(selected) == 2
    assert len(report['camera_input']['selected_image_ids']) == 2
    assert report['camera_input']['sha256'] == camera_fingerprint(load_camera_input(normalized))
    print('PASS 600 poses reduced to bounded, spatially diverse, preview-validated capture locations')

    source = root/'test.ply'; source.write_bytes(b'ply\nfixture')
    cfg.navigation = False; cfg.width = cfg.height = 64; cfg.min_votes = 1
    cfg.n_azimuth = 4; cfg.n_elevation = 1
    arrays = dict(means=shell, opacities=np.full(len(shell), 5.))
    with patch.object(render, 'get_renderer', return_value=PreviewRenderer()), patch.object(render, '_load_ply_arrays', return_value=arrays):
        output = render.render_views(str(source), str(root/'render'), cfg)
    transforms = json.loads(Path(output).read_text())
    assert transforms['camera_placement'] == 'provided_capture_surface_coverage'
    assert len(transforms['frames']) == 8
    assert len(transforms['camera_positions']) == 2
    for frame in transforms['frames']:
        assert (root/'render'/frame['file_path']).is_file()
        pose = np.array(frame['transform_matrix'])
        assert np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-6)
        assert np.linalg.det(pose[:3,:3]) > .99999
    cfg.camera_limit = 1
    try: render.render_views(str(source), str(root/'render'), cfg)
    except ValueError as error: assert 'stale analysis caches' in str(error)
    else: raise AssertionError('Changed capture settings reused an old output directory')
    cfg.camera_limit = 2
    normalized.write_text(json.dumps(dict(space='source', positions=[[1000,1000,1000]])))
    try: render._provided_camera_positions(shell, cfg)
    except ValueError as error: assert 'alignment' in str(error)
    else: raise AssertionError('Unusable capture locations silently accepted')
    print('PASS capture render path writes matched RGB/depth/matrices and rejects mismatched locations')

    auto = PipelineConfig(camera_candidates=12, n_positions=2, preview_size=32, navigation=False,
        width=64, height=64, n_azimuth=4, n_elevation=1)
    with patch.object(render, 'get_renderer', return_value=PreviewRenderer()), \
         patch.object(render, '_load_ply_arrays', return_value=arrays), \
         patch.object(render, 'load_camera_input', side_effect=AssertionError('Splat-only rendering parsed camera files')), \
         patch.object(render, '_provided_camera_positions', side_effect=AssertionError('Splat-only rendering required capture poses')):
        output = render.render_views(str(source), str(root/'automatic'), auto)
    automatic = json.loads(Path(output).read_text())
    assert automatic['camera_placement'] == 'preview_validated_surface_coverage'
    assert 'camera_input' not in automatic['camera_validation']
    assert len(automatic['camera_positions']) == 2 and len(automatic['frames']) == 8
    print('PASS splat-only rendering uses automatic placement without reading or requiring camera inputs')

if len(sys.argv) > 1:
    real = load_camera_input(sys.argv[1])
    assert len(real['poses']) == 600
    print(f'PASS supplied real model: {len(real["poses"])} poses')
print('COLMAP capture checks passed. Render-path checks above use a fixture renderer, not GPU inference.')
