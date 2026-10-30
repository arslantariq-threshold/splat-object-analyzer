"""Regression checks: retain >3 instances; never count duplicate image votes."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pipeline import _cluster_detections, _world_box_size, _postprocess_detection_groups
from auto_discover import normalise_tag
from config import PipelineConfig
from render_cameras import _generate_camera_positions


def detection(x, frame):
    return {'label': 'chair', 'position': np.array([x, 0., 0.]),
            'scale': np.ones(3), 'score': .9, 'frame_idx': frame,
            'box_2d': [x * 100, 0., x * 100 + 10, 10.]}


chairs = [detection(x, frame) for x in [0., 3., 6., 9.] for frame in range(8)]
assert len(_cluster_detections(chairs, eps_m=.5)) == 4, 'Instances are being truncated'
duplicates = [detection(0., 0) for _ in range(12)]
assert not _cluster_detections(duplicates), 'Repeated detections in one image counted as independent views'
neighbors = [detection(x, frame) for frame in range(8) for x in [0., .2]]
assert len(_cluster_detections(neighbors, eps_m=.5)) == 2, 'Simultaneously visible neighbors were merged'
repeated_object = [detection(x, frame) for frame in range(8) for x in [0., .4]]
for det in repeated_object:
    det['box_2d'] = [0., 0., 100., 100.]
assert len(_cluster_detections(repeated_object, eps_m=.5)) == 1, 'Overlapping duplicate instances were retained'
assert normalise_tag(' Bookcase ') == 'bookshelf'
assert normalise_tag('indoor') is None
# OWLv2 chooses one winning class per image patch. Added environment queries
# must not suppress the legacy object result from the same patch.
import torch
from types import SimpleNamespace
from transformers import Owlv2ImageProcessor
class Processor:
    def post_process_grounded_object_detection(self, outputs, **kwargs):
        return Owlv2ImageProcessor().post_process_object_detection(outputs, **kwargs)
outputs = SimpleNamespace(logits=torch.tensor([[[1., 2., 9.]]]), pred_boxes=torch.tensor([[[.5, .5, .4, .4]]]))
combined = _postprocess_detection_groups(Processor(), outputs, .1, torch.tensor([[64,64]]), 2)
legacy = _postprocess_detection_groups(Processor(), SimpleNamespace(logits=outputs.logits[..., :2], pred_boxes=outputs.pred_boxes), .1, torch.tensor([[64,64]]))
assert combined['labels'].tolist() == [1,2]
torch.testing.assert_close(combined['scores'][:1], legacy['scores'])
torch.testing.assert_close(combined['boxes'][:1], legacy['boxes'])
# A quarter-turn swaps camera X/Z extents in the exported world-aligned box.
quarter_turn = np.eye(4)
quarter_turn[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
np.testing.assert_allclose(_world_box_size(2., 4., quarter_turn), [3., 4., 2.])
# A room shell should produce cameras in open space rather than on its walls.
grid = np.linspace(-1., 1., 15)
volume = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
shell = volume[np.any(np.abs(volume) == 1, axis=1)]
cameras, _ = _generate_camera_positions(shell, 3, PipelineConfig())
clearance = np.min(np.linalg.norm(cameras[:, None, :] - shell[None, :, :], axis=2), axis=1)
assert (clearance > .5).all(), 'Room cameras are clipping through surfaces'

if len(sys.argv) > 1:
    result_path = Path(sys.argv[1])
    result = json.loads(result_path.read_text())
    assert result['discovery']['mode'] == 'automatic'
    assert result['discovery']['user_supplied_labels'] is False
    with Path(result['input']['path']).open('rb') as source:
        assert hashlib.file_digest(source, 'sha256').hexdigest() == result['input']['sha256']
    objects = result['objects']
    assert objects, 'The integration scene returned no objects'
    assert len({o['id'] for o in objects}) == len(objects)
    for obj in objects:
        assert obj['label'] in result['discovery']['labels']
        assert 0 <= obj['confidence'] <= 1
        assert obj['supporting_view_count'] == len({f['frame_idx'] for f in obj['frames']})
        assert obj['supporting_view_count'] >= result['settings']['min_votes']
        for axis in 'xyz':
            low, high = obj['bounding_box']['min'][axis], obj['bounding_box']['max'][axis]
            assert np.isfinite([low, high, obj['position'][axis]]).all()
            assert low < obj['position'][axis] < high
            assert np.isclose(high - low, obj['scale'][axis])
        assert all((result_path.parent / f['image']).exists() for f in obj['frames'])
    print(f'Validated automatic output: {len(objects)} candidates, verified source hash and box bounds.')

print('Automatic pipeline regression checks passed.')
