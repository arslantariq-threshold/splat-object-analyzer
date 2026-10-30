"""Optional capture locations, in the unmodified splat's source coordinate space.

Parsing is CPU-only and does not import a renderer/model. COLMAP orientations
are retained as provenance; analysis renders fresh perspective panoramas using
the same source-Y convention as automatic camera placement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

MAX_POSES = 200_000


def _positive_id(value):
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('COLMAP image and camera IDs must be positive integers') from error
    if str(number) != str(value) or number <= 0:
        raise ValueError('COLMAP image and camera IDs must be positive integers')
    return number


def _pose(image_id, quaternion, translation, camera_id, name):
    image_id, camera_id = _positive_id(image_id), _positive_id(camera_id)
    values = [float(v) for v in (*quaternion, *translation)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Camera poses must contain finite numbers')
    norm = math.hypot(*values[:4])
    if norm < 1e-8 or not math.isfinite(norm):
        raise ValueError('Camera quaternion must be nonzero')
    w, x, y, z = [v / norm for v in values[:4]]
    rotation = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    # COLMAP stores world-to-camera R,T; camera centre is -R.T @ T.
    position = [-sum(rotation[j][i] * values[4+j] for j in range(3)) for i in range(3)]
    if not all(math.isfinite(n) for n in position):
        raise ValueError('Camera position must contain finite source coordinates')
    return dict(id=image_id, camera_id=camera_id, name=name,
                position=position, world_to_camera_quaternion=[w, x, y, z])


def _read_text(path):
    poses, expected = [], None
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.lstrip().startswith('#'):
                match = re.match(r'# Number of images:\s*(\d+)', line.strip())
                if match:
                    expected = int(match[1])
                continue
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=9)
            if len(parts) != 10:
                raise ValueError('Invalid COLMAP image pose; provide images.txt, not cameras.txt')
            poses.append(_pose(parts[0], parts[1:5], parts[5:8], parts[8], parts[9]))
            # Discard potentially enormous feature lines without splitting them.
            if next(stream, None) is None:
                raise ValueError('Incomplete COLMAP images.txt: missing observations line')
            if len(poses) > MAX_POSES:
                raise ValueError('Too many camera poses')
    if expected is not None and expected != len(poses):
        raise ValueError(f'COLMAP declares {expected} images but contains {len(poses)} poses')
    return poses


def _read_binary(path):
    def unpack(stream, layout):
        size = struct.calcsize(layout)
        value = stream.read(size)
        if len(value) != size:
            raise ValueError('Incomplete COLMAP images.bin')
        return struct.unpack(layout, value)

    poses = []
    size = path.stat().st_size
    with path.open('rb') as stream:
        count, = unpack(stream, '<Q')
        if not 0 < count <= MAX_POSES:
            raise ValueError('Invalid COLMAP image count')
        for _ in range(count):
            values = unpack(stream, '<i7di')
            name = bytearray()
            while True:
                char = stream.read(1)
                if not char or len(name) > 16_384:
                    raise ValueError('Invalid COLMAP image name')
                if char == b'\0':
                    break
                name.extend(char)
            poses.append(_pose(values[0], values[1:5], values[5:8], values[8], name.decode('utf-8')))
            observations, = unpack(stream, '<Q')
            end = stream.tell() + observations * 24  # x,y: doubles; point3D_id: int64
            if end > size:
                raise ValueError('Incomplete COLMAP image observations')
            stream.seek(end)
        if stream.tell() != size:
            raise ValueError('Unexpected trailing bytes in images.bin')
    return poses


def _read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get('space') != 'source':
        raise ValueError('Camera JSON must declare space="source" and contain poses or positions')
    raw = value.get('poses', value.get('positions'))
    if not isinstance(raw, list) or not 0 < len(raw) <= MAX_POSES:
        raise ValueError('Camera JSON must contain a nonempty poses or positions array')
    poses = []
    for index, entry in enumerate(raw):
        item = entry if isinstance(entry, dict) else dict(position=entry)
        position = item.get('position')
        if (not isinstance(position, list) or len(position) != 3 or
            not all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in position)):
            raise ValueError('Every camera position must be an array of three finite source coordinates')
        image_id = item.get('id', index + 1)
        if not isinstance(image_id, (str, int)) or isinstance(image_id, bool) or not str(image_id) or len(str(image_id)) > 200:
            raise ValueError('Camera IDs must be nonempty strings or integers')
        name = item.get('name', str(image_id))
        if not isinstance(name, str) or len(name) > 16_384:
            raise ValueError('Invalid camera name')
        pose = dict(id=image_id, name=name, position=list(position), camera_id=_positive_id(item.get('camera_id', item.get('cameraId', 1))))
        # Normalized COLMAP uploads can be read again without discarding provenance.
        if 'world_to_camera_quaternion' in item:
            q = item['world_to_camera_quaternion']
            if not isinstance(q, list) or len(q) != 4 or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in q):
                raise ValueError('Invalid camera quaternion')
            pose['world_to_camera_quaternion'] = q
        poses.append(pose)
    return poses, value.get('format', 'camera_json')


def load_camera_input(path, calibration=None):
    path = Path(path)
    if path.is_dir():
        directory = path
        path = directory / ('images.bin' if (directory/'images.bin').is_file() else 'images.txt')
        calibration = calibration or (directory/'cameras.txt' if (directory/'cameras.txt').is_file() else None)
    if path.suffix.lower() == '.json':
        poses, kind = _read_json(path)
    elif path.suffix.lower() == '.txt':
        poses, kind = _read_text(path), 'colmap_text'
    elif path.suffix.lower() == '.bin':
        poses, kind = _read_binary(path), 'colmap_binary'
    else:
        raise ValueError('Provide COLMAP images.txt/images.bin, a model directory, or source-space camera JSON')
    if not isinstance(kind, str) or kind not in {'colmap_text', 'colmap_binary', 'camera_json'} or not poses:
        raise ValueError('No supported camera poses found')
    if len({str(p['id']) for p in poses}) != len(poses):
        raise ValueError('Duplicate camera image IDs')
    if calibration:
        ids = set()
        with Path(calibration).open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                fields = line.split()
                if len(fields) < 7:
                    raise ValueError('Invalid COLMAP cameras.txt')
                camera_id = _positive_id(fields[0])
                if camera_id in ids:
                    raise ValueError('Duplicate COLMAP calibration IDs')
                if int(fields[2]) <= 0 or int(fields[3]) <= 0 or not all(math.isfinite(float(n)) for n in fields[4:]):
                    raise ValueError('Invalid COLMAP camera calibration')
                ids.add(camera_id)
        if not ids or any(p['camera_id'] not in ids for p in poses):
            raise ValueError('Image pose references a camera missing from cameras.txt')
    return dict(schema_version='1.0', space='source', format=kind, poses=poses)


def camera_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def add_camera_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--colmap-model', dest='camera_input', help='Matching COLMAP directory or images.txt/images.bin')
    group.add_argument('--cameras', dest='camera_input', help='Source-space camera JSON or COLMAP images.txt/images.bin')
    parser.add_argument('--camera-limit', type=int, choices=range(1, 65), metavar='1..64',
                        help='Maximum selected capture locations; defaults to the quality preset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and normalize optional capture cameras')
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = load_camera_input(args.input)
    Path(args.output).write_text(json.dumps(value, allow_nan=False) + '\n')
    print(f'{len(value["poses"])} {value["format"]} poses; source coordinates; analysis renders fresh perspective orientations')
