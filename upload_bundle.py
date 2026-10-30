"""Bounded streaming multipart uploads shared by the API and local viewer."""
import hashlib
import json
from pathlib import Path

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from camera_inputs import load_camera_input, camera_fingerprint

MAX_CAMERA_UPLOAD = 512 * 1024**2


class UploadError(ValueError):
    def __init__(self, message, status=400, code='invalid_file'):
        super().__init__(message)
        self.status, self.code = status, code


class BundleUpload:
    """Only file parts are accepted. Never use client filenames as disk paths."""
    def __init__(self, content_type, destination, max_splat, max_cameras=MAX_CAMERA_UPLOAD):
        _, options = parse_options_header(content_type)
        boundary = options.get(b'boundary')
        if not boundary or len(boundary) > 200:
            raise UploadError('Missing or invalid multipart boundary')
        self.destination = Path(destination)
        self.max_splat, self.max_cameras = max_splat, max_cameras
        self.parts, self.files = {}, []
        self.size, self.camera_bytes, self.total = 0, 0, 0
        self.magic, self.digest = b'', hashlib.sha256()
        self.finished = False
        self.parser = MultipartParser(boundary, {name: getattr(self, name) for name in (
            'on_part_begin', 'on_header_field', 'on_header_value', 'on_header_end',
            'on_headers_finished', 'on_part_data', 'on_part_end', 'on_end')})

    def on_part_begin(self):
        self.headers, self.field, self.value = {}, bytearray(), bytearray()
        self.header_size, self.current, self.handle = 0, None, None

    def _header(self, data, start, end, target):
        self.header_size += end - start
        if self.header_size > 8192:
            raise UploadError('Multipart file headers are too large')
        target.extend(data[start:end])

    def on_header_field(self, data, start, end):
        self._header(data, start, end, self.field)

    def on_header_value(self, data, start, end):
        self._header(data, start, end, self.value)

    def on_header_end(self):
        self.headers[bytes(self.field).lower()] = bytes(self.value)
        self.field.clear(); self.value.clear()

    def on_headers_finished(self):
        _, options = parse_options_header(self.headers.get(b'content-disposition', b''))
        name = options.get(b'name', b'').decode('utf-8')
        filename = options.get(b'filename', b'').decode('utf-8')
        allowed = {'file', 'colmap_images', 'colmap_cameras', 'cameras'}
        if name not in allowed or name in self.parts or not filename:
            raise UploadError('Use one file part and optional colmap_images/colmap_cameras or cameras files')
        extension = Path(filename).suffix.lower()
        if name == 'colmap_images' and extension not in {'.txt', '.bin'}:
            raise UploadError('colmap_images must be images.txt or images.bin')
        if name == 'colmap_cameras' and extension != '.txt':
            raise UploadError('colmap_cameras must be cameras.txt')
        if name == 'cameras' and extension != '.json':
            raise UploadError('cameras must be source-space camera JSON')
        path = self.destination if name == 'file' else self.destination.parent / ('camera-upload-' + name + extension)
        self.parts[name] = path
        self.handle = path.open('xb')
        self.files.append(self.handle)
        self.current = name

    def on_part_data(self, data, start, end):
        chunk = data[start:end]
        if self.current == 'file':
            self.size += len(chunk)
            if self.size > self.max_splat:
                raise UploadError('File exceeds the configured upload limit', 413, 'file_too_large')
            self.magic = (self.magic + chunk[:4])[:4]
            self.digest.update(chunk)
        else:
            self.camera_bytes += len(chunk)
            if self.camera_bytes > self.max_cameras:
                raise UploadError('Camera files exceed the configured upload limit', 413, 'camera_files_too_large')
        self.handle.write(chunk)

    def on_part_end(self):
        self.handle.close()

    def on_end(self):
        self.finished = True

    def write(self, chunk):
        self.total += len(chunk)
        if self.total > self.max_splat + self.max_cameras + 65536:
            raise UploadError('Multipart request exceeds the configured upload limit', 413, 'file_too_large')
        self.parser.write(chunk)

    def finish(self):
        self.parser.finalize()
        if not self.finished or 'file' not in self.parts:
            raise UploadError('Incomplete multipart upload or missing file part')
        images, calibration, cameras = [self.parts.get(key) for key in ('colmap_images', 'colmap_cameras', 'cameras')]
        if cameras and (images or calibration) or calibration and not images:
            raise UploadError('Provide colmap_images with optional colmap_cameras, or cameras JSON', code='invalid_camera_input')
        if not images and not cameras:
            return None
        try:
            value = load_camera_input(images or cameras, calibration)
        except (ValueError, OSError, OverflowError) as error:
            raise UploadError(f'Invalid camera input: {error}', code='invalid_camera_input') from error
        target = self.destination.parent/'camera_input.json'
        target.write_text(json.dumps(value, allow_nan=False) + '\n')
        for key, path in self.parts.items():
            if key != 'file':
                path.unlink()
        return dict(path=target, sha256=camera_fingerprint(value), format=value['format'], input_pose_count=len(value['poses']))

    def close(self):
        for handle in self.files:
            handle.close()
