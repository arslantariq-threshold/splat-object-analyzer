"""One durable queue consumer; each model stage runs in a separate process."""
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from pipeline_progress import progress_from_line
from .results import analysis_result, query_result
from .storage import now, write_json

log = logging.getLogger(__name__)


class Stopping(Exception):
    pass


class Worker:
    def __init__(self, store, app_root, timeout=21600):
        self.store = store
        self.app_root = Path(app_root)
        self.timeout = timeout
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.process_lock = threading.Lock()
        self.process = None
        self.thread = None
        self.lock_file = None

    def start(self):
        self.lock_file = (self.store.root/'worker.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError('Another API worker owns this data directory. Run one Uvicorn worker per GPU.')
        try:
            self.store.recover()
            self.thread = threading.Thread(target=self.loop, name='splat-gpu-worker', daemon=True)
            self.thread.start()
        except Exception:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None
            raise

    @staticmethod
    def terminate(process):
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        except ProcessLookupError:
            pass

    def close(self):
        self.stop.set()
        self.wake.set()
        with self.process_lock:
            process = self.process
        self.terminate(process)
        if self.thread:
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                raise RuntimeError('GPU worker did not stop; retaining its exclusive lock')
        if self.lock_file:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None

    def loop(self):
        while not self.stop.is_set():
            record = self.store.claim()
            if not record:
                self.wake.wait(timeout=1)
                self.wake.clear()
                continue
            try:
                self.run(record)
                self.store.update(record['id'], status='succeeded', finished_at=now(),
                    progress=dict(stage='Results ready', completed=None, total=None))
            except Stopping:
                self.store.update(record['id'], status='queued', finished_at=None,
                    progress=dict(stage='Waiting for service restart', completed=None, total=None))
            except Exception:
                log.exception('Job %s failed', record['id'])
                self.store.update(record['id'], status='failed', finished_at=now(),
                    error=dict(code='processing_failed', message='Processing failed. Check this job’s server log for details.', details=[]),
                    progress=dict(stage='Processing failed', completed=None, total=None))

    def command(self, record, arguments):
        if self.stop.is_set():
            raise Stopping()
        environment = os.environ.copy()
        environment.setdefault('HF_HOME', str(self.app_root/'.cache/huggingface'))
        environment.setdefault('TORCH_HOME', str(self.app_root/'.cache/torch'))
        environment.setdefault('TORCH_EXTENSIONS_DIR', str(self.app_root/'.cache/torch-extensions'))
        environment.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        environment.setdefault('MAX_JOBS', '4')
        environment['PYTHONUNBUFFERED'] = '1'
        expired = threading.Event()
        with (self.store.directory(record['id'])/'worker.log').open('a') as log_file:
            with self.process_lock:
                if self.stop.is_set():
                    raise Stopping()
                process = subprocess.Popen([sys.executable, *map(str, arguments)], cwd=self.app_root,
                    env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    bufsize=1, start_new_session=True)
                self.process = process
            def timeout():
                expired.set()
                self.terminate(process)
            timer = threading.Timer(self.timeout, timeout)
            timer.daemon = True
            timer.start()
            previous = None
            try:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    progress = None
                    if line.startswith('[understand] '):
                        value = json.loads(line[len('[understand] '):])
                        progress = {k:value.get(k) for k in ('stage','completed','total')}
                    else:
                        progress = progress_from_line(line)
                    if progress and progress != previous:
                        self.store.update(record['id'], progress=progress)
                        previous = progress
                code = process.wait()
                if self.stop.is_set():
                    raise Stopping()
                if expired.is_set():
                    raise TimeoutError('Model stage exceeded its configured time limit')
                if code:
                    raise RuntimeError(f'Model stage exited with code {code}')
            finally:
                timer.cancel()
                self.terminate(process)
                process.stdout.close()
                with self.process_lock:
                    self.process = None

    def run(self, record):
        directory = self.store.directory(record['id'])
        if record['kind'] == 'analysis':
            output = directory/'analysis'
            output.mkdir(exist_ok=True)
            inventory_path = output/'interactions.json'
            # interactions.json is atomically published only after analysis finishes.
            # Restarted semantic jobs reuse that checkpoint and their caption cache.
            checkpoint = json.loads(inventory_path.read_text()) if inventory_path.exists() else None
            if (not checkpoint or checkpoint.get('input', {}).get('sha256') != record['sha256'] or
                checkpoint.get('input', {}).get('camera_input_sha256') != record.get('camera_input_sha256')):
                arguments = [self.app_root/'run_local.py', '--ply', directory/('source.'+record['format']),
                    '--quality', record['quality'], '--job_dir', output]
                if not record['understanding']:
                    arguments.append('--no-environment')
                if not record.get('navigation', True):
                    arguments.append('--no-navigation')
                if record.get('camera_input_sha256'):
                    arguments.extend(['--cameras', directory/'camera_input.json'])
                    if record.get('camera_limit') is not None:
                        arguments.extend(['--camera-limit', record['camera_limit']])
                self.command(record, arguments)
            inventory = json.loads(inventory_path.read_text())
            transforms = json.loads((output/'transforms.json').read_text())
            navigation = None
            if record.get('navigation', True):
                if not all((output/name).is_file() for name in ('navigation.json', 'navigation-grid.json', 'navigation.npz')):
                    self.command(record, [self.app_root/'navigation.py', '--splat', directory/('source.'+record['format']),
                        '--output', output])
                navigation = json.loads((output/'navigation.json').read_text())
            semantic = None
            if record['understanding']:
                response_path = directory/'semantic.json'
                self.command(record, [self.app_root/'scene_understanding.py', '--result', inventory_path,
                    '--prompt', '', '--output', response_path])
                semantic = json.loads(response_path.read_text())
            result = analysis_result(record, inventory, transforms, semantic, navigation=navigation)
        else:
            output = self.store.directory(record['analysis_id'])/'analysis'
            response_path = directory/'semantic.json'
            self.command(record, [self.app_root/'scene_understanding.py', '--result', output/'interactions.json',
                '--prompt', record['prompt'], '--output', response_path])
            result = query_result(record, json.loads(response_path.read_text()))
        write_json(directory/'result.json', result.model_dump(mode='json'))
