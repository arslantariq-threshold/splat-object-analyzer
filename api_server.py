"""Versioned, headless API. Run with serve-api.sh; see API.md and deploy/README.md."""
from contextlib import asynccontextmanager
import asyncio
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Annotated, Literal
import uuid

from fastapi import APIRouter, Depends, FastAPI, Path as PathParam, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import (AnalysisResult, ErrorResponse, Job, QueryRequest, QueryResult,
    NavigationSummary, NavigationGridResult, NavigationValidationRequest, NavigationValidation, NavigationPathRequest, NavigationPath)
from api.storage import QueueFull, Store, now
from api.worker import Worker
from upload_bundle import BundleUpload, UploadError, MAX_CAMERA_UPLOAD

ROOT = Path(__file__).resolve().parent
JobId = Annotated[str, PathParam(pattern=r'^[a-f0-9]{32}$')]


@lru_cache(maxsize=2)
def navigation_map(directory, fingerprint):
    from navigation import NavigationMap
    value = NavigationMap.load(directory)
    if value.metadata['fingerprint'] != fingerprint:
        raise ValueError('Navigation map fingerprint mismatch')
    return value


@dataclass
class Settings:
    data_dir: Path
    api_key: str
    max_upload_bytes: int = 2 * 1024**3
    max_pending_jobs: int = 8
    stage_timeout_seconds: int = 21600
    cors_origins: tuple[str, ...] = ()
    require_cuda: bool = False
    max_camera_upload_bytes: int = MAX_CAMERA_UPLOAD

    @classmethod
    def from_env(cls):
        return cls(data_dir=Path(os.getenv('SPLAT_DATA_DIR', ROOT/'outputs/api')),
            api_key=os.getenv('SPLAT_API_KEY', ''), max_upload_bytes=int(os.getenv('SPLAT_MAX_UPLOAD_BYTES', str(2*1024**3))),
            max_pending_jobs=int(os.getenv('SPLAT_MAX_PENDING_JOBS', '8')),
            stage_timeout_seconds=int(os.getenv('SPLAT_STAGE_TIMEOUT_SECONDS', '21600')),
            cors_origins=tuple(s.strip() for s in os.getenv('SPLAT_CORS_ORIGINS', '').split(',') if s.strip()),
            require_cuda=os.getenv('SPLAT_REQUIRE_CUDA', '0') == '1',
            max_camera_upload_bytes=int(os.getenv('SPLAT_MAX_CAMERA_UPLOAD_BYTES', str(MAX_CAMERA_UPLOAD))))


class APIError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


def public_job(record):
    result_url = ('/v1/analyses/' if record['kind'] == 'analysis' else '/v1/queries/') + record['id']
    return Job.model_validate({k:record.get(k) for k in (
        'id','kind','analysis_id','status','created_at','started_at','finished_at','progress','error')} |
        dict(status_url='/v1/jobs/'+record['id'], result_url=result_url))


def create_app(settings=None, app_root=ROOT):
    settings = settings or Settings.from_env()
    @asynccontextmanager
    async def lifespan(app):
        if len(settings.api_key) < 32 or settings.api_key.startswith('replace-'):
            raise RuntimeError('Set SPLAT_API_KEY to a random secret of at least 32 characters before starting the API')
        if min(settings.max_upload_bytes, settings.max_camera_upload_bytes, settings.max_pending_jobs, settings.stage_timeout_seconds) <= 0:
            raise RuntimeError('Upload, queue, and timeout limits must be positive')
        if settings.require_cuda:
            subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA GPU is unavailable"'], check=True)
        app.state.store = Store(settings.data_dir, settings.max_pending_jobs)
        app.state.upload_slots = asyncio.Semaphore(2)
        app.state.worker = Worker(app.state.store, app_root, settings.stage_timeout_seconds)
        app.state.worker.start()
        try:
            yield
        finally:
            app.state.worker.close()

    app = FastAPI(title='Splat Object Analyzer API', version='1.0.0', lifespan=lifespan,
        description='Upload a Gaussian splat, poll its asynchronous job, and retrieve objects, geometry, scene groups, and prompt-selected views.')
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
            allow_methods=['GET','POST'], allow_headers=['Authorization','Content-Type'], expose_headers=['Location','Retry-After'])
    bearer = HTTPBearer(auto_error=False)
    async def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if not credentials or not hmac.compare_digest(credentials.credentials.encode(), settings.api_key.encode()):
            raise APIError(401, 'unauthorized', 'Provide a valid Authorization: Bearer API key')

    @app.exception_handler(APIError)
    async def api_error(request, error):
        headers = {'WWW-Authenticate':'Bearer'} if error.status == 401 else {}
        if error.status in (409,429): headers['Retry-After'] = '2'
        return JSONResponse(dict(error=dict(code=error.code, message=error.message, details=[])), status_code=error.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        details = [dict(field='.'.join(map(str,e['loc'])), message=e['msg']) for e in error.errors()]
        return JSONResponse(dict(error=dict(code='invalid_request', message='Request validation failed', details=details)), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, error):
        return JSONResponse(dict(error=dict(code='http_error', message=str(error.detail), details=[])), status_code=error.status_code)

    @app.exception_handler(Exception)
    async def internal_error(request, error):
        logging.getLogger(__name__).error('Unhandled API error', exc_info=error)
        return JSONResponse(dict(error=dict(code='internal_error', message='Unexpected server error. Check the service logs.', details=[])), status_code=500)

    @app.get('/health/live', tags=['health'])
    def live():
        return dict(status='alive')

    @app.get('/health/ready', tags=['health'])
    def ready():
        worker = getattr(app.state, 'worker', None)
        ok = bool(worker and worker.thread and worker.thread.is_alive() and not worker.stop.is_set())
        return JSONResponse(dict(status='ready' if ok else 'unavailable'), status_code=200 if ok else 503)

    router = APIRouter(prefix='/v1', dependencies=[Depends(authenticate)], responses={
        code:dict(model=ErrorResponse) for code in (400,401,404,409,413,415,422,429,500)})

    def get_record(job_id, kind=None):
        record = app.state.store.get(job_id)
        if not record or (kind and record['kind'] != kind):
            raise APIError(404, 'not_found', 'Job not found')
        return record

    def completed(job_id, kind):
        record = get_record(job_id, kind)
        if record['status'] != 'succeeded':
            raise APIError(409, 'result_not_ready', f'Result is unavailable while job status is {record["status"]}. Poll /v1/jobs/{job_id}.')
        return record

    async def upload_slot():
        async with app.state.upload_slots:
            yield

    @router.post('/analyses', response_model=Job, status_code=202, tags=['analysis'], dependencies=[Depends(upload_slot)],
        openapi_extra={'requestBody':{'required':True,'content':{
            'application/octet-stream':{'schema':{'type':'string','format':'binary'}},
            'multipart/form-data':{'schema':{'type':'object','required':['file'],'properties':{
                name:{'type':'string','format':'binary','description':description} for name, description in {
                    'file':'Gaussian splat PLY or SPZ', 'colmap_images':'Optional matching COLMAP images.txt or images.bin',
                    'colmap_cameras':'Optional cameras.txt calibration; requires colmap_images',
                    'cameras':'Alternative source-space camera JSON; mutually exclusive with COLMAP files'}.items()}}}}}})
    async def submit(request: Request, filename: str = Query(min_length=5, max_length=180),
                     quality: Literal['low','medium','high'] = 'high', understanding: bool = True, navigation: bool = True,
                     camera_limit: int | None = Query(default=None, ge=1, le=64)):
        if app.state.store.pending_count() >= settings.max_pending_jobs:
            raise APIError(429, 'queue_full', 'The processing queue is full. Retry later.')
        if (Path(filename).name != filename or '\\' in filename or any(ord(c) < 32 for c in filename) or
            Path(filename).suffix.lower() not in ('.ply','.spz')):
            raise APIError(400, 'invalid_file', 'Provide a plain .ply or .spz filename')
        content_type = request.headers.get('content-type', '')
        multipart = content_type.split(';')[0].lower() == 'multipart/form-data'
        if not multipart and content_type.split(';')[0].lower() != 'application/octet-stream':
            raise APIError(415, 'invalid_content_type', 'Send application/octet-stream or multipart/form-data')
        if camera_limit is not None and not multipart:
            raise APIError(400, 'invalid_camera_input', 'camera_limit requires a multipart upload with camera files')
        try:
            length = int(request.headers.get('content-length', '0'))
        except ValueError:
            raise APIError(400, 'invalid_content_length', 'Invalid Content-Length')
        if length < 0:
            raise APIError(400, 'invalid_content_length', 'Invalid Content-Length')
        if length > settings.max_upload_bytes + (settings.max_camera_upload_bytes + 65536 if multipart else 0):
            raise APIError(413, 'file_too_large', 'File exceeds the configured upload limit')
        job_id = uuid.uuid4().hex
        directory = app.state.store.directory(job_id)
        directory.mkdir()
        suffix = Path(filename).suffix.lower()[1:]
        partial = directory/'upload.partial'
        accepted = False
        size, magic, digest = 0, b'', hashlib.sha256()
        bundle, camera_input = None, None
        try:
            if multipart:
                try:
                    bundle = BundleUpload(content_type, partial, settings.max_upload_bytes, settings.max_camera_upload_bytes)
                    async for chunk in request.stream():
                        await run_in_threadpool(bundle.write, chunk)
                    camera_input = await run_in_threadpool(bundle.finish)
                    size, magic, digest = bundle.size, bundle.magic, bundle.digest
                except (ValueError, OverflowError) as error:
                    raise APIError(getattr(error, 'status', 400), getattr(error, 'code', 'invalid_file'), str(error)) from error
                finally:
                    if bundle:
                        bundle.close()
            else:
                with partial.open('xb') as destination:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > settings.max_upload_bytes:
                            raise APIError(413, 'file_too_large', 'File exceeds the configured upload limit')
                        magic = (magic+chunk[:4])[:4]
                        digest.update(chunk)
                        await run_in_threadpool(destination.write, chunk)
            if not size or (suffix == 'ply' and not magic.startswith(b'ply')) or (suffix == 'spz' and not magic.startswith(b'\x1f\x8b')):
                raise APIError(400, 'invalid_file', 'The file contents do not match the requested Gaussian splat format')
            partial.replace(directory/('source.'+suffix))
            if camera_limit is not None and camera_input is None:
                raise APIError(400, 'invalid_camera_input', 'camera_limit requires camera files')
            record = dict(id=job_id, kind='analysis', analysis_id=job_id, status='queued', created_at=now(),
                started_at=None, finished_at=None, progress=dict(stage='Queued',completed=None,total=None), error=None,
                filename=filename, format=suffix, bytes=size, sha256=digest.hexdigest(), quality=quality, understanding=understanding,
                navigation=navigation)
            if camera_input:
                record.update(camera_input_sha256=camera_input['sha256'], camera_input_format=camera_input['format'],
                              camera_input_pose_count=camera_input['input_pose_count'], camera_limit=camera_limit)
            try:
                app.state.store.create(record)
            except QueueFull:
                raise APIError(429, 'queue_full', 'The processing queue is full. Retry later.')
            accepted = True
            app.state.worker.wake.set()
            return JSONResponse(public_job(record).model_dump(mode='json'), status_code=202,
                headers={'Location':'/v1/jobs/'+job_id, 'Retry-After':'2'})
        finally:
            if not accepted:
                shutil.rmtree(directory, ignore_errors=True)

    @router.get('/jobs/{job_id}', response_model=Job, tags=['jobs'])
    def status(job_id: JobId):
        return public_job(get_record(job_id))

    @router.get('/analyses/{analysis_id}', response_model=AnalysisResult, tags=['analysis'])
    def result(analysis_id: JobId):
        completed(analysis_id, 'analysis')
        return AnalysisResult.model_validate_json((app.state.store.directory(analysis_id)/'result.json').read_text())

    def navigation_assets(analysis_id):
        record = completed(analysis_id, 'analysis')
        root = app.state.store.directory(analysis_id)/'analysis'
        if not record.get('navigation', True) or not (root/'navigation.json').is_file():
            raise APIError(404, 'navigation_unavailable', 'This analysis has no navigation map. Submit a new analysis with navigation=true.')
        metadata = json.loads((root/'navigation.json').read_text())
        if metadata['source_sha256'] != record['sha256']:
            raise ValueError('Navigation source fingerprint mismatch')
        return root, metadata

    @router.get('/analyses/{analysis_id}/navigation', response_model=NavigationSummary, tags=['navigation'])
    def navigation_summary(analysis_id: JobId):
        _, metadata = navigation_assets(analysis_id)
        return {**metadata, 'grid_url':f'/v1/analyses/{analysis_id}/navigation/grid'}

    @router.get('/analyses/{analysis_id}/navigation/grid', response_class=FileResponse, tags=['navigation'],
        responses={200:{'model':NavigationGridResult}},
        description='Full candidate map as JSON. runs contains alternating state and count values in C-order with shape [nx,ny,nz]; z varies fastest. Cell centers are origin+(index+0.5)*voxel_size.')
    def navigation_grid(analysis_id: JobId):
        root, _ = navigation_assets(analysis_id)
        if not (root/'navigation-grid.json').is_file():
            raise APIError(404, 'navigation_unavailable', 'Navigation grid asset is unavailable')
        return FileResponse(root/'navigation-grid.json', media_type='application/json', filename='navigation-grid.json')

    @router.post('/analyses/{analysis_id}/navigation/validate', response_model=NavigationValidation, tags=['navigation'])
    def validate_navigation(analysis_id: JobId, value: NavigationValidationRequest):
        root, metadata = navigation_assets(analysis_id)
        return navigation_map(root, metadata['fingerprint']).validate(value.points)

    @router.post('/analyses/{analysis_id}/navigation/path', response_model=NavigationPath, tags=['navigation'])
    def plan_navigation(analysis_id: JobId, value: NavigationPathRequest):
        root, metadata = navigation_assets(analysis_id)
        return navigation_map(root, metadata['fingerprint']).find_path(value.start, value.goal)

    @router.post('/analyses/{analysis_id}/queries', response_model=Job, status_code=202, tags=['queries'])
    def query(analysis_id: JobId, value: QueryRequest):
        parent = completed(analysis_id, 'analysis')
        if not parent['understanding']:
            raise APIError(409, 'understanding_required', 'This analysis omitted scene understanding. Submit a new analysis with understanding=true.')
        prompt = value.prompt.strip()
        if not prompt:
            raise APIError(400, 'empty_prompt', 'Provide a non-empty prompt')
        query_id = uuid.uuid4().hex
        directory = app.state.store.directory(query_id)
        directory.mkdir()
        record = dict(id=query_id, kind='query', analysis_id=analysis_id, sha256=parent['sha256'], prompt=prompt,
            status='queued', created_at=now(), started_at=None, finished_at=None, error=None,
            progress=dict(stage='Queued',completed=None,total=None))
        try:
            app.state.store.create(record)
        except QueueFull:
            directory.rmdir()
            raise APIError(429, 'queue_full', 'The processing queue is full. Retry later.')
        app.state.worker.wake.set()
        return JSONResponse(public_job(record).model_dump(mode='json'), status_code=202,
            headers={'Location':'/v1/jobs/'+query_id, 'Retry-After':'2'})

    @router.get('/queries/{query_id}', response_model=QueryResult, tags=['queries'])
    def query_result(query_id: JobId):
        completed(query_id, 'query')
        return QueryResult.model_validate_json((app.state.store.directory(query_id)/'result.json').read_text())

    @router.get('/analyses/{analysis_id}/source', response_class=FileResponse, tags=['assets'])
    def source(analysis_id: JobId):
        record = completed(analysis_id, 'analysis')
        return FileResponse(app.state.store.directory(analysis_id)/('source.'+record['format']),
            media_type='application/octet-stream', filename=record['filename'])

    @router.get('/analyses/{analysis_id}/frames/{frame_index}', response_class=FileResponse, tags=['assets'])
    def frame(analysis_id: JobId, frame_index: Annotated[int, PathParam(ge=0)]):
        completed(analysis_id, 'analysis')
        root = app.state.store.directory(analysis_id)/'analysis'
        frames = json.loads((root/'transforms.json').read_text())['frames']
        if frame_index >= len(frames):
            raise APIError(404, 'not_found', 'Rendered frame not found')
        path = (root/frames[frame_index]['file_path']).resolve()
        if not path.is_relative_to(root.resolve()) or path.suffix.lower() != '.png' or not path.is_file():
            raise APIError(404, 'not_found', 'Rendered frame not found')
        return FileResponse(path, media_type='image/png')

    app.include_router(router)
    return app


app = create_app()
