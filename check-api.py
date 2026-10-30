"""Exercise the public API and real queue/process lifecycle without model downloads."""
import json
from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient
from api_server import create_app, Settings
from api.worker import Worker

KEY = 'test-api-key-' + 'x'*32
AUTH = {'Authorization':'Bearer '+KEY}
HEADERS = {**AUTH, 'Content-Type':'application/octet-stream'}


def wait(client, job_id, desired=None):
    deadline = time.monotonic()+12
    while time.monotonic() < deadline:
        response = client.get('/v1/jobs/'+job_id, headers=HEADERS)
        assert response.status_code == 200, response.text
        job = response.json()
        if desired and desired(job):
            return job
        if not desired and job['status'] in ('succeeded','failed'):
            return job
        time.sleep(.02)
    raise AssertionError('Job did not reach the expected state')


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    program = root/'program'; program.mkdir()
    (program/'run_local.py').write_text('''import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ply');p.add_argument('--quality');p.add_argument('--job_dir');p.add_argument('--no-navigation',action='store_true');p.add_argument('--no-environment',action='store_true');p.add_argument('--cameras');p.add_argument('--camera-limit',type=int);a=p.parse_args()
out=Path(a.job_dir);out.mkdir(exist_ok=True);source=Path(a.ply);content=source.read_bytes()
with (out/'pipeline-runs.txt').open('a') as log:log.write('run\\n')
print('[detect] Processed 1/2 views',flush=True)
time.sleep(.15)
if content.endswith(b'fail'): raise RuntimeError('fixture processing failure')
if content.endswith(b'timeout'): time.sleep(10)
digest=hashlib.sha256(content).hexdigest()
if content.endswith(b'bad-hash'):digest='wrong'
obj=dict(id='object_0001',label='desk',confidence=.8,supporting_view_count=2,geometry_supporting_view_count=2,
 position=dict(x=1,y=2,z=3),scale=dict(x=2,y=1,z=1),parent_id=None,children=[],
 oriented_box=dict(type='upright_visible_surface_estimate',position=dict(x=1,y=2,z=3),scale=dict(x=2,y=1,z=1),rotation=dict(x=0,y=.2,z=0,w=.9797958971)),
 bounding_box=dict(type='visible_surface_estimate',min=dict(x=0,y=1.5,z=2.5),max=dict(x=2,y=2.5,z=3.5)),
 frames=[dict(frame_idx=0,score=.8,box=[0,0,10,10])])
source_input=dict(path=str(source),sha256=digest);transforms=dict(frames=[dict(file_path='frames/frame_0000.png')])
if a.cameras:
 cameras=json.loads(Path(a.cameras).read_text());fingerprint=hashlib.sha256(json.dumps(cameras,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 source_input['camera_input_sha256']=fingerprint
 if content.endswith(b'bad-camera-hash'):source_input['camera_input_sha256']='wrong'
 chosen=cameras['poses'][:a.camera_limit or 2]
 transforms['camera_validation']=dict(candidate_count=len(cameras['poses']),selected_positions=len(chosen),camera_input=dict(format=cameras['format'],sha256=fingerprint,input_pose_count=len(cameras['poses']),selected_image_ids=[c['id'] for c in chosen],orientation='perspective_panorama_source_negative_y_up'))
(out/'interactions.json').write_text(json.dumps(dict(input=source_input,objects=[obj],limitations=['Test estimate'])))
(out/'transforms.json').write_text(json.dumps(transforms))
(out/'frames').mkdir(exist_ok=True);(out/'frames/frame_0000.png').write_bytes(b'\\x89PNG\\r\\n\\x1a\\nfixture')
''')
    (program/'scene_understanding.py').write_text('''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--result');p.add_argument('--prompt');p.add_argument('--output');a=p.parse_args()
r=json.loads(Path(a.result).read_text());out=Path(a.output)
print('[understand] '+json.dumps(dict(stage='Describing objects from multiple views',completed=1,total=1)),flush=True)
if next(Path(a.result).parent.parent.glob('source.*')).read_bytes().endswith(b'restart') and not (out.parent/'pause-marker').exists():
 (out.parent/'pause-marker').write_text('once');time.sleep(10)
if a.prompt=='fail-query':raise RuntimeError('fixture query failure')
objects=[dict(**o,category=o['label'],display_name='wooden desk',status='supported',description_source='vision_language_model',visual_priority='medium',attributes=[]) for o in r['objects']]
selection=dict(interpretation=a.prompt or 'Tour',object_ids=[] if 'piano' in a.prompt else ['object_0001'],group_ids=[],message='No matching subjects' if 'piano' in a.prompt else '')
shots=[dict(id='view_1',title='Subject overview',frame_idx=0,object_ids=['object_0001'],subjects=['wooden desk'],score=.9,camera=dict(position=[1,-2,-3],target=[1,-2,-2],fov=90),basis='existing_render_with_depth_visibility_check')] if selection['object_ids'] else []
out.write_text(json.dumps(dict(source_hash=r['input']['sha256'],selection=selection,shots=shots,unshown_object_ids=[],understanding=dict(objects=objects,groups=[],relations=[],hero_object_ids=['object_0001'],scene=dict(type='study',description='A workspace.',confidence='medium'),limitations=['Inferred descriptions']))))
''')
    (program/'navigation.py').write_text('import sys\nsys.path.insert(0, '+repr(str(Path(__file__).resolve().parent))+')\n'+'''import argparse,hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from navigation import build_navigation,NavigationConfig
p=argparse.ArgumentParser();p.add_argument('--splat');p.add_argument('--output');a=p.parse_args()
inside=np.zeros((24,24,24),dtype=bool);inside[4:20,4:20,4:20]=True
walls=ndimage.binary_dilation(inside,structure=np.ones((3,3,3))) & ~inside
means=np.array(np.nonzero(walls)).T.astype(float)+.5
arrays=dict(means=means,scales=np.full_like(means,np.log(.6)),quats=np.tile([1.,0,0,0],(len(means),1)),opacities=np.full(len(means),5.))
build_navigation(arrays,a.output,hashlib.sha256(Path(a.splat).read_bytes()).hexdigest(),NavigationConfig(resolution=32))
''')
    settings = Settings(root/'data', KEY, max_upload_bytes=512, cors_origins=('https://app.example.com',))
    app = create_app(settings, program)
    with TestClient(app) as client:
        assert client.get('/health/ready').status_code == 200
        assert client.get('/v1/jobs/'+'0'*32).status_code == 401
        spec = client.get('/openapi.json').json()
        assert spec['paths']['/v1/analyses']['post']['requestBody']['content']['application/octet-stream']
        assert 'AnalysisResult' in spec['components']['schemas'] and 'QueryResult' in spec['components']['schemas']
        assert spec['paths']['/v1/analyses']['post']['security']
        assert client.post('/v1/analyses?filename=scene.ply',content=b'ply\n',headers={}).status_code==401
        assert client.post('/v1/analyses?filename=../scene.ply',content=b'ply\n',headers=HEADERS).status_code==400
        assert client.post('/v1/analyses?filename=scene.ply',content=b'wrong',headers=HEADERS).status_code==400
        assert client.post('/v1/analyses?filename=scene.ply',content=b'ply\n',headers={**HEADERS,'Content-Type':'application/json'}).status_code==415
        assert client.post('/v1/analyses?filename=scene.ply',content=b'ply\n'+b'x'*513,headers=HEADERS).status_code==413
        assert client.post('/v1/analyses?filename=scene.ply',content=iter([b'ply\n',b'x'*513]),headers=HEADERS).status_code==413
        assert not list((settings.data_dir/'jobs').iterdir()), 'Rejected upload left data behind'
        response = client.post('/v1/analyses?filename=room.ply',content=b'ply\nvalid',headers=HEADERS)
        assert response.status_code==202, response.text
        job = response.json();analysis_id=job['id']
        assert response.headers['Location']==job['status_url']
        assert client.get(job['result_url'],headers=HEADERS).status_code==409
        assert wait(client,analysis_id)['status']=='succeeded'
        result=client.get(job['result_url'],headers=HEADERS)
        assert result.status_code==200, result.text
        data=result.json()
        assert data['objects'][0]['position']==dict(x=1,y=2,z=3), 'Source geometry was transformed'
        assert data['objects'][0]['oriented_box']['rotation']==dict(x=0,y=.2,z=0,w=.9797958971), 'Box rotation was changed'
        assert data['suggested_views'][0]['camera']['space']=='viewer'
        assert data['suggested_views'][0]['camera']['position']==[1,-2,-3]
        assert data['coordinates']['metric_scale_verified'] is False
        navigation=data['navigation']
        assert navigation['status']=='candidate_space' and navigation['space']=='source'
        assert navigation['source_sha256']==data['source']['sha256']
        nav_url='/v1/analyses/'+analysis_id+'/navigation'
        assert client.get(nav_url).status_code==401
        assert client.get(nav_url,headers=AUTH).json()['fingerprint']==navigation['fingerprint']
        grid=client.get(navigation['grid_url'],headers=AUTH).json()
        assert grid['encoding']=='rle_xyz_c' and sum(grid['runs'][1::2])==__import__('math').prod(grid['shape'])
        seed=navigation['regions'][0]['seed']
        validation=client.post(nav_url+'/validate',json={'points':[seed]},headers=AUTH)
        assert validation.status_code==200 and validation.json()['valid'], validation.text
        rejected=client.post(nav_url+'/validate',json={'points':[navigation['origin']]},headers=AUTH)
        assert rejected.status_code==200 and not rejected.json()['valid']
        assert client.post(nav_url+'/validate',json={'points':[[0,0]]},headers=AUTH).status_code==422
        assert client.post(nav_url+'/validate',json={'points':[seed],'space':'viewer'},headers=AUTH).status_code==422
        planned=client.post(nav_url+'/path',json={'start':seed,'goal':seed},headers=AUTH)
        assert planned.status_code==200 and planned.json()['found'], planned.text
        absent_path=client.post(nav_url+'/path',json={'start':seed,'goal':navigation['origin']},headers=AUTH)
        assert absent_path.status_code==200 and not absent_path.json()['found']
        assert str(root) not in result.text, 'Internal filesystem paths leaked into public results'
        image=client.get(data['suggested_views'][0]['image_url'],headers=HEADERS)
        assert image.status_code==200 and image.headers['Content-Type']=='image/png'
        assert client.get(data['source']['download_url'],headers=HEADERS).content==b'ply\nvalid'
        assert client.get('/v1/analyses/'+analysis_id+'/frames/99',headers=HEADERS).status_code==404
        metadata_path=settings.data_dir/'jobs'/analysis_id/'analysis/transforms.json'
        saved_metadata=metadata_path.read_text()
        metadata_path.write_text(json.dumps(dict(frames=[dict(file_path='../private.png')])) )
        assert client.get('/v1/analyses/'+analysis_id+'/frames/0',headers=HEADERS).status_code==404
        metadata_path.write_text(saved_metadata)
        assert client.get('/v1/jobs/not-a-job',headers=HEADERS).status_code==422
        assert client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':' '},headers=AUTH).status_code==400
        assert client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':'x'*1001},headers=AUTH).status_code==422
        query=client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':'show desk'},headers=AUTH)
        assert query.status_code==202
        assert wait(client,query.json()['id'])['status']=='succeeded'
        assert client.get(query.json()['result_url'],headers=HEADERS).json()['selection']['object_ids']==['object_0001']
        absent=client.post('/v1/analyses/'+analysis_id+'/queries',json={'prompt':'grand piano'},headers=AUTH).json()
        assert wait(client,absent['id'])['status']=='succeeded'
        absent_result=client.get(absent['result_url'],headers=HEADERS).json()
        assert not absent_result['selection']['object_ids'] and not absent_result['suggested_views']
        failure=client.post('/v1/analyses?filename=bad.ply',content=b'ply\nfail',headers=HEADERS).json()
        assert wait(client,failure['id'])['status']=='failed'
        assert str(root) not in json.dumps(wait(client,failure['id'])['error'])
        mismatch=client.post('/v1/analyses?filename=mismatch.ply&understanding=false',content=b'ply\nbad-hash',headers=HEADERS).json()
        assert wait(client,mismatch['id'])['status']=='failed', 'Mismatched source fingerprint was published'
        spz=client.post('/v1/analyses?filename=room.spz',content=b'\x1f\x8bfixture',headers=HEADERS).json()
        assert wait(client,spz['id'])['status']=='succeeded'
        assert client.get(spz['result_url'],headers=HEADERS).json()['source']['format']=='spz'
        fast=client.post('/v1/analyses?filename=fast.ply&understanding=false&navigation=false',content=b'ply\nvalid',headers=HEADERS).json()
        assert wait(client,fast['id'])['status']=='succeeded'
        assert client.get(fast['result_url'],headers=HEADERS).json()['understanding_included'] is False
        assert client.get(fast['result_url'],headers=HEADERS).json()['navigation'] is None
        assert client.get('/v1/analyses/'+fast['id']+'/navigation',headers=AUTH).status_code==404
        assert client.post('/v1/analyses/'+fast['id']+'/queries',json={'prompt':'desk'},headers=AUTH).status_code==409
        camera_text=b'# Number of images: 2\n1 1 0 0 0 0 0 0 1 front/a.jpg\n\n2 1 0 0 0 1 0 0 1 front/b.jpg\n\n'
        camera_json=json.dumps(dict(space='source',positions=[[0,0,0],[1,0,0]])).encode()
        endpoint='/v1/analyses?filename=capture.ply&understanding=false&navigation=false'
        for camera_part in [('colmap_images',('images.txt',camera_text)), ('cameras',('cameras.json',camera_json))]:
            capture_response=client.post(endpoint+'&camera_limit=1',files=[('file',('capture.ply',b'ply\nvalid')),camera_part],headers=AUTH)
            assert capture_response.status_code==202, capture_response.text
            capture=capture_response.json()
            assert wait(client,capture['id'])['status']=='succeeded'
            capture_result=client.get(capture['result_url'],headers=AUTH).json()
            assert capture_result['camera_sampling']['input_pose_count']==2
            assert capture_result['camera_sampling']['selected_position_count']==1
            assert capture_result['source']['sha256']==data['source']['sha256']
            saved=settings.data_dir/'jobs'/capture['id']
            assert (saved/'camera_input.json').exists() and not list(saved.glob('camera-upload-*'))
        plain_form=client.post(endpoint,files={'file':('capture.ply',b'ply\nvalid')},headers=AUTH).json()
        assert wait(client,plain_form['id'])['status']=='succeeded'
        assert client.get(plain_form['result_url'],headers=AUTH).json()['camera_sampling'] is None
        before=set((settings.data_dir/'jobs').iterdir())
        bad_cases=[ [('file',('capture.ply',b'ply\nvalid')),('cameras',('cameras.json',b'{bad'))],
                    [('file',('capture.ply',b'ply\nvalid')),('cameras',('cameras.json',camera_json)),('colmap_images',('images.txt',camera_text))],
                    [('file',('capture.ply',b'ply\nvalid')),('colmap_cameras',('cameras.txt',b'1 PINHOLE 800 600 500 500 400 300'))],
                    [('file',('capture.ply',b'ply\nvalid')),('file',('again.ply',b'ply\nvalid'))] ]
        for files in bad_cases:
            invalid=client.post(endpoint,files=files,headers=AUTH)
            assert invalid.status_code==400, invalid.text
        calibration=client.post(endpoint,files=[('file',('capture.ply',b'ply\nvalid')),('colmap_images',('images.txt',camera_text)),
            ('colmap_cameras',('cameras.txt',b'2 PINHOLE 800 600 500 500 400 300'))],headers=AUTH)
        assert calibration.status_code==400 and calibration.json()['error']['code']=='invalid_camera_input'
        encoded=client.build_request('POST',endpoint,files={'file':('capture.ply',b'ply\nvalid')})
        truncated=client.post(endpoint,content=encoded.read()[:-12],headers={**AUTH,'Content-Type':encoded.headers['Content-Type']})
        assert truncated.status_code==400
        settings.max_camera_upload_bytes=16
        try:
            camera_too_large=client.post(endpoint,files=[('file',('capture.ply',b'ply\nvalid')),('cameras',('cameras.json',camera_json))],headers=AUTH)
            assert camera_too_large.status_code==413
        finally:settings.max_camera_upload_bytes=512*1024**2
        too_large=client.post(endpoint,files={'file':('capture.ply',b'ply\n'+b'x'*513)},headers=AUTH)
        assert too_large.status_code==413
        assert client.post(endpoint+'&camera_limit=1',content=b'ply\nvalid',headers=HEADERS).status_code==400
        assert client.post(endpoint+'&camera_limit=0',files={'file':('capture.ply',b'ply\nvalid')},headers=AUTH).status_code==422
        assert set((settings.data_dir/'jobs').iterdir())==before, 'Rejected camera uploads left files behind'
        mismatch_camera=client.post(endpoint,files=[('file',('capture.ply',b'ply\nbad-camera-hash')),('cameras',('cameras.json',camera_json))],headers=AUTH).json()
        assert wait(client,mismatch_camera['id'])['status']=='failed', 'Mismatched camera fingerprint was published'
        cors=client.options('/v1/analyses',headers={'Origin':'https://app.example.com','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'Authorization,Content-Type'})
        assert cors.headers['Access-Control-Allow-Origin']=='https://app.example.com'
        second=Worker(app.state.store,program)
        try:second.start()
        except RuntimeError:pass
        else:raise AssertionError('A second GPU worker acquired the same queue')
        restart=client.post('/v1/analyses?filename=restart.ply',content=b'ply\nrestart',headers=HEADERS).json()
        wait(client,restart['id'],lambda j:j['progress']['stage']=='Describing objects from multiple views')
    with TestClient(create_app(settings,program)) as client:
        assert wait(client,restart['id'])['status']=='succeeded', 'Interrupted job was not resumed'
        assert (settings.data_dir/'jobs'/restart['id']/'analysis/pipeline-runs.txt').read_text().splitlines()==['run'], 'Completed geometry checkpoint was rerun'
        assert client.get(job['result_url'],headers=HEADERS).status_code==200, 'Completed result did not survive restart'
        capture_restart=client.post('/v1/analyses?filename=capture-restart.ply',files=[('file',('capture-restart.ply',b'ply\ncapture-restart')),('cameras',('cameras.json',camera_json))],headers=AUTH).json()
        wait(client,capture_restart['id'],lambda j:j['progress']['stage']=='Describing objects from multiple views')
    with TestClient(create_app(settings,program)) as client:
        assert wait(client,capture_restart['id'])['status']=='succeeded'
        assert (settings.data_dir/'jobs'/capture_restart['id']/'analysis/pipeline-runs.txt').read_text().splitlines()==['run'], 'Capture checkpoint was rerun after restart'
        assert client.get(capture_restart['result_url'],headers=AUTH).json()['camera_sampling']['input_pose_count']==2

    limited=Settings(root/'limited', KEY, max_pending_jobs=1, stage_timeout_seconds=1)
    with TestClient(create_app(limited,program)) as client:
        slow=client.post('/v1/analyses?filename=slow.ply',content=b'ply\ntimeout',headers=HEADERS).json()
        rejected=client.post('/v1/analyses?filename=extra.ply',content=b'ply\nvalid',headers=HEADERS)
        assert rejected.status_code==429 and len(list((limited.data_dir/'jobs').iterdir()))==1
        assert wait(client,slow['id'])['status']=='failed', 'Stage timeout did not release GPU worker'
        next_job=client.post('/v1/analyses?filename=next.ply',content=b'ply\nvalid',headers=HEADERS).json()
        assert wait(client,next_job['id'])['status']=='succeeded'

    try:
        with TestClient(create_app(Settings(root/'bad-key','short'),program)):pass
    except RuntimeError:pass
    else:raise AssertionError('API started with an unsafe default key')
print('API raw/multipart camera uploads, limits, queue, results, queries, coordinates, assets, restart recovery, failures, and timeout checks passed.')
