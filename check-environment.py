"""Environment discovery, surface identity, API compatibility, and evidence checks."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from environment_understanding import (normalize_subject, select_frames, discover_environment,
    ground_environment, attach_environment, associate, reprojection_fraction)
from refine_inventory import mask_points
from scene_understanding import SceneEvidence, rank_views, build_understanding
from semantic_binding import tour_subjects
from api.results import analysis_result
from pipeline_progress import progress_from_line


def patch_node(start, end, frame, proposal, scale=1):
    points = np.array([[x, y, 3.] for x in np.linspace(start, end, 41) for y in np.linspace(0, 1, 12)]) * scale
    return dict(points=points, frame=frame, mask=np.ones((64,64), bool), proposal=proposal,
                spacing=.01*scale, voxel=.01*scale)


# The same identity rules must hold across subject types, coordinate scales,
# observation order, partial views and ambiguous links between repeated subjects.
for kind, category in [('area','floor'), ('structure','wall'), ('structure','building'), ('area','courtyard')]:
    proposal = dict(kind=kind, category=category, name=category, attributes=[])
    for scale in (.001, 1, 1000):
        nodes = [patch_node(0,.6,0,proposal,scale), patch_node(.4,1,1,proposal,scale),
                 patch_node(.8,1.4,2,proposal,scale)]
        assert len(associate(nodes,.025*scale)) == 1, 'Linked partial views remained duplicate subjects'
        assert len(associate(list(reversed(nodes)),.025*scale)) == 1, 'Identity depends on observation order'
        nodes[2]['frame'] = 0
        nodes[0]['mask'][:,:32] = False
        nodes[2]['mask'][:,32:] = False
        assert len(associate(nodes,.025*scale)) == 2, 'A bridging observation joined distinct co-visible subjects'
white = dict(kind='structure',category='house',name='White house',attributes=['white'])
generic = dict(white,name='House',attributes=[])
red = dict(white,name='Red house',attributes=['red'])
nodes = [patch_node(0,.6,0,white), patch_node(.4,1,1,generic), patch_node(.8,1.4,2,red)]
assert len(associate(nodes,.025)) == 2, 'A generic observation bridged conflicting subject appearances'

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root/'frames').mkdir()
    frames = []
    for i in range(4):
        pose = np.eye(4); pose[0,3] = (i % 2) * .04
        Image.new('RGB', (64,64), '#779988').save(root/f'frames/{i}.png')
        depth = np.full((64,64), 3, dtype=np.float32)
        np.save(root/f'frames/{i}.npy', depth)
        frames.append(dict(file_path=f'frames/{i}.png', depth_path=f'frames/{i}.png',
            transform_matrix=pose.tolist(), position_idx=i % 2, quality=dict(accepted=True)))
    metadata = dict(w=64,h=64,fl_x=40,fl_y=40,cx=32,cy=32,scene_radius=2,frames=frames)
    (root/'transforms.json').write_text(json.dumps(metadata))
    base = dict(kind='structure',category='house',name='White house',description='Visible white house facade',
        attributes=['white'],box=[100,100,470,900])
    assert normalize_subject(dict(base,kind='invented'),0,metadata) is None
    assert normalize_subject(dict(base,category=''),0,metadata) is None
    assert len(select_frames(metadata,3)) == 3
    assert len(set(select_frames(metadata,3))) == 3
    rotated=copy.deepcopy(metadata)
    rotated['frames'][2]['transform_matrix'][2][2]=-1
    assert 2 in select_frames(rotated,2), 'Viewing direction did not affect frame coverage'
    class Model:
        calls=0
        def ask(self,prompt,images=(),tokens=0,schema=None):
            self.calls+=1
            if len(images)>1:
                return dict(scene_type='residential courtyard',description='Two houses surrounding a courtyard.',confidence='medium')
            return dict(scene_type='exterior',description='House facades.',subjects=[base,
                dict(base,name='Red house',attributes=['red'],box=[530,100,900,900]),
                dict(base,kind='area',category='lawn',name='Lawn',attributes=[],box=[200,950,800,1000])])
    model=Model()
    report=discover_environment(root,metadata,'source',4,lambda:model)
    assert model.calls == 5 and report['scene']['type']=='residential courtyard'
    assert len(report['observations'])==12
    calls=model.calls
    assert discover_environment(root,metadata,'source',4,lambda:(_ for _ in ()).throw(AssertionError('cache missed'))) == report
    assert model.calls == calls
    changed=copy.deepcopy(metadata);changed['frames'][0]['transform_matrix'][0][3]=.5
    again=discover_environment(root,changed,'source',4,lambda:model)
    assert again['fingerprint'] != report['fingerprint'], 'Camera changes reused stale environment observations'
    def segmenter(candidates,root,transforms,device,fingerprint,**kwargs):
        samples={}
        for candidate in candidates:
            frame=candidate['frames'][0];index=frame['frame_idx'];box=frame['box']
            mask=np.zeros((64,64),dtype=bool)
            x1,y1,x2,y2=map(int,box)
            if candidate['label'] != 'lawn':mask[y1:y2,x1:x2]=True
            points=mask_points(mask,np.full((64,64),3.),np.asarray(transforms['frames'][index]['transform_matrix']),transforms)
            path=f'frames/mask-{candidate["id"]}.npz'
            np.savez_compressed(root/path,mask=mask,quality=.9)
            samples[candidate['id']]=[dict(frame=index,mask=mask,points=points,quality=.9,path=path)] if len(points) else []
        return samples
    def detector(frames_dir,labels,transforms,radius,**kwargs):
        output=[]
        for i in range(len(transforms['frames'])):
            for label in labels:
                box=[6,6,30,58] if label=='white house' else [34,6,58,58] if label=='red house' else [12,60,52,64]
                output.append(dict(frame_idx=i,label=label,score=.8,box_2d=box))
        output.extend([dict(frame_idx=0,label='white house',score=.9,box_2d=[float('nan'),0,64,64]),
            dict(frame_idx=0,label='white house',score=.9,box_2d=[50,0,20,64])])
        return output
    with patch('refine_inventory._segment_candidates', segmenter):
        environment=ground_environment(report,root,metadata,'cpu',detector=detector)
    entities=environment['entities']
    assert len(entities)==2, [e['display_name'] for e in entities]
    assert {e['display_name'] for e in entities}=={'White house','Red house'}
    assert all(e['status']=='supported' and e['geometry_supporting_view_count']==4 for e in entities)
    assert all(e['independent_position_count']==2 for e in entities)
    assert environment['ungrounded'] and environment['status']=='partial'
    assert any(item['name'].lower()=='lawn' for item in environment['ungrounded'])
    assert all(len(e['surface_samples'])<=512 for e in entities)
    assert all(abs(e['position']['z']-3)<.02 for e in entities), 'Geometry did not come from depth'
    assert entities[0]['position']['x']*entities[1]['position']['x']<0, 'Distinct buildings were merged'
    # Co-visible, disjoint masks of same category/color still retain identity.
    points=np.array([[x,y,3.] for x in np.linspace(0,1,12) for y in np.linspace(0,1,12)])
    node=dict(points=points, frame=0, mask=np.ones((64,64),bool))
    assert reprojection_fraction(node,node,metadata,{0:np.full((64,64),3.)},.1) > .9
    assert reprojection_fraction(node,node,metadata,{0:np.ones((64,64))},.1) == 0, 'An occluded surface was associated'
    for frame in (0,1):np.save(root/f'frames/{frame}.npy',np.ones((64,64)))
    occluded=[dict(node,frame=frame,proposal=base) for frame in (0,1)]
    assert len(associate(occluded,.1,metadata,root)) == 2, '3D proximity bypassed rendered depth visibility'
    for frame in (0,1):np.save(root/f'frames/{frame}.npy',np.full((64,64),3.))
    a=np.zeros((64,64),bool);a[:,:30]=True
    b=~a
    nodes=[dict(points=points,frame=0,mask=a,proposal=base),dict(points=points,frame=0,mask=b,proposal=base)]
    assert len(associate(nodes,.1))==2
    nodes[1]['mask']=a;nodes[1]['proposal']=dict(base,name='Red house',attributes=['red'])
    assert len(associate(nodes,.1))==2, 'Conflicting identifying colors were merged'
    np.savez_compressed(root/'object_surfaces.npz')
    inventory=dict(input=dict(sha256='source'),objects=[],limitations=[])
    inventory=attach_environment(inventory,environment,root)
    (root/'interactions.json').write_text(json.dumps(inventory))
    scene=SceneEvidence(root/'interactions.json')
    assert set(scene.objects)=={e['id'] for e in entities}
    assert rank_views(scene,[entities[0]['id']]), 'Environment subjects could not yield visibility-checked views'
    class SemanticModel:
        def ask(self,prompt,images=(),tokens=0,schema=None):
            if 'Each labeled card' in prompt:
                return dict(objects=[dict(id=e['id'],category='house',display_name=e['display_name'],status='supported') for e in entities])
            if 'Propose up to 4 useful groups' in prompt:
                return dict(scene_type='outside',description='Two houses.',confidence='medium',groups=[],tour_categories=['house'])
            return dict(scene_type='outside',description='Two houses.',confidence='medium')
    index=build_understanding(scene,SemanticModel)
    assert len(index['objects'])==2 and index['scene']['type']=='residential courtyard'
    assert set(index['hero_object_ids'])=={e['id'] for e in entities}, 'A same-category building was dropped from the tour'
    record=dict(id='a'*32,sha256='source',filename='scene.ply',format='ply',bytes=123,quality='high')
    api=analysis_result(record,inventory,metadata).model_dump(mode='json')
    assert len(api['objects'])==2 and api['objects'][0]['kind']=='structure'
    assert api['environment']['entity_ids']==[e['id'] for e in entities]
    assert len(api['suggested_views'])==2
    assert all(v['camera']['space']=='viewer' for v in api['suggested_views'])
    assert api['suggested_views'][0]['camera']['position'][2]==0
    assert all(e['image_url'].startswith('/v1/analyses/') for e in api['objects'][0]['evidence'])
    assert api['objects'][0]['evidence'][0]['box']
    assert api['objects'][0]['evidence'][0]['mask_quality']==.9
    assert 'mask-' not in json.dumps(api), 'API leaked private mask paths'
    bad=copy.deepcopy(inventory);bad['environment']['source_sha256']='other'
    try:analysis_result(record,bad,metadata)
    except ValueError:pass
    else:raise AssertionError('Environment from another scene was accepted')
    bad=copy.deepcopy(inventory);bad['environment']['camera_input_sha256']='other'
    try:analysis_result(record,bad,metadata)
    except ValueError:pass
    else:raise AssertionError('Environment from another camera input was accepted')
    assert progress_from_line('[environment] '+json.dumps(dict(stage='Grounding environment',completed=2,total=4)))['completed']==2
    # Exercise orchestration: an empty object tagger must still allow grounded
    # environment results, and environment queries share one detector pass.
    import hashlib
    import shutil
    from pipeline import run_pipeline
    from config import PipelineConfig
    source=root/'scene.ply';source.write_bytes(b'ply\nfixture')
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    for enabled in (True,False):
        output=root/('full' if enabled else 'legacy');output.mkdir()
        shutil.copytree(root/'frames',output/'frames')
        (output/'transforms.json').write_text(json.dumps(metadata))
        fixture_report=dict(report,source_sha256=source_hash)
        calls=[]
        def model_detect(frames_dir,labels,transforms,radius,**kwargs):
            calls.append((labels,kwargs))
            return [dict(d,position=np.zeros(3),scale=np.ones(3)) for d in detector(frames_dir,labels,transforms,radius)]
        with patch('render_cameras.render_views',return_value=str(output/'transforms.json')), \
             patch('auto_discover.discover_labels',return_value=([],dict(mode='automatic',labels=[]))), \
             patch('environment_understanding.discover_environment',return_value=fixture_report) as discover, \
             patch('pipeline._run_owlv2',side_effect=model_detect), \
             patch('refine_inventory._segment_candidates',segmenter):
            run_pipeline(str(source), None if enabled else 'chair', str(output), PipelineConfig(environment=enabled,navigation=False))
        saved=json.loads((output/'interactions.json').read_text())
        assert len(calls)==1, 'Environment analysis loaded a second detector pass'
        assert saved['input']['sha256']==source_hash
        if enabled:
            assert len(saved['environment']['entities'])==2 and not saved['objects']
            assert 'house' in calls[0][0] and discover.call_count==1
        else:
            assert 'environment' not in saved and discover.call_count==0
print('Environment discovery, geometry, identity, evidence, and API checks passed.')
