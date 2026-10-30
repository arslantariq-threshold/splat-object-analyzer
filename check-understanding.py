"""Grounding, visibility, cache invalidation, and no-match regressions (no model download)."""
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from scene_understanding import (SceneEvidence, atomic_json, build_understanding, fingerprint,
                                 known_ids, rank_views, select_subjects, suggest_views, concept_schema)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root/'frames').mkdir()
    # Flat observed object at Z=3. Third view has an occluder in front of it.
    grid = np.array([[x,y,3.] for x in np.linspace(-.5,.5,15) for y in np.linspace(-.5,.5,15)], dtype=np.float32)
    objects = [dict(id='object_0001', label='monitor', position=dict(x=0,y=0,z=3),
                    scale=dict(x=1,y=1,z=.1), frames=[], children=[], parent_id=None)]
    result = dict(input=dict(sha256='scene-A'),objects=objects,frame_annotations={})
    atomic_json(root/'interactions.json', result)
    frames=[]
    for i, x in enumerate([0., .4, 0.]):
        pose=np.eye(4);pose[0,3]=x
        Image.new('RGB',(64,64),(50+i*30,100,80)).save(root/f'frames/frame_{i:04d}.png')
        depth=np.full((64,64),3. if i<2 else 1., dtype=np.float32)
        np.save(root/f'frames/depth_{i:04d}.npy',depth)
        frames.append(dict(file_path=f'frames/frame_{i:04d}.png',depth_path=f'frames/depth_{i:04d}.png',
                           transform_matrix=pose.tolist(),position_idx=i,quality=dict(accepted=True,detail=.05)))
    atomic_json(root/'transforms.json', dict(fl_x=40,fl_y=40,cx=32,cy=32,w=64,h=64,scene_radius=2,frames=frames))
    np.savez_compressed(root/'object_surfaces.npz',object_0001=grid)
    scene=SceneEvidence(root/'interactions.json')
    ranked=rank_views(scene,['object_0001'])
    assert {r['frame_idx'] for r in ranked} == {0,1}, 'Occluded object was recommended'
    assert known_ids(['object_0001','invented','object_0001',{}],{'object_0001'}) == ['object_0001']
    class FakeModel:
        calls=0
        fail_batch=False
        fail_single=False
        def ask(self,prompt,images=(),tokens=0,schema=None):
            self.calls+=1
            if 'Propose up to 4 useful groups' in prompt:
                return dict(scene_type='study',description='A workspace.',confidence='medium',groups=[dict(name='Desk setup',anchor_category='monitor',member_categories=[],description='A workspace',confidence='medium')],tour_categories=['monitor'])
            if 'scene_type' in prompt: return dict(scene_type='study',description='A workspace.',confidence='medium')
            if 'Each labeled card' in prompt:
                if self.fail_batch: raise ValueError('incomplete batch')
                return dict(objects=[dict(id='object_0001',category='monitor',display_name='Black monitor',status='supported',visual_priority='high')])
            if 'Describe the highlighted object' in prompt:
                if self.fail_single: raise ValueError('unresolved description')
                return dict(category='monitor',display_name='Black monitor',status='supported')
            if 'meaningful groups' in prompt:
                return dict(groups=[dict(name='Desk setup',object_ids=['object_0001','invented'])],hero_object_ids=['invented','object_0001'])
            if 'spaceship' in prompt: return dict(interpretation='Find a spaceship',target_groups=[],target_categories=[],tour=False,message='No spaceship is supported by this scene.')
            return dict(interpretation='Show the workspace',target_groups=['Desk setup'],target_categories=[],near_category='',attributes='',tour=False,message='')
    model=FakeModel()
    index=build_understanding(scene,lambda:model)
    assert len(index['objects'])==1 and index['objects'][0]['status']=='uncertain', 'Detector-only record was treated as verified'
    assert index['groups'][0]['object_ids']==['object_0001']
    calls=model.calls
    assert build_understanding(scene,lambda: (_ for _ in ()).throw(AssertionError('cache missed'))) == index
    # Interrupted publication must reuse completed batch caches without loading
    # the model again just to rebuild the final index.
    (root/'scene-understanding.json').unlink()
    assert build_understanding(scene,lambda: (_ for _ in ()).throw(AssertionError('Cached descriptions loaded the model'))) == index
    assert model.calls == calls
    selection=select_subjects(index,'Show my desk setup',lambda idx,prompt:model.ask(prompt))
    assert selection['object_ids']==['object_0001']
    missing=select_subjects(index,'Show a spaceship',lambda idx,prompt:model.ask(prompt))
    assert not missing['object_ids'] and not suggest_views(scene,missing,index)
    shots=suggest_views(scene,selection,index)
    assert len(shots)==2 and all(s['object_ids']==['object_0001'] for s in shots)
    assert shots[0]['camera']['target'][2] < shots[0]['camera']['position'][2], 'Wrong source/viewer camera conversion'
    before=fingerprint(scene.path)
    result['objects'][0]['label']='television';atomic_json(scene.path,result)
    assert fingerprint(scene.path)!=before, 'Reanalysis reused stale descriptions'
    model.fail_batch=True
    recovered=build_understanding(SceneEvidence(scene.path),lambda:model)
    assert recovered['review']['described_count']==1, 'Individual retry did not recover a failed batch'
    result['input']['sha256']='scene-B';atomic_json(scene.path,result)
    model.fail_single=True
    unresolved=build_understanding(SceneEvidence(scene.path),lambda:model)
    assert unresolved['review']['described_count']==0 and unresolved['objects'][0]['status']=='uncertain'
    try: scene.local_path('../outside.png')
    except ValueError: pass
    else: raise AssertionError('Evidence escaped analysis directory')
print('Scene grounding, occlusion, no-match, coordinate, and cache checks passed.')

# Check that constrained generation cannot emit invented IDs or oversized groups.
from lmformatenforcer import JsonSchemaParser

def allowed_response(value, schema):
    parser=JsonSchemaParser(schema)
    for char in json.dumps(value):
        if char not in parser.get_allowed_characters(): return False
        parser=parser.add_character(char)
    return parser.can_end()

schema=concept_schema({'desk','monitor'})
value=dict(scene_type='study',description='A desk and monitor.',confidence='medium',
           groups=[dict(name='Desk setup',description='A workspace',anchor_category='desk',member_categories=['monitor'],confidence='medium')],
           tour_categories=['desk'])
assert allowed_response(value,schema)
value['groups'][0]['anchor_category']='invented'
assert not allowed_response(value,schema)
value['groups'][0]['anchor_category']='desk'
value['groups'][0]['member_categories']=['monitor']*3
assert not allowed_response(value,schema), 'Group size limit was not enforced'
from semantic_binding import bind_groups
anchor=dict(id='opaque-desk-ID',category='desk',display_name='Wooden desk',status='supported',parent_id=None,
            position=dict(x=0,y=0,z=0),scale=dict(x=2,y=1,z=1))
near=dict(id='opaque-near-ID',category='monitor',display_name='Monitor',status='supported',parent_id=None,
          position=dict(x=.2,y=.8,z=0),scale=dict(x=.6,y=.5,z=.1))
far=dict(near,id='opaque-far-ID',position=dict(x=20,y=.8,z=0))
bound=bind_groups([far,anchor,near],[dict(name='Desk setup',anchor_category='desk',member_categories=['monitor'])],5)
assert bound[0]['object_ids']==['opaque-desk-ID','opaque-near-ID'], 'Roles were bound to an unrelated instance'
print('Generation bounds and spatial role-to-instance binding passed.')

computer=dict(near,id='ambiguous-computer-ID',category='computer',position=dict(x=0,y=0,z=0))
bound=bind_groups([computer,anchor,near],[dict(name='Desk setup',description='Desk with computer monitor',anchor_category='desk',member_categories=['computer'])],5)
assert bound[0]['object_ids']==['opaque-desk-ID','opaque-near-ID'], 'A generic computer role ignored the explicit monitor description'

from semantic_binding import direct_intent, resolve_intent
idx=dict(objects=[anchor,near],groups=[dict(id='g',name='desk setup',object_ids=[anchor['id'],near['id']])],hero_object_ids=[anchor['id']],scene_radius=5)
intent=direct_intent(idx,'Show off my desk setup')
assert resolve_intent(idx,intent)['object_ids']==[anchor['id'],near['id']]
assert resolve_intent(idx,direct_intent(idx,'Show around my bedroom'))['object_ids']==[anchor['id']]
assert direct_intent(idx,'Show a red monitor') is None, 'Attribute constraints were ignored'
