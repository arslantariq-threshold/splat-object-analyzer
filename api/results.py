"""Normalize existing pipeline output without changing object IDs or geometry."""
from .schemas import AnalysisResult, QueryResult


def views(response, analysis_id):
    return [dict(id=v['id'], title=v['title'], frame_index=v['frame_idx'],
        image_url=f'/v1/analyses/{analysis_id}/frames/{v["frame_idx"]}', object_ids=v['object_ids'],
        subjects=v['subjects'], score=v['score'], camera=v['camera'], basis=v['basis'])
        for v in response.get('shots', [])]


def analysis_result(record, inventory, transforms, semantic=None, navigation=None):
    if inventory['input']['sha256'] != record['sha256']:
        raise ValueError('Inventory source fingerprint mismatch')
    if inventory['input'].get('camera_input_sha256') != record.get('camera_input_sha256'):
        raise ValueError('Inventory camera input fingerprint mismatch')
    if semantic and semantic['source_hash'] != record['sha256']:
        raise ValueError('Scene understanding source fingerprint mismatch')
    if navigation and navigation['source_sha256'] != record['sha256']:
        raise ValueError('Navigation source fingerprint mismatch')
    environment = inventory.get('environment')
    if environment and environment['source_sha256'] != record['sha256']:
        raise ValueError('Environment source fingerprint mismatch')
    if environment and environment.get('camera_input_sha256') != record.get('camera_input_sha256'):
        raise ValueError('Environment camera input fingerprint mismatch')
    index = semantic.get('understanding', {}) if semantic else {}
    descriptions = {o['id']:o for o in index.get('objects', [])}
    objects = []
    for o in inventory['objects'] + (environment.get('entities', []) if environment else []):
        description = descriptions.get(o['id'], o if o.get('kind') in {'structure','area'} else {})
        evidence = description.get('evidence') or o.get('frames', [])[:4]
        mask_quality = {e['frame_idx']:e.get('mask_quality') for e in o.get('mask_evidence', [])}
        caption_reviewed = o.get('kind') in {'structure','area'} and description.get('description_source') == 'vision_language_model'
        objects.append(dict(id=o['id'], kind=o.get('kind', 'object'), description=description.get('description', '') if caption_reviewed else o.get('description', ''),
            geometry_basis=o.get('geometry_basis'), surface_samples=o.get('surface_samples', []),
            independent_position_count=o.get('independent_position_count'),
            confidence_type=o.get('confidence_type', 'peak_detector_score_not_calibrated_probability'), label=o['label'], category=description.get('category', o['label']),
            display_name=description.get('display_name', o['label']), status=description.get('status', 'unreviewed'),
            attributes=description.get('attributes', []), confidence=o['confidence'],
            supporting_view_count=o.get('supporting_view_count', 0),
            geometry_supporting_view_count=o.get('geometry_supporting_view_count', 0),
            position=o['position'], scale=o['scale'], bounding_box=o.get('bounding_box'), oriented_box=o.get('oriented_box'),
            parent_id=o.get('parent_id'), children=o.get('children', []), relationship=o.get('relationship'),
            description_source=description.get('description_source', 'detector_label_only'),
            visual_priority=description.get('visual_priority'),
            evidence=[dict(frame_index=e['frame_idx'], image_url=f'/v1/analyses/{record["id"]}/frames/{e["frame_idx"]}',
                box=e.get('box'), image_width=transforms.get('w'), image_height=transforms.get('h'),
                detector_score=e.get('score') if o.get('kind', 'object') == 'object' else None, mask_quality=e.get('mask_quality', mask_quality.get(e['frame_idx'])),
                geometry_consistency=e.get('geometry_consistency'), evidence_type=e.get('evidence_type', 'segmented_surface' if o.get('kind') in {'structure','area'} else 'detector_only'))
                for e in evidence]))
    relations = [dict(subject=r['subject'], relation=r['relation'], object=r['object'], basis=r['basis'],
                      gap_source_units=r.get('gap_source_units')) for r in index.get('relations', [])]
    if environment:
        for relation in environment['relations']:
            if relation not in relations:
                relations.append(relation)
    def issue(item):
        return dict(frame_index=item['frame_idx'], image_url=f'/v1/analyses/{record["id"]}/frames/{item["frame_idx"]}',
            reason=item['reason'], name=item.get('name'), kind=item.get('kind'))
    environment_summary = dict(version=environment['version'], fingerprint=environment['fingerprint'],
        status=environment['status'], entity_ids=[e['id'] for e in environment['entities']],
        analyzed_view_count=environment['analyzed_view_count'], evidence_frames=environment['selected_frames'],
        failed_views=[issue(v) for v in environment['failed_views']], ungrounded=[issue(v) for v in environment['ungrounded']],
        limitations=environment['limitations']) if environment else None
    environment_views = []
    if environment:
        import math
        for entity in environment['entities']:
            if not entity['frames']:
                continue
            frame_index = max(entity['frames'], key=lambda f: f['score'])['frame_idx']
            pose = transforms['frames'][frame_index]['transform_matrix']
            position = [pose[axis][3]*sign for axis, sign in enumerate([1,-1,-1])]
            target = [position[axis]+pose[axis][2]*sign for axis, sign in enumerate([1,-1,-1])]
            environment_views.append(dict(id='view_'+entity['id'], title=entity['display_name'], frame_index=frame_index,
                image_url=f'/v1/analyses/{record["id"]}/frames/{frame_index}', object_ids=[entity['id']],
                subjects=[entity['display_name']], score=entity['confidence'],
                camera=dict(position=position, target=target, fov=math.degrees(2*math.atan(transforms['h']/(2*transforms['fl_y'])))),
                basis='observed_segmented_surface_view'))
    summary = index.get('scene') or (environment['scene'] if environment else None)
    validation = transforms.get('camera_validation', {})
    cameras = validation.get('camera_input')
    if record.get('camera_input_sha256') and (not cameras or cameras.get('sha256') != record['camera_input_sha256']):
        raise ValueError('Rendered camera input fingerprint mismatch')
    sampling = dict(format=cameras['format'], input_pose_count=cameras['input_pose_count'],
        candidate_position_count=validation['candidate_count'], selected_position_count=validation['selected_positions'],
        selected_image_ids=cameras['selected_image_ids'], orientation=cameras['orientation']) if cameras else None
    return AnalysisResult.model_validate(dict(analysis_id=record['id'], quality=record['quality'],
        source=dict(filename=record['filename'], format=record['format'], bytes=record['bytes'], sha256=record['sha256'],
                    download_url=f'/v1/analyses/{record["id"]}/source'),
        rendered_view_count=len(transforms['frames']), understanding_included=semantic is not None,
        camera_sampling=sampling, environment=environment_summary,
        navigation={**navigation, 'grid_url':f'/v1/analyses/{record["id"]}/navigation/grid'} if navigation else None,
        scene={k:summary[k] for k in ('type','description','confidence')} if summary else None,
        objects=objects, groups=index.get('groups', []), relations=relations, hero_object_ids=index.get('hero_object_ids', []),
        default_selection=semantic.get('selection') if semantic else None, suggested_views=views(semantic or {}, record['id']) + environment_views,
        limitations=list(dict.fromkeys(inventory.get('limitations', []) + index.get('limitations', []) +
            ['Object confidence and semantic similarity scores are not calibrated probabilities.',
             'Geometry uses the source file coordinate space; suggested cameras use the documented viewer transform.']))))


def query_result(record, response):
    if response['source_hash'] != record['sha256']:
        raise ValueError('Query source fingerprint mismatch')
    return QueryResult.model_validate(dict(query_id=record['id'], analysis_id=record['analysis_id'],
        prompt=record['prompt'], selection=response['selection'], suggested_views=views(response, record['analysis_id']),
        unshown_object_ids=response.get('unshown_object_ids', [])))
