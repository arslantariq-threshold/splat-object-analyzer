"""Discover structures/areas from images, then ground them in observed depth.

No model-generated 3D coordinates are accepted. A proposed subject enters the
spatial inventory only after a segmented mask produces a coherent surface.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = 2
ASSOCIATION_VERSION = 2
MAX_SUBJECTS_PER_VIEW = 5


def emit(stage, completed=None, total=None):
    print('[environment] ' + json.dumps(dict(stage=stage, completed=completed, total=total)), flush=True)


def schema():
    from scene_understanding import record_schema
    text = lambda n: dict(type='string', maxLength=n)
    subject = record_schema(dict(kind=dict(type='string', enum=['structure', 'area']),
        category=text(40), name=text(70), description=text(140),
        attributes=dict(type='array', items=text(30), maxItems=4)))
    return record_schema(dict(scene_type=text(70), description=text(260),
        subjects=dict(type='array', items=subject, maxItems=MAX_SUBJECTS_PER_VIEW)))


def select_frames(transforms, limit=12):
    """Farthest-point sampling in position AND viewing direction, deterministically."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 32:
        raise ValueError('Environment view budget must be an integer from 1 to 32.')
    frames = transforms['frames']
    valid = [i for i, f in enumerate(frames) if f.get('quality', {}).get('accepted', True)]
    if not valid:
        raise ValueError('Environment discovery needs accepted rendered views.')
    poses = np.asarray([frames[i]['transform_matrix'] for i in valid], dtype=float)
    positions = poses[:, :3, 3]
    extent = max(float(np.linalg.norm(np.ptp(positions, axis=0))), .01)
    features = np.column_stack((positions / extent, poses[:, :3, 2] * .7))
    quality = np.asarray([frames[i].get('quality', {}).get('score', 1) for i in valid])
    chosen = [int(np.argmax(quality))]
    distance = np.full(len(valid), np.inf)
    while len(chosen) < min(limit, len(valid)):
        distance = np.minimum(distance, np.linalg.norm(features - features[chosen[-1]], axis=1))
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    return [valid[i] for i in chosen]


def normalize_subject(subject, frame_idx, transforms):
    if not isinstance(subject, dict) or subject.get('kind') not in {'structure', 'area'}:
        return None
    name = str(subject.get('name', '')).strip()[:70]
    category = ' '.join(str(subject.get('category', '')).lower().split())[:40]
    if not name or not category:
        return None
    if category in {'car', 'vehicle', 'chair', 'table', 'wheelbarrow', 'trash bin', 'tree', 'bush', 'plant',
                    'bed', 'desk', 'sofa', 'couch', 'monitor', 'television', 'lamp', 'pillow', 'book', 'stool',
                    'bookshelf', 'bookshelves', 'bookcase', 'cabinet', 'wardrobe'}:
        return None  # Individual objects have their own detection/identity stage.
    if category == 'ceiling' and 'light' in name.lower():
        return None
    if name.lower() in {'driveway', 'lawn', 'garden', 'courtyard', 'path'}:
        category = name.lower()
    aliases = {'building': 'house', 'house exterior': 'house', 'house facade': 'house',
               'house façade': 'house', 'grass': 'lawn'}
    # Only explicit house/building proposals are combined; walls stay walls.
    if category in aliases and (category != 'building' or 'house' in name.lower()):
        category = aliases[category]
    attributes = subject.get('attributes', [])
    attributes = [v.strip()[:30] for v in attributes if isinstance(v, str) and v.strip()][:4] if isinstance(attributes, list) else []
    areas = {'floor', 'lawn', 'garden', 'driveway', 'courtyard', 'backyard', 'front yard', 'path', 'walkway',
             'patio', 'terrace', 'hallway', 'corridor', 'room', 'kitchen', 'bedroom', 'living room', 'parking area'}
    structures = {'house', 'building', 'garage', 'shed', 'wall', 'facade', 'façade', 'roof', 'staircase', 'fence'}
    kind = 'area' if category in areas else 'structure' if category in structures else subject['kind']
    return dict(kind=kind, category=category, name=name,
        description=str(subject.get('description', ''))[:140], attributes=attributes,
        frame_idx=frame_idx)


def discover_environment(root, transforms, source_hash, limit=12, model_factory=None):
    from scene_understanding import LocalVLM, MODEL, REVISION, atomic_json
    root = Path(root)
    digest = hashlib.sha256(json.dumps([VERSION, MODEL, REVISION, source_hash, transforms, limit],
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    target = root / 'environment-proposals.json'
    if target.exists():
        saved = json.loads(target.read_text())
        camera_hash = transforms.get('camera_validation', {}).get('camera_input', {}).get('sha256')
        if (saved.get('fingerprint') == digest and saved.get('scene') and saved.get('summary_version') == 2
                and saved.get('camera_input_sha256') == camera_hash):
            saved['observations'] = [value for p in saved['observations']
                if (value := normalize_subject(p, p['frame_idx'], transforms)) is not None]
            emit('Reusing environment observations')
            return saved
    selected = select_frames(transforms, limit)
    asset = root / 'environment_observations'
    asset.mkdir(exist_ok=True)
    emit('Loading environment model')
    model = (model_factory or LocalVLM)()
    observations, summaries, failures = [], [], []
    overview = None
    prompt = ('Survey this rendered 3D scan image. Identify the surrounding environment, major distinct structures '
        'and spatial areas that a virtual tour should show. Include visible houses/buildings, façades, rooms, '
        'lawns, courtyards, driveways, terraces or hallways where supported. Each separate house is a separate subject; '
        'use distinguishing visible color/appearance in its name. Furniture and vehicles are handled separately. '
        'Do not label the entire picture as a room or courtyard: each subject needs a visible bounded region. '
        'Give at most 5 subjects, prioritizing major structures and areas. '
        'For each, return kind (structure|area), category (simple noun), a distinguishing name, short description and visible attributes. '
        'Return scene_type, description and subjects. Do not return coordinates or infer hidden surfaces.')
    try:
        for number, index in enumerate(selected):
            emit('Discovering structures and areas', number, len(selected))
            cache = asset / f'{digest[:16]}-{index}.json'
            try:
                if cache.exists():
                    answer = json.loads(cache.read_text())
                else:
                    with Image.open(root / transforms['frames'][index]['file_path']) as image:
                        answer = model.ask(prompt, [image.convert('RGB')], tokens=1350, schema=schema())
                    atomic_json(cache, answer)
                if not isinstance(answer.get('subjects'), list):
                    raise ValueError('No structured environment subjects returned')
                summaries.append(dict(frame_idx=index, type=str(answer.get('scene_type', ''))[:70],
                    description=str(answer.get('description', ''))[:260]))
                for subject in answer['subjects'][:MAX_SUBJECTS_PER_VIEW]:
                    normalized = normalize_subject(subject, index, transforms)
                    if normalized:
                        observations.append(normalized)
            except ValueError as error:
                failures.append(dict(frame_idx=index, reason=str(error)[:180]))
        if summaries:
            from scene_understanding import room_schema
            emit('Reconciling the environment across views')
            images = []
            try:
                for index in selected[:4]:
                    with Image.open(root / transforms['frames'][index]['file_path']) as image:
                        images.append(image.convert('RGB'))
                overview = model.ask('These are different views of ONE scanned environment. Describe the overall '
                    'setting and its major visible structures and spatial areas. Reconcile the views into one factual '
                    'description of at most TWO concise sentences. Mention only the setting and major structures/areas. '
                    'Do not describe weather or list small objects. Do not put confidence commentary in the description. '
                    'Do not assume the scan owner wants to showcase any particular object. '
                    'Do not infer hidden rooms, complete building shapes, or exact counts when views are ambiguous. '
                    'Return scene_type, description and confidence (high|medium|low).', images,
                    tokens=350, schema=room_schema())
            except ValueError as error:
                failures.append(dict(frame_idx=selected[0], reason='Multi-view summary: '+str(error)[:140]))
            finally:
                for image in images:
                    image.close()
        emit('Environment observations ready', len(selected), len(selected))
    finally:
        del model
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if not summaries:
        raise ValueError('Environment discovery failed in every selected view. Inspect the analysis log.')
    report = dict(version=VERSION, summary_version=2, fingerprint=digest, source_sha256=source_hash,
        model=dict(id=MODEL, revision=REVISION), selected_frames=selected,
        summaries=summaries, observations=observations, failed_views=failures,
        camera_input_sha256=transforms.get('camera_validation', {}).get('camera_input', {}).get('sha256'),
        scene=dict(type=str(overview.get('scene_type', 'unknown'))[:70],
            description=str(overview.get('description', ''))[:600],
            confidence=overview.get('confidence') if overview.get('confidence') in {'high','medium','low'} else 'low') if overview else None)
    atomic_json(target, report)
    return report


COLORS = {'white', 'red', 'blue', 'green', 'yellow', 'black', 'brown', 'gray', 'grey', 'beige', 'pink'}


def appearance_color(proposal):
    fields = [proposal['name']] + sorted(proposal['attributes'], key=lambda a: not any(t in a.lower() for t in ('exterior', 'siding', 'facade', 'façade', 'wall')))
    for text in fields:
        if any(t in text.lower() for t in ('sky', 'roof', 'door', 'window')):
            continue
        colors = [word for word in re.findall(r'\w+', text.lower()) if word in COLORS]
        if colors:
            return 'gray' if colors[0] == 'grey' else colors[0]
    return None


def compatible(a, b):
    categories_match = a['category'] == b['category'] or {a['category'], b['category']} == {'house', 'building'}
    if a['kind'] != b['kind'] or not categories_match:
        return False
    ca, cb = appearance_color(a), appearance_color(b)
    return not (ca and cb and ca != cb)


def sample_surface(points, voxel):
    """Downsample without moving an observed surface to voxel cell centers."""
    cells, inverse = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack([np.bincount(inverse, weights=points[:, axis], minlength=len(cells)) / counts for axis in range(3)])


def reprojection_fraction(source, target, transforms, depths, tolerance):
    points = source['points'][np.linspace(0, len(source['points'])-1, min(320, len(source['points'])), dtype=int)]
    pose = np.asarray(transforms['frames'][target['frame']]['transform_matrix'])
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    x = np.rint(camera[:, 0]/np.maximum(z, .001)*transforms['fl_x']+transforms['cx']).astype(int)
    y = np.rint(camera[:, 1]/np.maximum(z, .001)*transforms['fl_y']+transforms['cy']).astype(int)
    valid = (z > .01) & (x >= 0) & (y >= 0) & (x < transforms['w']) & (y < transforms['h'])
    selected = np.where(valid)[0]
    if len(selected) < 24:
        return 0.
    actual = depths[target['frame']][y[selected], x[selected]]
    match = target['mask'][y[selected], x[selected]] & (actual > .01)
    match &= np.abs(actual-z[selected]) <= np.maximum(tolerance, z[selected]*.04)
    return float(match.sum()/len(points))


def associate(nodes, tolerance, transforms=None, root=None):
    """Fuse matching observations across all views, with instance conflicts.

    Matching only the largest view strands patches that overlap a later view.
    Process the strongest pairwise links first, and check the whole component
    before joining it so a generic observation cannot bridge distinct houses.
    """
    from refine_inventory import surface_overlap
    from scipy.spatial import cKDTree
    depths = {}
    if transforms is not None and root is not None:
        for index in {n['frame'] for n in nodes}:
            path = Path(root) / transforms['frames'][index]['depth_path']
            depths[index] = np.load(path.with_suffix('.npy'), allow_pickle=False)
    ordered = sorted(nodes, key=lambda n: (-len(n['points']), n['frame'], n['proposal']['name'], tuple(n['points'].mean(0))))
    conflicts = [set() for _ in ordered]
    edges = []
    bounds = [(n['points'].min(0), n['points'].max(0)) for n in ordered]
    trees = {}
    for i, a in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            if not compatible(a['proposal'], b['proposal']):
                conflicts[i].add(j)
                conflicts[j].add(i)
                continue
            # Image pixel footprints set the uncertainty locally. The full scan
            # radius must not make neighboring small surfaces interchangeable.
            distance = max(tolerance, min(a.get('voxel', 0), b.get('voxel', 0)) * 2,
                           min(a.get('spacing', 0), b.get('spacing', 0)) * 1.5)
            mask_overlap = None
            if a['frame'] == b['frame']:
                mask_overlap = np.count_nonzero(a['mask'] & b['mask']) / max(1, min(a['mask'].sum(), b['mask'].sum()))
            if np.any(bounds[i][1] + distance * 2 < bounds[j][0]) or np.any(bounds[j][1] + distance * 2 < bounds[i][0]):
                if mask_overlap is not None and mask_overlap < .25:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                continue
            for index in (i, j):
                if index not in trees:
                    trees[index] = cKDTree(ordered[index]['points'])
            ab, ba = surface_overlap(a['points'], b['points'], distance, trees=(trees[i], trees[j]))
            if mask_overlap is not None:
                # A small crop of a larger segmented surface is not a second
                # instance. Comparable, disjoint detections remain cannot-links.
                contained = max(ab, ba) >= .9 and max(len(a['points']), len(b['points'])) >= min(len(a['points']), len(b['points'])) * 1.8
                if mask_overlap < .25 and not contained:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
            same_surface = min(ab, ba) >= .2 or max(ab, ba) >= .7
            if depths and a['frame'] != b['frame']:
                projected_ab = reprojection_fraction(a, b, transforms, depths, distance * 2)
                projected_ba = reprojection_fraction(b, a, transforms, depths, distance * 2)
                if same_surface:
                    same_surface = max(projected_ab, projected_ba) >= .12
                else:
                    ab, ba = projected_ab, projected_ba
                    same_surface = min(ab, ba) >= .45 or (max(ab, ba) >= .8 and min(ab, ba) >= .12)
            if same_surface:
                edges.append((min(ab, ba) + max(ab, ba), i, j))
    owners = list(range(len(ordered)))
    members = {i: {i} for i in owners}
    for _, i, j in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
        a, b = owners[i], owners[j]
        if a == b or any(conflicts[k] & members[b] for k in members[a]):
            continue
        for k in members[b]:
            owners[k] = a
        members[a].update(members.pop(b))
    return [[ordered[i] for i in sorted(group)] for group in sorted(members.values(), key=min)]


def environment_queries(report):
    by_query = {}
    proposals = list(report['observations'])
    setting = (report.get('scene') or {}).get('type', '').lower().replace('_', ' ')
    if any(word in setting for word in ('office', 'bedroom', 'living room', 'kitchen', 'interior', 'dining room', 'hallway')):
        # A search hypothesis only: it still needs an image detection, mask and
        # depth surface. A floor patch never asserts a complete room volume.
        proposals.append(dict(kind='area', category='floor', name='Floor area', attributes=[],
            description='Observed floor surface; room boundaries and camera clearance are not inferred.',
            frame_idx=report['selected_frames'][0]))
    for proposal in proposals:
        category = proposal['category']
        by_query.setdefault(category, []).append(proposal)
        color = appearance_color(proposal)
        if color:
            by_query.setdefault(color + ' ' + category, []).append(proposal)
    generic = {p['category'] for p in proposals}
    return by_query, generic


def localize_environment(report, root, transforms, detector=None, detections=None):
    """Dedicated open-vocabulary detector supplies pixel boxes, never the VLM."""
    from pipeline import _run_owlv2, OWL_MODEL_REVISION
    from scene_understanding import atomic_json
    root = Path(root)
    by_query, generic = environment_queries(report)
    # Bound model memory/work even if the overview proposes many synonyms.
    labels = sorted(by_query, key=lambda q: (q not in generic, -len(by_query[q]), q))[:24]
    if not labels:
        return [], []
    fingerprint = hashlib.sha256(json.dumps([report['fingerprint'], OWL_MODEL_REVISION, labels, .15, 4]).encode()).hexdigest()
    cache = root / 'environment-detections.json'
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    emit('Locating environment subjects in images')
    if detections is None and saved.get('fingerprint') == fingerprint:
        detections = saved['detections']
    elif detections is None:
        detections = (detector or _run_owlv2)(root / 'frames', labels, transforms,
            transforms['scene_radius'], score_threshold=.15)
    if saved.get('fingerprint') != fingerprint or detections is not saved.get('detections'):
        detections = [dict(label=d['label'], score=float(d['score']), frame_idx=d['frame_idx'], box_2d=d['box_2d'])
            for d in detections if np.isfinite(d['score']) and np.isfinite(d['box_2d']).all()]
        atomic_json(cache, dict(fingerprint=fingerprint, detections=detections))
    observations, found = [], set()
    eligible = [i for i, f in enumerate(transforms['frames']) if f.get('quality', {}).get('accepted', True)]
    if len(eligible) > 192:
        retained = set(report['selected_frames'])
        retained.update(eligible[i] for i in np.linspace(0, len(eligible)-1, 192-len(retained), dtype=int))
        eligible = sorted(retained)
    eligible = set(eligible)
    per_frame_limit = max(1, 384 // max(1, len(eligible)))
    frame_totals = Counter()
    per_view = Counter()
    for detection in sorted(detections, key=lambda d: -d['score']):
        query, frame_idx, box = detection['label'], detection['frame_idx'], detection['box_2d']
        if query not in labels or frame_idx not in eligible or len(box) != 4 or detection['score'] < .15:
            continue
        # A description observed from one side must not be assigned to every
        # detector match elsewhere in the scan (e.g. beige applied to all houses).
        if query not in generic and not any(p['frame_idx'] == frame_idx for p in by_query[query]):
            continue
        if not all(isinstance(v, (int,float)) and not isinstance(v, bool) and np.isfinite(v) for v in box):
            continue
        box = [max(0., min(float(value), float(transforms['w'] if i % 2 == 0 else transforms['h']))) for i, value in enumerate(box)]
        if box[2]-box[0] < 8 or box[3]-box[1] < 8 or per_view[(query,frame_idx)] >= 3:
            continue
        if frame_totals[frame_idx] >= per_frame_limit:
            continue
        per_view[(query,frame_idx)] += 1
        frame_totals[frame_idx] += 1
        proposal = next((p for p in by_query[query] if p['frame_idx']==frame_idx), by_query[query][0])
        if query in generic:
            proposal = dict(proposal, name=proposal['category'].capitalize(), attributes=[],
                description='Observed '+proposal['category']+' surface; appearance is not confirmed.')
        observations.append(dict(proposal, frame_idx=frame_idx, box=box, detector_score=detection['score']))
        found.add(query)
    missing = [dict(name=by_query[q][0]['name'],kind=by_query[q][0]['kind'],frame_idx=by_query[q][0]['frame_idx'],
        reason='No image detection passed localization thresholds') for q in by_query if q not in found]
    return observations, missing


def ground_environment(report, root, transforms, device, detector=None, detections=None):
    from refine_inventory import _segment_candidates, fit_oriented_box
    root = Path(root)
    observations, missing = localize_environment(report, root, transforms, detector, detections)
    candidates = []
    for number, proposal in enumerate(observations):
        candidates.append(dict(id=f'env_candidate_{number:04d}', label=proposal['category'],
            frames=[dict(frame_idx=proposal['frame_idx'], box=proposal['box'], score=proposal['detector_score'])]))
    emit('Segmenting environment subjects', 0, len(candidates))
    samples = _segment_candidates(candidates, root, transforms, device, report['fingerprint'], progress=emit)
    # Quantization follows observed image resolution, not the total scan size.
    # This keeps identity decisions comparable across rooms and outdoor scans.
    precision = 1e-6
    maximum_voxel = max(float(transforms['scene_radius']) * .008, precision)
    nodes, ungrounded = [], list(missing)
    for candidate, proposal in zip(candidates, observations):
        accepted = []
        for sample in samples[candidate['id']]:
            mask = sample['mask']
            x1, y1, x2, y2 = proposal['box']
            inside = np.zeros_like(mask)
            inside[max(0, int(y1)):min(mask.shape[0], int(np.ceil(y2))),
                   max(0, int(x1)):min(mask.shape[1], int(np.ceil(x2)))] = True
            if np.count_nonzero(mask & inside) < mask.sum() * .85:
                continue
            if mask.sum() < max(32, inside.sum() * .08):
                continue
            pose = np.asarray(transforms['frames'][sample['frame']]['transform_matrix'])
            median_z = float(np.median(np.abs((sample['points'] - pose[:3, 3]) @ pose[:3, 2])))
            pixel_spacing = median_z / min(transforms['fl_x'], transforms['fl_y'])
            from scipy.spatial import cKDTree
            neighbor_spacing = cKDTree(sample['points']).query(sample['points'], k=2)[0][:, 1]
            spacing = max(pixel_spacing, float(np.percentile(neighbor_spacing, 75)))
            voxel = max(precision, min(maximum_voxel, spacing * 2))
            points = sample_surface(sample['points'], voxel)
            if len(points) < 24:
                continue
            # Reject depth leaks that leave several disconnected surfaces. Keep
            # the largest connected patch, rather than bounding empty space.
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            edges = cKDTree(points).query_pairs(max(voxel * 3, pixel_spacing * 3), output_type='ndarray')
            graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points), len(points)))
            _, components = connected_components(graph, directed=False)
            largest = np.bincount(components).argmax()
            coherent = points[components == largest]
            if len(coherent) < 24 or len(coherent) < len(points) * .4:
                continue
            accepted.append(dict(sample, points=coherent, proposal=proposal, spacing=spacing, voxel=voxel))
        if accepted:
            nodes.extend(accepted)
        else:
            ungrounded.append(dict(name=proposal['name'], kind=proposal['kind'], frame_idx=proposal['frame_idx'],
                reason='No coherent segmented depth surface passed validation'))
    emit('Resolving environment identities', 0, len(nodes))
    groups = associate(nodes, precision * 2, transforms, root)
    entities, surfaces = [], {}
    for number, group in enumerate(groups):
        voxel = max(precision, float(np.percentile([n['voxel'] for n in group], 25)))
        points = sample_surface(np.vstack([n['points'] for n in group]), voxel)
        proposal = max((n['proposal'] for n in group), key=lambda p: (
            bool(appearance_color(p)), p['name'].lower() != p['category'].lower(), len(p['attributes'])))
        name = proposal['name']
        color = appearance_color(proposal)
        if color and re.fullmatch(r'(house|building)\s*\d*', name.lower()):
            name = color.capitalize() + ' ' + proposal['category']
        low, high = np.percentile(points, [1, 99], axis=0)
        high = np.maximum(high, low + voxel)
        key = f'environment_{number+1:04d}'
        surfaces[key] = points
        best = {}
        for node in group:
            if node['frame'] not in best or node['quality'] > best[node['frame']]['quality']:
                best[node['frame']] = node
        evidence = [dict(frame_idx=n['frame'], mask=n['path'], mask_quality=n['quality']) for n in best.values()]
        view_ids = {e['frame_idx'] for e in evidence}
        independent = {transforms['frames'][i].get('position_idx', i) for i in view_ids}
        entities.append(dict(id=key, kind=proposal['kind'], label=proposal['category'], category=proposal['category'],
            display_name=name, description=proposal['description'], attributes=proposal['attributes'],
            status='supported' if len(independent) >= 2 else 'uncertain',
            confidence=round(float(np.mean([n['quality'] for n in group])), 4),
            confidence_type='mean_mask_quality_not_calibrated_probability',
            supporting_view_count=len(view_ids), geometry_supporting_view_count=len(view_ids),
            independent_position_count=len(independent), position=dict(zip('xyz', map(float, (low+high)/2))),
            scale=dict(zip('xyz', map(float, high-low))),
            bounding_box=dict(type='observed_environment_surface', min=dict(zip('xyz', map(float, low))), max=dict(zip('xyz', map(float, high)))),
            oriented_box=fit_oriented_box(points, voxel), parent_id=None, children=[],
            frames=[dict(frame_idx=n['frame'], box=n['proposal']['box'], score=n['quality']) for n in best.values()],
            mask_evidence=evidence, description_source='vision_language_model_with_segmented_depth',
            geometry_basis='observed_segmented_surface',
            surface_samples=points[np.linspace(0, len(points)-1, min(512, len(points)), dtype=int)].tolist()))
    np.savez_compressed(root / 'environment_surfaces.npz', **surfaces)
    types = Counter(s['type'] for s in report['summaries'] if s['type'])
    dominant = types.most_common(1)[0][0] if types else 'unknown'
    summary = next((s for s in report['summaries'] if s['type'] == dominant), report['summaries'][0])
    result = dict(version=VERSION, association_version=ASSOCIATION_VERSION,
        fingerprint=hashlib.sha256(f"{report['fingerprint']}:{ASSOCIATION_VERSION}".encode()).hexdigest(), source_sha256=report['source_sha256'],
        camera_input_sha256=report.get('camera_input_sha256'),
        status='partial' if report['failed_views'] or ungrounded else 'ready',
        scene=report.get('scene') or dict(type=dominant, description=summary['description'], confidence='low'),
        selected_frames=report['selected_frames'], analyzed_view_count=len(report['summaries']),
        failed_views=report['failed_views'], entities=entities, relations=[], ungrounded=ungrounded,
        limitations=['Environment names and relationships are model estimates.',
            'Extents and surface samples describe observed surfaces, not complete hidden buildings or room volumes.',
            'Areas do not establish camera clearance; use the independent navigation map.',
            'Non-overlapping façades may remain separate until sufficient linking evidence exists.'])
    emit('Environment surfaces ready', len(entities), len(entities))
    return result


def attach_environment(result, environment, root):
    """Attach conservative part-of/located-in links supported by saved surfaces."""
    from refine_inventory import surface_overlap
    from scipy.spatial import cKDTree
    entities = environment['entities']
    if environment['source_sha256'] != result['input']['sha256']:
        raise ValueError('Environment source fingerprint mismatch')
    if environment.get('camera_input_sha256') != result['input'].get('camera_input_sha256'):
        raise ValueError('Environment camera input fingerprint mismatch')
    root = Path(root)
    with np.load(root / 'object_surfaces.npz', allow_pickle=False) as data:
        objects = {k: data[k] for k in data.files}
    with np.load(root / 'environment_surfaces.npz', allow_pickle=False) as data:
        surfaces = {k: data[k] for k in data.files}
    bounds = {key: (points.min(0), points.max(0)) for key, points in surfaces.items()}
    trees = {}
    def surface_tree(key):
        if key not in trees:
            trees[key] = cKDTree(surfaces[key])
        return trees[key]
    tolerance = max(float(result.get('instance_resolution', {}).get('voxel_size', .01)) * 2, 1e-5)
    relations = []
    structural_parts = {'window', 'door', 'roof', 'wall', 'exterior wall', 'house exterior',
                        'railing', 'balustrade', 'porch', 'staircase', 'chimney', 'garage door'}
    for obj in result['objects']:
        if obj['label'] not in structural_parts:
            continue
        points = objects.get(obj['id'], np.empty((0, 3)))
        if not len(points):
            continue
        object_tree = cKDTree(points)
        options = []
        for entity in entities:
            if entity['kind'] != 'structure':
                continue
            ab, _ = surface_overlap(points, surfaces[entity['id']], tolerance,
                trees=(object_tree, surface_tree(entity['id'])))
            if ab >= .7:
                options.append((ab, entity))
        if options:
            parent = max(options, key=lambda p: p[0])[1]
            relations.append(dict(subject=obj['id'], relation='observed_surface_part_of', object=parent['id'], basis='segmented_surface_overlap'))
            parent['children'].append(obj['id'])
    # Adjacent is symmetric and geometric; no unsupported semantic "front".
    for i, a in enumerate(entities):
        for b in entities[i+1:]:
            if a['kind'] == b['kind']:
                continue
            low_a, high_a = bounds[a['id']]
            low_b, high_b = bounds[b['id']]
            if np.any(high_a + tolerance * 3 < low_b) or np.any(high_b + tolerance * 3 < low_a):
                continue
            gap = float(surface_tree(a['id']).query(surfaces[b['id']])[0].min())
            if gap <= tolerance * 3:
                relations.append(dict(subject=a['id'], relation='adjacent_to', object=b['id'],
                    basis='observed_surface_distance', gap_source_units=round(gap, 5)))
    environment['relations'] = relations
    result['environment'] = environment
    result['limitations'] = list(dict.fromkeys(result.get('limitations', []) + environment['limitations']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='Existing interactions.json with RGB/depth views')
    parser.add_argument('--views', type=int, default=12)
    args = parser.parse_args()
    from scene_understanding import atomic_json
    from pipeline import _select_device
    target = Path(args.result)
    result = json.loads(target.read_text())
    transforms = json.loads((target.parent / 'transforms.json').read_text())
    report = discover_environment(target.parent, transforms, result['input']['sha256'], max(1, min(32, args.views)))
    environment = ground_environment(report, target.parent, transforms, _select_device())
    atomic_json(target, attach_environment(result, environment, target.parent))


if __name__ == '__main__':
    main()
