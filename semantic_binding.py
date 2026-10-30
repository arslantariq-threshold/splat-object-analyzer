"""Bind model-proposed semantic roles to observed instances; models never choose IDs."""
from collections import defaultdict
import math
import re

import numpy as np
from scipy.spatial.transform import Rotation

ARCHITECTURE = {'floor', 'wall', 'ceiling', 'door', 'doorway', 'window', 'curtain'}


def category_key(value):
    return ' '.join(str(value).lower().replace('_', ' ').split())


def compatible_category(actual, requested):
    if category_key(actual) == category_key(requested):
        return True
    from refine_inventory import family
    return family(category_key(actual)) == family(category_key(requested))


def catalog(objects):
    result = defaultdict(list)
    for obj in objects:
        if obj['status'] != 'rejected':
            names = result[category_key(obj['category'])]
            if obj['display_name'] not in names:
                names.append(obj['display_name'])
    return {key: names[:2] for key, names in sorted(result.items())}


def quality(obj):
    return (obj.get('source_confidence', .3) + .06*math.log1p(obj.get('geometry_supporting_view_count', 1)) +
            .2*max([v.get('score', 0) for v in obj.get('evidence', [])] or [0]))


def bounds(obj):
    box = obj.get('oriented_box') or obj
    center = np.array([box['position'][a] for a in 'xyz'])
    size = np.array([box.get('scale', {}).get(a, 0) for a in 'xyz'])
    rotation = box.get('rotation', dict(x=0,y=0,z=0,w=1))
    axes = Rotation.from_quat([rotation[a] for a in 'xyzw']).as_matrix()
    return center, np.abs(axes) @ (size/2)


def proximity(a, b, radius):
    ca, ea = bounds(a)
    cb, eb = bounds(b)
    gap = float(np.linalg.norm(np.maximum(np.abs(ca-cb)-ea-eb, 0)))
    distance = float(np.linalg.norm(ca-cb))
    return gap/radius + .25*distance/radius, gap, distance


def bind_groups(objects, prototypes, radius):
    available = [o for o in objects if o['status'] != 'rejected'
                 and (o.get('kind', 'object') == 'object' or o['status'] == 'supported')]
    by_category = defaultdict(list)
    for obj in available:
        by_category[category_key(obj['category'])].append(obj)
    groups = []
    for prototype in prototypes:
        context = category_key(prototype.get('name','') + ' ' + prototype.get('description',''))
        def specific_role(category):
            category = category_key(category)
            explicit = [key for key in by_category if key != category and compatible_category(key, category)
                        and re.search(r'\b'+re.escape(key)+r'\b', context)]
            return explicit[0] if len(explicit) == 1 else category
        requested_anchor = category_key(prototype.get('anchor_category', ''))
        anchor_type = specific_role(requested_anchor)
        member_types = list(dict.fromkeys(specific_role(c) for c in prototype.get('member_categories', [])))
        member_types = [c for c in member_types if c != anchor_type]
        candidates = []
        for anchor in [o for o in available if category_key(o['category']) == anchor_type]:
            if anchor.get('parent_id'):
                continue
            members = []
            for member_type in member_types:
                neighbors = []
                for obj in [o for o in available if category_key(o['category']) == member_type and o['id'] != anchor['id']]:
                    cost, gap, distance = proximity(anchor, obj, radius)
                    linked = obj.get('parent_id') == anchor['id']
                    if linked or (gap <= radius*.12 and distance <= radius*.65):
                        neighbors.append((cost - .03*quality(obj) - (1 if linked else 0), obj))
                if neighbors:
                    members.append(min(neighbors, key=lambda pair: pair[0])[1])
            candidates.append((len(members)*2 + quality(anchor), anchor, members))
        accepted_members = set()
        accepted_anchors = []
        for _, anchor, members in sorted(candidates, key=lambda item: -item[0]):
            member_ids = {obj['id'] for obj in members}
            # Shared focal accessories often reveal multiple desk/shelf fragments
            # belonging to one functional setup; inventory instances remain intact.
            if member_ids & accepted_members:
                continue
            if not members and any(proximity(anchor, other, radius)[2] < radius*.15 for other in accepted_anchors):
                continue
            accepted_members.update(member_ids)
            accepted_anchors.append(anchor)
            groups.append(dict(id=f'group_{len(groups)+1:03d}', name=str(prototype.get('name') or anchor_type)[:100],
                description=str(prototype.get('description') or '')[:400],
                anchor_id=anchor['id'], anchor_category=anchor_type,
                object_ids=[anchor['id']]+[o['id'] for o in members], confidence=prototype.get('confidence','low'),
                basis='model_roles_bound_to_observed_geometry'))
    counts = defaultdict(int)
    for group in groups:
        counts[group['name']] += 1
    numbered = defaultdict(int)
    for group in groups:
        name = group['name']
        if counts[name] > 1:
            numbered[name] += 1
            group['name'] = name + ' ' + str(numbered[name])
    return groups


def tour_subjects(objects, categories):
    heroes = []
    for category in categories:
        if category_key(category) in ARCHITECTURE:
            continue
        candidates = [o for o in objects if category_key(o['category']) == category_key(category)
                      and o['status'] != 'rejected' and not o.get('parent_id')
                      and (o.get('kind', 'object') == 'object' or o['status'] == 'supported')]
        if candidates:
            structures = [o for o in candidates if o.get('kind') == 'structure']
            if structures:
                heroes.extend(o['id'] for o in sorted(structures, key=quality, reverse=True)[:6])
            else:
                heroes.append(max(candidates, key=quality)['id'])
    return list(dict.fromkeys(heroes))


def resolve_intent(index, intent):
    available = [o for o in index['objects'] if o['status'] != 'rejected']
    by_id = {o['id']:o for o in available}
    groups = [g for g in index['groups'] if g['name'] in intent.get('target_groups', [])]
    ids = list(index['hero_object_ids']) if intent.get('tour') else []
    for group in groups:
        ids.extend(group['object_ids'])
    reference = category_key(intent.get('near_category', ''))
    references = [o for o in available if category_key(o['category']) == reference] if reference else []
    terms = set(category_key(intent.get('attributes', '')).split())
    for category in intent.get('target_categories', []):
        candidates = [o for o in available if category_key(o['category']) == category_key(category)]
        if terms:
            candidates = [o for o in candidates if terms & set(category_key(o['display_name']).split())]
        if references and candidates:
            radius = max(index.get('scene_radius', 1), .01)
            candidates = [min(candidates, key=lambda o:min(proximity(o, ref, radius)[0] for ref in references))]
        ids.extend(o['id'] for o in sorted(candidates, key=lambda o:-quality(o)))
    return dict(interpretation=intent.get('interpretation',''), object_ids=list(dict.fromkeys(i for i in ids if i in by_id)),
                group_ids=[g['id'] for g in groups], message=intent.get('message',''))


def direct_intent(index, prompt):
    """Resolve explicit group/category requests before using a model for paraphrases."""
    text = ' '.join(re.findall(r'\w+', prompt.lower()))
    if re.search(r"\b(not|except|without|avoid|exclude)\b|don't", prompt.lower()):
        return None
    names = [g['name'] for g in index['groups'] if re.search(r'\b'+re.escape(category_key(g['name']))+r'\b', text)]
    spatial = re.search(r'\b(near|beside|closest)\b|\bnext to\b', text)
    if names and not spatial:
        return dict(interpretation=prompt, target_groups=names, target_categories=[], tour=False, message='')
    if re.search(r'\b(tour|walkthrough)\b|\b(show|look|walk) around\b', text):
        return dict(interpretation=prompt, target_groups=[], target_categories=[], tour=True, message='')
    categories = set(catalog(index['objects']))
    found = []
    residual = text
    for category in sorted(categories, key=lambda value:-len(value)):
        pattern = r'\b'+re.escape(category)+r's?\b'
        if re.search(pattern, residual):
            found.append(category)
            residual = re.sub(pattern, ' ', residual)
    if spatial and len(found)>=2:
        ordered=sorted(found,key=lambda category:re.search(r'\b'+re.escape(category)+r's?\b',text).start())
        from semantic_search import COLORS
        attributes=' '.join(sorted(set(text.split()) & COLORS))
        return dict(interpretation=prompt,target_groups=[],target_categories=[ordered[0]],near_category=ordered[1],attributes=attributes,tour=False,message='')
    filler = {'show','find','me','my','the','a','an','all','some','of','please','off','and','objects','object','in','scene','this','that'}
    if found and not (set(residual.split()) - filler):
        return dict(interpretation=prompt, target_groups=[], target_categories=found, tour=False, message='')
    return None
