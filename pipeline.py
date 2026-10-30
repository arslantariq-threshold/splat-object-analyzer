"""
Master pipeline:
  1. Render spherical camera views from multiple positions (render_cameras.py)
  2. Discover labels with RAM++ (or use an optional supplied prompt)
  3. Run OWLv2 open-vocabulary detection on every frame
  4. Back-project 2D boxes to 3D using per-pixel depth maps
  5. Cluster per-label detections → single 3D bounding box per object
  6. Resolve instances with segmentation masks and link parts/contents
  7. Write interactions.json in WebXR format

Usage:
  python pipeline.py --ply scene.ply --job_dir /tmp/job123
"""

import argparse
import hashlib
import json
import math
import os
import time
from types import SimpleNamespace
from collections import defaultdict
from pathlib import Path

# Let OWLv2 ops not yet implemented on Apple MPS fall back to CPU instead of hard-erroring.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import batched_nms

import render_cameras
from config import PipelineConfig, QUALITY_PRESETS, DEFAULT_QUALITY
from camera_inputs import add_camera_arguments

OWL_MODEL_REVISION = 'cfd3195ba4ea9592eec887ded089f4c08eff231d'

def _select_device() -> str:
    """Compute device for OWLv2. Order: CUDA → Apple MPS → CPU.
    Set WMD_DEVICE=cuda|mps|cpu to override (e.g. WMD_DEVICE=cpu if MPS misbehaves)."""
    override = os.getenv("WMD_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 2-D → 3-D lifting helpers
# ---------------------------------------------------------------------------

def _unproject_box(box_2d, depth, K_inv, c2w):
    """
    Unproject a 2-D bounding box centre to a 3-D ray and place it at `depth`
    along that ray in world space.
    """
    x1, y1, x2, y2 = box_2d
    cx_px = (x1 + x2) / 2.0
    cy_px = (y1 + y2) / 2.0
    w_px = x2 - x1
    h_px = y2 - y1

    p_cam_norm = K_inv @ np.array([cx_px, cy_px, 1.0])
    ray_cam = p_cam_norm / np.linalg.norm(p_cam_norm)
    point_cam = ray_cam * (depth / ray_cam[2])
    point_world = (c2w[:3, :3] @ point_cam) + c2w[:3, 3]

    return point_world, (w_px, h_px)


def _pixel_size_to_world(w_px, h_px, depth, fl_x, fl_y):
    world_w = (w_px / fl_x) * depth
    world_h = (h_px / fl_y) * depth
    return world_w, world_h


def _world_box_size(world_w, world_h, c2w):
    # ponytail: thickness remains a heuristic until mask/depth fusion is added.
    camera_size = np.array([world_w, world_h, (world_w + world_h) / 2])
    return np.abs(c2w[:3, :3]) @ camera_size


def _bounds_iou(low_a, high_a, low_b, high_b):
    intersection = np.maximum(0, np.minimum(high_a, high_b) - np.maximum(low_a, low_b)).prod()
    union = np.prod(high_a - low_a) + np.prod(high_b - low_b) - intersection
    return float(intersection / union) if union > 0 else 0.


def _merge_duplicate_clusters(clusters):
    # ponytail: this geometric heuristic can still merge occluding instances.
    # Instance masks and multi-view association are needed for stronger identity.
    merged = []
    for peak, votes, label, pos, scale, members in clusters:
        for index, previous in enumerate(merged):
            prior_peak, _, prior_label, prior_pos, prior_scale, prior_members = previous
            if label != prior_label:
                continue
            if _bounds_iou(pos - scale / 2, pos + scale / 2,
                           prior_pos - prior_scale / 2, prior_pos + prior_scale / 2) < .3:
                continue
            by_frame = {m['frame_idx']: m for m in members}
            prior_by_frame = {m['frame_idx']: m for m in prior_members}
            shared = by_frame.keys() & prior_by_frame.keys()
            if not shared:
                continue
            overlaps = []
            for frame in shared:
                box = np.asarray(by_frame[frame]['box_2d'])
                prior_box = np.asarray(prior_by_frame[frame]['box_2d'])
                overlaps.append(_bounds_iou(box[:2], box[2:], prior_box[:2], prior_box[2:]))
            if np.median(overlaps) < .2:
                continue
            for frame, member in by_frame.items():
                if frame not in prior_by_frame or member['score'] > prior_by_frame[frame]['score']:
                    prior_by_frame[frame] = member
            combined = list(prior_by_frame.values())
            weights = np.array([m['score'] for m in combined])
            center = np.average([m['position'] for m in combined], axis=0, weights=weights)
            size = np.maximum(np.median([m['scale'] for m in combined], axis=0), .1)
            merged[index] = (max(peak, prior_peak), len(combined), label, center, size, combined)
            break
        else:
            merged.append((peak, votes, label, pos, scale, members))
    return merged


def _cluster_detections(detections, eps_m=0.5, max_per_label=None,
                        min_votes=8, min_peak_score=0.35):
    """
    Anchor-based greedy clustering with false-positive suppression.

    The seed detection's position is used as a FIXED anchor — no centroid drift.
    Drift was causing two nearby same-label objects (e.g. two sofas) to bleed
    into each other's clusters because the shifting centroid would migrate toward
    the second object and absorb its detections.
    """
    by_label = defaultdict(list)
    for det in detections:
        by_label[det["label"]].append(det)

    results = []
    for label, dets in by_label.items():
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        positions = np.array([d["position"] for d in dets])
        scales    = np.array([d["scale"]    for d in dets])
        scores    = np.array([d["score"]    for d in dets])

        used = [False] * len(dets)
        clusters = []
        for i in range(len(dets)):
            if used[i]:
                continue
            # Fixed anchor — never updated. Prevents cluster from drifting
            # toward a neighbouring object and stealing its detections.
            anchor = positions[i].copy()
            cluster_idx = [i]
            seen_frames = {dets[i]['frame_idx']}
            for j in range(i + 1, len(dets)):
                # Separate simultaneous instances and count each image once.
                if (not used[j] and dets[j]['frame_idx'] not in seen_frames
                        and np.linalg.norm(anchor - positions[j]) < eps_m):
                    cluster_idx.append(j)
                    seen_frames.add(dets[j]['frame_idx'])
            for j in cluster_idx:
                used[j] = True

            peak_score = float(scores[cluster_idx].max())
            vote_count = len(cluster_idx)

            if vote_count < min_votes or peak_score < min_peak_score:
                continue

            cluster_pos   = (positions[cluster_idx] * scores[cluster_idx, None]).sum(0) / scores[cluster_idx].sum()
            cluster_scale = np.median(scales[cluster_idx], axis=0)
            cluster_scale = np.maximum(cluster_scale, 0.1)
            # Carry the raw member dicts so callers can trace back to source frames
            member_dets = [dets[j] for j in cluster_idx]
            clusters.append((peak_score, vote_count, label, cluster_pos, cluster_scale, member_dets))

        clusters.sort(key=lambda c: c[0], reverse=True)
        clusters = _merge_duplicate_clusters(clusters)
        selected = clusters if max_per_label is None else clusters[:max_per_label]
        for peak, votes, lbl, pos, scale, members in selected:
            print(f"  [cluster] {lbl}: {votes} votes, peak={peak:.2f}, pos={pos.round(2)}")
            results.append((lbl, pos, scale, members))

    return results


# ---------------------------------------------------------------------------
# OWLv2 detection
# ---------------------------------------------------------------------------

def _postprocess_detection_groups(processor, outputs, threshold, target_sizes, independent_label_count=None):
    """Extra environment classes must not replace the existing object winner."""
    count = outputs.logits.shape[-1]
    if not independent_label_count or independent_label_count >= count:
        return processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    pieces = []
    for first, last in ((0, independent_label_count), (independent_label_count, count)):
        subset = SimpleNamespace(logits=outputs.logits[..., first:last], pred_boxes=outputs.pred_boxes)
        result = processor.post_process_grounded_object_detection(subset, threshold=threshold, target_sizes=target_sizes)[0]
        pieces.append({**result, 'labels': result['labels'] + first})
    return {key: torch.cat([piece[key] for piece in pieces]) for key in ('boxes', 'scores', 'labels')}


def _run_owlv2(frames_dir: Path, labels: list[str], transforms: dict, scene_radius: float,
               score_threshold: float = 0.12, frame_indices=None, independent_label_count=None):
    """
    Run OWLv2 on all rendered frames.
    Uses per-pixel depth maps (depth_XXXX.npy) for accurate 3D back-projection.
    Falls back to scene_radius if a depth file is missing or the sampled depth is zero.
    """
    device = _select_device()
    print(f"[pipeline] Loading OWLv2 on {device} …")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", revision=OWL_MODEL_REVISION)
    model     = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", revision=OWL_MODEL_REVISION).to(device)
    model.eval()

    fl_x = transforms["fl_x"]
    fl_y = transforms["fl_y"]
    cx   = transforms["cx"]
    cy   = transforms["cy"]
    W    = transforms["w"]
    H    = transforms["h"]

    K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    texts = [[f"a photo of a {lbl.strip()}" for lbl in labels]]

    raw_detections = []

    scene_center  = np.array(transforms.get("scene_center", [0.0, 0.0, 0.0]))
    stored_radius = transforms.get("scene_radius", scene_radius)

    indices = list(range(len(transforms["frames"]))) if frame_indices is None else list(frame_indices)
    for processed, frame_idx in enumerate(indices):
        frame_meta = transforms["frames"][frame_idx]
        frame_path = frames_dir.parent / frame_meta["file_path"]
        if not frame_path.exists():
            continue

        c2w = np.array(frame_meta["transform_matrix"], dtype=np.float64)

        # Fallback depth: distance from camera to scene centre (or scene_radius)
        cam_pos = c2w[:3, 3]
        cam_to_center = np.linalg.norm(cam_pos - scene_center)
        fallback_depth = cam_to_center if cam_to_center > stored_radius * 0.3 else stored_radius

        # Load per-pixel depth map if available
        depth_npy_path = (frames_dir.parent /
                          frame_meta.get("depth_path", "").replace(".png", ".npy"))
        depth_map = None
        if depth_npy_path.exists():
            depth_map = np.load(str(depth_npy_path)).astype(np.float64)

        image  = Image.open(frame_path).convert("RGB")
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[H, W]], device=device)
        results = _postprocess_detection_groups(processor, outputs, score_threshold, target_sizes, independent_label_count)

        # OWLv2 can emit overlapping proposals for one instance in one image.
        # Suppress these before separate-frame voting creates duplicate objects.
        boxes_cpu = results['boxes'].cpu()
        scores_cpu = results['scores'].cpu()
        labels_cpu = results['labels'].cpu()
        keep = batched_nms(boxes_cpu, scores_cpu, labels_cpu, iou_threshold=0.5)
        boxes = boxes_cpu[keep].numpy()
        scores = scores_cpu[keep].numpy()
        label_ids = labels_cpu[keep].numpy()

        for box, score, lid in zip(boxes, scores, label_ids):
            label = labels[lid].strip()
            bx1, by1, bx2, by2 = box          # original pixel coords (kept for box_2d)
            cx_px = int(np.clip((bx1 + bx2) / 2, 0, W - 1))
            cy_px = int(np.clip((by1 + by2) / 2, 0, H - 1))

            # Sample depth from a 5×5 patch around the box centre and take the
            # median of valid (non-zero) pixels — more robust than a single pixel.
            if depth_map is not None:
                h5, w5 = depth_map.shape
                py0 = max(0, cy_px - 2); py1 = min(h5, cy_px + 3)
                px0 = max(0, cx_px - 2); px1 = min(w5, cx_px + 3)
                patch = depth_map[py0:py1, px0:px1].ravel()
                valid = patch[patch > 0.01]
                sampled = float(np.median(valid)) if valid.size > 0 else 0.0
                box_depth = sampled if sampled > 0.01 else fallback_depth
            else:
                box_depth = fallback_depth

            world_pos, (w_px, h_px) = _unproject_box(box, box_depth, K_inv, c2w)
            world_w, world_h = _pixel_size_to_world(w_px, h_px, box_depth, fl_x, fl_y)

            raw_detections.append({
                "label":     label,
                "score":     float(score),
                "position":  world_pos,
                "scale":     _world_box_size(world_w, world_h, c2w),
                "frame_idx": frame_idx,
                "box_2d":    [float(bx1), float(by1), float(bx2), float(by2)],
            })
            print(f"  [detect] {label} ({score:.2f}) depth={box_depth:.2f} "
                  f"@ {world_pos.round(3)} {frame_path.name}")

        print(f'[detect] Processed {processed + 1}/{len(indices)} views', flush=True)

    return raw_detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(ply_path: str, prompt: str | None, job_dir: str,
                 cfg: PipelineConfig | None = None) -> list[dict]:
    started = time.perf_counter()
    cfg = cfg or PipelineConfig()
    job_dir = Path(job_dir)
    source = Path(ply_path).resolve()
    with source.open('rb') as f:
        source_hash = hashlib.file_digest(f, 'sha256').hexdigest()

    print("[pipeline] Step 1: Rendering camera views …")
    transforms_path = render_cameras.render_views(ply_path, str(job_dir), cfg, source_hash=source_hash)

    with open(transforms_path) as f:
        transforms = json.load(f)

    frames_dir = job_dir / "frames"

    cam_positions = np.array([
        frame["transform_matrix"]
        for frame in transforms["frames"]
    ])[:, :3, 3]
    scene_radius = float(np.linalg.norm(cam_positions, axis=1).mean())

    render_seconds = time.perf_counter() - started
    discovery_started = time.perf_counter()
    environment_report = None
    if cfg.environment:
        from environment_understanding import discover_environment
        environment_report = discover_environment(job_dir, transforms, source_hash, cfg.environment_views)
    if prompt and prompt.strip():
        labels = list(dict.fromkeys(l.strip() for l in prompt.split(',') if l.strip()))
        discovery = {'mode': 'manual', 'labels': labels, 'user_supplied_labels': True}
    else:
        from auto_discover import discover_labels
        labels, discovery = discover_labels(frames_dir, transforms, _select_device())
    discovery_seconds = time.perf_counter() - discovery_started
    detection_started = time.perf_counter()

    detection_labels = list(labels)
    if environment_report:
        from environment_understanding import environment_queries
        queries, generic = environment_queries(environment_report)
        environment_labels = sorted(queries, key=lambda q: (q not in generic, -len(queries[q]), q))[:24]
        detection_labels = list(dict.fromkeys(labels + environment_labels))
    if not detection_labels:
        raise ValueError('No scene subjects were discovered. Inspect the rendered views in the output folder.')
    print(f"[pipeline] Step 2: Detecting {detection_labels} in {len(transforms['frames'])} frames …")
    all_detections = _run_owlv2(frames_dir, detection_labels, transforms, scene_radius,
                                score_threshold=cfg.score_threshold, independent_label_count=len(labels))

    raw_detections = [d for d in all_detections if d['label'] in labels]

    frame_annotations: dict = {}   # frame_idx (str) → [{label, object_idx, box, score}]

    if not raw_detections:
        print("[pipeline] No detections above threshold.")
        interactions = []
    else:
        print(f"[pipeline] Step 3: Clustering {len(raw_detections)} raw detections …")
        clustered = _cluster_detections(
            raw_detections,
            eps_m=transforms.get("scene_radius", scene_radius) * 0.20,
            min_votes=cfg.min_votes,
            min_peak_score=cfg.min_peak_score,
        )

        interactions = []
        for obj_idx, (label, pos, scale, members) in enumerate(clustered):
            # Deduplicate members by frame_idx (keep highest-score per frame)
            best: dict = {}
            for m in members:
                fi = m["frame_idx"]
                if fi not in best or m["score"] > best[fi]["score"]:
                    best[fi] = m

            obj_frames = []
            for m in sorted(best.values(), key=lambda x: x["score"], reverse=True):
                fi = m["frame_idx"]
                obj_frames.append({
                    "frame_idx": fi,
                    "image": transforms['frames'][fi]['file_path'],
                    "box":       m["box_2d"],
                    "score":     round(m["score"], 4),
                })
                fkey = str(fi)
                frame_annotations.setdefault(fkey, []).append({
                    "label":      label,
                    "object_idx": obj_idx,
                    "box":        m["box_2d"],
                    "score":      round(m["score"], 4),
                })

            interactions.append({
                "id": f'object_{obj_idx + 1:04d}',
                "label":    label,
                "confidence": round(max(m['score'] for m in members), 4),
                "supporting_view_count": len(best),
                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "scale":    {"x": float(scale[0]), "y": float(scale[1]), "z": float(scale[2])},
                "bounding_box": {
                    "type": "axis_aligned_estimate",
                    "min": dict(zip('xyz', map(float, pos - scale / 2))),
                    "max": dict(zip('xyz', map(float, pos + scale / 2))),
                },
                "frames":   obj_frames,
            })

    output_path = job_dir / "interactions.json"
    result = {
        'input': {'path': str(source), 'sha256': source_hash},
        'coordinate_system': {'units': 'source_scene_units', 'metric_scale_verified': False},
        'camera_placement': transforms.get('camera_placement', 'upstream_density_sampler'),
        'navigation': transforms.get('navigation'),
        'detection_nms_iou': 0.5,
        'discovery': discovery,
        'settings': vars(cfg),
        'timings_seconds': {'render': render_seconds, 'discovery': discovery_seconds,
                            'detection_and_clustering': time.perf_counter() - detection_started,
                            'total': time.perf_counter() - started},
        'confidence_type': 'peak_detector_score_not_calibrated_probability',
        'limitations': ['Bounding box thickness is estimated from 2D detections.',
                       'Objects can be missed, misidentified, duplicated, or merged.'],
        'objects': interactions, 'frame_annotations': frame_annotations,
    }
    if transforms.get('camera_validation', {}).get('camera_input'):
        result['input']['camera_input_sha256'] = transforms['camera_validation']['camera_input']['sha256']
    candidates_path = job_dir / 'candidates.json'
    candidates_path.write_text(json.dumps(result, indent=2) + '\n')
    from refine_inventory import refine_inventory
    result = refine_inventory(result, job_dir, _select_device())
    result['timings_seconds']['instance_resolution'] = result['instance_resolution']['seconds']
    result['timings_seconds']['total'] = time.perf_counter() - started
    if environment_report is not None:
        from environment_understanding import ground_environment, attach_environment
        environment_started = time.perf_counter()
        environment = ground_environment(environment_report, job_dir, transforms, _select_device(), detections=all_detections)
        result = attach_environment(result, environment, job_dir)
        result['timings_seconds']['environment_grounding'] = time.perf_counter() - environment_started
        result['timings_seconds']['total'] = time.perf_counter() - started
    interactions = result['objects']
    temporary = output_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output_path)

    print(f"[pipeline] Done → {output_path} ({len(interactions)} objects)")
    return interactions


if __name__ == "__main__":
    _d = PipelineConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",              required=True)
    parser.add_argument("--prompt",           default=None)
    parser.add_argument("--job_dir",          required=True)
    parser.add_argument("--quality",          choices=list(QUALITY_PRESETS.keys()),
                        default=DEFAULT_QUALITY,
                        help="Camera-coverage preset (controls number of views)")
    parser.add_argument("--no-environment", action="store_true", help="Skip environment discovery and grounding")
    parser.add_argument("--score_threshold",  type=float, default=_d.score_threshold)
    parser.add_argument("--min_votes",        type=int,   default=_d.min_votes)
    parser.add_argument("--min_peak_score",   type=float, default=_d.min_peak_score)
    add_camera_arguments(parser)
    args = parser.parse_args()
    cfg = PipelineConfig.from_overrides(
        quality=args.quality,
        score_threshold=args.score_threshold,
        min_votes=args.min_votes,
        min_peak_score=args.min_peak_score,
        camera_input=args.camera_input, camera_limit=args.camera_limit, environment=not args.no_environment,
    )
    result = run_pipeline(args.ply, args.prompt, args.job_dir, cfg)
    print(json.dumps(result, indent=2))
