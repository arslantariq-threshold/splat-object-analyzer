"""
Local one-command runner for the WorldModelData pipeline.

Runs pipeline.run_pipeline on CUDA, Apple MPS, or CPU. Omitting --prompt adds
automatic image tagging before detection.

Example:
  python run_local.py --ply scene.spz --quality medium

Outputs <job_dir>/interactions.json plus the rendered frames/ and transforms.json.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pipeline
from config import PipelineConfig, QUALITY_PRESETS, DEFAULT_QUALITY
from camera_inputs import add_camera_arguments, load_camera_input, camera_fingerprint


def main():
    d = PipelineConfig()
    p = argparse.ArgumentParser(description="Run 3DGS object detection locally.")
    p.add_argument("--ply",     required=True, help="Path to a .ply or .spz Gaussian Splat file")
    p.add_argument("--prompt", default=None,
                   help='Optional comma-separated labels; omit to discover objects automatically')
    p.add_argument("--quality", choices=list(QUALITY_PRESETS.keys()), default=DEFAULT_QUALITY,
                   help="Camera-coverage preset (controls number of views)")
    p.add_argument('--no-environment', action='store_true', help='Skip environment discovery and grounding')
    p.add_argument('--no-navigation', action='store_true', help='Skip the geometry-only camera-space estimate')
    p.add_argument("--job_dir", default=None, help="Output directory (default: ./out_<name>_<sha256-prefix>)")
    p.add_argument("--score_threshold", type=float, default=d.score_threshold)
    p.add_argument("--min_votes",       type=int,   default=d.min_votes)
    p.add_argument("--min_peak_score",  type=float, default=d.min_peak_score)
    add_camera_arguments(p)
    args = p.parse_args()

    args.ply = str(Path(args.ply).expanduser().resolve())
    if not Path(args.ply).is_file():
        sys.exit(f"error: file not found: {args.ply}")

    if Path(args.ply).suffix.lower() not in {'.ply', '.spz'}:
        sys.exit('error: provide a Gaussian splat .ply or .spz file')
    capture_suffix = ''
    if args.camera_input:
        args.camera_input = str(Path(args.camera_input).expanduser().resolve())
        cameras = load_camera_input(args.camera_input)
        capture_suffix = f'_capture_{camera_fingerprint(cameras)[:8]}_{args.quality}_{args.camera_limit or "preset"}'
    elif args.camera_limit is not None:
        p.error('--camera-limit requires --colmap-model or --cameras')
    print(f'Compute device: {pipeline._select_device()}')

    with Path(args.ply).open('rb') as f:
        fingerprint = hashlib.file_digest(f, 'sha256').hexdigest()[:8]
    job_dir = str(Path(args.job_dir or f"out_{Path(args.ply).stem}_{fingerprint}{capture_suffix}").expanduser().resolve())
    cfg = PipelineConfig.from_overrides(
        quality=args.quality,
        navigation=not args.no_navigation, environment=not args.no_environment,
        score_threshold=args.score_threshold,
        min_votes=args.min_votes,
        min_peak_score=args.min_peak_score,
        camera_input=args.camera_input, camera_limit=args.camera_limit,
    )

    objects = pipeline.run_pipeline(args.ply, args.prompt, job_dir, cfg)

    out = Path(job_dir)
    roots = sum(not obj.get('parent_id') for obj in objects)
    print(f"\nDone — {roots} top-level objects; {len(objects) - roots} parts/contents.")
    print(f"  results: {out / 'interactions.json'}")
    print(f"  frames:  {out / 'frames'}")


if __name__ == "__main__":
    main()
