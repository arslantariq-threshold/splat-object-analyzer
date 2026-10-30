"""
Shared pipeline configuration — the single source of truth for defaults used by
both the local CLI (run_local.py / pipeline.py) and the FastAPI server (server.py).

Keeping every default here prevents the kind of drift that crept in before
(e.g. n_positions defaulting to 10 in one place and 5 in another).

The user-facing knob is `quality` (low/medium/high), which expands to the three
internal camera counts via `apply_quality()`. The granular counts remain on the
dataclass for internal use but are not exposed in the UI/API.
Optional capture inputs can override the selected-location budget via camera_limit.
"""

from dataclasses import dataclass, fields


# quality name -> (n_positions, n_azimuth, n_elevation)  → total frames
QUALITY_PRESETS = {
    "low":    (3, 4, 2),   # 24 frames  — fast preview
    "medium": (5, 6, 3),   # 90 frames  — balanced default
    "high":   (8, 8, 3),   # 192 frames — thorough coverage
}
DEFAULT_QUALITY = "medium"


@dataclass
class PipelineConfig:
    # ── rendering ──────────────────────────────────────────────────────────
    width: int = 512
    height: int = 512
    renderer: str = "auto"                  # "auto" | "gsplat" (CUDA) | "gsplat-metal" (Apple MPS)
    quality: str = DEFAULT_QUALITY          # drives the camera counts below
    n_positions: int = 5
    n_azimuth: int = 6
    n_elevation: int = 3
    fov_x_degrees: float = 90.0
    camera_input: str | None = None         # optional COLMAP model or source-space JSON
    camera_limit: int | None = None         # provided-camera locations only; quality default otherwise

    # ── automatic camera placement / preview validation ───────────────────
    bbox_pct_lo: float = 1.0                # robust bounding-box lower percentile
    bbox_pct_hi: float = 99.0               # robust bounding-box upper percentile
    min_sep_frac: float = 0.04              # min camera separation (fraction of bbox diagonal)
    camera_clearance_frac: float = 0.12     # clearance relative to shortest robust bbox side
    camera_candidates: int = 64
    preview_size: int = 128
    seed: int = 42                          # deterministic placement

    # Geometry-only navigation estimate; independent of view-count quality.
    navigation: bool = True
    navigation_resolution: int = 128

    # Environment discovery uses selected RGB views, then mask/depth grounding.
    environment: bool = True
    environment_views: int = 12

    # ── detection / clustering ─────────────────────────────────────────────
    score_threshold: float = 0.12
    min_votes: int = 8
    min_peak_score: float = 0.40

    def apply_quality(self):
        """Expand the `quality` preset into the three camera-count fields."""
        if self.quality in QUALITY_PRESETS:
            self.n_positions, self.n_azimuth, self.n_elevation = QUALITY_PRESETS[self.quality]
        return self

    @classmethod
    def from_overrides(cls, **kw):
        """Build a config from a loose dict of overrides (ignores unknown/None keys)."""
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kw.items() if k in valid and v is not None}
        return cls(**filtered).apply_quality()
