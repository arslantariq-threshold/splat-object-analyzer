"""Public contracts; source geometry and suggested-camera spaces are explicit."""
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class ErrorDetail(Model):
    code: str
    message: str
    details: list[dict[str, str]] = Field(default_factory=list)


class ErrorResponse(Model):
    error: ErrorDetail


class Progress(Model):
    stage: str
    completed: int | None = None
    total: int | None = None


class Job(Model):
    id: str
    kind: Literal['analysis', 'query']
    analysis_id: str
    status: Literal['queued', 'running', 'succeeded', 'failed']
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    progress: Progress
    error: ErrorDetail | None = None
    status_url: str
    result_url: str


class Vector3(Model):
    x: float
    y: float
    z: float


class Quaternion(Model):
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1


class AxisAlignedBox(Model):
    type: str
    min: Vector3
    max: Vector3


class OrientedBox(Model):
    type: str
    position: Vector3
    scale: Vector3
    rotation: Quaternion


class Evidence(Model):
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    image_width: int | None = None
    image_height: int | None = None
    frame_index: int
    image_url: str
    detector_score: float | None = None
    mask_quality: float | None = None
    geometry_consistency: float | None = None
    evidence_type: str


class ObjectRecord(Model):
    kind: Literal['object', 'structure', 'area'] = 'object'
    description: str = ''
    geometry_basis: str | None = None
    surface_samples: list[Annotated[list[float], Field(min_length=3, max_length=3)]] = Field(default_factory=list, max_length=512)
    independent_position_count: int | None = None
    id: str
    label: str
    category: str
    display_name: str
    status: Literal['supported', 'uncertain', 'rejected', 'unreviewed']
    attributes: list[str] = Field(default_factory=list)
    confidence: float
    confidence_type: str = 'peak_detector_score_not_calibrated_probability'
    supporting_view_count: int
    geometry_supporting_view_count: int
    position: Vector3
    scale: Vector3
    bounding_box: AxisAlignedBox | None = None
    oriented_box: OrientedBox | None = None
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    relationship: str | None = None
    description_source: str
    visual_priority: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class SceneSummary(Model):
    type: str
    description: str
    confidence: str


class ObjectGroup(Model):
    id: str
    name: str
    description: str
    anchor_id: str
    anchor_category: str
    object_ids: list[str]
    confidence: str
    basis: str


class Relation(Model):
    subject: str
    relation: str
    object: str
    basis: str
    gap_source_units: float | None = None


class CoordinateSystem(Model):
    geometry_space: Literal['source'] = 'source'
    units: Literal['source_scene_units'] = 'source_scene_units'
    metric_scale_verified: bool = False
    quaternion_order: Literal['xyzw'] = 'xyzw'
    box_scale: str = 'Full extents along box-local axes; oriented_box.rotation maps box-local to source space.'
    suggested_camera_space: Literal['viewer'] = 'viewer'
    source_to_viewer: list[list[float]] = Field(default_factory=lambda: [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])


class Camera(Model):
    space: Literal['viewer'] = 'viewer'
    position: list[float] = Field(min_length=3, max_length=3)
    target: list[float] = Field(min_length=3, max_length=3)
    fov: float
    fov_axis: Literal['vertical'] = 'vertical'
    fov_unit: Literal['degrees'] = 'degrees'


class SuggestedView(Model):
    id: str
    title: str
    frame_index: int
    image_url: str
    object_ids: list[str]
    subjects: list[str]
    score: float
    camera: Camera
    basis: str


class Selection(Model):
    interpretation: str
    object_ids: list[str]
    group_ids: list[str]
    message: str
    match: dict[str, Any] | None = None


class Source(Model):
    filename: str
    format: Literal['ply', 'spz']
    bytes: int
    sha256: str
    download_url: str


Point3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class NavigationRegion(Model):
    id: int
    cell_count: int
    volume_source_units_cubed: float
    seed: Point3
    max_clearance: float
    bounds: dict[str, Point3]


class NavigationSummary(Model):
    schema_version: Literal['1.0'] = '1.0'
    algorithm: str
    fingerprint: str
    source_sha256: str
    status: Literal['candidate_space', 'uncertain']
    space: Literal['source'] = 'source'
    units: Literal['source_scene_units'] = 'source_scene_units'
    metric_scale_verified: Literal[False] = False
    origin: Point3
    shape: list[int] = Field(min_length=3, max_length=3)
    voxel_size: float = Field(gt=0)
    camera_radius: float = Field(gt=0)
    voxel_clearance_margin: float
    state_codes: dict[str, int]
    cell_counts: dict[str, int]
    regions: list[NavigationRegion]
    settings: dict[str, Any]
    diagnostics: dict[str, int]
    seconds: float
    limitations: list[str]
    grid_url: str | None = None


class NavigationValidationRequest(Model):
    space: Literal['source'] = 'source'
    points: list[Point3] = Field(min_length=1, max_length=256)


class NavigationGridResult(NavigationSummary):
    encoding: Literal['rle_xyz_c']
    runs: list[int]


class NavigationValidation(Model):
    valid: bool
    reason: str
    invalid_point_index: int | None = None
    invalid_segment_index: int | None = None
    basis: Literal['geometry_candidate_map'] = 'geometry_candidate_map'


class NavigationPathRequest(Model):
    space: Literal['source'] = 'source'
    start: Point3
    goal: Point3


class NavigationPath(Model):
    found: bool
    reason: str
    points: list[Point3]
    basis: Literal['geometry_candidate_map'] = 'geometry_candidate_map'


class CameraSampling(Model):
    format: Literal['colmap_text', 'colmap_binary', 'camera_json']
    input_pose_count: int
    candidate_position_count: int
    selected_position_count: int
    selected_image_ids: list[int | str]
    orientation: Literal['perspective_panorama_source_negative_y_up']


class EnvironmentIssue(Model):
    frame_index: int
    image_url: str
    reason: str
    name: str | None = None
    kind: Literal['structure', 'area'] | None = None


class EnvironmentSummary(Model):
    version: int
    fingerprint: str
    status: Literal['ready', 'partial']
    entity_ids: list[str]
    analyzed_view_count: int
    evidence_frames: list[int]
    failed_views: list[EnvironmentIssue]
    ungrounded: list[EnvironmentIssue]
    limitations: list[str]


class AnalysisResult(Model):
    schema_version: Literal['1.0'] = '1.0'
    analysis_id: str
    quality: Literal['low', 'medium', 'high']
    source: Source
    coordinates: CoordinateSystem = Field(default_factory=CoordinateSystem)
    rendered_view_count: int
    understanding_included: bool
    camera_sampling: CameraSampling | None = None
    navigation: NavigationSummary | None = None
    scene: SceneSummary | None = None
    environment: EnvironmentSummary | None = None
    objects: list[ObjectRecord]
    groups: list[ObjectGroup] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    hero_object_ids: list[str] = Field(default_factory=list)
    default_selection: Selection | None = None
    suggested_views: list[SuggestedView] = Field(default_factory=list)
    limitations: list[str]


class QueryRequest(Model):
    prompt: str = Field(min_length=1, max_length=1000)


class QueryResult(Model):
    schema_version: Literal['1.0'] = '1.0'
    query_id: str
    analysis_id: str
    prompt: str
    coordinates: CoordinateSystem = Field(default_factory=CoordinateSystem)
    selection: Selection
    suggested_views: list[SuggestedView]
    unshown_object_ids: list[str]
