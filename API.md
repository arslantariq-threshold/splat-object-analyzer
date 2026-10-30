# Splat Object Analyzer API v1

The API runs the existing pipeline without serving the viewer. Upload one Gaussian
splat and receive an asynchronous job. By default the job includes automatic
object discovery, segmentation, positions, boxes, richer descriptions, functional
groups, scene context, hero subjects, relationships, and suggested still views.
It also estimates camera space from Gaussian geometry without original camera poses.

Interactive documentation: `/docs`. Machine-readable contract: `/openapi.json`.
A generated copy is checked in as `openapi.json` for frontend/client development.
All data endpoints require `Authorization: Bearer YOUR_API_KEY`.

## Start locally

After the normal GPU setup (`./setup.sh mac` or `./setup.sh cuda`), install the
API dependencies into the same environment:

```bash
uv pip install --python .venv/bin/python -r requirements-api.txt
export SPLAT_API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
./serve-api.sh
```

The default address is `http://127.0.0.1:8000`. Keep the generated key for client
requests. `SPLAT_DATA_DIR` defaults to `outputs/api/`. For GCP and HTTPS, see
[the deployment guide](deploy/README.md).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/analyses?filename=room.spz` | Upload a file and queue the full analysis; returns 202 |
| GET | `/v1/jobs/{job_id}` | Poll either an analysis or a prompt-query job |
| GET | `/v1/analyses/{analysis_id}` | Fetch the structured analysis result |
| GET | `/v1/analyses/{analysis_id}/navigation` | Camera-space metadata, connected regions, and automatic starting points |
| GET | `/v1/analyses/{analysis_id}/navigation/grid` | Full source-space candidate grid as RLE JSON |
| POST | `/v1/analyses/{analysis_id}/navigation/validate` | Validate points and every segment between them |
| POST | `/v1/analyses/{analysis_id}/navigation/path` | Find a bounded, collision-checked grid route |
| POST | `/v1/analyses/{analysis_id}/queries` | Queue a prompt such as “show off my desk setup”; returns 202 |
| GET | `/v1/queries/{query_id}` | Fetch selected object/group IDs and suggested views |
| GET | `/v1/analyses/{analysis_id}/frames/{frame_index}` | Fetch a supporting rendered PNG |
| GET | `/v1/analyses/{analysis_id}/source` | Download the original uploaded file |
| GET | `/health/live` | Public HTTP liveness check |
| GET | `/health/ready` | Public queue-worker readiness check |

Analyses and prompt queries share one persistent FIFO queue and one GPU worker.
Do not run multiple Uvicorn workers. The service rejects a second process using
the same data directory. Separate data directories are not a GPU isolation mechanism.

## Upload, poll, fetch

The existing splat-only upload body is the file's raw bytes. Supported
formats are standard 3DGS PLY and SPZ v1–v3. The default upload limit is 2 GiB.
`quality=high` is the default; `low` and `medium` are also accepted. Set
`understanding=false` to omit the slower description/grouping stage and return
the object inventory and geometry. Such an analysis cannot accept prompt queries.
Navigation is independent of semantic enrichment and enabled by default; set
`navigation=false` to omit it. Quality controls image coverage, not grid resolution.

```bash
export API_URL=http://127.0.0.1:8000

curl --fail-with-body "$API_URL/v1/analyses?filename=room.spz&quality=high&understanding=true" \
  -H "Authorization: Bearer $SPLAT_API_KEY" \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @/path/to/room.spz
```

The 202 response is a `Job`. The analysis ID is the job's `id`. Both `status_url`
and `result_url` are relative to your API base URL. `Location` points to the status
endpoint and `Retry-After` suggests the initial polling interval in seconds.

```json
{
  "id": "11111111111111111111111111111111",
  "kind": "analysis",
  "analysis_id": "11111111111111111111111111111111",
  "status": "queued",
  "created_at": "2026-10-28T00:00:00.000Z",
  "started_at": null,
  "finished_at": null,
  "progress": {"stage": "Queued", "completed": null, "total": null},
  "error": null,
  "status_url": "/v1/jobs/11111111111111111111111111111111",
  "result_url": "/v1/analyses/11111111111111111111111111111111"
}
```

Use the returned ID in subsequent calls:

```bash
curl --fail-with-body "$API_URL/v1/jobs/$ANALYSIS_ID" \
  -H "Authorization: Bearer $SPLAT_API_KEY"

curl --fail-with-body "$API_URL/v1/analyses/$ANALYSIS_ID" \
  -H "Authorization: Bearer $SPLAT_API_KEY"
```

Job states are `queued`, `running`, `succeeded`, and `failed`. Stop polling at a
terminal state. Progress counters apply to the current stage, not to an overall
percentage. A result request before success returns 409. A failed job includes a
structured error; operator diagnostics are in that job's `worker.log` on disk.

### Optional capture cameras

The same endpoint also accepts `multipart/form-data`. Keep `filename`, `quality`,
`understanding` and `navigation` in the query string. Send these file parts:

| Part | Contents |
| --- | --- |
| `file` | Required PLY or SPZ splat |
| `colmap_images` | Optional matching COLMAP `images.txt` or `images.bin` |
| `colmap_cameras` | Optional `cameras.txt`; requires `colmap_images` |
| `cameras` | Alternative source-space camera JSON; mutually exclusive with COLMAP parts |

```bash
curl --fail-with-body "$API_URL/v1/analyses?filename=scene.ply&quality=high&camera_limit=8" \
  -H "Authorization: Bearer $SPLAT_API_KEY" \
  -F 'file=@/path/to/scene.ply' \
  -F 'colmap_images=@/path/to/model/images.txt' \
  -F 'colmap_cameras=@/path/to/model/cameras.txt'
```

Camera JSON uses the original splat's **source** coordinates, before viewer
rotation, translation or scale:

```json
{"space":"source","positions":[[1.2,-0.8,2.1],[2.4,-0.8,1.7]]}
```

Alternatively use a `poses` array with `id`, `camera_id`, `name`, and `position`
on each entry. Original photos and `points3D.txt` are not required. COLMAP image
poses are inverted to obtain their camera centres; `cameras.txt` alone contains
no capture locations.

Only supplied locations are used when attached. Nearby centres (including
front/back lens pairs) are reduced through spatial sampling. At most 64 default
candidate locations receive low-resolution previews; useful surface coverage and
image quality decide the final subset. `camera_limit` optionally sets the maximum
selected locations, from 1 to 64; otherwise quality selects 3/5/8 for low/medium/high.
The existing panoramic sampling then attempts 8/18/24 perspective views per
selected location; unusable final views are discarded. `camera_sampling` reports
input, candidate and selected location counts and selected source image IDs.
Detailed measurements remain in `camera_diagnostics.json`.

The renderer generates fresh perspective orientations using the existing source
negative-Y up convention, rather than reproducing original fisheye projection or
capture roll. Cameras and the splat must share a coordinate frame. If no supplied
locations pass validation, the job fails with diagnostics rather than silently
switching to generated locations. Omitting cameras keeps automatic placement
and the raw-byte upload contract. Navigation remains geometry-based.

The combined camera-file limit is 512 MiB, configurable with
`SPLAT_MAX_CAMERA_UPLOAD_BYTES`; the splat has its own unchanged upload limit.

### Result fields

The result includes:

| Field | Contents |
| --- | --- |
| `schema_version` | Public result format, currently `1.0` |
| `source` | Original filename, format, bytes, SHA-256, download URL |
| `coordinates` | Geometry/camera spaces, units, quaternion order, transform |
| `rendered_view_count` | Number of accepted rendered analysis views |
| `camera_sampling` | Optional capture-camera source and sampling counts; null for automatic placement |
| `understanding_included` | Whether semantic enrichment was requested and completed |
| `navigation` | Candidate camera-space metadata and grid URL; null when omitted or absent from an older result |
| `scene` | Inferred space type, description, confidence; null without enrichment |
| `objects` | IDs, labels, categories, descriptions, review status, geometry, confidence, parent/child links, evidence URLs |
| `groups` | Functional areas such as a desk setup, with member object IDs |
| `relations` | Geometry-derived part/content and nearby relationships |
| `hero_object_ids` | Suggested primary subjects for a scene tour |
| `default_selection`, `suggested_views` | Default tour subjects and up to three saved camera views |
| `limitations` | Relevant model/geometry limitations |

`objects[].status` can be `supported`, `uncertain`, `rejected`, or `unreviewed`
(enrichment omitted). Rejected records are retained for inspection; clients can
filter them. Confidence is a detector score, not a calibrated probability.

## Prompt queries

```bash
curl --fail-with-body "$API_URL/v1/analyses/$ANALYSIS_ID/queries" \
  -H "Authorization: Bearer $SPLAT_API_KEY" \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"Show off my desk setup"}'
```

Poll the returned job's `status_url`, then fetch its `result_url`. The result
contains the original prompt, `selection.object_ids`, `selection.group_ids`, a
message, suggested views with image URLs/cameras, and `unshown_object_ids`.
Resolve those IDs against the analysis result. A no-match response succeeds with
empty subject and view lists. Prompt matching reuses the saved scene index;
it does not rerun detection or object descriptions.

All image and source URLs also require the bearer key. An HTML `<img>` cannot
attach that header automatically: proxy the image through your own application
backend, or fetch it with authorization and display a Blob URL.

## Coordinates

Object `position`, `scale`, `bounding_box`, and `oriented_box` remain in the
original **source** space. Units are source-scene units; metric scale is not
verified. Top-level `position`/`scale` describe the axis-aligned estimate. An
`oriented_box` has its own center, full local-axis extents, and XYZW quaternion.
Use those fields together when drawing a rotated box.

Suggested cameras are in **viewer** space, with vertical field of view in degrees.
The supplied 4×4 `source_to_viewer` matrix maps `[x,y,z]` to `[x,-y,-z]`. To use
these cameras, apply that transform to the splat and its annotations, composing
the box quaternion with the same rotation. Do not transform the cameras a second
time. Alternatively transform the cameras back into source space for your renderer.

Suggested views are existing analysis renders with depth visibility checks.
They are not camera animations or collision-checked paths.

## Candidate camera space

Navigation uses the **source** coordinate space, including its original rotation
and scale. No gravity direction, metric scale, original camera poses, or manually
selected starting point is required. `navigation.regions[].seed` is an automatic
starting position for each connected region. To place it in the prototype viewer,
apply the documented source-to-viewer transform once.

The map voxelizes a conservative superset of cutoff Gaussian ellipsoids, flood
excludes camera clearance margins, flood fills from the outer grid boundary, and
rejects small enclosed cavities. The fill operates in camera configuration space:
cracks smaller than the camera are blocked by clearance without adding obstacle
geometry. Outside flood fill uses 26 neighbors so diagonal openings that fit the
camera remain open. Paths use face-connected neighbors and continuous supercover checks
to prevent corner cutting. Tiny islands left after clearance are excluded.
Very large primitives use a conservative AABB fallback to bound work; the count
is included in diagnostics. No surfaces are reconstructed across scan holes.

`status=candidate_space` means at least one region passed these geometric checks.
It does **not** certify that a region is a room or that unrecorded obstacles are
absent. `status=uncertain` is a successful analysis with no approved camera region;
clients must not treat its unknown cells as free. A missing wall or roof may
produce this result. Clearances are source units, not meters. The default radius
is one voxel; `voxel_clearance_margin` accounts for both obstacle and camera-cell
discretization. The default grid targets 128 cells along the longest robust scene
side and is capped at two million cells. Resolution and conservative Gaussian
cutoffs can close very small openings; this is an estimate at the reported scale.

Fetch `navigation.grid_url` with the same bearer header as other assets. The JSON
contains metadata plus `encoding="rle_xyz_c"` and `runs=[state,count,state,count,...]`.
Expand the runs to `shape[0]*shape[1]*shape[2]` uint8 cells. The flat index for
`[i,j,k]` is `(i*shape[1]+j)*shape[2]+k`; z varies fastest. A cell center is
`origin + ([i,j,k]+0.5)*voxel_size`. The state codes are:

| Code | Meaning | Traversable |
| --- | --- | --- |
| 0 | Unknown, exterior, or rejected enclosed cavity | No |
| 1 | Modeled obstacle | No |
| 2 | Candidate camera space for the reported radius | Yes, relative to this map |
| 3 | Excluded clearance margins or isolated clearance pockets | No |

Position/path requests use arrays of three finite source coordinates:

```bash
curl --fail-with-body "$API_URL/v1/analyses/$ANALYSIS_ID/navigation/validate" \
  -H "Authorization: Bearer $SPLAT_API_KEY" -H 'Content-Type: application/json' \
  --data '{"space":"source","points":[[0,0,0],[1,0,0]]}'

curl --fail-with-body "$API_URL/v1/analyses/$ANALYSIS_ID/navigation/path" \
  -H "Authorization: Bearer $SPLAT_API_KEY" -H 'Content-Type: application/json' \
  --data '{"space":"source","start":[0,0,0],"goal":[1,0,0]}'
```

Replace the example points with source coordinates in your map. Validation
returns `valid`, `reason`, and an offending point or segment index when applicable.
A path response returns `found`, `reason`, `points`, and
`basis="geometry_candidate_map"`. A blocked/unreachable request returns 200 with
`valid=false` or `found=false`; it never substitutes an unchecked straight path.
Search stops after 50,000 expansions and reports `search_limit_reached` separately
from disconnected regions. These endpoints return 404 when navigation was omitted
or is absent from an older analysis. Existing completed analyses are not migrated.
Camera orientation, framing, render quality, animation and smoothed trajectories
are separate concerns; suggested still views have not been constrained to this map.

## Operations and errors

Jobs/results and SQLite queue state survive process/container restarts when the
data volume persists. Interrupted jobs are queued again. Completed geometry is
reused when only semantic enrichment was interrupted; saved caption batches are
also reused. An interrupted earlier analysis stage is rerun. Completed results
are retained until an operator removes them; this version has no retention or
delete endpoint. Monitor disk space on the VM.

Errors use `{"error":{"code":"...","message":"...","details":[]}}`.
401 means missing/invalid key; 400/415/422 mean invalid input; 413 means file too
large; 429 means the bounded queue is full; 404 means missing resource; 409 means
the result is not ready or semantic enrichment was omitted. Retry queued work by
polling, rather than uploading the same file repeatedly.

The API key identifies one trusted deployment, not separate end users. Keep it in
your application backend; add your own user authentication/authorization there.
`SPLAT_CORS_ORIGINS` can allow explicit browser origins for development. It is
empty by default. Model inference stays on the host; the first job downloads
the public pretrained weights into the persistent cache.

Run API lifecycle checks without downloading models:

```bash
uv pip install --python .venv/bin/python -r requirements-api-test.txt
.venv/bin/python check-api.py
.venv/bin/python check-navigation.py
node check-navigation.mjs
```

### Environment subjects

With `understanding=true` (the default), analysis also surveys the rendered views
for structures and areas. The existing `objects` list now includes all grounded
subjects, identified by `kind: "object" | "structure" | "area"`. Existing object
IDs and geometry remain unchanged. Environment IDs use `environment_####`.

Environment records include `description`, `attributes`, `geometry_basis`,
`surface_samples` (at most 512 source-space points), and
`independent_position_count`. Their `confidence_type` describes mean mask quality,
not semantic correctness. A subject observed at only one camera location remains
`uncertain`. Studio keeps those observations available for inspection and excludes
them from automatic tour targets. The later semantic stage may refine supported
subject names from masked views or mark unclear descriptions uncertain. Bounds measure visible surfaces, including partially captured façades.
They do not assert a complete building or free interior volume.

`environment` is null for older/object-only results, or includes the survey
version/fingerprint, `entity_ids`, analyzed view count, evidence frame IDs,
`failed_views`, `ungrounded` observations and limitations. Excluded observations
have no 3D identity and cannot be selected for a camera shot. Issue/image URLs use
the same authenticated frame endpoint. Evidence records may include a pixel-space
`box`, `image_width`, and `image_height` for inspection overlays.

`relations` includes observed-surface structural part links and geometric
structure/area adjacency. `suggested_views` also includes an observed camera for
each grounded environment entity. Query selection and depth-visibility checks
include environment surfaces. Navigation constraints remain independent.

The additional fields are additive to schema 1.0; clients that render all subjects
should display `kind`, distinguish uncertain observations, and avoid interpreting
area surfaces as permitted camera space. `understanding=false` retains object-only
processing. The first enabled analysis also requires the pinned local Qwen model.
