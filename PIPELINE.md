# Pipeline reference — what each module does

Use this when explaining the system in a demo, an interview, or a code
review. Every module has one job; the data flows top-to-bottom.

## End-to-end flow

```
video frame
   │
   ▼
 [1] video_source.VideoSource
   │
   ▼
 [2] detection_worker.DetectionWorker   ── async thread
   │   uses detector.YoloDetector
   ▼
 [3] detection_filter.DetectionFilter
   │
   ▼
 [4] tracker.CentroidTracker            ── stable IDs
   │
   ▼
 [5] motion.estimate_motion             ── velocity, predicted center
   │
   ▼
 [6] lane_model.LaneModel.assign_lane   ── which lane is this in?
   │
   ▼
 [7] scene_mapper.SceneMapper           ── lane + closeness → scene_x, scene_y + smoothing
   │
   ▼
 [8] scene_renderer.SceneRenderer       ── draw the road, ego, mini cars, planned path, trails
   │
   ▼
 [9] hud.*                              ── HUD overlays, side panels, action banner, debug panel
   │
   ▼
 displayed frame + (optional) recording
```

## Module-by-module

### 1. `video_source.py`
Opens a webcam index or a video file. Exposes `read()`, `fps`, width/height
through a uniform interface so `main.py` doesn't care which source it is.

### 2. `detection_worker.py` + `detector.py`
- **`detector.YoloDetector`** wraps Ultralytics YOLOv8. Filters predictions
  to road-relevant COCO classes (person, bicycle, car, motorcycle, bus,
  truck — plus stop sign / traffic light for the speed-limit hook).
  Auto-detects CUDA + FP16 if available.
- **`detection_worker.DetectionWorker`** is a background thread that pulls
  the most recent decoded frame, runs YOLO on a downscaled tensor, and
  writes detections into a thread-safe buffer. If inference is busy when a
  new frame arrives, the worker drops the request (and counts it). The
  display loop in `main.py` reads the latest detections non-blockingly,
  which is why display FPS stays decoupled from inference FPS.

### 3. `detection_filter.py`
Aggressive false-positive filter. Per-class confidence floors, minimum bbox
area / height, aspect-ratio sanity, road-paint rejection (long flat boxes
near the road surface), sky-band rejection, and a 3-frame temporal
consistency check. Without this filter, monocular dashcam footage produces a
lot of ghost "people" and "cars" stuck to lane paint or sun glare.

### 4. `tracker.py` + `geometry.py`
Hybrid centroid + IoU tracker keyed by class. For each new detection it
picks the cheapest matching active track (cost = weighted centroid distance
+ (1 − IoU)). Strong overlap can rescue a track through a centroid jump
(lane changes, jitter). Missing tracks decay over `MAX_MISSING_FRAMES` and
are dropped. Stronger than centroid-only; still not DeepSORT — dense crowds
can swap IDs.

### 5. `motion.py`
From the track's last few centers: velocity (px/sec), speed, bbox area
change rate, predicted center 1 s and 2 s ahead. Also provides
`shift_bbox_by_velocity` and `tracks_to_extrapolated_detections` for frames
where the async worker didn't fire.

### 6. `lane_model.py`
North-American multi-lane road geometry:

```
| sidewalk | oncoming_2 | oncoming_1 || ego_lane | right_lane | sidewalk |
                                       ▲▲▲▲▲▲▲▲▲▲
                                       double yellow on the LEFT of ego
```

The ego vehicle sits **in its own lane**, not on the canvas centerline. The
double-yellow divider sits to the **left** of the ego. Oncoming traffic is
drawn left of that divider.

`assign_lane(bbox, frame_w, frame_h, cls, velocity, track_id)` returns
`(lane_id, in_lane_offset)`. The logic is heuristic but matches what a
driver would intuit from a dashcam frame:

- Pedestrians + cyclists default to the **sidewalk** on whichever side their
  bbox suggests, unless they're clearly in the roadway (low closeness + over
  the curb). They migrate into the road only when motion + closeness say
  they've stepped into traffic — that's when their risk spikes.
- Vehicles get one of the carriageway lanes. Sign of lateral offset +
  closeness picks the lane. A sticky-history table keeps single-frame
  outliers from flipping a car from ego_lane into oncoming.

A `set_curve(strength)` knob is fed by `lane_estimator.py` so the whole
carriageway visibly bends with the real road.

### 7. `scene_mapper.py`
Where the lane-aware mapping actually happens. For each tracked detection:

1. `image_bbox_to_scene_plane(bbox)` returns `(lateral, closeness, area_norm)`
   where closeness is `0.65 * bottom_y_norm + 0.25 * height_norm + 0.10 * sqrt(area_norm)`.
2. `LaneModel.assign_lane(...)` gives `(lane_id, lane_offset)`.
3. `LaneModel.lane_to_scene_xy(lane_id, lane_offset, closeness)` gives
   `(scene_x, scene_y)`.
4. A per-track EMA on `(scene_x, scene_y, icon_w, icon_h)` kills jitter.
5. A trail buffer of the last ~14 positions per track feeds the breadcrumb
   rendering downstream.

**Why this matters:** if you map bbox-center-x directly to a fraction of
canvas width, every distant car drifts onto the canvas centerline because
the road *visually* converges there. By routing through a lane id first,
each car stays in its actual lane all the way to the horizon.

### 8. `scene_renderer.py`
Purely visual. Each frame it:

- Copies a cached `_bg_cache` (gradient + vignette baked once — a 30–50 ms/frame
  win) and draws the road poly, sidewalks, lane markings, perspective grid,
  and animated dashed lines.
- Draws the Tesla-blue planned-path ribbon out of the ego car, following the
  curved ego-lane centerline. Glow blends in an ROI, not full-canvas.
- Draws fading motion trails from the mapper's trail buffer.
- For each `SceneObject`, draws a mini 3D-ish silhouette: cars get a
  chamfered body + roof + windshield + 4 wheels + headlights or taillights
  depending on whether the lane is same-direction or oncoming; trucks get a
  cargo box + cab; buses get a long body with window strips; motorcycles get
  stacked wheels + rider; pedestrians get head/shoulders/torso/arms/legs;
  bicycles get two wheels + frame + rider.
- Draws the ego vehicle bottom-center with a faint headlight cone.

### 9. `hud.py`
All HUD overlays:

- **Scene mode**: top status bar (FPS or km/h, OBJECTS, PRIMARY, RISK, ACTION),
  side panels (AUTOPILOT, PRIMARY THREAT), speed-limit sign, action banner
  (only when ALERT/BRAKE), footer disclaimer, and an optional bottom-left
  debug telemetry panel.
- **Classic mode** (dashcam view): thin top strip + bbox overlays.
- **Debug mode**: vertical lane boundary lines + per-bbox mapping info
  (track id, class, normalized bbox center x, normalized bbox bottom y,
  assigned lane).
- **Composition**: `compose_split(left_frame, scene, W, H)` builds the
  side-by-side view.

### Supporting modules

- **`detection_filter.py`** — see above; lives between detector and tracker.
- **`speed_limit.py`** — manual `--speed-limit` flag for now, plus a
  stop-sign-aware HUD readout. A placeholder hook lets a future trained
  classifier replace `_classify_crop`.
- **`telemetry.py`** — EMA helpers for display FPS, detection FPS, YOLO
  latency, dropped-detection counter.
- **`utils.py`** — PALETTE, `risk_color_for`, color/math helpers, shared
  class-name sets.
- **`types.py`** — re-exports + new dataclasses (`LaneAssignment`,
  `SceneObject`, `SceneFrame`).
- **`manual_lanes.py`** — in-process helpers used by `main.py` to mutate
  `LANE_X_BOUNDARIES` via the `[`, `]`, `,`, `.` keys (and `P` to print).
- **`risk.py`** — legacy heuristic risk scoring that drives the GLOBAL RISK
  / ACTION strip on the HUD. The visualizer doesn't require it — if it
  imports cleanly it gets used, otherwise the HUD shows "--".

## Why async detection matters

Without an async worker the display loop runs at YOLO speed — maybe 3–5 FPS
on a CPU. With it, decode → render → present runs at whatever the monitor
can do (60 FPS by default, tunable with `--display-fps`), and YOLO catches
up asynchronously. The tracker keeps the boxes stable between fresh
detections, and `motion.tracks_to_extrapolated_detections` keeps positions
fresh on skip frames.

## In one sentence

> A real-time perception + tracking + lane-aware scene mapping pipeline with
> async YOLO detection, centroid/IoU tracking, and a Tesla-inspired 3D
> renderer — designed for smooth UI on CPU-first setups.
