# Tesla-Inspired Driving Scene Visualizer

> **Not real Tesla Vision.** This is an educational/portfolio project that
> reconstructs a Tesla-style driving scene from monocular dashcam video using
> YOLO object detection, simple object tracking, and heuristic lane-aware
> scene mapping. It does **not** perform vehicle control, navigation, or any
> certified safety function.

A real-time driving visualization system that uses dashcam or webcam video to
detect and track nearby vehicles, pedestrians, and road objects, then
reconstructs them into a Tesla-inspired 3D driving interface with lane-aware
object placement, HUD telemetry, and smooth real-time rendering.

---

## What it does

- **Detects** road users (person, bicycle, car, motorcycle, bus, truck, plus
  traffic light / stop sign hooks) with Ultralytics YOLOv8.
- **Tracks** them between frames with a centroid + IoU hybrid tracker so each
  object gets a stable ID.
- **Assigns** each tracked object to a lane (`far_left`, `left_lane`,
  `ego_lane`, `right_lane`, `far_right`) using a manual lane-boundary model
  so cars don't all cluster on the canvas centerline.
- **Maps** lane + perspective-depth into a Tesla-style 3D-ish scene:
  curved road, ego car bottom-center, mini cars/trucks/buses/people in their
  proper lanes, planned-path ribbon, motion trails, HUD.
- Decouples rendering from inference with an **async YOLO worker** so display
  FPS stays high even when YOLO is slow.

---

## Architecture pipeline

```
video frame
  -> YOLO detection (async, throttled by performance preset)
  -> detection filter (drops low-conf / tiny / road-paint / flickering boxes)
  -> centroid+IoU tracker (stable IDs, missing-frame handling)
  -> motion estimator (velocity, predicted center)
  -> lane assignment (LaneModel) ──┐
  -> closeness estimate            ├── scene_mapper.SceneMapper
  -> smoothed scene_x / scene_y ────┘
  -> scene_renderer.SceneRenderer
  -> hud overlays
  -> final frame -> window + optional recording
```

A more detailed walkthrough lives in [`PIPELINE.md`](PIPELINE.md).

---

## Install

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `ultralytics`, `opencv-python`, `numpy`, and
`PyQt6`. PyQt6 is **only** required for `--renderer qt`; if you skip it the
OpenCV renderer still works. YOLOv8n weights download automatically on
first run.

## Renderer options

The visualizer has **two** swappable frontends — the OpenCV one is the
stable default; the Qt one uses QPainter for smoother gradients,
anti-aliased polygons, and cleaner HUD typography.

```bash
# Stable OpenCV renderer (default — works without PyQt6)
python main.py --renderer opencv

# Qt / QPainter renderer (smoother visuals; needs `pip install PyQt6`)
python main.py --renderer qt
```

| Renderer | Pros | Cons |
| -------- | ---- | ---- |
| `opencv` | No extra deps, all four views (scene/dashcam/split/debug) | Slightly chunkier polygon edges |
| `qt`     | Anti-aliased QPainter polygons, gradient bodies, smoother HUD typography, dedicated 60 Hz QTimer | Scene-only view today (dashcam/debug fall back to a scene preview with a tag) |

Both renderers share the **same backend**: same YOLO detector, same async
detection worker, same tracker, same lane assignment, same scene mapper. The
backend writes a `SceneSnapshot` and either frontend draws from it. If
PyQt6 isn't installed, `--renderer qt` prints a hint and falls back to
OpenCV automatically.

---

## Run

```bash
# Defaults: scene view on data/dashcam.mp4 with the balanced preset
python main.py

# Pick a video / webcam / performance / view
python main.py --source data/dashcam.mp4 --view scene --performance fast
python main.py --source 0
python main.py --view split
python main.py --view debug

# Manual readouts on the HUD
python main.py --ego-speed-kmh 50 --speed-limit 60
```

If you omit `--source`, the app picks `data/dashcam.mp4` (a clean 720p/60fps
Toronto dashcam clip we ship as the default), then any other `*.mp4` in
`data/`, then webcam index 0.

### Performance presets

| Preset      | Model       | imgsz | Detect every N frames | Notes                       |
| ----------- | ----------- | ----- | --------------------- | --------------------------- |
| `fast`      | yolov8n.pt  | 416   | 3                     | Integrated graphics / CPU   |
| `balanced`  | yolov8n.pt  | 640   | 2                     | Default                     |
| `quality`   | yolov8s.pt  | 640   | 1                     | Slow on CPU; great on CUDA  |

CUDA is auto-detected — if `torch.cuda.is_available()`, the detector switches
to FP16 on the first device automatically.

---

## Controls

| Key       | Action                                                                  |
| --------- | ----------------------------------------------------------------------- |
| `Q`/`Esc` | Quit                                                                    |
| `V`       | Cycle view: scene → dashcam → split → debug → scene                     |
| `D`       | Toggle the debug telemetry panel inside scene mode                      |
| `R`       | Start/stop recording → `outputs/demo.mp4`                               |
| `F`       | Print detection-filter stats (kept vs dropped buckets)                  |
| `P`       | Print current lane boundaries (ready to paste into `config.py`)         |
| `[`       | Shift all inner lane boundaries left                                    |
| `]`       | Shift all inner lane boundaries right                                   |
| `,`       | Narrow the ego lane                                                     |
| `.`       | Widen the ego lane                                                      |

---

## Views

- **scene** *(default)* — full Tesla-inspired scene reconstruction. Curved
  road, ego car bottom-center, planned-path ribbon, mini 3D-ish vehicles in
  their proper lanes, motion trails, HUD.
- **dashcam** — raw dashcam frame with bounding boxes + classic top HUD.
- **split** — dashcam on the left (no boxes, so the eye reads "what the
  camera sees" vs "what the system reconstructs"), scene on the right.
- **debug** — raw dashcam + vertical lane boundary lines + per-bbox mapping
  info (track id, class, normalized bbox center x, normalized bbox bottom y,
  assigned lane). Use the `[`, `]`, `,`, `.` keys to tune the boundaries
  live, then press `P` to print them.

---

## Project layout

```
.
├── main.py                      # CLI + main loop + keyboard handling
├── requirements.txt
├── README.md
├── PIPELINE.md
├── run.bat                      # Windows convenience launcher
├── data/                        # Drop dashcam clips here (auto-picked)
├── outputs/                     # Recordings + smoke-test images
└── driving_scene/
    ├── __init__.py
    ├── config.py                # All tunables (thresholds, presets, lanes)
    ├── types.py                 # Shared dataclasses (Detection, Track, ...)
    ├── utils.py                 # PALETTE, color/math helpers
    ├── video_source.py          # File or webcam capture
    ├── detector.py              # YOLO wrapper (Ultralytics)
    ├── detection_worker.py      # Async YOLO thread + shared state
    ├── detection_filter.py      # Drops ghost / paint / low-conf detections
    ├── tracker.py               # Centroid + IoU hybrid tracker
    ├── motion.py                # Velocity + predicted-center estimator
    ├── lane_model.py            # Multi-lane road geometry + assign_lane
    ├── lane_estimator.py        # Optional curved-lane estimator (Canny+Hough)
    ├── manual_lanes.py          # In-process helpers for lane keyboard tuning
    ├── scene_mapper.py          # bbox -> lane -> scene_xy + smoothing
    ├── scene_renderer.py        # OpenCV visual layer
    ├── hud.py                   # OpenCV HUD / overlay drawing
    ├── qt_renderer.py           # PyQt6 / QPainter SceneCanvas (--renderer qt)
    ├── qt_app.py                # PyQt6 main window + backend thread bridge
    ├── speed_limit.py           # Manual + stop-sign-aware speed-limit
    ├── telemetry.py             # FPS / latency EMA helpers
    ├── risk.py                  # Heuristic risk scoring (legacy hook)
    ├── geometry.py              # IoU helper used by tracker
    └── ui.py                    # Legacy classic-mode overlay helpers
```

---

## How lane mapping works (and why it matters)

Mapping bbox-center-x **directly** to scene_x is the classic mistake — it
makes every car drift toward the canvas centerline as the perspective
narrows. Instead, this project does:

```
bbox center x
  -> LaneModel.assign_lane → lane id + in-lane offset
  -> closeness estimate (bbox bottom y + height + sqrt area)
  -> LaneModel.lane_to_scene_xy(lane_id, offset, closeness) → (scene_x, scene_y)
  -> per-track EMA on (scene_x, scene_y, icon_w, icon_h)
```

The result: same-direction cars stay on the right side of the road, oncoming
cars stay on the left, the ego car always sits in the ego lane, and far
objects fade naturally toward the horizon. Lane boundaries are tunable live
with `[`, `]`, `,`, `.` and snapshottable with `P`.

---

## Limitations

- **Monocular video.** No true metric depth — distances and TTC are
  approximations from bbox geometry.
- **No map / GPS / radar / LiDAR.** Lane curvature comes from a Canny+Hough
  heuristic; speed limit is manual or stop-sign-triggered.
- **Tracking is heuristic.** Not DeepSORT — ID swaps can happen in dense
  crowds or heavy occlusion.
- **Not certified safety software.** This is an educational visualization
  designed to demonstrate object detection, tracking, lane-aware mapping,
  and UI engineering concepts.
- **Performance is hardware-dependent.** On a laptop CPU expect ~3–5 FPS
  detection but 30–60 FPS display thanks to the async worker. On CUDA the
  `quality` preset opens up.

---

## Custom visual assets (optional)

The scene renderer ships with **procedural mini-3D icons** — rounded toy
cars with shaded bodies, dark windshields, soft shadows, and stable
per-track colors. No external assets are required; the project looks
polished out of the box.

If you'd like to swap in your own art (better looking, brand-consistent,
class-specific), drop transparent PNGs in:

```
assets/
  vehicles/
    car_white.png      car_silver.png    car_blue.png
    car_green.png      car_cream.png     car_lavender.png
    truck_white.png    bus_white.png     ...
  people/
    person.png
  bikes/
    bicycle.png        motorcycle.png
```

The asset loader (`driving_scene/assets.py`) picks them up automatically.
Naming convention: `{class}_{color}.png` where `{color}` is one of the
`VEHICLE_COLOR_PALETTE` slot names (`white`, `silver`, `blue`, `green`,
`cream`, `lavender`). If a file is missing, the renderer silently falls
back to the procedural icon — nothing breaks.

**Important:**

- The project does **not** ship any copyrighted assets — no Tesla
  Vision art, no Shutterstock watermarked images, no other licensed
  content. Default visuals are 100% procedural.
- Only use **royalty-free or self-created** PNGs in `assets/`.
- Don't redistribute downloaded copyrighted images bundled with this
  repo.

## Screenshots

*(Drop your own captures here — `outputs/scene_smoke.png` and
`outputs/scene_split.png` are auto-generated by `tools/smoke_test_scene.py`
and make good baselines.)*

---

## Future improvements

- Trained speed-limit OCR (GTSRB / LISA) to replace the stop-sign-only
  scaffold in `speed_limit.py`.
- DeepSORT or ByteTrack to replace the centroid tracker for heavy traffic.
- ONNX export of YOLO for cross-platform GPU inference.
- Real 3D vehicle meshes via OpenCV's wrapper for OpenGL or a tiny
  game-engine backend instead of 2D polygons.
- Optional CARLA simulation as a reproducible input source.

---

## Resume bullet

> Built a **Tesla-inspired real-time driving scene visualizer** in Python
> using **OpenCV** and **YOLOv8**, reconstructing dashcam detections into a
> **lane-aware 3D interface** with **async inference**, **object tracking**,
> **smooth rendering**, and **HUD telemetry**.
