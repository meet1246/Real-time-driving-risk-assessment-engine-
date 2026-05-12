"""
main.py — Tesla-Inspired Driving Scene Visualizer

Entry point that wires the driving_scene modules into the rendering loop.
Uses an async YOLO worker so display FPS is decoupled from inference FPS.

Views (cycle live with V, or pick at launch with --view):
  scene    full Tesla-inspired scene reconstruction (default)
  dashcam  raw dashcam + bounding boxes + classic HUD
  split    dashcam (no boxes) | scene
  debug    raw dashcam + lane boundary lines + per-bbox mapping info

Hotkeys:
  Q / Esc    quit
  V          cycle views
  D          toggle debug overlay panel inside scene mode
  R          start/stop recording -> outputs/demo.mp4
  F          print detection-filter stats
  P          print current lane boundaries (for pasting into config.py)
  [          shift inner lane boundaries left
  ]          shift inner lane boundaries right
  ,          narrow ego lane
  .          widen ego lane

This is NOT real Tesla Vision — it is an educational visualization built on
top of monocular dashcam video.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2


def _try_import(modname: str):
    try:
        return __import__(f"driving_scene.{modname}", fromlist=["*"])
    except Exception as e:  # noqa: BLE001
        print(f"[main] note: driving_scene.{modname} not available ({e})")
        return None


config_mod = _try_import("config")
video_source_mod = _try_import("video_source")
detector_mod = _try_import("detector")
worker_mod = _try_import("detection_worker")
tracker_mod = _try_import("tracker")
motion_mod = _try_import("motion")
risk_mod = _try_import("risk")
telemetry_mod = _try_import("telemetry")
scene_mod = _try_import("scene_renderer")
speed_limit_mod = _try_import("speed_limit")
filter_mod = _try_import("detection_filter")
lane_estimator_mod = _try_import("lane_estimator")
manual_lanes_mod = _try_import("manual_lanes")

if scene_mod is None:
    print("[main] FATAL: driving_scene/scene_renderer.py is missing.")
    sys.exit(1)


class _Telemetry:
    def __init__(self):
        self.display_fps = 0.0
        self.detection_fps = 0.0
        self.yolo_latency_ms = 0.0
        self.track_latency_ms = 0.0
        self.risk_latency_ms = 0.0
        self.dropped_detections = 0
        self._last_t = time.time()
        self._ema_alpha = 0.15

    def tick_display(self):
        now = time.time()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now
        inst = 1.0 / dt
        self.display_fps = (1 - self._ema_alpha) * self.display_fps + self._ema_alpha * inst


class _RiskState:
    def __init__(self):
        self.global_score = 0.0
        self.global_level = "LOW"
        self.global_action = "SAFE"
        self.primary_track_id: Optional[int] = None


def parse_args():
    p = argparse.ArgumentParser(description="Tesla-Inspired Driving Scene Visualizer")
    p.add_argument("--source", default=None, help="Video file path or webcam index (e.g. 0).")
    p.add_argument(
        "--view",
        default="scene",
        choices=["scene", "dashcam", "split", "debug"],
        help="scene (default) | dashcam | split | debug",
    )
    p.add_argument("--performance", default="balanced", choices=["fast", "balanced", "quality"])
    p.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold.")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size.")
    p.add_argument("--ego-speed-kmh", type=float, default=None, help="Static ego speed for HUD (km/h).")
    p.add_argument("--speed-limit", type=float, default=None, help="Posted speed limit (km/h) override.")
    p.add_argument("--canvas", default="1280x720", help="Canvas size, e.g. 1280x720.")
    p.add_argument("--display-fps", type=int, default=60, help="Target display FPS cap (waitKey delay).")
    p.add_argument("--model", default=None, help="Override YOLO model path (e.g. yolov8s.pt).")
    p.add_argument(
        "--renderer",
        default="opencv",
        choices=["opencv", "qt"],
        help="opencv (default, stable) | qt (PyQt6 QPainter — smoother, requires PyQt6)",
    )

    # Detection filter knobs — defaults are tuned to suppress ghost detections.
    p.add_argument("--filter-min-conf", type=float, default=0.45)
    p.add_argument("--filter-min-area", type=float, default=0.0006)
    p.add_argument("--filter-consecutive", type=int, default=3)
    p.add_argument("--no-filter", action="store_true")

    p.add_argument("--record", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--loop",
        action="store_true",
        help="When the source video ends, restart from frame 0 (default: stop and close).",
    )
    p.add_argument(
        "--debug-backend",
        action="store_true",
        help="Periodically print backend pipeline stats: detections, tracks, lane assignments, hidden objects.",
    )
    return p.parse_args()


def _resolve_source(args) -> Any:
    if args.source is not None:
        if str(args.source).isdigit():
            return int(args.source)
        return args.source

    if config_mod is not None:
        pref = getattr(config_mod, "PREFERRED_DEFAULT_DRIVE_VIDEO", None)
        if pref is not None and Path(pref).exists():
            return str(pref)
        if hasattr(config_mod, "DEFAULT_SOURCE"):
            s = getattr(config_mod, "DEFAULT_SOURCE", None)
            if s:
                return s

    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            vids = sorted(data_dir.glob(ext))
            if vids:
                return str(vids[0])
    return 0


def _open_writer(path: Path, w: int, h: int, vs) -> "cv2.VideoWriter":
    fps = float(getattr(vs, "fps", 30.0)) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


def _build_detector_and_worker(args):
    if detector_mod is None or not hasattr(detector_mod, "YoloDetector") or not hasattr(detector_mod, "RuntimeConfig"):
        return None, None, None
    if config_mod is None or not hasattr(config_mod, "PERFORMANCE_PRESETS"):
        return None, None, None

    preset = config_mod.PERFORMANCE_PRESETS.get(args.performance) or config_mod.PERFORMANCE_PRESETS.get("balanced")
    if preset is None:
        return None, None, None

    model_path = args.model or preset.model_path
    imgsz = int(args.imgsz) if args.imgsz else int(preset.imgsz)
    try:
        cfg = detector_mod.RuntimeConfig(model_path=model_path, conf_threshold=float(args.conf), imgsz=imgsz)
        detector = detector_mod.YoloDetector(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[main] could not build YoloDetector: {e}")
        return None, None, None

    state = None
    worker = None
    if worker_mod is not None and hasattr(worker_mod, "SharedDetectionState") and hasattr(worker_mod, "DetectionWorker"):
        try:
            state = worker_mod.SharedDetectionState()
            worker = worker_mod.DetectionWorker(detector, state, int(preset.detect_every_n_frames))
            worker.start()
        except Exception as e:  # noqa: BLE001
            print(f"[main] could not start DetectionWorker (sync inference only): {e}")
            state = None
            worker = None

    return detector, state, worker


def _tracks_for_render(
    tracks_dict: Dict[int, Any],
    raw_risk,
    motions: Optional[Dict[int, Any]] = None,
) -> List[Any]:
    risk_map = {}
    if raw_risk is not None and getattr(raw_risk, "per_object", None):
        risk_map = {o.track_id: o for o in raw_risk.per_object}

    out: List[Any] = []
    for tid, tr in tracks_dict.items():
        if getattr(tr, "missing_frames", 0) > 0:
            continue
        o = risk_map.get(tid)
        lvl = None
        if o is not None and getattr(o, "risk_level", None) is not None:
            lvl = o.risk_level.value if hasattr(o.risk_level, "value") else str(o.risk_level)
        vel = None
        if motions is not None:
            mo = motions.get(tid)
            if mo is not None:
                vel = (float(getattr(mo, "velocity_x", 0.0)), float(getattr(mo, "velocity_y", 0.0)))
        out.append(
            SimpleNamespace(
                track_id=tid,
                class_name=tr.class_name,
                bbox=tr.last_bbox,
                risk_score=getattr(o, "risk_score", None) if o is not None else None,
                risk_level=lvl,
                velocity=vel,
            )
        )
    return out


def _run_qt(args) -> int:
    """Dispatch to the Qt renderer. Falls back to OpenCV if PyQt6 is missing."""
    try:
        from driving_scene import qt_app  # noqa: WPS433
    except ImportError as e:
        print(f"[main] PyQt6 not installed ({e}). Run: pip install PyQt6")
        print("[main] Falling back to OpenCV renderer.")
        return -1
    # Bundle the lazily-imported backend modules + helpers Qt needs.
    modules = {
        "config_mod": config_mod,
        "video_source_mod": video_source_mod,
        "detector_mod": detector_mod,
        "worker_mod": worker_mod,
        "tracker_mod": tracker_mod,
        "motion_mod": motion_mod,
        "risk_mod": risk_mod,
        "telemetry_mod": telemetry_mod,
        "speed_limit_mod": speed_limit_mod,
        "filter_mod": filter_mod,
        "lane_estimator_mod": lane_estimator_mod,
        "manual_lanes_mod": manual_lanes_mod,
        "resolve_source": _resolve_source,
        "build_detector_and_worker": _build_detector_and_worker,
    }
    return int(qt_app.run_qt_app(args, modules))


def main():
    args = parse_args()

    # Qt path bypasses the OpenCV main loop entirely.
    if args.renderer == "qt":
        rc = _run_qt(args)
        if rc != -1:
            sys.exit(rc)
        # rc == -1 → fall through to the OpenCV renderer.

    try:
        cw, ch = (int(v) for v in args.canvas.lower().split("x"))
    except Exception:
        cw, ch = 1280, 720

    source = _resolve_source(args)
    print(f"[main] source = {source!r}")

    if (
        video_source_mod is not None
        and hasattr(video_source_mod, "VideoSource")
        and hasattr(video_source_mod.VideoSource, "open")
    ):
        vs = video_source_mod.VideoSource.open(source)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[main] FATAL: could not open video source.")
            sys.exit(1)

        class _VS:
            def __init__(self, c):
                self.cap = c
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

            def read(self):
                return self.cap.read()

            def release(self):
                self.cap.release()

        vs = _VS(cap)

    detector, det_state, worker = _build_detector_and_worker(args)

    tracker = tracker_mod.CentroidTracker() if tracker_mod and hasattr(tracker_mod, "CentroidTracker") else None
    telemetry = (
        telemetry_mod.FrameTelemetry()
        if telemetry_mod and hasattr(telemetry_mod, "FrameTelemetry")
        else _Telemetry()
    )
    risk_state = _RiskState()
    renderer = scene_mod.SceneRenderer(width=cw, height=ch)
    if args.debug:
        renderer.toggle_debug()

    # Manual lane boundaries (live-tunable via [, ], `,`, `.`, P, S).
    # Prefer data/lane_calibration.json if a calibration was saved for this dashcam.
    lane_boundaries: List[float]
    lane_labels: List[str]
    saved_b, saved_l = (None, None)
    if config_mod is not None and hasattr(config_mod, "load_lane_calibration"):
        saved_b, saved_l = config_mod.load_lane_calibration()
    if saved_b is not None and saved_l is not None:
        lane_boundaries, lane_labels = list(saved_b), list(saved_l)
        print(f"[main] loaded lane calibration from data/lane_calibration.json")
    else:
        lane_boundaries = list(getattr(config_mod, "LANE_X_BOUNDARIES", (0.0, 0.28, 0.43, 0.58, 0.73, 1.0)))
        lane_labels = list(getattr(config_mod, "LANE_LABELS", ("far_left", "left_lane", "ego_lane", "right_lane", "far_right")))
    renderer.set_lane_boundaries(lane_boundaries, lane_labels)

    slt = speed_limit_mod.SpeedLimitTracker() if speed_limit_mod and hasattr(speed_limit_mod, "SpeedLimitTracker") else None

    lane_est = None
    if lane_estimator_mod is not None and hasattr(lane_estimator_mod, "LaneEstimator"):
        try:
            lane_est = lane_estimator_mod.LaneEstimator()
        except Exception as e:  # noqa: BLE001
            print(f"[main] LaneEstimator unavailable: {e}")
            lane_est = None
    last_lane_curve: float = 0.0
    last_lane_conf: float = 0.0

    det_filter = None
    if filter_mod is not None and hasattr(filter_mod, "DetectionFilter") and not args.no_filter:
        per_class_conf = {
            k: max(float(v), float(args.filter_min_conf))
            for k, v in filter_mod.DEFAULT_MIN_CONFIDENCE.items()
        }
        det_filter = filter_mod.DetectionFilter(
            frame_w=cw,
            frame_h=ch,
            min_confidence=per_class_conf,
            min_box_area_frac=float(args.filter_min_area),
            require_consecutive=max(0, int(args.filter_consecutive)),
        )
        print(
            f"[main] detection filter ON: min_conf>={args.filter_min_conf}, "
            f"min_area>={args.filter_min_area}, consecutive>={args.filter_consecutive}"
        )
    elif args.no_filter:
        print("[main] detection filter OFF (--no-filter)")

    view_mode = args.view
    view_cycle = ["scene", "dashcam", "split", "debug"]

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "demo.mp4"
    writer: Optional[cv2.VideoWriter] = None
    recording = False
    if args.record:
        writer = _open_writer(out_path, cw, ch, vs)
        recording = True
        print(f"[main] recording to {out_path}")

    win = "Tesla-Inspired Driving Scene Visualizer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, cw, ch)

    last_detections: List[Any] = []
    last_tracks: Dict[int, Any] = {}
    frame_idx = 0
    stats_tick = 0

    # Display FPS cap: cv2.waitKey ms ~= 1000/target_fps (min 1 to keep window responsive).
    target_delay_ms = max(1, int(round(1000.0 / max(1, int(args.display_fps)))))

    try:
        while True:
            read_ret = vs.read()
            if isinstance(read_ret, tuple) and len(read_ret) == 2:
                ok, frame = read_ret
            else:
                ok, frame = True, read_ret
            if not ok or frame is None:
                cap = getattr(vs, "cap", None)
                if args.loop and isinstance(cap, cv2.VideoCapture):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                # End of clip — exit cleanly. Pass --loop to repeat.
                print("[main] end of video — exiting.")
                break

            frame_idx += 1

            # Detection
            if det_state is not None and worker is not None:
                with det_state.lock:
                    det_state.latest_video_frame = frame
                    det_state.latest_video_frame_id = frame_idx
                    if det_state.pending_ready:
                        last_detections = list(det_state.pending_dets)
                        det_state.pending_ready = False
                        if hasattr(telemetry, "yolo_latency_ms"):
                            telemetry.yolo_latency_ms = float(det_state.last_yolo_latency_ms)
            elif detector is not None:
                try:
                    last_detections, lat = detector.predict_resized_to_original(frame)
                    if hasattr(telemetry, "yolo_latency_ms"):
                        telemetry.yolo_latency_ms = float(lat * 1000.0)
                except Exception:
                    last_detections = []

            # Filter
            if det_filter is not None:
                det_filter.update_frame_size(frame.shape[1], frame.shape[0])
                last_detections = det_filter.apply(last_detections)

            # Track
            if tracker is not None:
                last_tracks = tracker.update(last_detections, frame_idx)
            else:
                last_tracks = {}

            # Motion + risk
            motions: Dict[int, Any] = {}
            if motion_mod is not None and hasattr(motion_mod, "estimate_motion"):
                motions = {
                    tid: motion_mod.estimate_motion(t, 1 / 30.0)
                    for tid, t in last_tracks.items()
                    if t.missing_frames == 0
                }

            raw_risk = None
            if risk_mod is not None and hasattr(risk_mod, "compute_frame_risks"):
                try:
                    raw_risk = risk_mod.compute_frame_risks(last_tracks, motions, frame.shape)
                except Exception:
                    raw_risk = None

            if raw_risk is not None:
                risk_state.global_score = float(raw_risk.global_score)
                risk_state.global_level = raw_risk.global_risk_level.value
                risk_state.global_action = raw_risk.global_decision.value
                risk_state.primary_track_id = raw_risk.primary_threat_id

            # Speed limit scaffold
            if slt is not None:
                try:
                    slt.observe(detections=last_detections, frame=frame)
                except Exception:
                    pass
            posted = args.speed_limit if args.speed_limit is not None else (slt.current() if slt is not None else None)

            # Lane curvature estimation (throttled — every 4th frame).
            if lane_est is not None and (frame_idx % 4 == 0):
                try:
                    lp = lane_est.estimate(frame)
                    last_lane_curve = float(getattr(lp, "curve_strength", 0.0))
                    last_lane_conf = float(getattr(lp, "confidence", 0.0))
                except Exception:
                    pass

            # Render
            canvas = renderer.render(
                frame_bgr=frame,
                detections=last_detections,
                tracks=_tracks_for_render(last_tracks, raw_risk, motions),
                telemetry=telemetry,
                risk_state=risk_state,
                ego_speed_kmh=args.ego_speed_kmh,
                view_mode=view_mode,
                speed_limit_kmh=posted,
                lane_curve=last_lane_curve,
                lane_confidence=last_lane_conf,
            )

            if hasattr(telemetry, "tick_display"):
                telemetry.tick_display()

            if args.debug and det_filter is not None and stats_tick % 60 == 0:
                print(f"[filter] {det_filter.stats()}  kept_this_frame={len(last_detections)}")
            if args.debug_backend and stats_tick % 30 == 0:
                # Per-track snapshot. The renderer's mapper holds the items
                # we just produced (it stores per-track scene points).
                visible = list(renderer.mapper.track_points.keys())
                hidden = renderer.mapper.hidden_items
                print(
                    f"[backend] frame={frame_idx} dets={len(last_detections)} "
                    f"tracks={len(last_tracks)} visible={len(visible)} hidden={len(hidden)}"
                )
                for tid, tr in list(last_tracks.items())[:5]:
                    bbox = tr.last_bbox
                    cxn = ((bbox[0] + bbox[2]) * 0.5) / max(1, frame.shape[1])
                    _id, label, _c, _w = renderer.lanes.assign_lane_from_boundaries(cxn)
                    print(
                        f"  #{tid:3d} {tr.class_name:8s} cx={cxn:0.2f} lane={label:10s} "
                        f"q={tr.track_quality:0.2f} hits={tr.hits} misses={tr.misses}"
                    )
            stats_tick += 1

            cv2.imshow(win, canvas)
            if recording and writer is not None:
                writer.write(canvas)

            key = cv2.waitKey(target_delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                renderer.toggle_debug()
            elif key == ord("f"):
                if det_filter is not None:
                    print(f"[filter] {det_filter.stats()}")
                else:
                    print("[filter] disabled (--no-filter)")
            elif key == ord("v"):
                view_mode = view_cycle[(view_cycle.index(view_mode) + 1) % len(view_cycle)]
                print(f"[main] view -> {view_mode}")
            elif key == ord("r"):
                if recording:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"[main] stopped recording -> {out_path}")
                else:
                    writer = _open_writer(out_path, cw, ch, vs)
                    recording = True
                    print(f"[main] recording -> {out_path}")
            elif key == ord("p"):
                if manual_lanes_mod is not None and hasattr(manual_lanes_mod, "format_boundaries_for_config"):
                    print(manual_lanes_mod.format_boundaries_for_config(lane_boundaries))
                else:
                    print(f"LANE_X_BOUNDARIES = {lane_boundaries}")
            elif key in (ord("["), ord("]")):
                if manual_lanes_mod is not None:
                    delta = -0.012 if key == ord("[") else 0.012
                    manual_lanes_mod.shift_inner_boundaries(lane_boundaries, delta)
                    renderer.set_lane_boundaries(lane_boundaries, lane_labels)
                    print(f"[main] lanes shifted -> {lane_boundaries}")
            elif key in (ord(","), ord(".")):
                if manual_lanes_mod is not None:
                    widen = key == ord(".")
                    manual_lanes_mod.adjust_ego_lane_width(lane_boundaries, widen=widen)
                    renderer.set_lane_boundaries(lane_boundaries, lane_labels)
                    word = "wider" if widen else "narrower"
                    print(f"[main] ego lane {word} -> {lane_boundaries}")
            elif key == ord("s"):
                if config_mod is not None and hasattr(config_mod, "save_lane_calibration"):
                    try:
                        config_mod.save_lane_calibration(lane_boundaries, lane_labels)
                        print("[main] lane calibration saved -> data/lane_calibration.json")
                    except Exception as e:  # noqa: BLE001
                        print(f"[main] save failed: {e}")
    finally:
        if det_state is not None:
            det_state.stop_event.set()
        if worker is not None:
            try:
                worker.join(timeout=5.0)
            except Exception:
                pass
        if writer is not None:
            writer.release()
        try:
            vs.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
