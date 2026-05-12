"""
qt_app.py — PyQt6 main-loop entry point for the Tesla-Inspired
Driving Scene Visualizer.

Architecture:

  ┌────────────────────────────────────────────────────────────┐
  │ BackendThread (Python threading.Thread, NOT QThread)       │
  │   • Reads frames from VideoSource                          │
  │   • Hands frames to DetectionWorker (separate thread)      │
  │   • Applies detection filter, tracker, motion estimator    │
  │   • Calls SceneMapper.collect_renderables()                │
  │   • Writes a SceneSnapshot to a lock-guarded slot          │
  │                                                            │
  │   YOLO does NOT run on this thread; DetectionWorker does.  │
  └─────────────────────┬──────────────────────────────────────┘
                        │  snapshot slot (mutex)
                        ▼
  ┌────────────────────────────────────────────────────────────┐
  │ Qt main thread (QApplication)                              │
  │   • QTimer at ~60 fps                                      │
  │   • Each tick: pull latest SceneSnapshot, push to canvas   │
  │   • SceneCanvas.paintEvent draws with QPainter             │
  └────────────────────────────────────────────────────────────┘

Public entry point:

    run_qt_app(args, modules) -> int

It returns the Qt event loop exit code so main.py can `sys.exit(...)`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QMainWindow

from .qt_renderer import SceneCanvas, SceneSnapshot


# ---------------------------------------------------------------------- #
# BackendThread                                                          #
# ---------------------------------------------------------------------- #

class _Telemetry:
    def __init__(self):
        self.display_fps = 0.0
        self.detection_fps = 0.0
        self.yolo_latency_ms = 0.0
        self.track_latency_ms = 0.0
        self.risk_latency_ms = 0.0
        self.dropped_detections = 0
        self._last_t = time.time()
        self._alpha = 0.15

    def tick(self):
        now = time.time()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now
        inst = 1.0 / dt
        self.display_fps = (1 - self._alpha) * self.display_fps + self._alpha * inst


class _RiskState:
    def __init__(self):
        self.global_score = 0.0
        self.global_level = "LOW"
        self.global_action = "SAFE"
        self.primary_track_id: Optional[int] = None


class BackendThread(threading.Thread):
    """Runs the perception pipeline; writes the latest SceneSnapshot."""

    def __init__(self, args, modules: Dict[str, Any], canvas_size):
        super().__init__(daemon=True)
        self.args = args
        self.modules = modules
        self.W, self.H = canvas_size
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[SceneSnapshot] = None
        # Mutable lane boundaries (Qt sends keystrokes that mutate this list).
        # Prefer data/lane_calibration.json over config defaults.
        cfg = modules.get("config_mod")
        saved_b, saved_l = (None, None)
        if cfg is not None and hasattr(cfg, "load_lane_calibration"):
            saved_b, saved_l = cfg.load_lane_calibration()
        if saved_b is not None and saved_l is not None:
            self.lane_boundaries: List[float] = list(saved_b)
            self.lane_labels: List[str] = list(saved_l)
            print("[qt-backend] loaded lane calibration from data/lane_calibration.json")
        else:
            self.lane_boundaries: List[float] = list(
                getattr(cfg, "LANE_X_BOUNDARIES",
                        (0.0, 0.28, 0.43, 0.58, 0.73, 1.0))
            )
            self.lane_labels: List[str] = list(
                getattr(cfg, "LANE_LABELS",
                        ("far_left", "left_lane", "ego_lane", "right_lane", "far_right"))
            )
        # Public flags Qt main thread can flip.
        self.view_mode = args.view
        self.toggled_debug = False

        # Lane model + mapper are created eagerly so the keyboard handler on
        # the Qt main thread can call self.lanes.set_boundaries(...) live.
        from . import lane_model as _lm  # noqa: WPS433
        from .scene_mapper import SceneMapper  # noqa: WPS433
        self.lanes = _lm.LaneModel(self.W, self.H)
        self.lanes.set_boundaries(self.lane_boundaries, self.lane_labels)
        self.mapper = SceneMapper(self.lanes)

    def stop(self):
        self._stop.set()

    def get_snapshot(self) -> Optional[SceneSnapshot]:
        with self._lock:
            return self._latest

    def _publish(self, snap: SceneSnapshot):
        with self._lock:
            self._latest = snap

    def run(self):
        args = self.args
        m = self.modules
        cw, ch = self.W, self.H

        # Open the video source.
        source = m["resolve_source"](args)
        print(f"[qt-backend] source = {source!r}")
        vs_mod = m.get("video_source_mod")
        if vs_mod is not None and hasattr(vs_mod, "VideoSource") and hasattr(vs_mod.VideoSource, "open"):
            vs = vs_mod.VideoSource.open(source)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print("[qt-backend] FATAL: could not open video source.")
                return

            class _VS:
                def __init__(self, c):
                    self.cap = c
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

                def read(self):
                    return self.cap.read()

                def release(self):
                    self.cap.release()

            vs = _VS(cap)

        # Build detector + async worker.
        detector, det_state, worker = m["build_detector_and_worker"](args)

        tracker_mod = m.get("tracker_mod")
        tracker = tracker_mod.CentroidTracker() if tracker_mod and hasattr(tracker_mod, "CentroidTracker") else None

        telemetry_mod = m.get("telemetry_mod")
        telemetry = (
            telemetry_mod.FrameTelemetry()
            if telemetry_mod and hasattr(telemetry_mod, "FrameTelemetry")
            else _Telemetry()
        )

        motion_mod = m.get("motion_mod")
        risk_mod = m.get("risk_mod")
        risk_state = _RiskState()

        slt_mod = m.get("speed_limit_mod")
        slt = slt_mod.SpeedLimitTracker() if slt_mod and hasattr(slt_mod, "SpeedLimitTracker") else None

        filter_mod = m.get("filter_mod")
        det_filter = None
        if filter_mod is not None and hasattr(filter_mod, "DetectionFilter") and not args.no_filter:
            per_class_conf = {
                k: max(float(v), float(args.filter_min_conf))
                for k, v in filter_mod.DEFAULT_MIN_CONFIDENCE.items()
            }
            det_filter = filter_mod.DetectionFilter(
                frame_w=cw, frame_h=ch,
                min_confidence=per_class_conf,
                min_box_area_frac=float(args.filter_min_area),
                require_consecutive=max(0, int(args.filter_consecutive)),
            )

        # SceneMapper + LaneModel — already created in __init__ so the Qt
        # keyboard handler can call lanes.set_boundaries(...) live without
        # racing with this thread's startup.
        lanes = self.lanes
        mapper = self.mapper

        lane_est_mod = m.get("lane_estimator_mod")
        lane_est = None
        if lane_est_mod is not None and hasattr(lane_est_mod, "LaneEstimator"):
            try:
                lane_est = lane_est_mod.LaneEstimator()
            except Exception as e:  # noqa: BLE001
                print(f"[qt-backend] LaneEstimator unavailable: {e}")

        last_lane_curve = 0.0
        last_lane_conf = 0.0
        last_detections: List[Any] = []
        last_tracks: Dict[int, Any] = {}
        frame_idx = 0
        last_t = time.time()

        try:
            while not self._stop.is_set():
                read_ret = vs.read()
                if isinstance(read_ret, tuple) and len(read_ret) == 2:
                    ok, frame = read_ret
                else:
                    ok, frame = True, read_ret
                if not ok or frame is None:
                    cap = getattr(vs, "cap", None)
                    if getattr(args, "loop", False) and isinstance(cap, cv2.VideoCapture):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    print("[qt-backend] end of video — stopping.")
                    break

                frame_idx += 1

                # Detection (async worker if available, sync fallback).
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

                if det_filter is not None:
                    det_filter.update_frame_size(frame.shape[1], frame.shape[0])
                    last_detections = det_filter.apply(last_detections)

                if tracker is not None:
                    last_tracks = tracker.update(last_detections, frame_idx)
                else:
                    last_tracks = {}

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

                if slt is not None:
                    try:
                        slt.observe(detections=last_detections, frame=frame)
                    except Exception:
                        pass
                posted = args.speed_limit if args.speed_limit is not None else (slt.current() if slt is not None else None)

                # Lane curvature (throttled).
                if lane_est is not None and (frame_idx % 4 == 0):
                    try:
                        lp = lane_est.estimate(frame)
                        last_lane_curve = float(getattr(lp, "curve_strength", 0.0))
                        last_lane_conf = float(getattr(lp, "confidence", 0.0))
                    except Exception:
                        pass

                # Map → scene items.
                mapper.set_curve(last_lane_curve)
                mapper.set_lane_confidence(last_lane_conf)
                now = time.time()
                dt = max(0.001, min(0.1, now - last_t))
                last_t = now
                mapper.advance(dt, args.ego_speed_kmh)

                # Build SimpleNamespace-like records to feed mapper, since
                # last_tracks gives TrackedObjects but we want velocity too.
                tracks_with_vel = []
                from types import SimpleNamespace
                for tid, tr in last_tracks.items():
                    if getattr(tr, "missing_frames", 0) > 0:
                        continue
                    o = (raw_risk.per_object if raw_risk and getattr(raw_risk, "per_object", None) else None)
                    lvl = None
                    score = None
                    if o:
                        match = next((x for x in o if x.track_id == tid), None)
                        if match:
                            lvl = match.risk_level.value if hasattr(match.risk_level, "value") else str(match.risk_level)
                            score = getattr(match, "risk_score", None)
                    mo = motions.get(tid)
                    vel = (float(mo.velocity_x), float(mo.velocity_y)) if mo else None
                    tracks_with_vel.append(SimpleNamespace(
                        track_id=tid,
                        class_name=tr.class_name,
                        bbox=tr.last_bbox,
                        risk_score=score,
                        risk_level=lvl,
                        velocity=vel,
                    ))

                items = mapper.collect_renderables(tracks_with_vel, last_detections, frame.shape[1], frame.shape[0])
                items.sort(key=lambda it: it["closeness"])
                mapper.update_trails(items)
                mapper.prune_stale()

                if hasattr(telemetry, "tick_display"):
                    telemetry.tick_display()
                elif hasattr(telemetry, "tick"):
                    telemetry.tick()

                # Publish snapshot.
                snap = SceneSnapshot(
                    items=items,
                    primary_threat_id=risk_state.primary_track_id,
                    lane_curve=last_lane_curve,
                    lane_confidence=last_lane_conf,
                    lane_phase=mapper.lane_phase,
                    track_trails=dict(mapper.track_trails),
                    ego_speed_kmh=args.ego_speed_kmh,
                    speed_limit_kmh=posted,
                    view_mode=self.view_mode,
                    n_objects=len(items),
                    display_fps=float(getattr(telemetry, "display_fps", 0.0) or 0.0),
                    detection_fps=float(getattr(telemetry, "detection_fps", 0.0) or 0.0),
                    yolo_latency_ms=float(getattr(telemetry, "yolo_latency_ms", 0.0) or 0.0),
                    dropped_detections=int(getattr(telemetry, "dropped_detections", 0) or 0),
                    global_score=risk_state.global_score,
                    global_level=risk_state.global_level,
                    global_action=risk_state.global_action,
                    lane_boundaries=list(self.lane_boundaries),
                    lane_labels=list(self.lane_labels),
                )
                self._publish(snap)
        finally:
            if det_state is not None:
                det_state.stop_event.set()
            if worker is not None:
                try:
                    worker.join(timeout=5.0)
                except Exception:
                    pass
            try:
                vs.release()
            except Exception:
                pass


# ---------------------------------------------------------------------- #
# MainWindow                                                             #
# ---------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    def __init__(self, args, modules: Dict[str, Any]):
        super().__init__()
        self.setWindowTitle("Tesla-Inspired Driving Scene Visualizer (Qt)")
        try:
            cw, ch = (int(v) for v in args.canvas.lower().split("x"))
        except Exception:
            cw, ch = 1280, 720
        self.canvas = SceneCanvas(cw, ch, parent=self)
        self.canvas.debug = bool(args.debug)
        self.setCentralWidget(self.canvas)
        self.setFixedSize(cw, ch)

        self.args = args
        self.modules = modules
        self.view_cycle = ["scene", "dashcam", "split", "debug"]

        self.backend = BackendThread(args, modules, (cw, ch))
        self.backend.start()

        # Pull snapshots @ ~60 fps and trigger repaint.
        target_fps = max(15, int(getattr(args, "display_fps", 60)))
        interval_ms = max(1, int(round(1000.0 / target_fps)))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(interval_ms)

    def _tick(self):
        snap = self.backend.get_snapshot()
        if snap is not None:
            self.canvas.set_snapshot(snap)

    def closeEvent(self, event):  # noqa: N802
        self.backend.stop()
        self.backend.join(timeout=2.0)
        super().closeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        text = event.text().lower()
        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
            return
        if text == "v":
            idx = self.view_cycle.index(self.backend.view_mode) if self.backend.view_mode in self.view_cycle else 0
            new = self.view_cycle[(idx + 1) % len(self.view_cycle)]
            self.backend.view_mode = new
            print(f"[qt] view -> {new}")
            return
        if text == "d":
            self.canvas.toggle_debug()
            print(f"[qt] debug -> {self.canvas.debug}")
            return
        if text == "p":
            manual_mod = self.modules.get("manual_lanes_mod")
            if manual_mod is not None and hasattr(manual_mod, "format_boundaries_for_config"):
                print(manual_mod.format_boundaries_for_config(self.backend.lane_boundaries))
            else:
                print(f"LANE_X_BOUNDARIES = {self.backend.lane_boundaries}")
            return
        manual_mod = self.modules.get("manual_lanes_mod")
        if manual_mod is not None:
            mutated = False
            if text == "[":
                manual_mod.shift_inner_boundaries(self.backend.lane_boundaries, -0.012)
                mutated = True
            elif text == "]":
                manual_mod.shift_inner_boundaries(self.backend.lane_boundaries, +0.012)
                mutated = True
            elif text == ",":
                manual_mod.adjust_ego_lane_width(self.backend.lane_boundaries, widen=False)
                mutated = True
            elif text == ".":
                manual_mod.adjust_ego_lane_width(self.backend.lane_boundaries, widen=True)
                mutated = True
            if mutated:
                self.backend.lanes.set_boundaries(
                    self.backend.lane_boundaries, self.backend.lane_labels
                )
                print(f"[qt] lanes -> {self.backend.lane_boundaries}")
                return
        if text == "s":
            # Persist current boundaries to data/lane_calibration.json.
            try:
                from . import config as _cfg
                _cfg.save_lane_calibration(
                    self.backend.lane_boundaries, self.backend.lane_labels
                )
                print("[qt] lane calibration saved -> data/lane_calibration.json")
            except Exception as e:  # noqa: BLE001
                print(f"[qt] save failed: {e}")
            return


# ---------------------------------------------------------------------- #
# Public entry point                                                     #
# ---------------------------------------------------------------------- #

def run_qt_app(args, modules: Dict[str, Any]) -> int:
    """Block on the Qt event loop until the window is closed; return exit code."""
    app = QApplication([])
    win = MainWindow(args, modules)
    win.show()
    return app.exec()
