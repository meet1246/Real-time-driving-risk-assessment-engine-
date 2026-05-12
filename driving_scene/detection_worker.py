"""Background YOLO worker: never blocks the display thread."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .detector import Detection, YoloDetector


@dataclass
class SharedDetectionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)

    # Latest decoded video frame (BGR, no HUD) — updated only on new camera/video frames
    latest_video_frame: Optional[np.ndarray] = None
    latest_video_frame_id: int = 0

    # Completed inference (consumer clears pending_ready)
    pending_ready: bool = False
    pending_dets: List = field(default_factory=list)
    pending_video_frame_id: int = -1
    last_yolo_latency_ms: float = 0.0

    inference_busy: bool = False
    dropped_detection_requests: int = 0
    detections_completed: int = 0


class DetectionWorker(threading.Thread):
    """
    Runs YOLO on a downsized tensor; skips work when inference is already running (drops frame).
    """

    def __init__(
        self,
        detector: YoloDetector,
        state: SharedDetectionState,
        detect_every_n_frames: int,
    ) -> None:
        super().__init__(daemon=True)
        self._detector = detector
        self._state = state
        self._every_n = max(1, detect_every_n_frames)
        self._last_consumed_vfid: int = -1

    def run(self) -> None:
        while not self._state.stop_event.is_set():
            with self._state.lock:
                vfid = self._state.latest_video_frame_id
                frame = self._state.latest_video_frame

            if frame is None:
                time.sleep(0.002)
                continue

            if vfid <= self._last_consumed_vfid:
                time.sleep(0.001)
                continue

            if vfid % self._every_n != 0:
                self._last_consumed_vfid = vfid
                time.sleep(0.0005)
                continue

            if self._state.inference_busy:
                with self._state.lock:
                    self._state.dropped_detection_requests += 1
                time.sleep(0.0005)
                continue

            with self._state.lock:
                snap = self._state.latest_video_frame
                if snap is None:
                    continue
                snap = snap.copy()
                target_vfid = self._state.latest_video_frame_id

            self._state.inference_busy = True
            ok_infer = False
            try:
                dets, lat = self._detector.predict_resized_to_original(snap)
                with self._state.lock:
                    self._state.pending_dets = list(dets)
                    self._state.pending_video_frame_id = target_vfid
                    self._state.pending_ready = True
                    self._state.last_yolo_latency_ms = float(lat * 1000.0)
                    self._state.detections_completed += 1
                ok_infer = True
            finally:
                self._state.inference_busy = False
            if ok_infer:
                self._last_consumed_vfid = target_vfid

            time.sleep(0.0005)
