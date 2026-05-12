"""Per-frame timing and summary telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameTelemetry:
    # Legacy / total loop
    fps_ema: float = 0.0
    yolo_latency_ms: float = 0.0
    tracking_latency_ms: float = 0.0
    risk_latency_ms: float = 0.0
    total_frame_latency_ms: float = 0.0
    num_detections: int = 0
    num_tracked_objects: int = 0
    highest_risk_track_id: Optional[int] = None
    highest_risk_score: float = 0.0

    display_fps_ema: float = 0.0
    detection_fps_ema: float = 0.0
    render_latency_ms: float = 0.0
    dropped_detections: int = 0

    _ema_alpha: float = field(default=0.12, repr=False)

    def update_fps(self, frame_dt_sec: float) -> None:
        if frame_dt_sec <= 0:
            return
        inst = 1.0 / frame_dt_sec
        self.fps_ema = (
            inst if self.fps_ema <= 0 else self._ema_alpha * inst + (1 - self._ema_alpha) * self.fps_ema
        )

    def update_display_fps(self, dt_sec: float) -> None:
        if dt_sec <= 0:
            return
        inst = 1.0 / dt_sec
        self.display_fps_ema = (
            inst
            if self.display_fps_ema <= 0
            else self._ema_alpha * inst + (1 - self._ema_alpha) * self.display_fps_ema
        )

    def note_detection_fire(self, dt_since_last_det_sec: float) -> None:
        if dt_since_last_det_sec <= 0:
            return
        inst = 1.0 / dt_since_last_det_sec
        self.detection_fps_ema = (
            inst
            if self.detection_fps_ema <= 0
            else self._ema_alpha * inst + (1 - self._ema_alpha) * self.detection_fps_ema
        )
