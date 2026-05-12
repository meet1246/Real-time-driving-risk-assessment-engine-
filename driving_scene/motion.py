"""Motion estimation from track histories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import MIN_DT_SEC, MOTION_HISTORY_FOR_VEL
from .detector import Detection
from .tracker import TrackedObject


@dataclass
class MotionState:
    track_id: int
    velocity_x: float
    velocity_y: float
    speed_px_per_sec: float
    bbox_area_change_rate: float  # d(area)/dt (px^2 / sec)
    movement_direction_deg: float  # 0 = +x, 90 = +y (image coords, y down)
    pred_center_1s: Tuple[float, float]
    pred_center_2s: Tuple[float, float]
    dt_used: float


def tracks_to_extrapolated_detections(
    tracks: Dict[int, TrackedObject],
    motions: Dict[int, MotionState],
    dt: float,
    frame_w: int,
    frame_h: int,
) -> List[Detection]:
    """
    Build synthetic Detection list from active tracks + velocities (YOLO skip frames).
    Keeps centroid tracker fed without running inference every frame.
    """
    out: List[Detection] = []
    if dt <= 0:
        dt = MIN_DT_SEC
    for tid, tr in tracks.items():
        if getattr(tr, "missing_frames", 0) > 0:
            continue
        mo = motions.get(tid)
        bbox = tr.last_bbox
        if mo is not None:
            bbox = shift_bbox_by_velocity(bbox, mo.velocity_x, mo.velocity_y, dt, frame_w, frame_h)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        out.append(
            Detection(
                class_id=tr.last_class_id,
                class_name=tr.class_name,
                confidence=float(tr.last_confidence) * 0.99,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(cx, cy),
                area=w * h,
            )
        )
    return out


def shift_bbox_by_velocity(
    bbox: Tuple[float, float, float, float],
    vx: float,
    vy: float,
    dt: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, float, float]:
    """Translate bbox by constant-velocity motion for dt seconds; clamp to frame bounds."""
    if dt <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    dx, dy = vx * dt, vy * dt
    nx1, ny1, nx2, ny2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
    fw, fh = float(max(frame_w - 1, 1)), float(max(frame_h - 1, 1))

    def clamp_bb(
        a: float, b: float, c: float, d: float,
    ) -> Tuple[float, float, float, float]:
        a, c = max(0.0, min(a, fw)), max(0.0, min(c, fw))
        b, d = max(0.0, min(b, fh)), max(0.0, min(d, fh))
        if c < a:
            a, c = c, a
        if d < b:
            b, d = d, b
        return (a, b, c, d)

    return clamp_bb(nx1, ny1, nx2, ny2)


def _poly_line_points(
    center: Tuple[float, float],
    pred: Tuple[float, float],
    steps: int = 5,
) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for i in range(steps + 1):
        t = i / steps
        x = center[0] + t * (pred[0] - center[0])
        y = center[1] + t * (pred[1] - center[1])
        pts.append((int(round(x)), int(round(y))))
    return pts


def estimate_motion(track: TrackedObject, dt_sec: float) -> MotionState:
    """
    Velocity from last two centers (or short regression if enough history).
    Area change rate from last two bbox areas.
    """
    dt = max(dt_sec, MIN_DT_SEC)
    centers = track.center_history
    bboxes = track.bbox_history

    vx, vy = 0.0, 0.0
    if len(centers) >= 2:
        n = min(MOTION_HISTORY_FOR_VEL, len(centers))
        c0 = centers[-n]
        c1 = centers[-1]
        vx = (c1[0] - c0[0]) / (dt * (n - 1) if n > 1 else 1.0)
        vy = (c1[1] - c0[1]) / (dt * (n - 1) if n > 1 else 1.0)

    speed = float(np.hypot(vx, vy))
    direction = math.degrees(math.atan2(vy, vx)) % 360.0

    area_rate = 0.0
    if len(bboxes) >= 2:
        def area(bb: Tuple[float, float, float, float]) -> float:
            return max(0.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))

        a0 = area(bboxes[-2])
        a1 = area(bboxes[-1])
        area_rate = (a1 - a0) / dt

    cx, cy = track.last_center
    p1 = (cx + vx * 1.0, cy + vy * 1.0)
    p2 = (cx + vx * 2.0, cy + vy * 2.0)

    return MotionState(
        track_id=track.track_id,
        velocity_x=vx,
        velocity_y=vy,
        speed_px_per_sec=speed,
        bbox_area_change_rate=area_rate,
        movement_direction_deg=direction,
        pred_center_1s=p1,
        pred_center_2s=p2,
        dt_used=dt,
    )


def trajectory_polyline(track: TrackedObject, motion: MotionState) -> List[Tuple[int, int]]:
    return _poly_line_points(track.last_center, motion.pred_center_2s)
