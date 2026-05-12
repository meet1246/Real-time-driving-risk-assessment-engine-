"""
scene_mapper.py — Tesla-Inspired Driving Scene Visualizer

Converts tracked detections into scene-space items the renderer can draw.

WHY THIS MODULE EXISTS
======================
A bbox in image space has two pieces of information we need:
    1. horizontal position  -> which lane the object is in
    2. vertical position    -> how far ahead the object is ("closeness")

Mapping bbox-center-x DIRECTLY to scene-x is the classic mistake — it makes
every car drift toward the canvas centerline as the road perspective
narrows. Instead we:

    bbox center x  -> lane assignment (LaneModel.assign_lane)
    closeness      -> scene_y (perspective interpolation)
    lane id + closeness + small in-lane offset
                   -> scene_x (LaneModel.lane_to_scene_xy)

Then we EMA-smooth (scene_x, scene_y, icon_w, icon_h) per track to kill jitter.
Trails are kept in the same module since they're position history, not visual
state.

PUBLIC API
==========
    image_bbox_to_scene_plane(bbox, fw, fh) -> (lateral, closeness, area_norm)
    SceneMapper(lanes, ema_alpha=0.45, stale_after=1.2, min_height_ratio,
                min_area_ratio)
        .advance(dt, ego_speed_kmh)         step the lane phase (animated dashes)
        .set_curve(c)                       forward to LaneModel.set_curve
        .set_lane_confidence(c)
        .collect_renderables(tracks, detections, fw, fh) -> List[dict]
        .update_trails(items)
        .prune_stale()
        .reset()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import ema, lerp


# ---------------------------------------------------------------------- #
# bbox -> normalized scene plane                                         #
# ---------------------------------------------------------------------- #

def image_bbox_to_scene_plane(
    bbox_xyxy: Sequence[float],
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float, float]:
    """
    Returns (lateral, closeness, area_norm) all in [0, 1] (lateral in [-1.4, 1.4]).

    Closeness blends bbox_bottom_y, bbox_height, and sqrt(area) — closer
    objects sit lower in the image and look bigger. The weights match the
    project spec:

        closeness = 0.65 * bbox_bottom_y_norm
                  + 0.25 * bbox_height_norm
                  + 0.10 * sqrt(area_norm)
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = 0.5 * (x1 + x2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    lateral = (cx - frame_w * 0.5) / (frame_w * 0.5 + 1e-6)
    lateral = max(-1.4, min(1.4, lateral))

    bottom_y_norm = y2 / max(1.0, frame_h)
    area_norm = (bw * bh) / max(1.0, frame_w * frame_h)
    height_norm = bh / max(1.0, frame_h)

    closeness = (
        0.65 * bottom_y_norm
        + 0.25 * min(1.0, height_norm * 2.2)
        + 0.10 * (area_norm ** 0.5)
    )
    closeness = max(0.0, min(1.0, closeness))
    return float(lateral), float(closeness), float(area_norm)


# ---------------------------------------------------------------------- #
# Internal per-track state                                               #
# ---------------------------------------------------------------------- #

@dataclass
class _ScenePoint:
    x: float
    y: float
    w: float
    h: float
    last_seen: float


@dataclass
class _MapperState:
    track_points: Dict[int, _ScenePoint] = field(default_factory=dict)
    track_trails: Dict[int, List[Tuple[float, float, float]]] = field(default_factory=dict)
    lane_phase: float = 0.0
    last_render_t: float = 0.0
    lane_confidence: Optional[float] = None


# ---------------------------------------------------------------------- #
# SceneMapper                                                            #
# ---------------------------------------------------------------------- #

class SceneMapper:
    """Stateful mapper: turns tracks + detections into smoothed scene items."""

    def __init__(
        self,
        lanes,
        ema_alpha: float = 0.45,
        stale_after: float = 1.2,
        min_height_ratio: float = 0.025,
        min_area_ratio: float = 0.0005,
        min_track_quality: float = 0.0,
    ):
        self.lanes = lanes
        self.ema_alpha = ema_alpha
        self.stale_after = stale_after
        # Visibility gates — far/tiny detections get hidden to keep the scene clean.
        self.min_height_ratio = min_height_ratio
        self.min_area_ratio = min_area_ratio
        self.min_track_quality = min_track_quality
        # If non-None, hidden items are appended here as dicts with
        # `hidden_reason` set so the debug overlay / probe tool can show them.
        self.hidden_items: List[Dict[str, Any]] = []
        self.state = _MapperState()

    # ---- runtime state knobs -------------------------------------------- #

    def advance(self, dt: float, ego_speed_kmh: Optional[float]) -> None:
        speed_for_flow = ego_speed_kmh if ego_speed_kmh is not None else 30.0
        flow_rate = 0.4 + 0.025 * float(speed_for_flow)
        self.state.lane_phase = (self.state.lane_phase + flow_rate * dt) % 1.0

    @property
    def lane_phase(self) -> float:
        return self.state.lane_phase

    def set_curve(self, curve_strength: float) -> None:
        self.lanes.set_curve(curve_strength)

    def set_lane_confidence(self, conf: Optional[float]) -> None:
        self.state.lane_confidence = (
            None if conf is None else float(conf)
        )

    @property
    def lane_confidence(self) -> Optional[float]:
        return self.state.lane_confidence

    @property
    def track_points(self) -> Dict[int, _ScenePoint]:
        return self.state.track_points

    @property
    def track_trails(self) -> Dict[int, List[Tuple[float, float, float]]]:
        return self.state.track_trails

    def reset(self) -> None:
        self.state = _MapperState()

    # ---- main entry ----------------------------------------------------- #

    def collect_renderables(
        self,
        tracks: Optional[Iterable[Any]],
        detections: Optional[Iterable[Any]],
        fw: int,
        fh: int,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        self.hidden_items = []
        seen_ids: set = set()
        for t in (tracks or []):
            tid = getattr(t, "track_id", None)
            bbox = (
                getattr(t, "bbox_xyxy", None)
                or getattr(t, "bbox", None)
                or getattr(t, "last_bbox", None)
            )
            if bbox is None:
                continue
            cls = getattr(t, "class_name", None) or getattr(t, "cls", None) or "car"
            score = getattr(t, "risk_score", None)
            if score is None:
                score = getattr(t, "score", None)
            level = getattr(t, "risk_level", None)
            velocity = getattr(t, "velocity", None)
            quality = float(getattr(t, "track_quality", 1.0) or 0.0)
            item = self._build_item(tid, bbox, cls, score, level, fw, fh, velocity, track_quality=quality)
            if item is not None:
                items.append(item)
                if tid is not None:
                    seen_ids.add(tid)
        for d in (detections or []):
            tid = getattr(d, "track_id", None)
            if tid is not None and tid in seen_ids:
                continue
            bbox = getattr(d, "bbox_xyxy", None) or getattr(d, "bbox", None)
            if bbox is None:
                continue
            cls = getattr(d, "class_name", None) or getattr(d, "cls", None) or "car"
            item = self._build_item(tid, bbox, cls, None, None, fw, fh, velocity=None)
            if item is not None:
                items.append(item)
        return items

    def _build_item(
        self,
        tid: Optional[int],
        bbox: Sequence[float],
        cls: str,
        score: Optional[float],
        level: Optional[str],
        fw: int,
        fh: int,
        velocity: Optional[Tuple[float, float]] = None,
        track_quality: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        height_ratio = bh / max(1.0, fh)
        area_ratio = (bw * bh) / max(1.0, fw * fh)
        cx_norm = ((x1 + x2) * 0.5) / max(1.0, fw)
        by_norm = y2 / max(1.0, fh)

        # Visibility gates: far/tiny / flickery detections are hidden so the
        # scene doesn't clutter with phantom dots. Each gate records WHY it
        # dropped the item so the debug overlay and backend probe can show it.
        hidden_reason: Optional[str] = None
        if self.min_height_ratio > 0 and height_ratio < self.min_height_ratio:
            hidden_reason = f"height_ratio<{self.min_height_ratio:.3f}"
        elif self.min_area_ratio > 0 and area_ratio < self.min_area_ratio:
            hidden_reason = f"area_ratio<{self.min_area_ratio:.4f}"
        elif self.min_track_quality > 0 and track_quality < self.min_track_quality:
            hidden_reason = f"track_quality<{self.min_track_quality:.2f}"
        if hidden_reason is not None:
            self.hidden_items.append({
                "track_id": tid,
                "class": str(cls).lower(),
                "bbox": bbox,
                "cx_norm": cx_norm,
                "by_norm": by_norm,
                "height_ratio": height_ratio,
                "area_ratio": area_ratio,
                "track_quality": track_quality,
                "hidden_reason": hidden_reason,
            })
            return None

        lane_id, lane_offset = self.lanes.assign_lane(
            bbox=bbox,
            frame_w=fw,
            frame_h=fh,
            cls=str(cls),
            velocity=velocity,
            track_id=tid,
        )
        lateral, closeness, area_norm = image_bbox_to_scene_plane(bbox, fw, fh)
        sx, sy = self.lanes.lane_to_scene_xy(lane_id, lane_offset, closeness)
        base_w = lerp(28.0, 150.0, closeness)
        base_h = lerp(18.0, 95.0, closeness)
        size_boost = 0.6 + 1.4 * min(1.0, area_norm * 5.0)
        icon_w = base_w * size_boost
        icon_h = base_h * size_boost

        if tid is not None:
            prev = self.state.track_points.get(tid)
            if prev is not None:
                sx = ema(prev.x, sx, self.ema_alpha)
                sy = ema(prev.y, sy, self.ema_alpha)
                icon_w = ema(prev.w, icon_w, self.ema_alpha)
                icon_h = ema(prev.h, icon_h, self.ema_alpha)
            self.state.track_points[tid] = _ScenePoint(
                x=sx, y=sy, w=icon_w, h=icon_h, last_seen=time.time()
            )

        # Lane label for debug overlays.
        _id, lane_label, lane_center_norm, lane_width_norm = self.lanes.assign_lane_from_boundaries(cx_norm)
        return {
            "track_id": tid,
            "class": str(cls).lower(),
            "score": score,
            "level": level,
            "scene_x": sx,
            "scene_y": sy,
            "icon_w": icon_w,
            "icon_h": icon_h,
            "closeness": closeness,
            "lateral": lateral,
            "lane": lane_id,
            "lane_label": lane_label,
            "lane_center_norm": lane_center_norm,
            "lane_width_norm": lane_width_norm,
            "cx_norm": cx_norm,
            "by_norm": by_norm,
            "bbox": bbox,
            "velocity": velocity,
            "height_ratio": height_ratio,
            "area_ratio": area_ratio,
            "track_quality": track_quality,
            "hidden_reason": None,
        }

    # ---- trails / pruning ---------------------------------------------- #

    def update_trails(self, items: List[Dict[str, Any]]) -> None:
        now = time.time()
        for it in items:
            tid = it["track_id"]
            if tid is None:
                continue
            trail = self.state.track_trails.setdefault(tid, [])
            trail.append((float(it["scene_x"]), float(it["scene_y"]), now))
            if len(trail) > 14:
                del trail[: len(trail) - 14]
        cutoff = now - self.stale_after
        for tid in list(self.state.track_trails.keys()):
            trail = self.state.track_trails[tid]
            if not trail or trail[-1][2] < cutoff:
                self.state.track_trails.pop(tid, None)

    def prune_stale(self) -> None:
        now = time.time()
        for tid in list(self.state.track_points.keys()):
            if now - self.state.track_points[tid].last_seen > self.stale_after:
                self.state.track_points.pop(tid, None)
        self.lanes.prune_tracks(set(self.state.track_points.keys()))
