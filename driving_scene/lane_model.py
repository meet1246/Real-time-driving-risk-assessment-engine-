"""
lane_model.py — North-American-style multi-lane road geometry +
per-object lane-assignment heuristics for the Tesla-Vision scene renderer.

PROBLEM THIS FIXES
==================
The original scene_view placed every object at canvas-center + a raw
lateral offset from its bbox center. That had two bad consequences:

  * The ego vehicle sat on the canvas centerline, which lands directly
    on the road's dashed center divider — making it look like the ego
    is straddling the lane divider rather than driving in a lane.
  * Other vehicles got dropped wherever their image-space x-offset
    pointed, with no notion of which lane they're actually in. Cars
    in your travel direction floated into oncoming traffic; cars
    coming the other way landed wherever the bbox sat.

This module rebuilds the road as a real **multi-lane carriageway**
laid out in NORTH-AMERICAN right-hand-drive convention:

    | sidewalk | oncoming lane(s) || ego lane | right lane | shoulder/sidewalk |
                                    ^^^^^^^^^^^^^^^^^^^^^^^
                                    double-yellow on the LEFT of ego
                                    white dashed lane line on the RIGHT

The ego vehicle sits centered IN ITS OWN LANE, not on the canvas
centerline. The road's double-yellow divider sits to the LEFT of the
ego. Oncoming traffic is drawn LEFT of that divider, NOT in the
ego's lane.

LANE ASSIGNMENT
===============
For each tracked object we decide which lane it belongs in using
heuristics that approximate what a North-American driver would
intuit from a dashcam frame:

  1. Pedestrians + cyclists default to the SIDEWALK on the side of
     the road their bbox-x suggests, UNLESS they are clearly in the
     roadway (low closeness + already over the curb). They migrate
     into the road only when their lateral motion + closeness say
     they have actually stepped into traffic — that's when their
     risk should spike.

  2. Vehicles get assigned to ONE of the carriageway lanes (ego,
     right-adjacent, oncoming-1, oncoming-2) using:
       - sign of their lateral image offset
       - whether their motion suggests they're closing (same
         direction) or approaching (opposite direction)
       - their closeness — a vehicle far ahead and roughly centered
         is in the ego lane; one closing fast from a strong left
         offset is oncoming.

  3. A vehicle is only drawn DIRECTLY in front of the ego when
     it is BOTH (a) close AND (b) lateral-aligned with the ego
     lane. Vehicles passing left or right are kept in their own
     lane and never overlap the ego.

PUBLIC API
==========
    layout = LaneModel(width, height)
    layout.draw_road(canvas)                  # static + lane markings
    lane_id, lane_x = layout.assign_lane(
        bbox=(x1,y1,x2,y2), frame_w=w, frame_h=h,
        cls='car',
        velocity=(vx,vy) or None,             # pixels/sec, optional
        track_id=1 or None,                   # for sticky decisions
    )
    sx, sy = layout.lane_to_scene_xy(lane_id, lane_x, closeness)
    ego_cx, ego_cy = layout.ego_position()

The renderer just calls these instead of computing positions itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple


# ---------------------------------------------------------------------- #
# Lane identifiers — small integer "slots" along the road cross-section. #
# Negative = oncoming (left of double-yellow), 0 = ego lane,             #
# Positive = same direction as ego (right of ego).                       #
# Special values for off-road slots.                                     #
# ---------------------------------------------------------------------- #

LANE_SIDEWALK_LEFT = -99  # far left of the canvas
LANE_ONCOMING_2 = -2  # outer oncoming lane (rare, multi-lane roads)
LANE_ONCOMING_1 = -1  # inner oncoming lane, just left of double-yellow
LANE_EGO = 0  # ego's own travel lane
LANE_RIGHT_1 = 1  # same direction, one lane right of ego
LANE_SIDEWALK_RIGHT = 99  # far right of the canvas


# Label -> internal lane id. Used by assign_lane_from_boundaries() so that
# config-driven labels resolve to the same lane slots the renderer draws.
LABEL_TO_LANE_ID: Dict[str, int] = {
    "far_left": -2,        # LANE_ONCOMING_2 — outer oncoming lane
    "left_lane": -1,       # LANE_ONCOMING_1 — inner oncoming lane
    "ego_lane": 0,         # LANE_EGO
    "right_lane": 1,       # LANE_RIGHT_1 — same-direction right of ego
    "far_right": 1,        # also LANE_RIGHT_1 (we only render one right slot)
}


# Where each lane's CENTER sits as a fraction of canvas width.
LANE_CENTER_FRAC = {
    LANE_SIDEWALK_LEFT: 0.05,
    LANE_ONCOMING_2: 0.16,
    LANE_ONCOMING_1: 0.34,
    LANE_EGO: 0.58,
    LANE_RIGHT_1: 0.82,
    LANE_SIDEWALK_RIGHT: 0.96,
}

# Half-width of each lane as a fraction of canvas width.
LANE_HALF_WIDTH_FRAC = {
    LANE_SIDEWALK_LEFT: 0.04,
    LANE_ONCOMING_2: 0.08,
    LANE_ONCOMING_1: 0.10,
    LANE_EGO: 0.12,
    LANE_RIGHT_1: 0.10,
    LANE_SIDEWALK_RIGHT: 0.04,
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class _TrackHistory:
    """Per-track memory so lane decisions are sticky and don't flicker."""

    last_lane: int = LANE_EGO
    last_closeness: float = 0.0
    last_lateral: float = 0.0
    stickiness: float = 0.0


class LaneModel:
    """
    Owns the road geometry, draws the multi-lane road, and assigns
    each tracked object to a lane.
    """

    def __init__(self, width: int, height: int):
        self.W = int(width)
        self.H = int(height)
        self.horizon_y_frac = 0.32
        self.ego_y_frac = 0.86
        self.horizon_compression = 0.18
        self._history: Dict[int, _TrackHistory] = {}
        self._stickiness_decay = 0.05
        self._max_missing_frames = 30
        # Signed road curvature in [-1, 1]; positive = road bends right.
        # Set every frame by SceneRenderer from LaneEstimator output.
        self._curve: float = 0.0
        self._curve_smoothed: float = 0.0
        # Manual lane boundaries used to bucket bbox cx into a lane label.
        # Mirror the defaults in config.py so the model works standalone, but
        # main.py / qt_app override these via set_boundaries() at startup.
        self._boundaries: List[float] = [0.0, 0.28, 0.43, 0.58, 0.73, 1.0]
        self._labels: List[str] = ["far_left", "left_lane", "ego_lane", "right_lane", "far_right"]

    def set_boundaries(self, boundaries: Sequence[float], labels: Sequence[str]) -> None:
        """Replace the manual lane boundary buckets used by assign_lane_from_boundaries.

        Boundaries are normalized (0..1) cx cut-points; labels name each bucket.
        len(labels) must be len(boundaries) - 1.
        """
        if len(boundaries) < 2 or len(labels) != len(boundaries) - 1:
            return
        self._boundaries = [float(b) for b in boundaries]
        self._labels = [str(s) for s in labels]

    def get_boundaries(self) -> Tuple[List[float], List[str]]:
        return list(self._boundaries), list(self._labels)

    def set_curve(self, curve_strength: float, smooth_alpha: float = 0.18) -> None:
        c = max(-1.0, min(1.0, float(curve_strength)))
        self._curve_smoothed = smooth_alpha * c + (1.0 - smooth_alpha) * self._curve_smoothed
        self._curve = self._curve_smoothed

    def horizon_y(self) -> float:
        return self.H * self.horizon_y_frac

    def ego_y(self) -> float:
        return self.H * self.ego_y_frac

    def ego_position(self) -> Tuple[float, float]:
        return (self.W * LANE_CENTER_FRAC[LANE_EGO], self.ego_y() + 18)

    def lane_center_x_at_y(self, lane: int, y: float) -> float:
        bottom_x = self.W * LANE_CENTER_FRAC.get(lane, 0.5)
        vp_x_base = self.W * 0.5 * (LANE_CENTER_FRAC[LANE_EGO] + LANE_CENTER_FRAC[LANE_ONCOMING_1])
        # Shift the vanishing point laterally with detected road curvature
        # so the whole carriageway visibly bends with the real road.
        vp_x = vp_x_base + self._curve * self.W * 0.32
        t = self._perspective_t(y)
        straight = _lerp(vp_x, bottom_x, t)
        # Add a parabolic bow so the road bends smoothly (not just a wedge).
        bow = (1.0 - t) ** 2 * self._curve * self.W * 0.10
        return straight + bow

    def lane_half_width_at_y(self, lane: int, y: float) -> float:
        bottom = self.W * LANE_HALF_WIDTH_FRAC.get(lane, 0.08)
        t = self._perspective_t(y)
        return _lerp(bottom * self.horizon_compression, bottom, t)

    def lane_to_scene_xy(self, lane: int, lane_offset: float, closeness: float) -> Tuple[float, float]:
        hy = self.horizon_y()
        ey = self.ego_y()
        sy = _lerp(hy + 6.0, ey - 28.0, max(0.0, min(1.0, closeness)))
        center_x = self.lane_center_x_at_y(lane, sy)
        half_w = self.lane_half_width_at_y(lane, sy)
        sx = center_x + lane_offset * half_w
        return sx, sy

    def _perspective_t(self, y: float) -> float:
        hy = self.horizon_y()
        ey = self.ego_y()
        return max(0.0, min(1.0, (y - hy) / max(1.0, ey - hy)))

    def draw_road(self, canvas, palette: Dict[str, tuple]) -> None:
        import cv2
        import numpy as np

        hy = int(self.horizon_y())
        ey = int(self.ego_y() + 40)

        left_x_top = self.lane_center_x_at_y(LANE_ONCOMING_2, hy) - self.lane_half_width_at_y(LANE_ONCOMING_2, hy)
        right_x_top = self.lane_center_x_at_y(LANE_RIGHT_1, hy) + self.lane_half_width_at_y(LANE_RIGHT_1, hy)
        left_x_bot = self.lane_center_x_at_y(LANE_ONCOMING_2, ey) - self.lane_half_width_at_y(LANE_ONCOMING_2, ey)
        right_x_bot = self.lane_center_x_at_y(LANE_RIGHT_1, ey) + self.lane_half_width_at_y(LANE_RIGHT_1, ey)

        road_poly = np.array(
            [
                [int(left_x_top), hy],
                [int(right_x_top), hy],
                [int(right_x_bot), ey],
                [int(left_x_bot), ey],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [road_poly], palette["road"])

        sidewalk_color = (52, 54, 60)
        for sw_lane in (LANE_SIDEWALK_LEFT, LANE_SIDEWALK_RIGHT):
            xt = self.lane_center_x_at_y(sw_lane, hy)
            ht = self.lane_half_width_at_y(sw_lane, hy)
            xb = self.lane_center_x_at_y(sw_lane, ey)
            hb = self.lane_half_width_at_y(sw_lane, ey)
            poly = np.array(
                [
                    [int(xt - ht), hy],
                    [int(xt + ht), hy],
                    [int(xb + hb), ey],
                    [int(xb - hb), ey],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(canvas, [poly], sidewalk_color)

        self._draw_perspective_line(
            canvas,
            self.lane_center_x_at_y(LANE_ONCOMING_2, hy) - self.lane_half_width_at_y(LANE_ONCOMING_2, hy),
            hy,
            self.lane_center_x_at_y(LANE_ONCOMING_2, ey) - self.lane_half_width_at_y(LANE_ONCOMING_2, ey),
            ey,
            palette["lane_solid"],
            thickness_top=1,
            thickness_bot=3,
        )
        self._draw_perspective_line(
            canvas,
            self.lane_center_x_at_y(LANE_RIGHT_1, hy) + self.lane_half_width_at_y(LANE_RIGHT_1, hy),
            hy,
            self.lane_center_x_at_y(LANE_RIGHT_1, ey) + self.lane_half_width_at_y(LANE_RIGHT_1, ey),
            ey,
            palette["lane_solid"],
            thickness_top=1,
            thickness_bot=3,
        )

        self._draw_dashed_perspective_line(
            canvas,
            self.lane_center_x_at_y(LANE_ONCOMING_2, hy) + self.lane_half_width_at_y(LANE_ONCOMING_2, hy),
            hy,
            self.lane_center_x_at_y(LANE_ONCOMING_2, ey) + self.lane_half_width_at_y(LANE_ONCOMING_2, ey),
            ey,
            palette["lane_solid"],
        )

        self._draw_dashed_perspective_line(
            canvas,
            self.lane_center_x_at_y(LANE_EGO, hy) + self.lane_half_width_at_y(LANE_EGO, hy),
            hy,
            self.lane_center_x_at_y(LANE_EGO, ey) + self.lane_half_width_at_y(LANE_EGO, ey),
            ey,
            palette["lane_solid"],
        )

        boundary_top = self.lane_center_x_at_y(LANE_ONCOMING_1, hy) + self.lane_half_width_at_y(LANE_ONCOMING_1, hy)
        boundary_bot = self.lane_center_x_at_y(LANE_ONCOMING_1, ey) + self.lane_half_width_at_y(LANE_ONCOMING_1, ey)
        yellow = palette.get("lane_yellow", (40, 200, 230))
        for offset in (-3, +3):
            self._draw_perspective_line(
                canvas,
                boundary_top + offset,
                hy,
                boundary_bot + offset,
                ey,
                yellow,
                thickness_top=1,
                thickness_bot=3,
            )

    def draw_animated_markings(self, canvas, palette, lane_phase: float) -> None:
        for line_top, line_bot in (
            (
                self.lane_center_x_at_y(LANE_EGO, self.horizon_y()) + self.lane_half_width_at_y(LANE_EGO, self.horizon_y()),
                self.lane_center_x_at_y(LANE_EGO, self.ego_y() + 40) + self.lane_half_width_at_y(LANE_EGO, self.ego_y() + 40),
            ),
            (
                self.lane_center_x_at_y(LANE_ONCOMING_2, self.horizon_y()) + self.lane_half_width_at_y(LANE_ONCOMING_2, self.horizon_y()),
                self.lane_center_x_at_y(LANE_ONCOMING_2, self.ego_y() + 40) + self.lane_half_width_at_y(LANE_ONCOMING_2, self.ego_y() + 40),
            ),
        ):
            self._draw_animated_dashes(
                canvas,
                line_top,
                self.horizon_y(),
                line_bot,
                self.ego_y() + 40,
                palette["lane_dash"],
                palette.get("lane_glow", (255, 200, 120)),
                lane_phase,
            )

    def assign_lane(
        self,
        bbox: Sequence[float],
        frame_w: int,
        frame_h: int,
        cls: str,
        velocity: Optional[Tuple[float, float]] = None,
        track_id: Optional[int] = None,
    ) -> Tuple[int, float]:
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        by = y2
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        lateral = (cx - frame_w * 0.5) / (frame_w * 0.5 + 1e-6)
        closeness = (
            0.55 * (by / max(1.0, frame_h))
            + 0.20 * min(1.0, (bw * bh) / max(1.0, frame_w * frame_h) * 6.0)
            + 0.15 * (cy / max(1.0, frame_h))
            + 0.10 * min(1.0, bh / max(1.0, frame_h) * 2.2)
        )
        closeness = max(0.0, min(1.0, closeness))

        cls_l = (cls or "").lower()
        cx_norm = cx / max(1.0, float(frame_w))

        if cls_l in ("person", "pedestrian", "bicycle", "bike"):
            lane = self._assign_vru_lane(lateral, closeness, velocity, cls_l)
            offset = self._vru_lane_offset(lane, cls_l)
            return self._apply_stickiness(track_id, lane, offset, closeness, lateral)

        lane = self._assign_vehicle_lane(cx_norm, closeness, velocity)
        # In-lane offset: how far from the lane's centerline the object is,
        # clamped to ±0.35 so cars don't crowd the lane edge in the scene.
        _id, _label, lane_center, lane_width = self.assign_lane_from_boundaries(cx_norm)
        offset = (cx_norm - lane_center) / max(1e-3, lane_width * 0.5)
        offset = max(-0.35, min(0.35, offset))
        return self._apply_stickiness(track_id, lane, offset, closeness, lateral)

    def _assign_vru_lane(
        self,
        lateral: float,
        closeness: float,
        velocity: Optional[Tuple[float, float]],
        cls_l: str,
    ) -> int:
        on_right = lateral >= -0.05
        default_lane = LANE_SIDEWALK_RIGHT if on_right else LANE_SIDEWALK_LEFT

        if cls_l in ("bicycle", "bike"):
            if lateral < -0.55 and closeness < 0.45:
                return LANE_SIDEWALK_LEFT
            return LANE_RIGHT_1

        in_road_static = closeness > 0.55 and abs(lateral) < 0.45
        moving_into_road = False
        if velocity is not None:
            vx, _ = velocity
            moving_into_road = (
                0.10 < abs(lateral) < 0.45
                and abs(vx) > 30.0
                and ((lateral < 0 and vx > 0) or (lateral > 0 and vx < 0))
            )
        if in_road_static or moving_into_road:
            if lateral < -0.25:
                return LANE_ONCOMING_1
            if lateral > 0.25:
                return LANE_RIGHT_1
            return LANE_EGO

        return default_lane

    def _vru_lane_offset(self, lane: int, cls_l: str) -> float:
        if cls_l in ("bicycle", "bike") and lane == LANE_RIGHT_1:
            return 0.6
        return 0.0

    def assign_lane_from_boundaries(
        self,
        cx_norm: float,
    ) -> Tuple[int, str, float, float]:
        """Bucket cx_norm (in [0, 1]) into one of the configured lanes.

        Returns (lane_id, lane_label, lane_center_norm, lane_width_norm).
        This is the authoritative lane assignment — replaces the older
        lateral-only heuristic so cars on the right of the dashcam actually
        end up in the right lane on the scene, not pulled back to center.
        """
        b = self._boundaries
        lbls = self._labels
        cx = max(0.0, min(1.0, float(cx_norm)))
        n = len(b) - 1
        # Linear scan — n is small (5), no need for bisect.
        for i in range(n):
            if cx <= b[i + 1] or i == n - 1:
                label = lbls[i] if i < len(lbls) else "ego_lane"
                lane_id = LABEL_TO_LANE_ID.get(label, LANE_EGO)
                lane_center = 0.5 * (b[i] + b[i + 1])
                lane_width = max(1e-3, b[i + 1] - b[i])
                return lane_id, label, lane_center, lane_width
        return LANE_EGO, "ego_lane", 0.5, 0.15

    def _assign_vehicle_lane(
        self,
        cx_norm: float,
        closeness: float,
        velocity: Optional[Tuple[float, float]],
    ) -> int:
        """Vehicle lane is now derived purely from the configured boundaries.

        We deliberately ignore velocity here — for steady-state vehicles,
        position alone is the right signal. The track-stickiness layer
        downstream prevents single-frame jitter from flipping lanes.
        """
        lane_id, _label, _center, _width = self.assign_lane_from_boundaries(cx_norm)
        return lane_id

    def _apply_stickiness(
        self,
        track_id: Optional[int],
        new_lane: int,
        offset: float,
        closeness: float,
        lateral: float,
    ) -> Tuple[int, float]:
        if track_id is None:
            return new_lane, offset

        h = self._history.get(track_id)
        if h is None:
            self._history[track_id] = _TrackHistory(
                last_lane=new_lane,
                last_closeness=closeness,
                last_lateral=lateral,
                stickiness=0.6,
            )
            return new_lane, offset

        if new_lane == h.last_lane:
            h.stickiness = min(1.0, h.stickiness + 0.15)
            h.last_closeness = closeness
            h.last_lateral = lateral
            return new_lane, offset

        switch_threshold = 0.55 - 0.35 * closeness
        if h.stickiness < switch_threshold:
            h.last_lane = new_lane
            h.stickiness = 0.5
            h.last_closeness = closeness
            h.last_lateral = lateral
            return new_lane, offset
        h.stickiness = max(0.0, h.stickiness - self._stickiness_decay)
        return h.last_lane, offset

    def prune_tracks(self, active_ids: set) -> None:
        for tid in list(self._history.keys()):
            if tid not in active_ids:
                self._history.pop(tid, None)

    def reset(self) -> None:
        self._history.clear()

    def _draw_perspective_line(self, canvas, x_top, y_top, x_bot, y_bot, color, thickness_top=1, thickness_bot=3):
        import cv2

        steps = 6
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 1) / steps
            x0 = int(_lerp(x_top, x_bot, t0))
            y0 = int(_lerp(y_top, y_bot, t0))
            x1 = int(_lerp(x_top, x_bot, t1))
            y1 = int(_lerp(y_top, y_bot, t1))
            thick = max(1, int(_lerp(thickness_top, thickness_bot, (t0 + t1) * 0.5)))
            cv2.line(canvas, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)

    def _draw_dashed_perspective_line(self, canvas, x_top, y_top, x_bot, y_bot, color, n_dashes=16):
        import cv2

        for i in range(n_dashes):
            t0 = i / n_dashes
            t1 = t0 + (0.5 / n_dashes)
            x0 = int(_lerp(x_top, x_bot, t0))
            y0 = int(_lerp(y_top, y_bot, t0))
            x1 = int(_lerp(x_top, x_bot, t1))
            y1 = int(_lerp(y_top, y_bot, t1))
            thick = max(1, int(_lerp(1, 3, (t0 + t1) * 0.5)))
            cv2.line(canvas, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)

    def _draw_animated_dashes(self, canvas, x_top, y_top, x_bot, y_bot, color, glow_color, phase):
        import cv2

        n_dashes = 14
        for i in range(n_dashes):
            t0 = ((i / n_dashes) + phase) % 1.0
            t1 = t0 + (0.40 / n_dashes)
            if t1 > 1.0:
                continue
            x0 = int(_lerp(x_top, x_bot, t0))
            y0 = int(_lerp(y_top, y_bot, t0))
            x1 = int(_lerp(x_top, x_bot, t1))
            y1 = int(_lerp(y_top, y_bot, t1))
            thick = max(2, int(_lerp(2, 6, (t0 + t1) * 0.5)))
            cv2.line(canvas, (x0, y0), (x1, y1), glow_color, thick + 4, cv2.LINE_AA)
            cv2.line(canvas, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)

