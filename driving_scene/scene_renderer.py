"""
scene_renderer.py — Tesla-Inspired Driving Scene Visualizer

Visual layer. Composes the road plate, lanes, ego car, mini 3D vehicles,
pedestrians, planned path, motion trails, glow, etc. The HUD is delegated to
hud.py and mapping/smoothing is delegated to scene_mapper.SceneMapper.

ENTRY POINT
===========
    SceneRenderer(width, height).render(
        frame_bgr, detections, tracks, telemetry, risk_state,
        ego_speed_kmh=None, debug=False, view_mode='scene',
        primary_threat_id=None, speed_limit_kmh=None,
        lane_curve=None, lane_confidence=None,
    ) -> np.ndarray

VIEW MODES
==========
    scene    full Tesla-inspired scene (default)
    dashcam  raw frame + bounding boxes + classic HUD
    split    dashcam (left, no boxes) + scene (right)
    debug    raw frame + lane boundary lines + per-bbox mapping info

The renderer caches the heavy parts (gradient + vignette) so the curved road
re-rendered each frame stays cheap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from . import assets as _assets
from . import hud
from . import lane_model as _lm
from .scene_mapper import SceneMapper
from .utils import (
    BIKE_CLASSES,
    PALETTE,
    PED_CLASSES,
    darken,
    ema,
    lerp,
    lighten,
    risk_color_for,
    safe_get,
    tint_vehicle,
)


@dataclass
class _RendererState:
    debug: bool = False


class SceneRenderer:
    """Compose a Tesla-inspired driving scene from detections + tracks."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.W = int(width)
        self.H = int(height)
        self.state = _RendererState()
        # Heavy gradient + vignette baked once; only the cheap road geometry
        # is re-drawn per frame.
        self._bg_cache: Optional[np.ndarray] = None
        self._bg_cache_size: Tuple[int, int] = (-1, -1)
        # Lane geometry + per-track scene mapping/smoothing.
        self.lanes = _lm.LaneModel(self.W, self.H)
        self.mapper = SceneMapper(self.lanes)
        # Cached "manual" lane boundaries for the debug overlay; main.py can
        # update this list at runtime via the [, ], `,`, `.` keyboard tuning.
        self.lane_boundaries: Optional[List[float]] = None
        self.lane_labels: Optional[List[str]] = None

    # ---- public API ---------------------------------------------------- #

    def toggle_debug(self) -> None:
        self.state.debug = not self.state.debug

    def reset(self) -> None:
        self.state = _RendererState()
        self.mapper.reset()

    def set_lane_boundaries(self, boundaries: List[float], labels: List[str]) -> None:
        """Push the manual lane boundaries into the LaneModel (authoritative
        lane assignment) AND cache them for the debug overlay.
        """
        self.lane_boundaries = list(boundaries)
        self.lane_labels = list(labels)
        self.lanes.set_boundaries(boundaries, labels)

    def render(
        self,
        frame_bgr: Optional[np.ndarray],
        detections: Optional[Iterable[Any]] = None,
        tracks: Optional[Iterable[Any]] = None,
        telemetry: Optional[Any] = None,
        risk_state: Optional[Any] = None,
        ego_speed_kmh: Optional[float] = None,
        debug: Optional[bool] = None,
        view_mode: str = "scene",
        primary_threat_id: Optional[int] = None,
        speed_limit_kmh: Optional[float] = None,
        lane_curve: Optional[float] = None,
        lane_confidence: Optional[float] = None,
    ) -> np.ndarray:
        if debug is not None:
            self.state.debug = bool(debug)

        # Advance lane-dash animation phase based on real wall clock.
        now = time.time()
        dt = 0.016 if self.mapper.state.last_render_t == 0.0 else (now - self.mapper.state.last_render_t)
        dt = max(0.001, min(0.1, dt))
        self.mapper.state.last_render_t = now
        self.mapper.advance(dt, ego_speed_kmh)

        if lane_curve is not None:
            self.mapper.set_curve(float(lane_curve))
        self.mapper.set_lane_confidence(lane_confidence)

        view_mode = (view_mode or "scene").lower()

        if view_mode == "dashcam":
            if frame_bgr is None:
                return self._blank_canvas()
            canvas = self._resize_to_canvas(frame_bgr)
            fw, fh = canvas.shape[1], canvas.shape[0]
            items = self.mapper.collect_renderables(tracks, detections, fw, fh)
            primary = getattr(risk_state, "primary_track_id", None) if risk_state else None
            hud.draw_classic_overlays(canvas, items, primary)
            hud.draw_classic_hud(canvas, telemetry, risk_state, ego_speed_kmh)
            return canvas

        if view_mode == "debug":
            if frame_bgr is None:
                return self._blank_canvas()
            canvas = self._resize_to_canvas(frame_bgr)
            fw, fh = canvas.shape[1], canvas.shape[0]
            items = self.mapper.collect_renderables(tracks, detections, fw, fh)
            hud.draw_debug_overlay(canvas, items, self.lane_boundaries, self.lane_labels)
            hud.draw_classic_hud(canvas, telemetry, risk_state, ego_speed_kmh)
            return canvas

        scene = self._render_scene(
            frame_bgr=frame_bgr,
            tracks=tracks,
            detections=detections,
            telemetry=telemetry,
            risk_state=risk_state,
            ego_speed_kmh=ego_speed_kmh,
            primary_threat_id=primary_threat_id,
            speed_limit_kmh=speed_limit_kmh,
        )

        if view_mode == "split" and frame_bgr is not None:
            return hud.compose_split(frame_bgr, scene, self.W, self.H)
        return scene

    # ---- scene render -------------------------------------------------- #

    def _render_scene(
        self,
        frame_bgr: Optional[np.ndarray],
        tracks: Optional[Iterable[Any]],
        detections: Optional[Iterable[Any]],
        telemetry: Optional[Any],
        risk_state: Optional[Any],
        ego_speed_kmh: Optional[float],
        primary_threat_id: Optional[int],
        speed_limit_kmh: Optional[float],
    ) -> np.ndarray:
        canvas = self._make_road_plate()
        if frame_bgr is not None:
            fh, fw = frame_bgr.shape[:2]
        else:
            fh, fw = self.H, self.W

        if primary_threat_id is None and risk_state is not None:
            primary_threat_id = getattr(risk_state, "primary_track_id", None)

        items = self.mapper.collect_renderables(tracks, detections, fw, fh)
        items.sort(key=lambda it: it["closeness"])
        self.mapper.update_trails(items)

        self._draw_planned_path(canvas)
        self._draw_trails(canvas)
        for it in items:
            is_primary = it["track_id"] is not None and it["track_id"] == primary_threat_id
            self._draw_scene_object(canvas, it, is_primary=is_primary)
            self._draw_trajectory_arc(canvas, it, fw, fh)
        self.mapper.prune_stale()
        self._draw_ego_vehicle(canvas)

        hud.draw_top_hud(canvas, self.W, self.H, telemetry, risk_state,
                         ego_speed_kmh, primary_threat_id, len(items))
        hud.draw_side_panels(canvas, self.W, self.H, risk_state, ego_speed_kmh,
                             speed_limit_kmh, items, primary_threat_id)
        hud.draw_action_banner(canvas, self.W, self.H, risk_state)
        hud.draw_footer_disclaimer(canvas, self.W, self.H)
        if self.state.debug:
            hud.draw_debug_panel(canvas, self.W, self.H, telemetry, len(items))
        return canvas

    # ---- road plate (background + road + perspective grid) ------------ #

    def _make_road_plate(self) -> np.ndarray:
        if self._bg_cache is None or self._bg_cache_size != (self.W, self.H):
            self._bg_cache = self._build_static_bg()
            self._bg_cache_size = (self.W, self.H)
        canvas = self._bg_cache.copy()
        horizon_y = int(self.lanes.horizon_y())
        cv2.rectangle(canvas, (0, horizon_y - 14), (self.W, horizon_y + 14),
                      PALETTE["horizon"], thickness=cv2.FILLED)
        ego_y = int(self.lanes.ego_y())
        self.lanes.draw_road(canvas, PALETTE)
        top_half = int(
            abs(
                self.lanes.lane_center_x_at_y(_lm.LANE_RIGHT_1, horizon_y)
                - self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_2, horizon_y)
            )
            * 0.5
            + self.lanes.lane_half_width_at_y(_lm.LANE_RIGHT_1, horizon_y)
        )
        bot_half = int(
            abs(
                self.lanes.lane_center_x_at_y(_lm.LANE_RIGHT_1, ego_y)
                - self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_2, ego_y)
            )
            * 0.5
            + self.lanes.lane_half_width_at_y(_lm.LANE_RIGHT_1, ego_y)
        )
        cx = int(
            0.5
            * (
                self.lanes.lane_center_x_at_y(_lm.LANE_EGO, ego_y)
                + self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_1, ego_y)
            )
        )
        self._draw_perspective_grid(canvas, horizon_y, ego_y, top_half, bot_half, cx)
        self.lanes.draw_animated_markings(canvas, PALETTE, self.mapper.lane_phase)
        return canvas

    def _build_static_bg(self) -> np.ndarray:
        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        top = np.array(PALETTE["bg_top"], dtype=np.float32)
        bot = np.array(PALETTE["bg_bottom"], dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.H, dtype=np.float32)[:, None]
        gradient = top * (1.0 - ys) + bot * ys
        canvas[:] = gradient[:, None, :].astype(np.uint8).repeat(self.W, axis=1)
        self._apply_vignette(canvas)
        return canvas

    def _apply_vignette(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = h * 0.55, w * 0.5
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        d /= d.max() + 1e-6
        mask = np.clip(1.0 - 0.55 * (d ** 2), 0.45, 1.0)
        canvas[:] = (canvas.astype(np.float32) * mask[..., None]).astype(np.uint8)

    def _draw_perspective_grid(self, canvas, horizon_y, ego_y, top_half, bot_half, cx):
        for i in range(1, 7):
            t = i / 7.0
            y = int(lerp(horizon_y, ego_y + 40, t))
            half = int(lerp(top_half, bot_half, t))
            color = (
                int(lerp(PALETTE["road_edge"][0], 80, 0.3)),
                int(lerp(PALETTE["road_edge"][1], 80, 0.3)),
                int(lerp(PALETTE["road_edge"][2], 90, 0.3)),
            )
            cv2.line(canvas, (cx - half, y), (cx + half, y), color, 1, cv2.LINE_AA)
        for frac in (-0.66, -0.22, 0.22, 0.66):
            x_top = int(cx + frac * top_half)
            x_bot = int(cx + frac * bot_half)
            cv2.line(canvas, (x_top, horizon_y), (x_bot, ego_y + 40), (50, 52, 58), 1, cv2.LINE_AA)

    # ---- ego car ------------------------------------------------------- #

    def _draw_ego_vehicle(self, canvas: np.ndarray) -> None:
        ego_cx_f, ego_cy_f = self.lanes.ego_position()
        cx = int(ego_cx_f)
        cy = int(ego_cy_f)
        w = int(self.W * 0.07)
        h = int(self.H * 0.075)
        # Headlight cone.
        cone = canvas.copy()
        pts = np.array([
            [cx - int(w * 1.1), cy - int(h * 4)],
            [cx + int(w * 1.1), cy - int(h * 4)],
            [cx + int(w * 0.45), cy - int(h * 0.55)],
            [cx - int(w * 0.45), cy - int(h * 0.55)],
        ], dtype=np.int32)
        cv2.fillPoly(cone, [pts], PALETTE["ego_beam"])
        cv2.addWeighted(cone, 0.10, canvas, 0.90, 0, dst=canvas)
        # Body / roof / glass.
        body_color = PALETTE["ego"]
        outline = PALETTE["ego_outline"]
        roof_color = (220, 225, 230)
        glass = (140, 170, 200)
        body = np.array([
            [cx - w, cy + int(h * 0.45)],
            [cx - int(w * 0.9), cy - int(h * 0.5)],
            [cx + int(w * 0.9), cy - int(h * 0.5)],
            [cx + w, cy + int(h * 0.45)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [body], body_color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [body], True, outline, 2, cv2.LINE_AA)
        roof = np.array([
            [cx - int(w * 0.55), cy + int(h * 0.15)],
            [cx - int(w * 0.5), cy - int(h * 0.25)],
            [cx + int(w * 0.5), cy - int(h * 0.25)],
            [cx + int(w * 0.55), cy + int(h * 0.15)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [roof], roof_color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [roof], True, outline, 1, cv2.LINE_AA)
        wind = np.array([
            [cx - int(w * 0.45), cy - int(h * 0.05)],
            [cx - int(w * 0.42), cy - int(h * 0.22)],
            [cx + int(w * 0.42), cy - int(h * 0.22)],
            [cx + int(w * 0.45), cy - int(h * 0.05)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [wind], glass, lineType=cv2.LINE_AA)
        # Headlights.
        cv2.circle(canvas, (cx - int(w * 0.7), cy - int(h * 0.45)), 3, (255, 240, 200), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx + int(w * 0.7), cy - int(h * 0.45)), 3, (255, 240, 200), -1, cv2.LINE_AA)

    # ---- planned path / trails / trajectory arcs ---------------------- #

    def _draw_planned_path(self, canvas: np.ndarray) -> None:
        ego_x, ego_y = self.lanes.ego_position()
        horizon_y = self.lanes.horizon_y()
        n = 18
        pts: List[Tuple[int, int]] = []
        for i in range(n + 1):
            t = i / n
            y = lerp(ego_y - 30, horizon_y + 12, t)
            x = self.lanes.lane_center_x_at_y(0, y)
            pts.append((int(x), int(y)))
        if len(pts) < 2:
            return
        conf = self.mapper.lane_confidence
        conf_mul = 1.0 if conf is None else max(0.25, min(1.0, conf))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 18
        x0 = max(0, min(xs) - pad)
        x1 = min(self.W, max(xs) + pad)
        y0 = max(0, min(ys) - pad)
        y1 = min(self.H, max(ys) + pad)
        if x1 > x0 and y1 > y0:
            roi = canvas[y0:y1, x0:x1]
            overlay = roi.copy()
            for i in range(1, len(pts)):
                depth_t = i / float(len(pts))
                base_thick = int(lerp(14, 4, depth_t))
                cv2.line(
                    overlay,
                    (pts[i - 1][0] - x0, pts[i - 1][1] - y0),
                    (pts[i][0] - x0, pts[i][1] - y0),
                    PALETTE["planned_path_glow"],
                    base_thick,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(overlay, 0.22 * conf_mul, roi, 1 - 0.22 * conf_mul, 0, dst=roi)
        for i in range(1, len(pts)):
            depth_t = i / float(len(pts))
            thick = max(2, int(lerp(9, 2, depth_t)))
            cv2.line(canvas, pts[i - 1], pts[i], PALETTE["planned_path"], thick, cv2.LINE_AA)
        cv2.circle(canvas, pts[-1], 3, PALETTE["planned_path"], -1, cv2.LINE_AA)

    def _draw_trails(self, canvas: np.ndarray) -> None:
        for tid, trail in self.mapper.track_trails.items():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                a = trail[i - 1]
                b = trail[i]
                age_t = i / float(len(trail))
                alpha = age_t ** 1.4
                color = PALETTE["trail"]
                bg = PALETTE["road"]
                c = (
                    int(bg[0] + (color[0] - bg[0]) * alpha),
                    int(bg[1] + (color[1] - bg[1]) * alpha),
                    int(bg[2] + (color[2] - bg[2]) * alpha),
                )
                thick = max(1, int(1 + 2 * age_t))
                cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, thick, cv2.LINE_AA)

    def _draw_trajectory_arc(self, canvas: np.ndarray, item: Dict[str, Any], fw: int, fh: int) -> None:
        velocity = item.get("velocity")
        if velocity is None:
            return
        try:
            vx, vy = float(velocity[0]), float(velocity[1])
        except Exception:
            return
        if item["class"] in PED_CLASSES or item["class"] in BIKE_CLASSES:
            return
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < 12.0:
            return
        bbox = item.get("bbox")
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        px = cx + vx * 1.0
        py = cy + vy * 1.0
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pred_bbox = (px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2)
        from .scene_mapper import image_bbox_to_scene_plane
        _, closeness_p, _ = image_bbox_to_scene_plane(pred_bbox, fw, fh)
        sx_a = int(item["scene_x"])
        sy_a = int(item["scene_y"])
        lane_id = item.get("lane", 0)
        sx_b, sy_b = self.lanes.lane_to_scene_xy(lane_id, 0.0, closeness_p)
        color = risk_color_for(item.get("level"), item.get("score"))
        cv2.line(canvas, (sx_a, sy_a), (int(sx_b), int(sy_b)), color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(sx_b), int(sy_b)), 3, color, -1, cv2.LINE_AA)

    # ---- per-object icons --------------------------------------------- #

    def _draw_scene_object(self, canvas: np.ndarray, item: Dict[str, Any], is_primary: bool) -> None:
        cls = item["class"]
        sx = int(item["scene_x"])
        sy = int(item["scene_y"])
        w = max(8, int(item["icon_w"]))
        h = max(6, int(item["icon_h"]))
        risk_col = risk_color_for(item["level"], item["score"])
        lane_id = item.get("lane", 0)
        oncoming = isinstance(lane_id, int) and lane_id < 0
        tid = item.get("track_id")

        # Soft shadow underneath every object so the icon "sits" on the road
        # instead of floating. Slightly larger for vehicles than peds.
        shadow_w = w * (1.1 if cls in PED_CLASSES else 1.05)
        shadow_h = h * (0.16 if cls in PED_CLASSES else 0.22)
        self._draw_soft_shadow(canvas, sx, sy + h * 0.45, shadow_w, shadow_h)

        if cls in PED_CLASSES:
            self._draw_ped_icon(canvas, sx, sy, h, risk_col, is_primary)
        elif cls in BIKE_CLASSES:
            self._draw_bike_icon(canvas, sx, sy, w, h, risk_col, is_primary)
        elif cls == "bus":
            self._draw_vehicle_icon(canvas, sx, sy, w, int(h * 1.45), risk_col, is_primary, kind="bus", oncoming=oncoming, track_id=tid)
        elif cls == "truck":
            self._draw_vehicle_icon(canvas, sx, sy, w, int(h * 1.25), risk_col, is_primary, kind="truck", oncoming=oncoming, track_id=tid)
        elif cls in ("motorcycle", "motorbike"):
            self._draw_vehicle_icon(canvas, sx, sy, int(w * 0.55), h, risk_col, is_primary, kind="moto", oncoming=oncoming, track_id=tid)
        else:
            self._draw_vehicle_icon(canvas, sx, sy, w, h, risk_col, is_primary, kind="sedan", oncoming=oncoming, track_id=tid)

    def _draw_soft_shadow(self, canvas, cx, cy, w, h, alpha: float = 0.35) -> None:
        """Soft elliptical shadow beneath an object. ROI-blended for speed."""
        rx, ry = max(4.0, w * 0.5), max(2.0, h * 0.5)
        x0 = max(0, int(cx - rx - 2))
        x1 = min(canvas.shape[1], int(cx + rx + 2))
        y0 = max(0, int(cy - ry - 2))
        y1 = min(canvas.shape[0], int(cy + ry + 2))
        if x1 <= x0 or y1 <= y0:
            return
        roi = canvas[y0:y1, x0:x1]
        overlay = roi.copy()
        cv2.ellipse(overlay, (int(cx - x0), int(cy - y0)), (int(rx), int(ry)),
                    0, 0, 360, (8, 8, 10), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

    @staticmethod
    def _rounded_rect_poly(cx, cy, w, h, r, n_corner: int = 5) -> np.ndarray:
        """Polygon vertices approximating a rounded rectangle (used for car body).

        Returns an int32 (N, 2) array suitable for cv2.fillPoly / polylines.
        """
        r = float(min(r, w * 0.5, h * 0.5))
        hx = w * 0.5
        hy = h * 0.5
        pts = []
        # Top-left corner arc: center (cx-hx+r, cy-hy+r), angles 180..270
        for i in range(n_corner + 1):
            a = np.pi + i * (np.pi * 0.5) / n_corner
            pts.append((cx - hx + r + r * np.cos(a), cy - hy + r + r * np.sin(a)))
        # Top-right
        for i in range(n_corner + 1):
            a = -np.pi * 0.5 + i * (np.pi * 0.5) / n_corner
            pts.append((cx + hx - r + r * np.cos(a), cy - hy + r + r * np.sin(a)))
        # Bottom-right
        for i in range(n_corner + 1):
            a = 0.0 + i * (np.pi * 0.5) / n_corner
            pts.append((cx + hx - r + r * np.cos(a), cy + hy - r + r * np.sin(a)))
        # Bottom-left
        for i in range(n_corner + 1):
            a = np.pi * 0.5 + i * (np.pi * 0.5) / n_corner
            pts.append((cx - hx + r + r * np.cos(a), cy + hy - r + r * np.sin(a)))
        return np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)

    @staticmethod
    def _fill_vertical_gradient(canvas, poly, top_color, bot_color) -> None:
        """Fill a polygon with a vertical color gradient (top -> bot)."""
        ys = poly[:, 1]
        y_top, y_bot = int(ys.min()), int(ys.max())
        if y_bot <= y_top:
            cv2.fillPoly(canvas, [poly], tuple(int(c) for c in top_color), lineType=cv2.LINE_AA)
            return
        # Render the gradient into a tight ROI, then mask-blit through the
        # polygon. This is cheaper than per-row fillPoly.
        x_top = int(poly[:, 0].min())
        x_bot_lim = int(poly[:, 0].max()) + 1
        h = y_bot - y_top + 1
        w = x_bot_lim - x_top + 1
        if h <= 0 or w <= 0:
            return
        # Per-row interpolated color band.
        ts = np.linspace(0.0, 1.0, h, dtype=np.float32)
        row = (
            np.asarray(top_color, dtype=np.float32)[None, :] * (1.0 - ts[:, None])
            + np.asarray(bot_color, dtype=np.float32)[None, :] * ts[:, None]
        )
        band = np.broadcast_to(row[:, None, :], (h, w, 3)).astype(np.uint8).copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - np.array([x_top, y_top])], 255, lineType=cv2.LINE_AA)
        roi = canvas[y_top:y_top + h, x_top:x_top + w]
        # Apply mask: roi = band where mask, else roi.
        m3 = (mask.astype(np.float32) / 255.0)[..., None]
        out = roi.astype(np.float32) * (1.0 - m3) + band.astype(np.float32) * m3
        canvas[y_top:y_top + h, x_top:x_top + w] = out.astype(np.uint8)

    def _draw_vehicle_icon(self, canvas, sx, sy, w, h, color, is_primary, kind="sedan",
                            oncoming=False, track_id: Optional[int] = None) -> None:
        # Risk glow (only visually meaningful when risk is med/high — for LOW
        # risk it's a barely-visible halo).
        self._draw_glow(canvas, sx, sy, w, h, color, primary=is_primary)
        # Stable body color per track so a car keeps its identity through the
        # whole clip instead of flickering as risk changes.
        body_color = _assets.stable_vehicle_color(track_id)
        outline = PALETTE["vehicle_outline"]
        if kind == "moto":
            self._draw_moto(canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming)
            return
        if kind == "bus":
            self._draw_bus(canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming)
            return
        if kind == "truck":
            self._draw_truck(canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming)
            return
        self._draw_car(canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming)

    def _draw_car(self, canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming):
        """Polished mini 3D toy car.

        Rounded body, raised cabin with dark windshield strip, vertical body
        gradient (lighter top, darker bottom), soft front/rear lights, dark
        underbody shadow for the "wheels at the corners" feel.
        """
        radius = max(2, int(min(w, h) * 0.28))
        # Body — rounded rect with vertical gradient.
        body_poly = self._rounded_rect_poly(sx, sy, w, h, radius)
        top = lighten(body_color, 0.18)
        bot = darken(body_color, 0.18)
        self._fill_vertical_gradient(canvas, body_poly, top, bot)
        cv2.polylines(canvas, [body_poly], True, outline, 1, cv2.LINE_AA)

        # Side highlight — a faint lighter band across the upper third for a
        # "polished plastic" sheen.
        sheen_y = int(sy - h * 0.22)
        cv2.line(
            canvas,
            (int(sx - w * 0.42), sheen_y),
            (int(sx + w * 0.42), sheen_y),
            lighten(body_color, 0.35),
            max(1, int(h * 0.03)),
            cv2.LINE_AA,
        )

        # Cabin / roof — narrower rounded rect on top of body, with dark glass.
        cabin_w = int(w * 0.62)
        cabin_h = int(h * 0.50)
        cabin_cx = sx
        cabin_cy = int(sy - h * 0.05)
        cabin_r = max(2, int(min(cabin_w, cabin_h) * 0.30))
        cabin_poly = self._rounded_rect_poly(cabin_cx, cabin_cy, cabin_w, cabin_h, cabin_r)
        cabin_top = lighten(body_color, 0.05)
        cabin_bot = darken(body_color, 0.10)
        self._fill_vertical_gradient(canvas, cabin_poly, cabin_top, cabin_bot)
        cv2.polylines(canvas, [cabin_poly], True, outline, 1, cv2.LINE_AA)

        # Windshield / rear-window glass: a single dark band across the cabin.
        glass_h = max(2, int(cabin_h * 0.32))
        glass_y0 = cabin_cy - int(cabin_h * 0.05) - glass_h // 2
        glass_x0 = int(sx - cabin_w * 0.42)
        glass_x1 = int(sx + cabin_w * 0.42)
        cv2.rectangle(
            canvas,
            (glass_x0, glass_y0),
            (glass_x1, glass_y0 + glass_h),
            (45, 55, 70),
            -1,
            cv2.LINE_AA,
        )

        # Dark underbody band (sells the "wheels" without drawing them
        # individually — works much better at small icon sizes).
        under_y0 = int(sy + h * 0.30)
        under_y1 = int(sy + h * 0.48)
        cv2.rectangle(
            canvas,
            (int(sx - w * 0.42), under_y0),
            (int(sx + w * 0.42), under_y1),
            (28, 28, 32),
            -1,
            cv2.LINE_AA,
        )

        # Headlights / taillights — small bright dots, oriented by traffic dir.
        light_y = int(sy - h * 0.42) if oncoming else int(sy + h * 0.42)
        light_color = (210, 245, 255) if oncoming else (70, 70, 240)
        for dx_frac in (-0.30, 0.30):
            cv2.circle(canvas, (int(sx + dx_frac * w), light_y), max(1, int(h * 0.05)),
                       light_color, -1, cv2.LINE_AA)

        if is_primary:
            cv2.polylines(canvas, [body_poly], True, PALETTE["primary_ring"], 2, cv2.LINE_AA)

    def _draw_truck(self, canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming):
        """Polished mini 3D toy truck: tall rounded cargo + small cab."""
        # Cargo: taller rounded rect occupying upper ~70% of icon.
        cargo_h = int(h * 0.68)
        cargo_cy = int(sy - h * 0.16)
        radius_c = max(2, int(min(w, cargo_h) * 0.18))
        cargo_poly = self._rounded_rect_poly(sx, cargo_cy, w, cargo_h, radius_c)
        top = lighten(body_color, 0.14)
        bot = darken(body_color, 0.18)
        self._fill_vertical_gradient(canvas, cargo_poly, top, bot)
        cv2.polylines(canvas, [cargo_poly], True, outline, 1, cv2.LINE_AA)
        # Seam line across cargo (mid).
        cv2.line(canvas, (int(sx - w * 0.45), int(cargo_cy - cargo_h * 0.08)),
                 (int(sx + w * 0.45), int(cargo_cy - cargo_h * 0.08)),
                 darken(body_color, 0.30), 1, cv2.LINE_AA)

        # Cab — narrower rounded rect in the lower 30%, lighter color.
        cab_w = int(w * 0.78)
        cab_h = int(h * 0.30)
        cab_cx = sx
        cab_cy = int(sy + h * 0.28)
        radius_cab = max(2, int(min(cab_w, cab_h) * 0.28))
        cab_color = lighten(body_color, 0.18)
        cab_poly = self._rounded_rect_poly(cab_cx, cab_cy, cab_w, cab_h, radius_cab)
        self._fill_vertical_gradient(canvas, cab_poly,
                                     lighten(cab_color, 0.10), darken(cab_color, 0.10))
        cv2.polylines(canvas, [cab_poly], True, outline, 1, cv2.LINE_AA)
        # Cab windshield.
        wsx0 = int(sx - cab_w * 0.40)
        wsx1 = int(sx + cab_w * 0.40)
        wsy0 = int(cab_cy - cab_h * 0.25)
        wsy1 = int(cab_cy + cab_h * 0.10)
        cv2.rectangle(canvas, (wsx0, wsy0), (wsx1, wsy1), (45, 55, 70), -1, cv2.LINE_AA)

        # Dark underbody.
        cv2.rectangle(canvas,
                      (int(sx - w * 0.42), int(sy + h * 0.40)),
                      (int(sx + w * 0.42), int(sy + h * 0.50)),
                      (28, 28, 32), -1, cv2.LINE_AA)

        # Light at the front edge.
        light_y = int(sy - h * 0.45) if oncoming else int(sy + h * 0.42)
        light_color = (210, 245, 255) if oncoming else (70, 70, 240)
        for dx_frac in (-0.32, 0.32):
            cv2.circle(canvas, (int(sx + dx_frac * w), light_y), max(1, int(h * 0.045)),
                       light_color, -1, cv2.LINE_AA)

        if is_primary:
            cv2.polylines(canvas, [cargo_poly], True, PALETTE["primary_ring"], 2, cv2.LINE_AA)

    def _draw_bus(self, canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming):
        """Polished mini 3D toy bus: long rounded body with row of windows."""
        # Use a yellow-ish tint for the bus body so it reads visually as a bus.
        bus_body = (60, 200, 240)  # warm yellow-orange in BGR
        # Blend with the stable body color so different buses still look distinct.
        bus_body = (
            int(0.55 * bus_body[0] + 0.45 * body_color[0]),
            int(0.55 * bus_body[1] + 0.45 * body_color[1]),
            int(0.55 * bus_body[2] + 0.45 * body_color[2]),
        )
        radius = max(2, int(min(w, h) * 0.18))
        body_poly = self._rounded_rect_poly(sx, sy, w, h, radius)
        self._fill_vertical_gradient(canvas, body_poly,
                                     lighten(bus_body, 0.16), darken(bus_body, 0.18))
        cv2.polylines(canvas, [body_poly], True, outline, 1, cv2.LINE_AA)

        # Window strip — dark band across the upper half.
        win_y0 = int(sy - h * 0.30)
        win_y1 = int(sy - h * 0.05)
        cv2.rectangle(canvas, (int(sx - w * 0.42), win_y0),
                      (int(sx + w * 0.42), win_y1), (40, 50, 65), -1, cv2.LINE_AA)
        # Window dividers — three vertical light lines so it reads as separate windows.
        for frac in (-0.20, 0.00, 0.20):
            x = int(sx + frac * w)
            cv2.line(canvas, (x, win_y0), (x, win_y1), bus_body, 1, cv2.LINE_AA)

        # Dark underbody.
        cv2.rectangle(canvas,
                      (int(sx - w * 0.42), int(sy + h * 0.30)),
                      (int(sx + w * 0.42), int(sy + h * 0.48)),
                      (28, 28, 32), -1, cv2.LINE_AA)

        # Light at the front edge.
        light_y = int(sy - h * 0.42) if oncoming else int(sy + h * 0.42)
        light_color = (210, 245, 255) if oncoming else (70, 70, 240)
        for dx_frac in (-0.32, 0.32):
            cv2.circle(canvas, (int(sx + dx_frac * w), light_y), max(1, int(h * 0.05)),
                       light_color, -1, cv2.LINE_AA)

        if is_primary:
            cv2.polylines(canvas, [body_poly], True, PALETTE["primary_ring"], 2, cv2.LINE_AA)

    def _draw_moto(self, canvas, sx, sy, w, h, body_color, outline, is_primary, oncoming):
        r = max(3, min(w, h) // 4)
        cv2.circle(canvas, (sx, sy - h // 3), r, (30, 30, 34), -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy + h // 3), r, (30, 30, 34), -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy - h // 3), r, outline, 1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy + h // 3), r, outline, 1, cv2.LINE_AA)
        cv2.line(canvas, (sx, sy - h // 3), (sx, sy + h // 3), body_color, max(2, w // 6), cv2.LINE_AA)
        head_r = max(2, r - 1)
        cv2.circle(canvas, (sx, sy - int(h * 0.05)), head_r, (220, 230, 240), -1, cv2.LINE_AA)
        cv2.line(canvas, (sx - head_r, sy + 4), (sx + head_r, sy + 4), (220, 230, 240), 2, cv2.LINE_AA)
        if is_primary:
            cv2.rectangle(canvas, (sx - w // 2, sy - h // 2), (sx + w // 2, sy + h // 2),
                          PALETTE["primary_ring"], 2, cv2.LINE_AA)

    def _draw_wheels(self, canvas, sx, sy, w, h, scale=1.0, extra_axle=False):
        wheel_color = (28, 28, 32)
        wheel_w = max(2, int(w * 0.08 * scale))
        wheel_h = max(3, int(h * 0.16 * scale))
        positions = [
            (sx - w // 2 - 1, sy - int(h * 0.30)),
            (sx + w // 2 - wheel_w + 1, sy - int(h * 0.30)),
            (sx - w // 2 - 1, sy + int(h * 0.30) - wheel_h),
            (sx + w // 2 - wheel_w + 1, sy + int(h * 0.30) - wheel_h),
        ]
        for (x, y) in positions:
            cv2.rectangle(canvas, (x, y), (x + wheel_w, y + wheel_h), wheel_color, -1, cv2.LINE_AA)
        if extra_axle:
            for (x, y) in [
                (sx - w // 2 - 1, sy),
                (sx + w // 2 - wheel_w + 1, sy),
            ]:
                cv2.rectangle(canvas, (x, y - wheel_h // 2), (x + wheel_w, y + wheel_h // 2),
                              wheel_color, -1, cv2.LINE_AA)

    def _draw_headlights(self, canvas, sx, sy, w):
        off = max(3, int(w * 0.28))
        for dx in (-off, off):
            cv2.circle(canvas, (sx + dx, sy), 2, (200, 240, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx + dx, sy), 4, (180, 220, 255), 1, cv2.LINE_AA)

    def _draw_taillights(self, canvas, sx, sy, w):
        off = max(3, int(w * 0.30))
        for dx in (-off, off):
            cv2.circle(canvas, (sx + dx, sy), 2, (60, 60, 230), -1, cv2.LINE_AA)
            cv2.circle(canvas, (sx + dx, sy), 4, (40, 40, 200), 1, cv2.LINE_AA)

    def _draw_ped_icon(self, canvas, sx, sy, h, color, is_primary):
        """Polished mini 3D toy pedestrian.

        Filled head + tapered torso + short legs (filled, not a stick). The
        figure reads as a human from far away because of the silhouette
        proportions (head-to-body ratio matches a real pedestrian).
        """
        skin = (170, 195, 220)           # warm peach (BGR)
        torso = (200, 130, 80)           # soft blue-grey for the torso
        torso_top = lighten(torso, 0.20)
        torso_bot = darken(torso, 0.20)
        legs = (60, 70, 100)             # darker for legs
        outline = (20, 22, 28)
        self._draw_glow(canvas, sx, sy, int(h * 0.65), h, color, primary=is_primary)

        # Head — filled circle, slightly rounded.
        head_r = max(3, int(h * 0.13))
        head_cy = sy - h // 2 + head_r + max(1, h // 30)
        cv2.circle(canvas, (sx, head_cy), head_r, skin, -1, cv2.LINE_AA)
        cv2.circle(canvas, (sx, head_cy), head_r, outline, 1, cv2.LINE_AA)

        # Torso — tapered rounded shape (shoulders wider than waist).
        torso_top_y = head_cy + head_r
        torso_bot_y = int(sy + h * 0.18)
        shoulder_w = max(3, int(h * 0.18))
        waist_w = max(2, int(shoulder_w * 0.75))
        torso_poly = np.array([
            [sx - shoulder_w, torso_top_y + 1],
            [sx + shoulder_w, torso_top_y + 1],
            [sx + waist_w, torso_bot_y],
            [sx - waist_w, torso_bot_y],
        ], dtype=np.int32)
        self._fill_vertical_gradient(canvas, torso_poly, torso_top, torso_bot)
        cv2.polylines(canvas, [torso_poly], True, outline, 1, cv2.LINE_AA)

        # Legs — two filled rounded shapes side by side (no stick lines).
        leg_top_y = torso_bot_y
        leg_bot_y = sy + h // 2 - max(1, h // 30)
        leg_w = max(2, int(waist_w * 0.55))
        for cx_leg in (sx - leg_w - 1, sx + 1):
            leg_poly = np.array([
                [cx_leg, leg_top_y],
                [cx_leg + leg_w, leg_top_y],
                [cx_leg + leg_w, leg_bot_y],
                [cx_leg, leg_bot_y],
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [leg_poly], legs, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [leg_poly], True, outline, 1, cv2.LINE_AA)

        if is_primary:
            cv2.circle(canvas, (sx, sy), max(head_r * 3, 14),
                       PALETTE["primary_ring"], 2, cv2.LINE_AA)

    def _draw_bike_icon(self, canvas, sx, sy, w, h, color, is_primary):
        self._draw_glow(canvas, sx, sy, w, h, color, primary=is_primary)
        frame_color = PALETTE["bike"]
        wheel_color = (28, 28, 32)
        r = max(4, min(w, h) // 3)
        lwx, lwy = sx - int(w * 0.32), sy + int(h * 0.28)
        rwx, rwy = sx + int(w * 0.32), sy + int(h * 0.28)
        cv2.circle(canvas, (lwx, lwy), r, wheel_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (rwx, rwy), r, wheel_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (lwx, lwy), r, frame_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (rwx, rwy), r, frame_color, 2, cv2.LINE_AA)
        seat = (sx, sy - int(h * 0.05))
        head = (sx, sy - int(h * 0.30))
        cv2.line(canvas, (lwx, lwy), seat, frame_color, 2, cv2.LINE_AA)
        cv2.line(canvas, (rwx, rwy), seat, frame_color, 2, cv2.LINE_AA)
        cv2.line(canvas, seat, head, frame_color, 2, cv2.LINE_AA)
        cv2.line(canvas, (sx - int(w * 0.15), head[1]), (sx + int(w * 0.15), head[1]),
                 frame_color, 2, cv2.LINE_AA)
        rider_head_r = max(3, r // 2)
        cv2.circle(canvas, (sx, head[1] - rider_head_r - 1), rider_head_r,
                   (220, 230, 240), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (sx, head[1] + 2),
                    (int(r * 0.9), max(2, rider_head_r)), 0, 0, 360,
                    (220, 230, 240), -1, cv2.LINE_AA)
        if is_primary:
            cv2.rectangle(canvas, (sx - w // 2, sy - h // 2),
                          (sx + w // 2, sy + h // 2),
                          PALETTE["primary_ring"], 2, cv2.LINE_AA)

    def _draw_glow(self, canvas, sx, sy, w, h, color, primary=False):
        radius = int(max(w, h) * (1.05 if primary else 0.85))
        ry = max(6, radius // 2)
        rx = max(6, radius)
        x0 = max(0, int(sx - rx - 3))
        x1 = min(canvas.shape[1], int(sx + rx + 3))
        y0 = max(0, int(sy - ry - 3))
        y1 = min(canvas.shape[0], int(sy + ry + 3))
        if x1 <= x0 or y1 <= y0:
            return
        roi = canvas[y0:y1, x0:x1]
        overlay = roi.copy()
        cv2.ellipse(overlay, (int(sx - x0), int(sy - y0)), (int(rx), int(ry)),
                    0, 0, 360, color, -1, cv2.LINE_AA)
        alpha = 0.30 if primary else 0.18
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

    # ---- helpers ------------------------------------------------------- #

    def _resize_to_canvas(self, frame_bgr: np.ndarray, target_w: Optional[int] = None,
                          target_h: Optional[int] = None) -> np.ndarray:
        target_w = target_w or self.W
        target_h = target_h or self.H
        return cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _blank_canvas(self) -> np.ndarray:
        return np.zeros((self.H, self.W, 3), dtype=np.uint8)
