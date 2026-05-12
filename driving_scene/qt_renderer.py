"""
qt_renderer.py — PyQt6 / QPainter frontend for the Tesla-Inspired
Driving Scene Visualizer.

This module is OPTIONAL. It is only imported when `--renderer qt` is passed.
If PyQt6 isn't installed, main.py prints a clear hint and falls back to the
OpenCV renderer.

Responsibilities (frontend-only):
  * Draw the background gradient + faint vignette.
  * Draw the road, lane lines (curved when lane_curve != 0), planned path.
  * Draw the ego car bottom-center.
  * Draw each SceneObject as a polished QPainter polygon (cars, trucks,
    buses, motos, pedestrians, bicycles) using gradients and anti-aliasing.
  * Draw the HUD (top status bar, side panels, footer disclaimer, action
    banner).
  * Draw debug labels when debug mode is on.

NOT responsibilities:
  * Running YOLO. Tracking. Lane assignment. Smoothing.
  * Those all live in the backend thread (see qt_app.BackendThread).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

from . import lane_model as _lm


# ---------------------------------------------------------------------- #
# Snapshot the backend hands to the canvas                               #
# ---------------------------------------------------------------------- #

@dataclass
class SceneSnapshot:
    """A frame-worth of state for the Qt canvas to paint.

    `items` are the raw dicts from SceneMapper.collect_renderables — kept
    as dicts (not full dataclasses) to avoid one more conversion hop.
    """

    items: List[Dict[str, Any]] = field(default_factory=list)
    primary_threat_id: Optional[int] = None
    lane_curve: float = 0.0
    lane_confidence: float = 0.0
    lane_phase: float = 0.0
    track_trails: Dict[int, List[Tuple[float, float, float]]] = field(default_factory=dict)
    ego_speed_kmh: Optional[float] = None
    speed_limit_kmh: Optional[float] = None
    view_mode: str = "scene"
    n_objects: int = 0
    # Telemetry / HUD strings
    display_fps: float = 0.0
    detection_fps: float = 0.0
    yolo_latency_ms: float = 0.0
    dropped_detections: int = 0
    # Risk-state summary
    global_score: Optional[float] = None
    global_level: Optional[str] = None
    global_action: Optional[str] = None
    # Lane debug
    lane_boundaries: Optional[List[float]] = None
    lane_labels: Optional[List[str]] = None


# ---------------------------------------------------------------------- #
# Palette — RGB (Qt uses RGB, not BGR like OpenCV)                       #
# ---------------------------------------------------------------------- #

QPALETTE = {
    "bg_top": QColor(22, 18, 18),
    "bg_bottom": QColor(10, 8, 8),
    "road": QColor(44, 40, 38),
    "road_edge": QColor(68, 62, 60),
    "lane_solid": QColor(220, 215, 210),
    "lane_dash": QColor(245, 240, 235),
    "lane_yellow": QColor(230, 200, 40),
    "horizon": QColor(36, 30, 28),
    "ego": QColor(248, 245, 245),
    "ego_outline": QColor(150, 130, 110),
    "ego_beam": QColor(255, 200, 90, 90),     # alpha for soft beam
    "vehicle_base": QColor(180, 175, 170),
    "vehicle_outline": QColor(48, 42, 40),
    "ped": QColor(255, 200, 120),
    "bike": QColor(120, 220, 180),
    "risk_low": QColor(120, 220, 160),
    "risk_med": QColor(240, 200, 80),
    "risk_high": QColor(240, 120, 60),
    "risk_crit": QColor(240, 60, 60),
    "primary_ring": QColor(255, 255, 255),
    "hud_panel": QColor(24, 26, 32, 200),
    "hud_panel_edge": QColor(84, 75, 70),
    "hud_text": QColor(245, 240, 235),
    "hud_dim": QColor(160, 150, 140),
    "hud_accent": QColor(120, 200, 255),
    "speed_sign_bg": QColor(248, 245, 245),
    "speed_sign_ring": QColor(220, 40, 40),
    "speed_sign_text": QColor(24, 20, 20),
    "planned_path": QColor(60, 180, 255),
    "planned_path_glow": QColor(120, 220, 255, 70),
    "trail": QColor(110, 170, 200),
    "shadow": QColor(0, 0, 0, 80),
}


def _risk_color(level: Optional[str], score: Optional[float]) -> QColor:
    by_level = {
        "LOW": QPALETTE["risk_low"],
        "MEDIUM": QPALETTE["risk_med"],
        "HIGH": QPALETTE["risk_high"],
        "CRITICAL": QPALETTE["risk_crit"],
    }
    if level and level in by_level:
        return by_level[level]
    if score is None:
        return QPALETTE["risk_low"]
    s = max(0.0, min(100.0, float(score))) / 100.0
    if s < 0.33:
        return QPALETTE["risk_low"]
    if s < 0.66:
        return QPALETTE["risk_med"]
    if s < 0.88:
        return QPALETTE["risk_high"]
    return QPALETTE["risk_crit"]


def _lighten(c: QColor, amount: float) -> QColor:
    return QColor(
        min(255, int(c.red() + (255 - c.red()) * amount)),
        min(255, int(c.green() + (255 - c.green()) * amount)),
        min(255, int(c.blue() + (255 - c.blue()) * amount)),
        c.alpha(),
    )


def _darken(c: QColor, amount: float) -> QColor:
    k = max(0.0, 1.0 - amount)
    return QColor(int(c.red() * k), int(c.green() * k), int(c.blue() * k), c.alpha())


def _tint(base: QColor, accent: QColor, mix: float = 0.45) -> QColor:
    return QColor(
        int(base.red() * (1 - mix) + accent.red() * mix),
        int(base.green() * (1 - mix) + accent.green() * mix),
        int(base.blue() * (1 - mix) + accent.blue() * mix),
    )


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------------------------------------------------- #
# SceneCanvas — the QWidget we draw on                                   #
# ---------------------------------------------------------------------- #

class SceneCanvas(QWidget):
    """Tesla-inspired scene canvas. Receives SceneSnapshot, paints with QPainter."""

    def __init__(self, width: int = 1280, height: int = 720, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setMouseTracking(False)
        self.W = int(width)
        self.H = int(height)
        self.snapshot: Optional[SceneSnapshot] = None
        # Local LaneModel mirrors the backend's geometry config so we can
        # query lane center / half-width at any y to draw curves.
        self.lanes = _lm.LaneModel(self.W, self.H)
        self.debug = False

    # ---- API ----------------------------------------------------------- #

    def set_snapshot(self, snap: SceneSnapshot) -> None:
        self.snapshot = snap
        # Keep our LaneModel curve synced so the road bends with the road.
        self.lanes.set_curve(snap.lane_curve)
        self.update()

    def toggle_debug(self) -> None:
        self.debug = not self.debug
        self.update()

    # ---- paint --------------------------------------------------------- #

    def paintEvent(self, event):  # noqa: N802 (Qt naming)
        snap = self.snapshot
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self._draw_background(painter)

        # Backend hasn't published a snapshot yet — still paint the static
        # parts so the window doesn't look broken during startup.
        if snap is None:
            self._draw_road(painter)
            self._draw_lanes(painter)
            self._draw_planned_path(painter)
            self._draw_ego(painter)
            self._draw_footer(painter)
            self._draw_mode_tag(painter, "WAITING FOR BACKEND…")
            return

        view_mode = snap.view_mode
        if view_mode in ("dashcam", "debug"):
            # We don't carry the raw frame into the Qt path right now; show
            # the scene + a tag so the user can switch back. The OpenCV
            # renderer is the right tool for dashcam/debug views since it
            # already has the raw frame in hand.
            self._draw_road(painter)
            self._draw_lanes(painter)
            self._draw_planned_path(painter)
            self._draw_objects(painter, snap)
            self._draw_ego(painter)
            self._draw_hud(painter, snap)
            self._draw_mode_tag(painter, f"{view_mode.upper()} (Qt: use OpenCV renderer for raw frame)")
            return

        self._draw_road(painter)
        self._draw_lanes(painter)
        self._draw_planned_path(painter)
        self._draw_trails(painter, snap)
        self._draw_objects(painter, snap)
        self._draw_ego(painter)
        self._draw_hud(painter, snap)
        self._draw_footer(painter)
        if self.debug:
            self._draw_debug_panel(painter, snap)

    # ---- background / road / lanes ------------------------------------ #

    def _draw_background(self, p: QPainter) -> None:
        grad = QLinearGradient(0, 0, 0, self.H)
        grad.setColorAt(0.0, QPALETTE["bg_top"])
        grad.setColorAt(1.0, QPALETTE["bg_bottom"])
        p.fillRect(self.rect(), QBrush(grad))
        # Faint radial vignette: a darker brush over the corners.
        vg = QRadialGradient(QPointF(self.W * 0.5, self.H * 0.55), max(self.W, self.H) * 0.7)
        vg.setColorAt(0.0, QColor(0, 0, 0, 0))
        vg.setColorAt(1.0, QColor(0, 0, 0, 140))
        p.fillRect(self.rect(), QBrush(vg))

    def _draw_road(self, p: QPainter) -> None:
        hy = int(self.lanes.horizon_y())
        ey = int(self.lanes.ego_y() + 40)
        # Carriageway extents from LaneModel (so curve is respected).
        left_top = self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_2, hy) - self.lanes.lane_half_width_at_y(_lm.LANE_ONCOMING_2, hy)
        right_top = self.lanes.lane_center_x_at_y(_lm.LANE_RIGHT_1, hy) + self.lanes.lane_half_width_at_y(_lm.LANE_RIGHT_1, hy)
        left_bot = self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_2, ey) - self.lanes.lane_half_width_at_y(_lm.LANE_ONCOMING_2, ey)
        right_bot = self.lanes.lane_center_x_at_y(_lm.LANE_RIGHT_1, ey) + self.lanes.lane_half_width_at_y(_lm.LANE_RIGHT_1, ey)

        # Horizon strip.
        p.fillRect(QRectF(0, hy - 14, self.W, 28), QPALETTE["horizon"])

        # Sample the carriageway polygon as curves (left + right edges).
        steps = 20
        path = QPainterPath()
        path.moveTo(QPointF(left_top, hy))
        for i in range(1, steps + 1):
            y = _lerp(hy, ey, i / steps)
            lx = self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_2, y) - self.lanes.lane_half_width_at_y(_lm.LANE_ONCOMING_2, y)
            path.lineTo(QPointF(lx, y))
        for i in range(steps, -1, -1):
            y = _lerp(hy, ey, i / steps)
            rx = self.lanes.lane_center_x_at_y(_lm.LANE_RIGHT_1, y) + self.lanes.lane_half_width_at_y(_lm.LANE_RIGHT_1, y)
            path.lineTo(QPointF(rx, y))
        path.closeSubpath()
        p.fillPath(path, QBrush(QPALETTE["road"]))

        # Soft far-fog: paint a translucent gradient near the horizon.
        fog = QLinearGradient(0, hy - 30, 0, hy + 80)
        fog.setColorAt(0.0, QColor(40, 38, 36, 110))
        fog.setColorAt(1.0, QColor(40, 38, 36, 0))
        p.fillPath(path, QBrush(fog))

    def _draw_lanes(self, p: QPainter) -> None:
        hy = self.lanes.horizon_y()
        ey = self.lanes.ego_y() + 40
        # Outer solid edges.
        for lane, side in ((_lm.LANE_ONCOMING_2, -1), (_lm.LANE_RIGHT_1, +1)):
            self._draw_lane_curve(
                p,
                lambda y, lane=lane, side=side: self.lanes.lane_center_x_at_y(lane, y) + side * self.lanes.lane_half_width_at_y(lane, y),
                hy, ey,
                QPALETTE["lane_solid"], width_top=1.2, width_bot=3.0,
            )
        # Double yellow between ego and oncoming_1.
        for offset in (-3.0, 3.0):
            self._draw_lane_curve(
                p,
                lambda y, off=offset: self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_1, y) + self.lanes.lane_half_width_at_y(_lm.LANE_ONCOMING_1, y) + off,
                hy, ey,
                QPALETTE["lane_yellow"], width_top=1.2, width_bot=3.0,
            )
        # Dashed line between ego and right_lane.
        self._draw_lane_dashes(
            p,
            lambda y: self.lanes.lane_center_x_at_y(_lm.LANE_EGO, y) + self.lanes.lane_half_width_at_y(_lm.LANE_EGO, y),
            hy, ey,
        )
        # Dashed line between oncoming_2 and oncoming_1.
        self._draw_lane_dashes(
            p,
            lambda y: self.lanes.lane_center_x_at_y(_lm.LANE_ONCOMING_2, y) + self.lanes.lane_half_width_at_y(_lm.LANE_ONCOMING_2, y),
            hy, ey,
        )

    def _draw_lane_curve(self, p, x_of_y, y_top, y_bot, color, width_top=1.0, width_bot=3.0) -> None:
        # Draw the lane as a stroked QPainterPath, varying width by depth.
        steps = 12
        prev_pt = None
        for i in range(steps + 1):
            t = i / steps
            y = _lerp(y_top, y_bot, t)
            x = x_of_y(y)
            pt = QPointF(x, y)
            if prev_pt is not None:
                w = _lerp(width_top, width_bot, t)
                pen = QPen(color, w)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(prev_pt, pt)
            prev_pt = pt

    def _draw_lane_dashes(self, p, x_of_y, y_top, y_bot) -> None:
        snap = self.snapshot
        phase = snap.lane_phase if snap is not None else 0.0
        n_dashes = 14
        color = QPALETTE["lane_dash"]
        for i in range(n_dashes):
            t0 = ((i / n_dashes) + phase) % 1.0
            t1 = t0 + (0.42 / n_dashes)
            if t1 > 1.0:
                continue
            y0 = _lerp(y_top, y_bot, t0)
            y1 = _lerp(y_top, y_bot, t1)
            x0 = x_of_y(y0)
            x1 = x_of_y(y1)
            w = max(2.0, _lerp(2.0, 6.0, (t0 + t1) * 0.5))
            pen = QPen(color, w)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    # ---- planned path -------------------------------------------------- #

    def _draw_planned_path(self, p: QPainter) -> None:
        snap = self.snapshot
        if snap is None:
            return
        ego_x, ego_y = self.lanes.ego_position()
        horizon_y = self.lanes.horizon_y()
        n = 20
        path = QPainterPath()
        first = True
        for i in range(n + 1):
            t = i / n
            y = _lerp(ego_y - 30, horizon_y + 12, t)
            x = self.lanes.lane_center_x_at_y(_lm.LANE_EGO, y)
            pt = QPointF(x, y)
            if first:
                path.moveTo(pt)
                first = False
            else:
                path.lineTo(pt)
        # Glow underlay.
        conf = max(0.25, min(1.0, snap.lane_confidence)) if snap.lane_confidence else 0.6
        glow = QColor(QPALETTE["planned_path_glow"])
        glow.setAlpha(int(120 * conf))
        glow_pen = QPen(glow, 14)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(glow_pen)
        p.drawPath(path)
        # Crisp line.
        main = QPen(QPALETTE["planned_path"], 6)
        main.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(main)
        p.drawPath(path)
        # Horizon end-cap.
        end_x = self.lanes.lane_center_x_at_y(_lm.LANE_EGO, horizon_y + 12)
        p.setBrush(QBrush(QPALETTE["planned_path"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(end_x, horizon_y + 12), 4, 4)

    # ---- motion trails ------------------------------------------------- #

    def _draw_trails(self, p: QPainter, snap: SceneSnapshot) -> None:
        for tid, trail in (snap.track_trails or {}).items():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                a = trail[i - 1]
                b = trail[i]
                age_t = i / float(len(trail))
                color = QColor(QPALETTE["trail"])
                color.setAlpha(int(180 * (age_t ** 1.3)))
                pen = QPen(color, max(1.0, 1.0 + 2.0 * age_t))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

    # ---- objects ------------------------------------------------------- #

    def _draw_objects(self, p: QPainter, snap: SceneSnapshot) -> None:
        items = list(snap.items)
        items.sort(key=lambda it: it.get("closeness", 0.0))
        # Lazy import to avoid hard-coupling Qt path on the OpenCV side.
        from . import assets as _assets
        for it in items:
            cls = it.get("class", "car").lower()
            sx = float(it.get("scene_x", self.W * 0.5))
            sy = float(it.get("scene_y", self.H * 0.5))
            w = max(8.0, float(it.get("icon_w", 40.0)))
            h = max(6.0, float(it.get("icon_h", 26.0)))
            lane = it.get("lane", 0)
            oncoming = isinstance(lane, int) and lane < 0
            tid = it.get("track_id")
            is_primary = snap.primary_threat_id is not None and tid == snap.primary_threat_id
            risk_col = _risk_color(it.get("level"), it.get("score"))

            # Soft shadow first (under everything).
            self._draw_shadow(p, sx, sy + h * 0.42, w * (1.10 if cls not in ("person", "pedestrian") else 1.15),
                              h * (0.20 if cls not in ("person", "pedestrian") else 0.14))
            # Risk glow ring (subtle, primary threats stronger).
            self._draw_glow(p, sx, sy, w, h, risk_col, is_primary)

            # Stable per-track body color in Qt RGB.
            color_idx = _assets.stable_color_index(tid)
            bgr = _assets.VEHICLE_COLOR_PALETTE[color_idx]
            body_color = QColor(bgr[2], bgr[1], bgr[0])  # BGR -> RGB for Qt

            if cls in ("person", "pedestrian"):
                self._draw_ped(p, sx, sy, w, h, risk_col, is_primary)
            elif cls in ("bicycle", "bike"):
                self._draw_bike(p, sx, sy, w, h, risk_col, is_primary)
            elif cls == "bus":
                self._draw_bus(p, sx, sy, w, h * 1.45, body_color, is_primary, oncoming)
            elif cls == "truck":
                self._draw_truck(p, sx, sy, w, h * 1.25, body_color, is_primary, oncoming)
            elif cls in ("motorcycle", "motorbike"):
                self._draw_moto(p, sx, sy, w * 0.55, h, risk_col, is_primary)
            else:
                self._draw_car(p, sx, sy, w, h, body_color, is_primary, oncoming)

            if self.debug:
                self._draw_object_debug_label(p, sx, sy, h, it)

    def _draw_glow(self, p: QPainter, sx: float, sy: float, w: float, h: float, color: QColor, is_primary: bool) -> None:
        radius = max(w, h) * (1.4 if is_primary else 1.1)
        glow = QRadialGradient(QPointF(sx, sy), radius)
        c = QColor(color)
        c.setAlpha(120 if is_primary else 70)
        glow.setColorAt(0.0, c)
        glow.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sx, sy), radius, radius * 0.55)

    def _draw_shadow(self, p: QPainter, sx: float, sy: float, w: float, h: float) -> None:
        p.setBrush(QBrush(QPALETTE["shadow"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sx, sy), w * 0.5, h * 0.5)

    def _draw_car(self, p, sx, sy, w, h, body_color, is_primary, oncoming) -> None:
        """Polished mini 3D toy car (Qt). Rounded body + cabin + glass + soft sheen."""
        outline = QPALETTE["vehicle_outline"]
        radius = max(2.0, min(w, h) * 0.28)

        # Body — rounded rect with vertical gradient.
        body_rect = QRectF(sx - w * 0.5, sy - h * 0.5, w, h)
        body_path = QPainterPath()
        body_path.addRoundedRect(body_rect, radius, radius)
        grad = QLinearGradient(0, sy - h * 0.5, 0, sy + h * 0.5)
        grad.setColorAt(0.0, _lighten(body_color, 0.18))
        grad.setColorAt(1.0, _darken(body_color, 0.18))
        p.fillPath(body_path, QBrush(grad))
        p.setPen(QPen(outline, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(body_path)

        # Sheen — faint lighter band across the upper third.
        sheen_pen = QPen(_lighten(body_color, 0.35), max(1.0, h * 0.04))
        sheen_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(sheen_pen)
        p.drawLine(QPointF(sx - w * 0.42, sy - h * 0.22),
                   QPointF(sx + w * 0.42, sy - h * 0.22))

        # Cabin — narrower rounded rect on top half.
        cabin_w = w * 0.62
        cabin_h = h * 0.50
        cabin_r = max(2.0, min(cabin_w, cabin_h) * 0.30)
        cabin_rect = QRectF(sx - cabin_w * 0.5, sy - h * 0.30, cabin_w, cabin_h)
        cabin_path = QPainterPath()
        cabin_path.addRoundedRect(cabin_rect, cabin_r, cabin_r)
        grad_cabin = QLinearGradient(0, cabin_rect.top(), 0, cabin_rect.bottom())
        grad_cabin.setColorAt(0.0, _lighten(body_color, 0.05))
        grad_cabin.setColorAt(1.0, _darken(body_color, 0.10))
        p.fillPath(cabin_path, QBrush(grad_cabin))
        p.setPen(QPen(outline, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(cabin_path)

        # Windshield / rear glass — single dark band across cabin.
        glass_h = max(2.0, cabin_h * 0.32)
        glass_y = sy - h * 0.10 - glass_h * 0.5
        p.fillRect(QRectF(sx - cabin_w * 0.42, glass_y, cabin_w * 0.84, glass_h),
                   QColor(45, 55, 70))

        # Dark underbody band (suggests wheels at this icon scale).
        p.fillRect(QRectF(sx - w * 0.42, sy + h * 0.30, w * 0.84, h * 0.18),
                   QColor(28, 28, 32))

        # Headlight / taillight at the front edge.
        light_y = sy - h * 0.42 if oncoming else sy + h * 0.42
        light_color = QColor(255, 245, 210) if oncoming else QColor(240, 70, 70)
        r_light = max(1.5, h * 0.05)
        p.setBrush(QBrush(light_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sx - w * 0.30, light_y), r_light, r_light)
        p.drawEllipse(QPointF(sx + w * 0.30, light_y), r_light, r_light)

        if is_primary:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QPALETTE["primary_ring"], 2.2))
            p.drawPath(body_path)

    def _draw_truck(self, p, sx, sy, w, h, body_color, is_primary, oncoming) -> None:
        """Polished mini 3D toy truck (Qt)."""
        outline = QPALETTE["vehicle_outline"]

        # Cargo — large rounded rect in the top 70%.
        cargo_h = h * 0.68
        cargo_cy = sy - h * 0.16
        radius = max(2.0, min(w, cargo_h) * 0.18)
        cargo_rect = QRectF(sx - w * 0.5, cargo_cy - cargo_h * 0.5, w, cargo_h)
        cargo_path = QPainterPath()
        cargo_path.addRoundedRect(cargo_rect, radius, radius)
        grad = QLinearGradient(0, cargo_rect.top(), 0, cargo_rect.bottom())
        grad.setColorAt(0.0, _lighten(body_color, 0.14))
        grad.setColorAt(1.0, _darken(body_color, 0.18))
        p.fillPath(cargo_path, QBrush(grad))
        p.setPen(QPen(outline, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(cargo_path)
        # Seam line.
        p.setPen(QPen(_darken(body_color, 0.30), 1))
        p.drawLine(QPointF(sx - w * 0.45, cargo_cy - cargo_h * 0.08),
                   QPointF(sx + w * 0.45, cargo_cy - cargo_h * 0.08))

        # Cab — lighter rounded rect at the bottom.
        cab_color = _lighten(body_color, 0.18)
        cab_w = w * 0.78
        cab_h = h * 0.30
        cab_rect = QRectF(sx - cab_w * 0.5, sy + h * 0.13, cab_w, cab_h)
        cab_path = QPainterPath()
        cab_path.addRoundedRect(cab_rect, max(2.0, min(cab_w, cab_h) * 0.28),
                                max(2.0, min(cab_w, cab_h) * 0.28))
        cab_grad = QLinearGradient(0, cab_rect.top(), 0, cab_rect.bottom())
        cab_grad.setColorAt(0.0, _lighten(cab_color, 0.10))
        cab_grad.setColorAt(1.0, _darken(cab_color, 0.10))
        p.fillPath(cab_path, QBrush(cab_grad))
        p.setPen(QPen(outline, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(cab_path)
        # Cab windshield.
        p.fillRect(QRectF(sx - cab_w * 0.40, sy + h * 0.21, cab_w * 0.80, cab_h * 0.35),
                   QColor(45, 55, 70))

        # Dark underbody + lights.
        p.fillRect(QRectF(sx - w * 0.42, sy + h * 0.40, w * 0.84, h * 0.10),
                   QColor(28, 28, 32))
        light_y = sy - h * 0.45 if oncoming else sy + h * 0.42
        light_color = QColor(255, 245, 210) if oncoming else QColor(240, 70, 70)
        r_light = max(1.5, h * 0.045)
        p.setBrush(QBrush(light_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sx - w * 0.32, light_y), r_light, r_light)
        p.drawEllipse(QPointF(sx + w * 0.32, light_y), r_light, r_light)

        if is_primary:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QPALETTE["primary_ring"], 2.2))
            p.drawPath(cargo_path)

    def _draw_bus(self, p, sx, sy, w, h, body_color, is_primary, oncoming) -> None:
        """Polished mini 3D toy bus (Qt). Long rounded body + window strip."""
        outline = QPALETTE["vehicle_outline"]
        # Mix the stable per-track body color with a bus-yellow signature.
        bus_yellow = QColor(240, 200, 60)
        bus_body = QColor(
            int(0.45 * body_color.red() + 0.55 * bus_yellow.red()),
            int(0.45 * body_color.green() + 0.55 * bus_yellow.green()),
            int(0.45 * body_color.blue() + 0.55 * bus_yellow.blue()),
        )
        radius = max(2.0, min(w, h) * 0.18)
        body_rect = QRectF(sx - w * 0.5, sy - h * 0.5, w, h)
        body_path = QPainterPath()
        body_path.addRoundedRect(body_rect, radius, radius)
        grad = QLinearGradient(0, body_rect.top(), 0, body_rect.bottom())
        grad.setColorAt(0.0, _lighten(bus_body, 0.16))
        grad.setColorAt(1.0, _darken(bus_body, 0.18))
        p.fillPath(body_path, QBrush(grad))
        p.setPen(QPen(outline, 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(body_path)

        # Window strip across the upper half.
        p.fillRect(QRectF(sx - w * 0.42, sy - h * 0.30, w * 0.84, h * 0.25),
                   QColor(40, 50, 65))
        # Window dividers.
        p.setPen(QPen(bus_body, 1))
        for frac in (-0.20, 0.00, 0.20):
            x = sx + frac * w
            p.drawLine(QPointF(x, sy - h * 0.30), QPointF(x, sy - h * 0.05))

        # Underbody + lights.
        p.fillRect(QRectF(sx - w * 0.42, sy + h * 0.30, w * 0.84, h * 0.18),
                   QColor(28, 28, 32))
        light_y = sy - h * 0.42 if oncoming else sy + h * 0.42
        light_color = QColor(255, 245, 210) if oncoming else QColor(240, 70, 70)
        r_light = max(1.5, h * 0.05)
        p.setBrush(QBrush(light_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sx - w * 0.30, light_y), r_light, r_light)
        p.drawEllipse(QPointF(sx + w * 0.30, light_y), r_light, r_light)

        if is_primary:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QPALETTE["primary_ring"], 2.2))
            p.drawPath(body_path)

    def _draw_moto(self, p, sx, sy, w, h, risk_col, is_primary) -> None:
        outline = QPALETTE["vehicle_outline"]
        body = _tint(QPALETTE["vehicle_base"], risk_col, 0.45)
        r = max(3.0, min(w, h) * 0.25)
        p.setBrush(QBrush(QColor(30, 30, 34)))
        p.setPen(QPen(outline, 0.8))
        p.drawEllipse(QPointF(sx, sy - h / 3), r, r)
        p.drawEllipse(QPointF(sx, sy + h / 3), r, r)
        p.setPen(QPen(body, max(2.0, w / 6.0)))
        p.drawLine(QPointF(sx, sy - h / 3), QPointF(sx, sy + h / 3))
        head_r = max(2.0, r - 1)
        p.setBrush(QBrush(QColor(220, 230, 240)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sx, sy - h * 0.05), head_r, head_r)
        if is_primary:
            p.setPen(QPen(QPALETTE["primary_ring"], 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(sx - w / 2, sy - h / 2, w, h))

    def _draw_ped(self, p, sx, sy, w, h, risk_col, is_primary) -> None:
        """Polished mini 3D toy pedestrian (Qt). Filled head + tapered torso + legs."""
        skin = QColor(220, 195, 170)         # warm peach
        torso = QColor(80, 130, 200)         # blue torso
        legs = QColor(100, 70, 60)           # dark legs
        outline = QColor(28, 22, 20)

        # Head — filled circle.
        head_r = max(3.0, h * 0.13)
        head_cy = sy - h * 0.5 + head_r + max(1.0, h / 30)
        p.setBrush(QBrush(skin))
        p.setPen(QPen(outline, 1.0))
        p.drawEllipse(QPointF(sx, head_cy), head_r, head_r)

        # Torso — tapered shape with vertical gradient.
        torso_top_y = head_cy + head_r
        torso_bot_y = sy + h * 0.18
        shoulder_w = max(3.0, h * 0.18)
        waist_w = max(2.0, shoulder_w * 0.75)
        torso_poly = QPolygonF([
            QPointF(sx - shoulder_w, torso_top_y + 1),
            QPointF(sx + shoulder_w, torso_top_y + 1),
            QPointF(sx + waist_w, torso_bot_y),
            QPointF(sx - waist_w, torso_bot_y),
        ])
        torso_grad = QLinearGradient(0, torso_top_y, 0, torso_bot_y)
        torso_grad.setColorAt(0.0, _lighten(torso, 0.20))
        torso_grad.setColorAt(1.0, _darken(torso, 0.20))
        p.setBrush(QBrush(torso_grad))
        p.setPen(QPen(outline, 1.0))
        p.drawPolygon(torso_poly)

        # Legs — two filled rounded rectangles.
        leg_top_y = torso_bot_y
        leg_bot_y = sy + h * 0.5 - max(1.0, h / 30)
        leg_w = max(2.0, waist_w * 0.55)
        p.setBrush(QBrush(legs))
        for cx_leg in (sx - leg_w - 1, sx + 1):
            leg_rect = QRectF(cx_leg, leg_top_y, leg_w, leg_bot_y - leg_top_y)
            p.drawRect(leg_rect)

        if is_primary:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QPALETTE["primary_ring"], 2.0))
            r = max(head_r * 3, 14.0)
            p.drawEllipse(QPointF(sx, sy), r, r)

    def _draw_bike(self, p, sx, sy, w, h, risk_col, is_primary) -> None:
        frame = QPALETTE["bike"]
        outline = QPALETTE["vehicle_outline"]
        r = max(4.0, min(w, h) / 3)
        lwx, lwy = sx - w * 0.32, sy + h * 0.28
        rwx, rwy = sx + w * 0.32, sy + h * 0.28
        p.setBrush(QBrush(QColor(28, 28, 32)))
        p.setPen(QPen(frame, 2))
        p.drawEllipse(QPointF(lwx, lwy), r, r)
        p.drawEllipse(QPointF(rwx, rwy), r, r)
        p.setPen(QPen(frame, 2))
        seat = QPointF(sx, sy - h * 0.05)
        head = QPointF(sx, sy - h * 0.30)
        p.drawLine(QPointF(lwx, lwy), seat)
        p.drawLine(QPointF(rwx, rwy), seat)
        p.drawLine(seat, head)
        p.drawLine(QPointF(sx - w * 0.15, head.y()), QPointF(sx + w * 0.15, head.y()))
        rider_head = max(3.0, r / 2)
        p.setBrush(QBrush(QColor(220, 230, 240)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sx, head.y() - rider_head - 1), rider_head, rider_head)
        p.drawEllipse(QPointF(sx, head.y() + 2), r * 0.9, max(2.0, rider_head))
        if is_primary:
            p.setPen(QPen(QPALETTE["primary_ring"], 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(sx - w / 2, sy - h / 2, w, h))

    def _draw_wheels(self, p, sx, sy, w, h, extra_axle=False, scale=1.0) -> None:
        wheel_color = QColor(28, 28, 32)
        ww = max(2.0, w * 0.08 * scale)
        wh = max(3.0, h * 0.16 * scale)
        p.setBrush(QBrush(wheel_color))
        p.setPen(Qt.PenStyle.NoPen)
        for (dx, dy) in (
            (-w / 2 - 1, -h * 0.30),
            (w / 2 - ww + 1, -h * 0.30),
            (-w / 2 - 1, h * 0.30 - wh),
            (w / 2 - ww + 1, h * 0.30 - wh),
        ):
            p.drawRect(QRectF(sx + dx, sy + dy, ww, wh))
        if extra_axle:
            for dx in (-w / 2 - 1, w / 2 - ww + 1):
                p.drawRect(QRectF(sx + dx, sy - wh / 2, ww, wh))

    def _draw_lights(self, p: QPainter, sx: float, sy: float, w: float, headlight: bool) -> None:
        # Headlight = warm white; taillight = red.
        off = max(3.0, w * 0.28)
        inner = QColor(255, 240, 200) if headlight else QColor(230, 60, 60)
        ring = QColor(255, 220, 160, 150) if headlight else QColor(200, 40, 40, 150)
        p.setBrush(QBrush(inner))
        p.setPen(Qt.PenStyle.NoPen)
        for dx in (-off, off):
            p.drawEllipse(QPointF(sx + dx, sy), 2.4, 2.4)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(ring, 1.2))
        for dx in (-off, off):
            p.drawEllipse(QPointF(sx + dx, sy), 4.5, 4.5)

    # ---- ego car ------------------------------------------------------- #

    def _draw_ego(self, p: QPainter) -> None:
        ex, ey = self.lanes.ego_position()
        w = self.W * 0.07
        h = self.H * 0.075
        # Faint headlight cone.
        cone = QPainterPath()
        cone.moveTo(QPointF(ex - w * 1.1, ey - h * 4))
        cone.lineTo(QPointF(ex + w * 1.1, ey - h * 4))
        cone.lineTo(QPointF(ex + w * 0.45, ey - h * 0.55))
        cone.lineTo(QPointF(ex - w * 0.45, ey - h * 0.55))
        cone.closeSubpath()
        beam_grad = QLinearGradient(ex, ey - h * 0.55, ex, ey - h * 4)
        beam_grad.setColorAt(0.0, QColor(255, 200, 90, 70))
        beam_grad.setColorAt(1.0, QColor(255, 200, 90, 0))
        p.setBrush(QBrush(beam_grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(cone)

        # Body.
        body = QPolygonF([
            QPointF(ex - w, ey + h * 0.45),
            QPointF(ex - w * 0.9, ey - h * 0.5),
            QPointF(ex + w * 0.9, ey - h * 0.5),
            QPointF(ex + w, ey + h * 0.45),
        ])
        grad = QLinearGradient(ex, ey - h, ex, ey + h)
        grad.setColorAt(0.0, _lighten(QPALETTE["ego"], 0.04))
        grad.setColorAt(1.0, _darken(QPALETTE["ego"], 0.08))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QPALETTE["ego_outline"], 1.5))
        p.drawPolygon(body)
        # Roof.
        roof = QPolygonF([
            QPointF(ex - w * 0.55, ey + h * 0.15),
            QPointF(ex - w * 0.5, ey - h * 0.25),
            QPointF(ex + w * 0.5, ey - h * 0.25),
            QPointF(ex + w * 0.55, ey + h * 0.15),
        ])
        p.setBrush(QBrush(QColor(225, 220, 215)))
        p.setPen(QPen(QPALETTE["ego_outline"], 0.8))
        p.drawPolygon(roof)
        # Windshield.
        wind = QPolygonF([
            QPointF(ex - w * 0.45, ey - h * 0.05),
            QPointF(ex - w * 0.42, ey - h * 0.22),
            QPointF(ex + w * 0.42, ey - h * 0.22),
            QPointF(ex + w * 0.45, ey - h * 0.05),
        ])
        p.setBrush(QBrush(QColor(140, 170, 200)))
        p.drawPolygon(wind)
        # Headlights.
        p.setBrush(QBrush(QColor(255, 240, 200)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(ex - w * 0.7, ey - h * 0.45), 3, 3)
        p.drawEllipse(QPointF(ex + w * 0.7, ey - h * 0.45), 3, 3)

    # ---- HUD ----------------------------------------------------------- #

    def _draw_hud(self, p: QPainter, snap: SceneSnapshot) -> None:
        if snap is None:
            return
        # Top status bar.
        p.fillRect(QRectF(0, 0, self.W, 70), QPALETTE["hud_panel"])
        p.setPen(QPen(QPALETTE["hud_panel_edge"], 1))
        p.drawLine(0, 70, self.W, 70)

        if snap.ego_speed_kmh is not None:
            head_main = f"{int(round(snap.ego_speed_kmh))}"
            head_unit = "KM/H"
        else:
            head_main = f"{snap.display_fps:0.0f}" if snap.display_fps else "--"
            head_unit = "FPS"
        big_font = QFont("Segoe UI", 28, QFont.Weight.Bold)
        small_font = QFont("Segoe UI", 9)
        p.setFont(big_font)
        p.setPen(QPen(QPALETTE["hud_text"]))
        p.drawText(QPointF(28, 50), head_main)
        head_w = p.fontMetrics().horizontalAdvance(head_main)
        p.setFont(small_font)
        p.setPen(QPen(QPALETTE["hud_dim"]))
        p.drawText(QPointF(28 + head_w + 10, 48), head_unit)

        # Center stats.
        cx = self.W // 2
        risk_col = _risk_color(snap.global_level, snap.global_score)
        self._stat(p, cx - 240, 22, "OBJECTS", str(snap.n_objects), QPALETTE["hud_text"])
        self._stat(p, cx - 80, 22, "PRIMARY",
                   f"#{snap.primary_threat_id}" if snap.primary_threat_id is not None else "--",
                   QPALETTE["hud_accent"])
        self._stat(p, cx + 80, 22, "RISK",
                   f"{snap.global_score:0.0f}" if isinstance(snap.global_score, (int, float)) else "--",
                   risk_col)
        if snap.global_action:
            self._action_pill(p, self.W - 220, 16, snap.global_action, risk_col)

        # Side panels.
        self._panel(p, 16, 96, 220, 130, "AUTOPILOT")
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        p.setPen(QPen(QPALETTE["hud_accent"]))
        p.drawText(QPointF(28, 150), "LANE-CENTERED")
        p.setPen(QPen(QPALETTE["risk_low"]))
        p.drawText(QPointF(28, 174), "VISION OK")
        if snap.ego_speed_kmh is not None:
            p.setPen(QPen(QPALETTE["hud_text"]))
            p.drawText(QPointF(28, 200), f"SET: {int(round(snap.ego_speed_kmh))} KM/H")
        else:
            p.setPen(QPen(QPALETTE["hud_dim"]))
            p.drawText(QPointF(28, 200), "TARGET: ---")

        rx = self.W - 240
        self._panel(p, rx, 96, 224, 150, "PRIMARY THREAT")
        prim = next((it for it in snap.items if it.get("track_id") == snap.primary_threat_id), None) if snap.primary_threat_id is not None else None
        if prim is None:
            p.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
            p.setPen(QPen(QPALETTE["risk_low"]))
            p.drawText(QPointF(rx + 14, 150), "NONE")
            p.setFont(QFont("Segoe UI", 9))
            p.setPen(QPen(QPALETTE["hud_dim"]))
            p.drawText(QPointF(rx + 14, 174), "PATH CLEAR")
        else:
            cls = prim.get("class", "?").upper()
            tid = prim.get("track_id")
            score = prim.get("score")
            level = prim.get("level")
            col = _risk_color(level, score)
            p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            p.setPen(QPen(col))
            p.drawText(QPointF(rx + 14, 144), f"{cls} #{tid}")
            p.setFont(QFont("Segoe UI", 10))
            p.setPen(QPen(QPALETTE["hud_text"]))
            score_text = f"{score:0.0f}" if isinstance(score, (int, float)) else "--"
            p.drawText(QPointF(rx + 14, 168), f"RISK {score_text}")
            if level:
                p.setPen(QPen(col))
                p.drawText(QPointF(rx + 14, 192), str(level))
            # Bar.
            bar_y = 210
            bar_w = 200
            p.fillRect(QRectF(rx + 14, bar_y, bar_w, 6), QColor(50, 52, 58))
            if isinstance(score, (int, float)):
                fill = int(bar_w * max(0.0, min(100.0, float(score))) / 100.0)
                p.fillRect(QRectF(rx + 14, bar_y, fill, 6), col)

        # Speed limit sign.
        self._speed_limit_sign(p, rx + 174, 270, snap.speed_limit_kmh)

        # Action banner.
        if snap.global_action and snap.global_action.upper() in {"ALERT", "BRAKE"}:
            bar_h = 44
            y0 = self.H - bar_h - 32
            banner = QColor(_risk_color(snap.global_level, snap.global_score))
            banner.setAlpha(150)
            p.fillRect(QRectF(0, y0, self.W, bar_h), banner)
            p.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
            p.setPen(QPen(QColor(20, 20, 24)))
            p.drawText(QPointF(32, y0 + 30), f"!  {snap.global_action.upper()}")

    def _stat(self, p: QPainter, x: int, y: int, label: str, value: str, color: QColor) -> None:
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QPALETTE["hud_dim"]))
        p.drawText(QPointF(x, y + 12), label)
        p.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        p.setPen(QPen(color))
        p.drawText(QPointF(x, y + 36), value)

    def _action_pill(self, p: QPainter, x: int, y: int, text: str, color: QColor) -> None:
        rect = QRectF(x, y, 200, 38)
        path = QPainterPath()
        path.addRoundedRect(rect, 8, 8)
        p.fillPath(path, color)
        p.setPen(QPen(QPALETTE["hud_panel_edge"], 1))
        p.drawPath(path)
        p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        p.setPen(QPen(QColor(20, 20, 24)))
        ts = p.fontMetrics().horizontalAdvance(text)
        p.drawText(QPointF(x + (200 - ts) / 2, y + 25), text)

    def _panel(self, p: QPainter, x: int, y: int, w: int, h: int, title: str) -> None:
        path = QPainterPath()
        path.addRoundedRect(QRectF(x, y, w, h), 8, 8)
        p.fillPath(path, QPALETTE["hud_panel"])
        p.setPen(QPen(QPALETTE["hud_panel_edge"], 1))
        p.drawPath(path)
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QPen(QPALETTE["hud_dim"]))
        p.drawText(QPointF(x + 12, y + 22), title)

    def _speed_limit_sign(self, p: QPainter, cx: int, cy: int, kmh: Optional[float]) -> None:
        r = 32
        p.setBrush(QBrush(QPALETTE["speed_sign_bg"]))
        p.setPen(QPen(QPALETTE["speed_sign_ring"], 4))
        p.drawEllipse(QPointF(cx, cy), r, r)
        text = f"{int(round(kmh))}" if isinstance(kmh, (int, float)) else "--"
        p.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        p.setPen(QPen(QPALETTE["speed_sign_text"]))
        ts = p.fontMetrics().horizontalAdvance(text)
        p.drawText(QPointF(cx - ts / 2, cy + 6), text)
        p.setFont(QFont("Segoe UI", 7))
        p.setPen(QPen(QPALETTE["hud_dim"]))
        p.drawText(QPointF(cx - 36, cy + r + 18), "SPEED LIMIT")

    def _draw_footer(self, p: QPainter) -> None:
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QPen(QPALETTE["hud_dim"]))
        msg = "TESLA-INSPIRED EDUCATIONAL VISUALIZATION  -  NOT A REAL AUTONOMOUS SYSTEM"
        ts = p.fontMetrics().horizontalAdvance(msg)
        p.drawText(QPointF(self.W / 2 - ts / 2, self.H - 10), msg)

    def _draw_debug_panel(self, p: QPainter, snap: SceneSnapshot) -> None:
        lines = [
            f"DISP {snap.display_fps:0.1f}",
            f"DET {snap.detection_fps:0.1f}",
            f"YOLO {snap.yolo_latency_ms:0.1f}ms",
            f"DROP {int(snap.dropped_detections)}",
            f"OBJ {snap.n_objects}",
            f"LANE Δ {snap.lane_curve:+0.2f}",
        ]
        x = 16
        y = self.H - 32 - 16 * len(lines)
        p.setFont(QFont("Consolas", 9))
        p.setPen(QPen(QPALETTE["hud_text"]))
        for line in lines:
            p.drawText(QPointF(x, y), line)
            y += 16

    def _draw_object_debug_label(self, p, sx, sy, h, it) -> None:
        p.setFont(QFont("Consolas", 7))
        p.setPen(QPen(QPALETTE["hud_dim"]))
        tag = (
            f"#{it.get('track_id')}  {it.get('class', '?')}  "
            f"lane={it.get('lane', '?')}  "
            f"close={it.get('closeness', 0):0.2f}"
        )
        p.drawText(QPointF(sx - 40, sy - h / 2 - 4), tag)

    def _draw_mode_tag(self, p, text: str) -> None:
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.setPen(QPen(QPALETTE["hud_accent"]))
        p.drawText(QPointF(self.W - 380, self.H - 32), text)
