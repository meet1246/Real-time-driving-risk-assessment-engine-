"""HUD and object overlays — clean minimal layout (details in debug mode)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import (
    COLOR_ALERT,
    COLOR_BRAKE,
    COLOR_HUD_BG,
    COLOR_MONITOR,
    COLOR_PRIMARY_THREAT,
    COLOR_SAFE,
    COLOR_TEXT_DIM,
)
from .motion import MotionState, shift_bbox_by_velocity, trajectory_polyline
from .risk import FrameRiskResult, RiskLevel
from .tracker import TrackedObject

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _risk_palette(level: RiskLevel) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if level == RiskLevel.LOW:
        return COLOR_SAFE, (20, 60, 20)
    if level == RiskLevel.MEDIUM:
        return COLOR_MONITOR, (40, 80, 80)
    if level == RiskLevel.HIGH:
        return COLOR_ALERT, (40, 70, 100)
    return COLOR_BRAKE, (40, 40, 80)


def _poly_line_points_from_centers(
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


def _ttc_short(ttc: float) -> str:
    if math.isinf(ttc) or ttc > 1e5:
        return "—"
    return f"{ttc:.0f}s"


def _draw_label_chip(
    frame: np.ndarray,
    x: int,
    y_top: int,
    text: str,
    accent: Tuple[int, int, int],
    bold: bool,
) -> None:
    """Single-line label with soft dark backing (readable on bright video)."""
    scale = 0.52 if bold else 0.48
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thick)
    pad_x, pad_y = 6, 4
    x2 = min(frame.shape[1] - 2, x + tw + pad_x * 2)
    y1 = max(2, y_top - th - pad_y)
    y2 = y_top + baseline + 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y1), (x2, y2), (22, 26, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (x, y1), (x2, y1), accent, 2 if bold else 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (x + pad_x, y2 - pad_y),
        _FONT,
        scale,
        (235, 238, 242),
        thick,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    tel_fps: float,
    frame_latency_ms: float,
    n_tracks: int,
    risk_result: FrameRiskResult,
    bottom_status: str,
    perf_warning: str | None = None,
) -> None:
    """
    Compact top-left card (2 lines) + single bottom status line.
    """
    h, w = frame.shape[:2]
    pad = 12
    box_w = min(340, w - 2 * pad)
    line_h = 22
    box_h = 52

    x0, y0 = pad, pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)

    accent, _ = _risk_palette(risk_result.global_risk_level)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + 2), accent, -1)

    primary = "—"
    if risk_result.primary_threat_id is not None:
        primary = f"#{risk_result.primary_threat_id}"

    # Line 1: risk + action (main readout)
    line1 = f"{risk_result.global_risk_level.value}  {risk_result.global_score:.0f}  →  {risk_result.global_decision.value}"
    cv2.putText(
        frame,
        line1,
        (x0 + 10, y0 + line_h),
        _FONT,
        0.58,
        accent,
        1,
        cv2.LINE_AA,
    )
    # Line 2: fps · tracks · threat (quiet)
    line2 = f"{tel_fps:.0f} fps   ·   {n_tracks} obj   ·   threat {primary}   ·   {frame_latency_ms:.0f} ms"
    cv2.putText(
        frame,
        line2,
        (x0 + 10, y0 + line_h + 18),
        _FONT,
        0.42,
        COLOR_TEXT_DIM,
        1,
        cv2.LINE_AA,
    )

    if perf_warning:
        short = perf_warning.replace("PERF WARNING: ", "").strip()
        tw = int(cv2.getTextSize(short, _FONT, 0.48, 1)[0][0])
        tx = max(pad, min((w - tw) // 2, w - tw - pad))
        cv2.putText(
            frame,
            short,
            (tx, y0 + box_h + 18),
            _FONT,
            0.48,
            (70, 95, 255),
            1,
            cv2.LINE_AA,
        )

    # Single slim bottom bar
    yb = h - 28
    bw = min(w - 2 * pad, max(400, len(bottom_status) * 7 + 40))
    bot = frame.copy()
    cv2.rectangle(bot, (pad, yb - 4), (pad + bw, yb + 18), (14, 16, 20), -1)
    cv2.addWeighted(bot, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame,
        bottom_status,
        (pad + 8, yb + 12),
        _FONT,
        0.40,
        (155, 165, 175),
        1,
        cv2.LINE_AA,
    )


def draw_objects(
    frame: np.ndarray,
    risk_result: FrameRiskResult,
    motions: Dict[int, MotionState],
    tracks: Dict[int, TrackedObject],
    debug: bool,
    extrapolate_sec: float = 0.0,
    trajectory_primary_only: bool = True,
) -> None:
    primary_id = risk_result.primary_threat_id
    by_id = {o.track_id: o for o in risk_result.per_object}
    fh, fw = frame.shape[0], frame.shape[1]

    for tid, tr in tracks.items():
        if tr.missing_frames > 0:
            continue
        orisk = by_id.get(tid)
        if orisk is None:
            continue
        mo = motions.get(tid)
        bbox_draw = orisk.bbox
        if extrapolate_sec > 0 and mo is not None:
            bbox_draw = shift_bbox_by_velocity(orisk.bbox, mo.velocity_x, mo.velocity_y, extrapolate_sec, fw, fh)
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_draw]
        color, _ = _risk_palette(orisk.risk_level)
        is_primary = tid == primary_id
        if is_primary:
            color = COLOR_PRIMARY_THREAT
        thick = 2 if is_primary else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # One compact label above box (debug adds second line)
        if debug:
            label = f"#{tid}  {orisk.class_name}  {orisk.risk_level.value}  {orisk.risk_score:.0f}  ·  {_ttc_short(orisk.ttc_sec)}"
            _draw_label_chip(frame, x1, y1 - 2, label, color, bold=is_primary)
        else:
            label = f"{orisk.class_name}  #{tid}  ·  {orisk.risk_score:.0f}"
            _draw_label_chip(frame, x1, y1 - 2, label, color, bold=is_primary)

        # Trajectory lines only on primary threat (keeps scene readable)
        draw_traj = mo is not None and (is_primary or not trajectory_primary_only)
        if draw_traj and mo is not None:
            if extrapolate_sec > 0:
                dx = mo.velocity_x * extrapolate_sec
                dy = mo.velocity_y * extrapolate_sec
                c0 = (bbox_draw[0] + bbox_draw[2]) * 0.5, (bbox_draw[1] + bbox_draw[3]) * 0.5
                c1 = mo.pred_center_2s[0] + dx, mo.pred_center_2s[1] + dy
                pts = _poly_line_points_from_centers(c0, c1)
            else:
                pts = trajectory_polyline(tr, mo)
            tc = tuple(int(c * 0.55 + 120 * 0.45) for c in color)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], tc, 1, cv2.LINE_AA)

        if debug and mo is not None:
            cx = int(round((bbox_draw[0] + bbox_draw[2]) * 0.5))
            cy = int(round((bbox_draw[1] + bbox_draw[3]) * 0.5))
            vx, vy = mo.velocity_x, mo.velocity_y
            scale = 0.28
            cv2.arrowedLine(
                frame,
                (cx, cy),
                (int(cx + vx * scale), int(cy + vy * scale)),
                (160, 175, 220),
                1,
                tipLength=0.22,
            )
            growth = mo.bbox_area_change_rate
            area = max(1.0, (bbox_draw[2] - bbox_draw[0]) * (bbox_draw[3] - bbox_draw[1]))
            gfrac = growth / area
            dbg = f"v {mo.speed_px_per_sec:.0f}px/s  ΔA {growth:.0f}  g {gfrac:.2f}/s"
            cv2.putText(
                frame,
                dbg,
                (x1, min(fh - 4, y2 + 14)),
                _FONT,
                0.36,
                (140, 155, 185),
                1,
                cv2.LINE_AA,
            )
            c = orisk.components
            comp = f"cw{c.class_weight:.0f} ctr{c.center_proximity:.0f} dep{c.vertical_depth_proxy:.0f} app{c.approach_velocity:.0f} ar{c.area_growth:.0f} ttc{c.ttc_component:.0f}"
            cv2.putText(
                frame,
                comp,
                (x1, min(fh - 4, y2 + 28)),
                _FONT,
                0.34,
                (120, 135, 165),
                1,
                cv2.LINE_AA,
            )


def draw_debug_help(frame: np.ndarray, recording: bool, debug: bool, *, manual_lane_help: bool = False) -> None:
    h, w = frame.shape[:2]
    msg = f"Q quit   ·   D debug   ·   R rec"
    if manual_lane_help:
        msg += "   ·   [ ] lane shift   ·   , . ego width   ·   P print bounds"
    if recording:
        msg += " ●"
    if debug:
        msg += "   (detail)"
    tw = cv2.getTextSize(msg, _FONT, 0.38, 1)[0][0]
    x = w - tw - 14
    cv2.putText(
        frame,
        msg,
        (x, h - 12),
        _FONT,
        0.38,
        (130, 140, 150),
        1,
        cv2.LINE_AA,
    )
