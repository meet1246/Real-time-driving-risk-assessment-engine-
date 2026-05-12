"""
hud.py — Tesla-Inspired Driving Scene Visualizer

All HUD / overlay drawing functions. The HUD doesn't carry state — every
function takes `canvas` and the data it needs, and draws on top.

Three flavors of overlay live here:

    1. Scene-mode HUD       — top status bar, autopilot/primary panels,
                              speed-limit sign, action banner, footer.
    2. Classic-mode HUD     — single thin top bar + bbox overlays for
                              dashcam mode.
    3. Debug-mode panels    — per-track mapping info for `--view debug`.

Plus utility composition:
    compose_split(left_frame, scene, W, H)   side-by-side dashcam | scene

Functions take width/height (W, H) explicitly rather than holding a reference
to a renderer, so they're testable in isolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .utils import PALETTE, risk_color_for, safe_get


# ---------------------------------------------------------------------- #
# Scene-mode HUD                                                         #
# ---------------------------------------------------------------------- #

def draw_top_hud(
    canvas: np.ndarray,
    W: int,
    H: int,
    telemetry: Any,
    risk_state: Any,
    ego_speed_kmh: Optional[float],
    primary_threat_id: Optional[int],
    n_objects: int,
) -> None:
    h = 70
    panel = canvas.copy()
    cv2.rectangle(panel, (0, 0), (W, h), PALETTE["hud_panel"], -1)
    cv2.addWeighted(panel, 0.78, canvas, 0.22, 0, dst=canvas)
    cv2.line(canvas, (0, h), (W, h), PALETTE["hud_panel_edge"], 1)
    if ego_speed_kmh is not None:
        head_main = f"{int(round(ego_speed_kmh))}"
        head_unit = "KM/H"
    else:
        disp_fps = safe_get(telemetry, "display_fps", "fps", default=None)
        head_main = f"{disp_fps:0.0f}" if isinstance(disp_fps, (int, float)) else "--"
        head_unit = "FPS"
    cv2.putText(canvas, head_main, (28, 50), cv2.FONT_HERSHEY_DUPLEX, 1.6,
                PALETTE["hud_text"], 2, cv2.LINE_AA)
    head_w = cv2.getTextSize(head_main, cv2.FONT_HERSHEY_DUPLEX, 1.6, 2)[0][0]
    cv2.putText(canvas, head_unit, (28 + head_w + 10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                PALETTE["hud_dim"], 1, cv2.LINE_AA)
    center_x = W // 2
    global_score = safe_get(risk_state, "global_score", "score", default=None)
    global_level = safe_get(risk_state, "global_level", "level", default=None)
    action = safe_get(risk_state, "global_action", "action", default=None)
    rcolor = risk_color_for(global_level, global_score)
    _stat_block(canvas, center_x - 240, 22, "OBJECTS", str(n_objects), PALETTE["hud_text"])
    primary_label = f"#{primary_threat_id}" if primary_threat_id is not None else "--"
    _stat_block(canvas, center_x - 80, 22, "PRIMARY", primary_label, PALETTE["hud_accent"])
    risk_text = f"{global_score:0.0f}" if isinstance(global_score, (int, float)) else "--"
    _stat_block(canvas, center_x + 80, 22, "RISK", risk_text, rcolor)
    if action:
        _action_pill(canvas, W - 220, 16, str(action), rcolor)


def _stat_block(canvas, x, y, label, value, color):
    cv2.putText(canvas, label, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                PALETTE["hud_dim"], 1, cv2.LINE_AA)
    cv2.putText(canvas, value, (x, y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                color, 1, cv2.LINE_AA)


def _action_pill(canvas, x, y, text, color):
    w = 200
    h = 38
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PALETTE["hud_panel_edge"], 1)
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
    tx = x + (w - ts[0]) // 2
    ty = y + (h + ts[1]) // 2 - 2
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 24),
                2, cv2.LINE_AA)


def draw_side_panels(
    canvas: np.ndarray,
    W: int,
    H: int,
    risk_state: Any,
    ego_speed_kmh: Optional[float],
    speed_limit_kmh: Optional[float],
    items: List[Dict[str, Any]],
    primary_threat_id: Optional[int],
) -> None:
    _panel(canvas, 16, 96, 220, 130, "AUTOPILOT")
    cv2.putText(canvas, "LANE-CENTERED", (28, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                PALETTE["hud_accent"], 1, cv2.LINE_AA)
    cv2.putText(canvas, "VISION OK", (28, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                PALETTE["risk_low"], 1, cv2.LINE_AA)
    if ego_speed_kmh is not None:
        cv2.putText(canvas, f"SET: {int(round(ego_speed_kmh))} KM/H", (28, 206),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PALETTE["hud_text"], 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "TARGET: ---", (28, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    PALETTE["hud_dim"], 1, cv2.LINE_AA)

    rx = W - 240
    _panel(canvas, rx, 96, 224, 150, "PRIMARY THREAT")
    primary_item = next((it for it in items if it["track_id"] == primary_threat_id), None)
    if primary_item is None:
        cv2.putText(canvas, "NONE", (rx + 14, 150), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    PALETTE["risk_low"], 1, cv2.LINE_AA)
        cv2.putText(canvas, "PATH CLEAR", (rx + 14, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    PALETTE["hud_dim"], 1, cv2.LINE_AA)
    else:
        cls = primary_item["class"].upper()
        tid = primary_item["track_id"]
        score = primary_item["score"]
        level = primary_item["level"]
        color = risk_color_for(level, score)
        cv2.putText(canvas, f"{cls} #{tid}", (rx + 14, 144), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    color, 1, cv2.LINE_AA)
        score_text = f"{score:0.0f}" if isinstance(score, (int, float)) else "--"
        cv2.putText(canvas, f"RISK {score_text}", (rx + 14, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    PALETTE["hud_text"], 1, cv2.LINE_AA)
        if level:
            cv2.putText(canvas, str(level), (rx + 14, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
        bar_y = 220
        bar_w = 200
        cv2.rectangle(canvas, (rx + 14, bar_y), (rx + 14 + bar_w, bar_y + 6), (50, 52, 58), -1)
        if isinstance(score, (int, float)):
            fill = int(bar_w * max(0.0, min(100.0, float(score))) / 100.0)
            cv2.rectangle(canvas, (rx + 14, bar_y), (rx + 14 + fill, bar_y + 6), color, -1)
    draw_speed_limit_sign(canvas, rx + 174, 270, speed_limit_kmh)


def _panel(canvas, x, y, w, h, title):
    panel = canvas.copy()
    cv2.rectangle(panel, (x, y), (x + w, y + h), PALETTE["hud_panel"], -1)
    cv2.addWeighted(panel, 0.78, canvas, 0.22, 0, dst=canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PALETTE["hud_panel_edge"], 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (x + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                PALETTE["hud_dim"], 1, cv2.LINE_AA)


def draw_speed_limit_sign(canvas, cx, cy, speed_limit_kmh):
    r = 32
    cv2.circle(canvas, (cx, cy), r + 3, PALETTE["speed_sign_ring"], -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r, PALETTE["speed_sign_bg"], -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r, PALETTE["speed_sign_ring"], 3, cv2.LINE_AA)
    text = f"{int(round(speed_limit_kmh))}" if isinstance(speed_limit_kmh, (int, float)) else "--"
    ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0]
    tx = cx - ts[0] // 2
    ty = cy + ts[1] // 2 - 2
    cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.85,
                PALETTE["speed_sign_text"], 2, cv2.LINE_AA)
    cv2.putText(canvas, "SPEED LIMIT", (cx - 36, cy + r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                PALETTE["hud_dim"], 1, cv2.LINE_AA)


def draw_action_banner(canvas: np.ndarray, W: int, H: int, risk_state: Any) -> None:
    action = safe_get(risk_state, "global_action", "action", default=None)
    level = safe_get(risk_state, "global_level", "level", default=None)
    score = safe_get(risk_state, "global_score", "score", default=None)
    if not action:
        return
    color = risk_color_for(level, score)
    if str(action).upper() not in {"ALERT", "BRAKE"}:
        return
    bar_h = 44
    y0 = H - bar_h - 32
    bar = canvas.copy()
    cv2.rectangle(bar, (0, y0), (W, y0 + bar_h), color, -1)
    cv2.addWeighted(bar, 0.55, canvas, 0.45, 0, dst=canvas)
    text = f"!  {str(action).upper()}"
    cv2.putText(canvas, text, (32, y0 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.95,
                (20, 20, 24), 2, cv2.LINE_AA)


def draw_footer_disclaimer(canvas: np.ndarray, W: int, H: int) -> None:
    msg = "TESLA-INSPIRED EDUCATIONAL VISUALIZATION  -  NOT A REAL AUTONOMOUS SYSTEM"
    ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    cv2.putText(canvas, msg, (W // 2 - ts[0] // 2, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                PALETTE["hud_dim"], 1, cv2.LINE_AA)


# ---------------------------------------------------------------------- #
# Debug panel                                                            #
# ---------------------------------------------------------------------- #

def draw_debug_panel(
    canvas: np.ndarray,
    W: int,
    H: int,
    telemetry: Any,
    n_objects: int,
) -> None:
    lines = []
    for fname, attr in [
        ("DISP", "display_fps"),
        ("DET", "detection_fps"),
        ("YOLO", "yolo_latency_ms"),
        ("TRK", "track_latency_ms"),
        ("RISK", "risk_latency_ms"),
        ("DROP", "dropped_detections"),
    ]:
        v = safe_get(telemetry, attr, default=None)
        if isinstance(v, (int, float)):
            if "fps" in attr:
                lines.append(f"{fname} {v:0.1f}")
            elif "latency" in attr:
                lines.append(f"{fname} {v:0.1f}ms")
            else:
                lines.append(f"{fname} {int(v)}")
        else:
            lines.append(f"{fname} --")
    lines.append(f"OBJ {n_objects}")
    x = 16
    y = H - 32 - 16 * len(lines)
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    PALETTE["hud_text"], 1, cv2.LINE_AA)
        y += 16


def draw_debug_overlay(
    canvas: np.ndarray,
    items: List[Dict[str, Any]],
    lane_boundaries: Optional[List[float]] = None,
    lane_labels: Optional[List[str]] = None,
) -> None:
    """For `--view debug` (raw dashcam + mapping info + lane boundary lines).

    Draws on the original dashcam canvas (frame_w, frame_h match canvas).
    """
    h, w = canvas.shape[:2]
    # Vertical lane boundary lines.
    if lane_boundaries:
        for b in lane_boundaries:
            x = int(b * w)
            cv2.line(canvas, (x, int(h * 0.45)), (x, h), (40, 220, 240), 1, cv2.LINE_AA)
    if lane_labels:
        for i, label in enumerate(lane_labels):
            if not lane_boundaries or i + 1 >= len(lane_boundaries):
                continue
            cx = int(0.5 * (lane_boundaries[i] + lane_boundaries[i + 1]) * w)
            cv2.putText(canvas, label, (cx - 30, int(h * 0.48)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 220, 240), 1, cv2.LINE_AA)
    # Per-track mapping info above each bbox.
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        color = risk_color_for(it.get("level"), it.get("score"))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        info = (
            f"#{it['track_id']} {it['class']}  "
            f"cx={(0.5 * (x1 + x2) / max(1, w)):.2f} "
            f"by={(y2 / max(1, h)):.2f}  "
            f"lane={it.get('lane', '?')}"
        )
        cv2.putText(canvas, info, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------- #
# Classic-mode HUD (dashcam-with-boxes)                                  #
# ---------------------------------------------------------------------- #

def draw_classic_overlays(
    canvas: np.ndarray,
    items: List[Dict[str, Any]],
    primary_id: Optional[int],
) -> None:
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        color = risk_color_for(it["level"], it["score"])
        thick = 3 if it["track_id"] == primary_id else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        tag = it["class"].upper()
        if it["track_id"] is not None:
            tag += f" #{it['track_id']}"
        if isinstance(it["score"], (int, float)):
            tag += f"  R:{it['score']:0.0f}"
        cv2.putText(canvas, tag, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_classic_hud(
    canvas: np.ndarray,
    telemetry: Any,
    risk_state: Any,
    ego_speed_kmh: Optional[float],
) -> None:
    h = 36
    bar = canvas.copy()
    cv2.rectangle(bar, (0, 0), (canvas.shape[1], h), PALETTE["hud_panel"], -1)
    cv2.addWeighted(bar, 0.7, canvas, 0.3, 0, dst=canvas)
    gs = safe_get(risk_state, "global_score", "score", default=None)
    gl = safe_get(risk_state, "global_level", "level", default=None)
    ga = safe_get(risk_state, "global_action", "action", default=None)
    df = safe_get(telemetry, "display_fps", "fps", default=None)
    col = risk_color_for(gl, gs)
    s_txt = f"{gs:0.0f}" if isinstance(gs, (int, float)) else "--"
    f_txt = f"{df:0.0f}" if isinstance(df, (int, float)) else "--"
    cv2.putText(canvas, f"FPS {f_txt}   RISK {s_txt}   {gl or ''}   {ga or ''}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    if ego_speed_kmh is not None:
        cv2.putText(canvas, f"{int(round(ego_speed_kmh))} KM/H",
                    (canvas.shape[1] - 130, 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, PALETTE["hud_text"], 1, cv2.LINE_AA)


# ---------------------------------------------------------------------- #
# Composition                                                            #
# ---------------------------------------------------------------------- #

def compose_split(
    left_frame: np.ndarray,
    scene: np.ndarray,
    W: int,
    H: int,
) -> np.ndarray:
    """Side-by-side: raw dashcam on the left, scene on the right.

    The left pane is intentionally box-free so the viewer can compare "what the
    camera sees" vs "what the system reconstructs."
    """
    left = cv2.resize(left_frame, (W // 2, H), interpolation=cv2.INTER_AREA)
    right = cv2.resize(scene, (W // 2, H), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:, : W // 2] = left
    out[:, W // 2 :] = right
    cv2.line(out, (W // 2, 0), (W // 2, H), PALETTE["hud_panel_edge"], 2)
    return out
