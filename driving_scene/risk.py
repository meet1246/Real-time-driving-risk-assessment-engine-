"""
Approximate TTC and risk scoring (monocular — educational prototype).

Time-to-collision (TTC) here is NOT from ranging or LiDAR. It uses bbox **area
growth rate** as a weak proxy for “getting closer.” When area is flat or shrinking,
we treat TTC as **infinity** (no closing cue from this signal). Do not use for
real vehicle safety decisions.
"""

from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .config import (
    RISK_REL_AREA_REFERENCE,
    TTC_ALERT_SEC,
    TTC_AREA_GROWTH_EPS,
    TTC_CAP_FROM_GROWTH,
    TTC_HIGH_SEC,
    TTC_MAX_SEC,
    RISK_HIGH_MAX,
    RISK_LOW_MAX,
    RISK_MED_MAX,
)
from .motion import MotionState
from .tracker import TrackedObject


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Decision(str, Enum):
    SAFE = "SAFE"
    MONITOR = "MONITOR"
    ALERT = "ALERT"
    BRAKE = "BRAKE"


@dataclass
class RiskComponents:
    """Populated for debug overlay."""

    class_weight: float
    center_proximity: float
    vertical_depth_proxy: float
    approach_velocity: float
    area_growth: float
    ttc_component: float


@dataclass
class ObjectRisk:
    track_id: int
    class_name: str
    risk_level: RiskLevel
    risk_score: float  # 0–100
    ttc_sec: float  # math.inf when not approaching by area proxy
    decision: Decision
    components: RiskComponents
    bbox: Tuple[float, float, float, float]


@dataclass
class FrameRiskResult:
    per_object: List[ObjectRisk]
    global_risk_level: RiskLevel
    global_decision: Decision
    global_score: float
    primary_threat_id: Optional[int]


def _level_from_score(score: float) -> RiskLevel:
    if score <= RISK_LOW_MAX:
        return RiskLevel.LOW
    if score <= RISK_MED_MAX:
        return RiskLevel.MEDIUM
    if score <= RISK_HIGH_MAX:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def _decision_from_level(level: RiskLevel) -> Decision:
    return {
        RiskLevel.LOW: Decision.SAFE,
        RiskLevel.MEDIUM: Decision.MONITOR,
        RiskLevel.HIGH: Decision.ALERT,
        RiskLevel.CRITICAL: Decision.BRAKE,
    }[level]


def approximate_ttc_from_area(
    area: float,
    area_change_rate: float,
) -> float:
    """
    Approximate time-to-collision using bbox area growth as a proxy for approach.

    Heuristic: positive fractional growth rate (1/A * dA/dt) suggests closing;
    rough scale: ttc ≈ 1 / growth_rate_sec when growth_rate_sec > eps.

    This is NOT geometrically accurate — monocular video has no metric depth.
    If area is not increasing, returns +inf.
    """
    if area <= 1.0 or area_change_rate <= TTC_AREA_GROWTH_EPS:
        return float("inf")
    growth_rate = area_change_rate / area  # 1/sec scale
    if growth_rate <= TTC_AREA_GROWTH_EPS:
        return float("inf")
    ttc = 1.0 / growth_rate
    return float(min(max(ttc, 0.05), TTC_CAP_FROM_GROWTH))


def _ego_forward_image_dy_deg(frame_h: int, motion: MotionState) -> float:
    """Angle of velocity vs 'into scene' (roughly downward in image)."""
    # Forward in driving dash view ≈ increasing y (down). Unit forward = (0, 1)
    vx, vy = motion.velocity_x, motion.velocity_y
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        return 90.0
    mag = math.hypot(vx, vy)
    if mag < 1e-6:
        return 90.0
    # cos between v and (0,1)
    cos_sim = max(-1.0, min(1.0, vy / mag))
    return math.degrees(math.acos(cos_sim))


def score_object(
    track: TrackedObject,
    motion: MotionState,
    frame_w: int,
    frame_h: int,
) -> ObjectRisk:
    x1, y1, x2, y2 = track.last_bbox
    cx, cy = track.last_center
    area = max(1.0, (x2 - x1) * (y2 - y1))

    # Class weight (person highest)
    cw = {
        "person": 28.0,
        "bicycle": 22.0,
        "motorcycle": 20.0,
        "car": 14.0,
        "bus": 12.0,
        "truck": 12.0,
    }.get(track.class_name, 10.0)

    # Near center of frame (normalized distance)
    fcx, fcy = frame_w * 0.5, frame_h * 0.5
    dist = math.hypot(cx - fcx, cy - fcy) / max(frame_w, frame_h)
    center_score = max(0.0, 22.0 * (1.0 - min(dist * 2.2, 1.0)))

    # Lower in frame → closer proxy (y normalized)
    vert = cy / max(float(frame_h), 1.0)
    vertical_score = 18.0 * vert

    # Moving toward ego (downward in image)
    angle_to_forward = _ego_forward_image_dy_deg(frame_h, motion)
    approach_score = max(0.0, 16.0 * (1.0 - min(angle_to_forward / 90.0, 1.0)))

    # Area growing quickly
    growth = motion.bbox_area_change_rate
    growth_norm = max(0.0, growth) / max(area, 1.0)  # per second
    area_score = min(22.0, 120.0 * growth_norm)

    ttc = approximate_ttc_from_area(area, growth)
    if math.isinf(ttc):
        ttc_component = 0.0
    else:
        if ttc < TTC_HIGH_SEC:
            ttc_component = 30.0
        elif ttc < TTC_ALERT_SEC:
            ttc_component = 18.0 + 12.0 * (TTC_ALERT_SEC - ttc) / max(TTC_ALERT_SEC - TTC_HIGH_SEC, 1e-6)
        else:
            ttc_component = max(0.0, 10.0 * (1.0 - min(ttc / TTC_MAX_SEC, 1.0)))

    # Monocular "far" proxy: tiny bbox → weak geometry / TTC cues (reduces far-car → HIGH false alarms)
    frame_area_px = float(frame_w * frame_h)
    rel_area = area / max(frame_area_px, 1.0)
    near_gate = min(1.0, math.sqrt(rel_area / max(RISK_REL_AREA_REFERENCE, 1e-9)))
    geom_core = center_score + vertical_score + approach_score
    geom_weighted = geom_core * near_gate
    area_score_w = area_score * near_gate
    ttc_component_w = ttc_component * near_gate

    raw = cw + geom_weighted + area_score_w + ttc_component_w
    score = float(max(0.0, min(100.0, raw)))
    level = _level_from_score(score)
    decision = _decision_from_level(level)

    comps = RiskComponents(
        class_weight=cw,
        center_proximity=center_score * near_gate,
        vertical_depth_proxy=vertical_score * near_gate,
        approach_velocity=approach_score * near_gate,
        area_growth=area_score_w,
        ttc_component=ttc_component_w,
    )

    return ObjectRisk(
        track_id=track.track_id,
        class_name=track.class_name,
        risk_level=level,
        risk_score=score,
        ttc_sec=ttc,
        decision=decision,
        components=comps,
        bbox=track.last_bbox,
    )


def aggregate_frame_risks(per_object: List[ObjectRisk]) -> Tuple[RiskLevel, Decision, float, Optional[int]]:
    if not per_object:
        return RiskLevel.LOW, Decision.SAFE, 0.0, None
    best = max(per_object, key=lambda o: o.risk_score)
    # Global score: emphasize max but blend mean of top 2
    sorted_o = sorted(per_object, key=lambda o: -o.risk_score)
    top1 = sorted_o[0].risk_score
    top2 = sorted_o[1].risk_score if len(sorted_o) > 1 else 0.0
    global_score = float(min(100.0, 0.72 * top1 + 0.18 * top2))
    g_level = _level_from_score(global_score)
    g_dec = _decision_from_level(g_level)
    return g_level, g_dec, global_score, best.track_id


def compute_frame_risks(
    tracks: Dict[int, TrackedObject],
    motions: Dict[int, MotionState],
    frame_shape: Tuple[int, int, int],
) -> FrameRiskResult:
    h, w = frame_shape[0], frame_shape[1]
    per: List[ObjectRisk] = []
    for tid, tr in tracks.items():
        if tr.missing_frames > 0:
            continue
        mo = motions.get(tid)
        if mo is None:
            continue
        per.append(score_object(tr, mo, w, h))
    gl, gd, gs, pid = aggregate_frame_risks(per)
    return FrameRiskResult(
        per_object=per,
        global_risk_level=gl,
        global_decision=gd,
        global_score=gs,
        primary_threat_id=pid,
    )


def smooth_global_risk_display(
    prev_ema: Optional[float],
    raw: FrameRiskResult,
    alpha: float,
) -> Tuple[FrameRiskResult, float]:
    """
    EMA-smooth **global** score and derived level/decision for stable HUD.
    Per-object scores and box colors still come from the raw `FrameRiskResult` passed to draw_objects.
    """
    if not raw.per_object:
        # Preserve EMA when scene briefly empty (no objects)
        return raw, float(prev_ema) if prev_ema is not None else 0.0
    ema = raw.global_score if prev_ema is None else (alpha * raw.global_score + (1.0 - alpha) * prev_ema)
    ema = max(0.0, min(100.0, float(ema)))
    lvl = _level_from_score(ema)
    dec = _decision_from_level(lvl)
    out = dataclasses.replace(
        raw,
        global_score=ema,
        global_risk_level=lvl,
        global_decision=dec,
    )
    return out, ema
