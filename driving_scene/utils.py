"""
utils.py — Tesla-Inspired Driving Scene Visualizer

Shared palette, color helpers, math helpers, and class-name groupings used by
both the scene renderer and the HUD. Keeping these in one place lets the
renderer and the HUD agree on what "risk_high" looks like without circular
imports.

Public:
    PALETTE                     dict[str, BGR tuple]
    RISK_LEVEL_COLOR            dict mapping risk levels to PALETTE colors
    VEHICLE_CLASSES             set of class names treated as vehicles
    PED_CLASSES                 set of class names treated as pedestrians
    BIKE_CLASSES                set of class names treated as bicycles/cyclists
    risk_color_for(level, score) -> BGR tuple
    lerp(a, b, t) -> float
    ema(prev, new, alpha) -> float
    safe_get(obj, *attrs, default=None) -> Any
    lighten(c, amount) -> BGR tuple
    darken(c, amount) -> BGR tuple
    tint_vehicle(risk_color) -> BGR tuple

All colors are BGR (OpenCV convention).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


# ---------------------------------------------------------------------- #
# Palette                                                                #
# ---------------------------------------------------------------------- #

PALETTE = {
    "bg_top": (18, 18, 22),
    "bg_bottom": (8, 8, 10),
    "road": (38, 40, 44),
    "road_edge": (60, 62, 68),
    "lane_solid": (210, 215, 220),
    "lane_dash": (235, 240, 245),
    "lane_glow": (255, 200, 120),
    "lane_yellow": (40, 200, 230),
    "horizon": (28, 30, 36),
    "ego": (245, 245, 248),
    "ego_outline": (110, 130, 150),
    "ego_beam": (90, 200, 255),
    "vehicle_base": (170, 175, 180),
    "vehicle_outline": (40, 42, 48),
    "ped": (120, 200, 255),
    "bike": (180, 220, 120),
    "risk_low": (160, 220, 120),
    "risk_med": (80, 200, 240),
    "risk_high": (60, 120, 240),
    "risk_crit": (60, 60, 240),
    "primary_ring": (255, 255, 255),
    "hud_panel": (24, 26, 32),
    "hud_panel_edge": (70, 75, 84),
    "hud_text": (235, 240, 245),
    "hud_dim": (140, 150, 160),
    "hud_accent": (255, 200, 120),
    "speed_sign_bg": (245, 245, 248),
    "speed_sign_ring": (40, 40, 220),
    "speed_sign_text": (20, 20, 24),
    "planned_path": (255, 180, 60),
    "planned_path_glow": (255, 220, 120),
    "trail": (200, 170, 110),
}


RISK_LEVEL_COLOR = {
    "LOW": PALETTE["risk_low"],
    "MEDIUM": PALETTE["risk_med"],
    "HIGH": PALETTE["risk_high"],
    "CRITICAL": PALETTE["risk_crit"],
}


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "motorbike"}
PED_CLASSES = {"person", "pedestrian"}
BIKE_CLASSES = {"bicycle", "bike"}


# ---------------------------------------------------------------------- #
# Math helpers                                                           #
# ---------------------------------------------------------------------- #

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ema(prev: Optional[float], new: float, alpha: float) -> float:
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


# ---------------------------------------------------------------------- #
# Color helpers                                                          #
# ---------------------------------------------------------------------- #

def risk_color_for(level: Optional[str], score: Optional[float]) -> Tuple[int, int, int]:
    if level and level in RISK_LEVEL_COLOR:
        return RISK_LEVEL_COLOR[level]
    if score is None:
        return PALETTE["risk_low"]
    s = max(0.0, min(100.0, float(score))) / 100.0
    if s < 0.33:
        return PALETTE["risk_low"]
    if s < 0.66:
        return PALETTE["risk_med"]
    if s < 0.88:
        return PALETTE["risk_high"]
    return PALETTE["risk_crit"]


def lighten(c: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    return (
        min(255, int(c[0] + (255 - c[0]) * amount)),
        min(255, int(c[1] + (255 - c[1]) * amount)),
        min(255, int(c[2] + (255 - c[2]) * amount)),
    )


def darken(c: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    k = max(0.0, 1.0 - amount)
    return (int(c[0] * k), int(c[1] * k), int(c[2] * k))


def tint_vehicle(risk_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    base = PALETTE["vehicle_base"]
    return (
        int(0.55 * base[0] + 0.45 * risk_color[0]),
        int(0.55 * base[1] + 0.45 * risk_color[1]),
        int(0.55 * base[2] + 0.45 * risk_color[2]),
    )


# ---------------------------------------------------------------------- #
# Misc helpers                                                           #
# ---------------------------------------------------------------------- #

def safe_get(obj: Any, *attrs: str, default: Any = None) -> Any:
    """Try attrs in order; return first non-None hit, else default.

    Handles both objects (uses getattr) and dicts (uses [] access). Lets HUD
    code work whether the caller hands it a dataclass, a SimpleNamespace, or
    a plain dict.
    """
    if obj is None:
        return default
    for a in attrs:
        if hasattr(obj, a):
            v = getattr(obj, a)
            if v is not None:
                return v
        if isinstance(obj, dict) and a in obj:
            return obj[a]
    return default
