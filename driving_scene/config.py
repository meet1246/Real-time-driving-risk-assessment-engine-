"""Central configuration and COCO class filtering for road-relevant objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

# When --source is omitted: if this path exists on disk, it is used first (then data/*.mp4, then webcam).
# Defaults to data/dashcam.mp4 — a clean 10s Toronto 720p/60fps urban clip with cars,
# pedestrians, and visible lane markings (matches the project description).
PREFERRED_DEFAULT_DRIVE_VIDEO = Path(__file__).resolve().parent.parent / "data" / "dashcam.mp4"

# COCO class ids used by YOLOv8 (same as COCO 80-class)
COCO_CLASS_IDS: FrozenSet[int] = frozenset({0, 1, 2, 3, 5, 7})

COCO_ID_TO_NAME: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Centroid + IoU hybrid tracking (lower match_cost = better)
MAX_MATCH_DISTANCE_PX: float = 120.0
MAX_MISSING_FRAMES: int = 15
TRACK_HISTORY_LEN: int = 30
TRACK_MATCH_DIST_WEIGHT: float = 0.48
TRACK_MATCH_IOU_WEIGHT: float = 0.52
TRACK_MATCH_MAX_COST: float = 0.88  # above this → new ID

# Motion
MIN_DT_SEC: float = 1e-6
MOTION_HISTORY_FOR_VEL: int = 5

# Scene visualization calibration (frontend only)
# These tune the bbox→scene mapping and smoothing for the Tesla-inspired scene view.
SCENE_HORIZON_Y: float = 0.10  # fraction of scene height
SCENE_EGO_Y: float = 0.93  # fraction of scene height
SCENE_LATERAL_GAIN: float = 1.00  # >1 = wider lateral spread, <1 = tighter
SCENE_DEPTH_BOTTOM_Y_WEIGHT: float = 0.58  # bbox bottom-y dominance
SCENE_DEPTH_AREA_WEIGHT: float = 0.22  # area contributes to depth proxy
SCENE_DEPTH_HEIGHT_WEIGHT: float = 0.20  # bbox height contributes to depth proxy
SCENE_SMOOTH_ALPHA: float = 0.20  # position responsiveness (lower = smoother)
SCENE_SCALE_SMOOTH_ALPHA: float = 0.12  # scale responsiveness

# Backwards-compatible alias (older name)
SCENE_DEPTH_Y_WEIGHT: float = SCENE_DEPTH_BOTTOM_Y_WEIGHT

# Lane/path estimation (frontend only; heuristic)
LANE_RESIZE_WIDTH: int = 640  # smaller = faster, keep >=480 for stability
LANE_CANNY1: int = 60
LANE_CANNY2: int = 150
LANE_HOUGH_THRESHOLD: int = 32
LANE_HOUGH_MIN_LINE_LEN: int = 24
LANE_HOUGH_MAX_LINE_GAP: int = 30
LANE_ROI_Y_TOP: float = 0.55  # fraction of image height (lower half)
LANE_SMOOTH_ALPHA: float = 0.22
LANE_KEEP_FRAMES: int = 18
LANE_CURVE_GAIN: float = 0.72  # damped vs older builds — less over-curve in FSD
LANE_OBJECT_CURVE_FAR_GAIN: float = 0.55  # far objects follow curve more than near

# Ego lane calibration (image space, frontend only)
EGO_LANE_CENTER_BIAS: float = 0.0  # in approximate lane-width units; shifts ego lane center x
LANE_WIDTH_ESTIMATE_RATIO: float = 0.22  # fallback lane width as fraction of frame width
LANE_ASSIGN_SMOOTH_FRAMES: int = 8  # require this many consistent frames before lane_id changes

# Lane curvature output smoothing / FSD limits (frontend only)
LANE_CURVE_SMOOTH_ALPHA: float = 0.12  # EMA on curve_strength (lower = smoother)
LANE_CURVE_MAX_OFFSET_RATIO: float = 0.18  # max lateral curve offset as fraction of scene width
LANE_CONFIDENCE_MIN_FOR_CURVE: float = 0.45  # below this, blend FSD lanes toward straight

# Manual dashcam lane calibration (normalized bbox center x = cx / frame_width).
# Five lanes separated by six x-cut positions; tune visually using overlay + keyboard in app.
LANE_X_BOUNDARIES: Tuple[float, ...] = (
    0.00,
    0.28,
    0.43,
    0.58,
    0.73,
    1.00,
)
LANE_LABELS: Tuple[str, ...] = (
    "far_left",
    "left_lane",
    "ego_lane",
    "right_lane",
    "far_right",
)


# ---------------------------------------------------------------------- #
# Visual asset configuration                                             #
# ---------------------------------------------------------------------- #

# When True, the scene renderer will look in ASSET_DIR for transparent PNGs
# (e.g. assets/vehicles/car_white.png) and use them in place of procedural
# icons. If a PNG isn't there, it falls back to procedural drawing — so the
# project ships and runs fine with zero assets.
USE_ICON_ASSETS: bool = True
ASSET_DIR: Path = Path(__file__).resolve().parent.parent / "assets"

# Stable per-track body colors (BGR for OpenCV). The renderer picks
# palette[track_id % len(palette)] so a car keeps the same color across the
# whole clip, instead of flickering as risk fluctuates. Risk is now
# indicated by the glow halo + the primary-threat ring + the HUD readouts.
VEHICLE_COLOR_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (235, 235, 240),  # off-white
    (205, 205, 210),  # silver
    (190, 165, 130),  # muted blue
    (160, 200, 170),  # soft sage
    (160, 195, 220),  # warm cream
    (200, 175, 200),  # dusty lavender
)


# Calibration persistence. If data/lane_calibration.json exists, load_lane_calibration()
# returns the saved boundaries; main.py / qt_app prefer that over the hard-coded defaults.
LANE_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "lane_calibration.json"


def load_lane_calibration() -> Tuple[Optional[List[float]], Optional[List[str]]]:
    """Return (boundaries, labels) from data/lane_calibration.json if it exists, else (None, None)."""
    try:
        import json
        if not LANE_CALIBRATION_PATH.exists():
            return None, None
        with LANE_CALIBRATION_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        b = data.get("boundaries")
        l = data.get("labels")
        if not isinstance(b, list) or not isinstance(l, list):
            return None, None
        if len(b) < 2 or len(l) != len(b) - 1:
            return None, None
        return [float(x) for x in b], [str(x) for x in l]
    except Exception:
        return None, None


def save_lane_calibration(boundaries: List[float], labels: List[str]) -> None:
    """Persist boundaries + labels to data/lane_calibration.json (atomic)."""
    import json
    LANE_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"boundaries": list(boundaries), "labels": list(labels)}
    tmp = LANE_CALIBRATION_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(LANE_CALIBRATION_PATH)

# FSD scene clutter + fade (frontend only)
MIN_RENDER_BBOX_HEIGHT_RATIO: float = 0.035
MIN_RENDER_BBOX_AREA_RATIO: float = 0.0008
FAR_OBJECT_FADE_START: float = 0.05  # start fading when bbox height ratio falls below this (above hard min)
MIN_SCENE_DEPTH_CONFIDENCE: float = 0.16  # lane_conf × depth proxy gate for hiding

# Aliases (older names)
MIN_SCENE_BBOX_HEIGHT_RATIO: float = MIN_RENDER_BBOX_HEIGHT_RATIO
MIN_SCENE_BBOX_AREA_RATIO: float = MIN_RENDER_BBOX_AREA_RATIO

# Vehicle-only depth tuning (reduce “lead car too close” from oversized height/area)
SCENE_VEHICLE_DEPTH_GAMMA: float = 0.88  # <1 pushes typical vehicles slightly farther in scene_y

# Risk: dampen geometry when bbox is tiny (monocular "far" proxy) — avoids far car → HIGH
RISK_REL_AREA_REFERENCE: float = 0.04  # fraction of frame area where geometry cues reach full weight
# Smooth only HUD global score/action (per-object boxes stay instantaneous)
RISK_GLOBAL_DISPLAY_SMOOTH_ALPHA: float = 0.18

# TTC (approximate — monocular, no depth)
# Fractional area growth per second above this suggests finite TTC
TTC_AREA_GROWTH_EPS: float = 0.02
TTC_MAX_SEC: float = 60.0
TTC_CAP_FROM_GROWTH: float = 30.0  # cap raw estimate before blending with other signals

# Risk thresholds (0–100 score)
RISK_LOW_MAX: float = 35.0
RISK_MED_MAX: float = 55.0
RISK_HIGH_MAX: float = 75.0
# Above RISK_HIGH_MAX → CRITICAL

# TTC contribution (seconds)
TTC_ALERT_SEC: float = 4.0
TTC_HIGH_SEC: float = 2.0

# UI colors (BGR)
COLOR_SAFE = (60, 200, 80)
COLOR_MONITOR = (0, 220, 255)
COLOR_ALERT = (0, 140, 255)
COLOR_BRAKE = (60, 60, 255)
COLOR_HUD_BG = (24, 28, 32)
COLOR_TEXT_DIM = (160, 170, 180)
COLOR_PRIMARY_THREAT = (100, 80, 255)

# Recording
OUTPUT_DIR = "outputs"
OUTPUT_VIDEO_NAME = "risk_demo.mp4"
RECORD_FPS_FALLBACK = 20.0


@dataclass(frozen=True)
class PerformancePreset:
    """Built-in CPU-friendly profiles (override via CLI flags)."""

    name: str
    model_path: str
    imgsz: int
    detect_every_n_frames: int  # run YOLO every N **decoded video** frames


PERFORMANCE_PRESETS: Dict[str, PerformancePreset] = {
    "fast": PerformancePreset("fast", "yolov8n.pt", 416, 3),
    # Default — keeps yolov8n for CPU but pairs with stricter conf + 3-frame
    # consecutive filter to keep false positives down. Use --performance quality
    # if you have CUDA or accept a 2-5x slowdown for yolov8s accuracy.
    "balanced": PerformancePreset("balanced", "yolov8n.pt", 640, 2),
    "quality": PerformancePreset("quality", "yolov8s.pt", 640, 1),
}

DISPLAY_TARGET_FPS: float = 60.0


@dataclass
class RuntimeConfig:
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.35
    device: str | None = None  # None = ultralytics default
    imgsz: int = 416  # smaller = faster (try 512 or 640 if you need accuracy)
    half: bool = False  # FP16 inference when True (use on CUDA for speed)

    def __post_init__(self) -> None:
        if not (0.0 < self.conf_threshold <= 1.0):
            raise ValueError("conf_threshold must be in (0, 1]")
        if not (320 <= self.imgsz <= 1280):
            raise ValueError("imgsz must be between 320 and 1280")
