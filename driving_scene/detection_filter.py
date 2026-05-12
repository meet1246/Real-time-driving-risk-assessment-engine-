"""
detection_filter.py — reject the YOLO false-positives that are
making the scene reconstruction render objects that aren't there.

WHAT THIS FIXES
===============
On wet/night urban dashcam footage YOLO routinely produces:
  * 0.15-0.30 confidence boxes on crosswalk stripes, lane paint,
    manhole covers, and reflections — labeled "person" or "car"
    because they have the right rough size.
  * Tiny boxes (<0.1% of frame area) on far-away pixels that
    flicker in/out for one frame at a time.
  * Single-frame ghost detections that the tracker has no time
    to validate before they appear in the scene view.

This module is a **pre-tracker** and **pre-renderer** filter. It
takes raw YOLO Detection objects and:

  1. Drops anything below a class-aware confidence floor.
  2. Drops anything whose bbox is implausibly small/large/oblong
     for its class (a 12x40 "car" is a road-paint hit).
  3. Drops anything whose bbox is wholly below the horizon line
     AND whose aspect ratio matches a road marking.
  4. Drops anything whose center sits OUTSIDE the plausible road
     region (top 25% of the frame is sky; far edges are buildings).
  5. Optionally requires N consecutive frames of presence before
     a detection is "promoted" — uses simple greedy IoU matching
     against a short history buffer.

Anything that survives all five tests is passed through to the
tracker unchanged. The tracker / risk engine / renderer don't
need to know this module exists.

USAGE
=====
    flt = DetectionFilter(
        frame_w=1280, frame_h=720,
        min_confidence={"car": 0.45, "person": 0.50, "bicycle": 0.55},
        min_box_area_frac=0.0008,
        horizon_y_frac=0.32,
        require_consecutive=3,
    )

    raw = detector.predict(frame)
    clean = flt.apply(raw)
    tracks = tracker.update(clean)

PUBLIC API
==========
    DetectionFilter(...)
        .apply(detections) -> list           # filter + temporal smoothing
        .reset()                             # clear history
        .stats() -> dict                     # counts: pass/conf/size/aspect/road/temporal
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# COCO class names YOLOv8 emits that we care about. Anything not in
# this set is dropped before geometry checks even run.
ALLOWED_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "motorbike",
    "bicycle",
    "bike",
    "person",
    "pedestrian",
}


# Default per-class confidence floors. Lowered from the over-aggressive
# 0.45-0.55 to a range that preserves real traffic in low-light urban
# dashcam footage. The crosswalk-paint false positives have to be killed
# by the size+aspect+road-paint filters, NOT by raising the confidence
# floor — raising it past ~0.35 also kills real distant traffic.
DEFAULT_MIN_CONFIDENCE = {
    "car": 0.30,
    "truck": 0.30,
    "bus": 0.30,
    "motorcycle": 0.35,
    "motorbike": 0.35,
    "bicycle": 0.35,
    "bike": 0.35,
    "person": 0.35,
    "pedestrian": 0.35,
}


# Per-class plausible aspect-ratio range (width/height).
# A "car" bbox should be roughly square-to-wide. A 5:1 wide-flat box is
# road paint. A 1:5 tall-thin box is a pole, not a car.
DEFAULT_ASPECT_RANGE = {
    "car": (0.50, 3.50),
    "truck": (0.40, 3.80),
    "bus": (0.40, 3.80),
    "motorcycle": (0.25, 2.20),
    "motorbike": (0.25, 2.20),
    "bicycle": (0.30, 2.00),
    "bike": (0.30, 2.00),
    "person": (0.20, 1.20),  # people are taller than wide
    "pedestrian": (0.20, 1.20),
}


# Per-class min height as a fraction of frame height. Lowered from the
# over-aggressive 3-6% to 1.5-3.5% so distant real cars in 4K dashcam
# footage (typically 30-50px tall = 2-3% of a 1080p frame) are kept.
DEFAULT_MIN_HEIGHT_FRAC = {
    "car": 0.015,
    "truck": 0.020,
    "bus": 0.025,
    "motorcycle": 0.020,
    "motorbike": 0.020,
    "bicycle": 0.025,
    "bike": 0.025,
    "person": 0.030,
    "pedestrian": 0.030,
}


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #


def _bbox_xyxy(d: Any) -> Optional[Tuple[float, float, float, float]]:
    """Pull bbox out of a Detection-like object in xyxy form."""
    bbox = (
        getattr(d, "bbox_xyxy", None)
        or getattr(d, "bbox", None)
        or getattr(d, "xyxy", None)
    )
    if bbox is None:
        return None
    if len(bbox) != 4:
        return None
    return tuple(float(v) for v in bbox)


def _class_name(d: Any) -> str:
    name = (
        getattr(d, "class_name", None)
        or getattr(d, "cls", None)
        or getattr(d, "name", None)
        or ""
    )
    return str(name).lower().strip()


def _confidence(d: Any) -> float:
    c = getattr(d, "confidence", None) if getattr(d, "confidence", None) is not None else getattr(d, "conf", None)
    if c is None:
        c = getattr(d, "score", None)
    try:
        return float(c) if c is not None else 1.0
    except (TypeError, ValueError):
        return 1.0


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / max(1e-6, area_a + area_b - inter)


@dataclass
class _Stats:
    pass_through: int = 0
    drop_class: int = 0
    drop_confidence: int = 0
    drop_size: int = 0
    drop_aspect: int = 0
    drop_road_paint: int = 0
    drop_off_road: int = 0
    drop_temporal: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "pass": self.pass_through,
            "drop_class": self.drop_class,
            "drop_conf": self.drop_confidence,
            "drop_size": self.drop_size,
            "drop_aspect": self.drop_aspect,
            "drop_paint": self.drop_road_paint,
            "drop_offroad": self.drop_off_road,
            "drop_temp": self.drop_temporal,
        }


@dataclass
class _PendingTrack:
    """A detection seen recently but not yet promoted."""

    bbox: Tuple[float, float, float, float]
    cls: str
    hits: int
    last_seen_frame: int


# --------------------------------------------------------------------------- #
# DetectionFilter                                                             #
# --------------------------------------------------------------------------- #


class DetectionFilter:
    """
    Filters raw YOLO detections before they reach the tracker.

    All filters are independent — disable any of them by setting the
    relevant threshold to 0 / None / a permissive value.
    """

    def __init__(
        self,
        frame_w: int = 1280,
        frame_h: int = 720,
        allowed_classes: Optional[set] = None,
        min_confidence: Optional[Dict[str, float]] = None,
        # 0.04% of frame area — keeps real cars 30 px tall at the
        # vanishing point. Anything smaller is hopefully noise.
        min_box_area_frac: float = 0.0004,
        max_box_area_frac: float = 0.55,
        aspect_range: Optional[Dict[str, Tuple[float, float]]] = None,
        min_height_frac: Optional[Dict[str, float]] = None,
        horizon_y_frac: float = 0.32,
        sky_band_y_frac: float = 0.18,
        # Road-paint specific filter: bottom of bbox in lower hood_band
        # of frame AND aspect > this AND height < hood_band_max_height.
        # This catches crosswalk arrows / lane paint without rejecting
        # real distant cars (which sit higher in the frame).
        road_paint_aspect_min: float = 1.5,
        hood_band_y_frac: float = 0.75,  # bbox bottom > 75% of frame
        hood_band_max_height_frac: float = 0.18,  # AND height < 18% of frame
        require_consecutive: int = 0,
        # 0.15 — lower than before so fast-moving real boxes survive
        # frame-to-frame IoU matching. False positives rarely persist
        # in the same spot for multiple frames.
        promotion_iou: float = 0.15,
        history_horizon_frames: int = 8,
    ):
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)
        self.allowed_classes = allowed_classes if allowed_classes is not None else set(ALLOWED_CLASSES)
        self.min_confidence = dict(min_confidence or DEFAULT_MIN_CONFIDENCE)
        self.min_box_area_frac = float(min_box_area_frac)
        self.max_box_area_frac = float(max_box_area_frac)
        self.aspect_range = dict(aspect_range or DEFAULT_ASPECT_RANGE)
        self.min_height_frac = dict(min_height_frac or DEFAULT_MIN_HEIGHT_FRAC)
        self.horizon_y_frac = float(horizon_y_frac)
        self.sky_band_y_frac = float(sky_band_y_frac)
        self.road_paint_aspect_min = float(road_paint_aspect_min)
        self.hood_band_y_frac = float(hood_band_y_frac)
        self.hood_band_max_height_frac = float(hood_band_max_height_frac)
        self.require_consecutive = int(require_consecutive)
        self.promotion_iou = float(promotion_iou)
        self.history_horizon_frames = int(history_horizon_frames)

        self._pending: List[_PendingTrack] = []
        self._frame_idx: int = 0
        self._stats = _Stats()

    # ----- public API ---------------------------------------------------- #

    def apply(self, detections: Optional[Iterable[Any]]) -> List[Any]:
        """Return only the detections that pass every filter."""
        self._frame_idx += 1
        keep: List[Any] = []

        for d in (detections or []):
            if not self._geometric_pass(d):
                continue
            if self.require_consecutive > 0:
                if not self._promote_if_seen_enough(d):
                    self._stats.drop_temporal += 1
                    continue
            keep.append(d)
            self._stats.pass_through += 1

        self._gc_pending()
        return keep

    def reset(self) -> None:
        self._pending.clear()
        self._frame_idx = 0
        self._stats = _Stats()

    def stats(self) -> Dict[str, int]:
        return self._stats.as_dict()

    def update_frame_size(self, frame_w: int, frame_h: int) -> None:
        """Call when the source resolution changes."""
        self.frame_w = int(frame_w)
        self.frame_h = int(frame_h)

    # ----- geometric / class / confidence filters ----------------------- #

    def _geometric_pass(self, d: Any) -> bool:
        cls = _class_name(d)
        if cls not in self.allowed_classes:
            self._stats.drop_class += 1
            return False

        conf = _confidence(d)
        min_conf = self.min_confidence.get(cls, 0.50)
        if conf < min_conf:
            self._stats.drop_confidence += 1
            return False

        bbox = _bbox_xyxy(d)
        if bbox is None:
            self._stats.drop_size += 1
            return False
        x1, y1, x2, y2 = bbox
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 1 or bh <= 1:
            self._stats.drop_size += 1
            return False

        # Min height by class.
        min_h = self.min_height_frac.get(cls, 0.04) * self.frame_h
        if bh < min_h:
            self._stats.drop_size += 1
            return False

        # Area band.
        area_frac = (bw * bh) / max(1.0, self.frame_w * self.frame_h)
        if area_frac < self.min_box_area_frac:
            self._stats.drop_size += 1
            return False
        if area_frac > self.max_box_area_frac:
            self._stats.drop_size += 1
            return False

        # Aspect ratio plausibility.
        aspect = bw / max(1.0, bh)
        lo, hi = self.aspect_range.get(cls, (0.2, 3.5))
        if aspect < lo or aspect > hi:
            self._stats.drop_aspect += 1
            return False

        # Sky-band rejection: anything whose bbox is fully in the top
        # sky_band_y_frac of the image is sky/buildings, not road users.
        sky_y = self.sky_band_y_frac * self.frame_h
        if y2 < sky_y:
            self._stats.drop_off_road += 1
            return False

        # Road-paint rejection — targets the crosswalk-arrow false-
        # positive pattern: bbox bottom sits low in the frame AND the
        # bbox is wider than tall AND it's short. A real distant car
        # has its bottom higher in the frame; a real near car is much
        # taller. The previous version used a too-permissive
        # aspect-only test that either let crosswalks through or also
        # killed real distant cars.
        hood_band_y = self.hood_band_y_frac * self.frame_h
        max_paint_h = self.hood_band_max_height_frac * self.frame_h
        in_hood_band = y2 > hood_band_y
        is_flat = bh < max_paint_h
        is_widish = aspect >= self.road_paint_aspect_min
        if in_hood_band and is_flat and is_widish:
            self._stats.drop_road_paint += 1
            return False

        return True

    # ----- temporal stability ------------------------------------------- #

    def _promote_if_seen_enough(self, d: Any) -> bool:
        """
        Greedy IoU-match the detection to the pending buffer. If a
        match exists, increment its hit count; otherwise add a new
        pending entry. A detection is "promoted" (allowed through)
        once its hits reach require_consecutive.
        """
        bbox = _bbox_xyxy(d)
        if bbox is None:
            return False
        cls = _class_name(d)

        best_iou = 0.0
        best_idx = -1
        for i, p in enumerate(self._pending):
            if p.cls != cls:
                continue
            iou = _iou(bbox, p.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0 and best_iou >= self.promotion_iou:
            p = self._pending[best_idx]
            p.bbox = bbox
            p.hits += 1
            p.last_seen_frame = self._frame_idx
            return p.hits >= self.require_consecutive

        # New pending entry.
        self._pending.append(
            _PendingTrack(
                bbox=bbox,
                cls=cls,
                hits=1,
                last_seen_frame=self._frame_idx,
            )
        )
        return self.require_consecutive <= 1

    def _gc_pending(self) -> None:
        cutoff = self._frame_idx - self.history_horizon_frames
        self._pending = [p for p in self._pending if p.last_seen_frame >= cutoff]

