"""
Monocular lane/path estimator (frontend-only heuristic).

This is NOT autonomy and not production-grade. It uses classical OpenCV
edges + Hough line segments in an ROI, then fits a simple polynomial in
image space for left/right boundaries and derives an ego center path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    LANE_CANNY1,
    LANE_CANNY2,
    LANE_CONFIDENCE_MIN_FOR_CURVE,
    LANE_CURVE_GAIN,
    LANE_CURVE_MAX_OFFSET_RATIO,
    LANE_CURVE_SMOOTH_ALPHA,
    LANE_HOUGH_MAX_LINE_GAP,
    LANE_HOUGH_MIN_LINE_LEN,
    LANE_HOUGH_THRESHOLD,
    LANE_KEEP_FRAMES,
    LANE_RESIZE_WIDTH,
    LANE_ROI_Y_TOP,
    LANE_SMOOTH_ALPHA,
)


class CurveDir(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT = "STRAIGHT"


@dataclass
class LanePacket:
    left_pts: List[Tuple[int, int]]
    right_pts: List[Tuple[int, int]]
    center_pts: List[Tuple[int, int]]
    confidence: float  # 0..1
    direction: CurveDir
    curve_strength: float  # signed, ~[-1,1]
    # Debug artefacts (in original frame coords)
    debug_segments: List[Tuple[int, int, int, int]] = field(default_factory=list)


def _polyfit_x_of_y(pts: np.ndarray, deg: int = 2) -> Optional[np.ndarray]:
    """Fit x = a*y^2 + b*y + c. Return coeffs [a,b,c] or None."""
    if pts.shape[0] < max(12, deg + 5):
        return None
    y = pts[:, 1]
    x = pts[:, 0]
    try:
        c = np.polyfit(y, x, deg)
        return c.astype(np.float32)
    except Exception:
        return None


def _poly_eval(coeffs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return coeffs[0] * y * y + coeffs[1] * y + coeffs[2]


@dataclass
class LaneEstimator:
    """
    Stateful lane estimator with EMA smoothing of polynomial coefficients.
    """

    smooth_alpha: float = LANE_SMOOTH_ALPHA
    keep_frames: int = LANE_KEEP_FRAMES
    _left_c: Optional[np.ndarray] = None
    _right_c: Optional[np.ndarray] = None
    _missing: int = 0
    _curve_smoothed: float = 0.0

    def _ema(self, prev: Optional[np.ndarray], new: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if new is None:
            return prev
        if prev is None:
            return new
        a = float(self.smooth_alpha)
        return (a * new + (1.0 - a) * prev).astype(np.float32)

    def estimate(self, frame_bgr: np.ndarray) -> LanePacket:
        h, w = frame_bgr.shape[:2]
        # Resize for speed
        rw = int(LANE_RESIZE_WIDTH)
        scale = rw / max(w, 1)
        rh = int(round(h * scale))
        small = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, LANE_CANNY1, LANE_CANNY2)

        # ROI mask (lower half / trapezoid-ish)
        y_top = int(rh * float(LANE_ROI_Y_TOP))
        mask = np.zeros_like(edges)
        poly = np.array(
            [
                (int(rw * 0.05), rh - 1),
                (int(rw * 0.95), rh - 1),
                (int(rw * 0.62), y_top),
                (int(rw * 0.38), y_top),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [poly], 255)
        roi = cv2.bitwise_and(edges, mask)

        segs = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=LANE_HOUGH_THRESHOLD,
            minLineLength=LANE_HOUGH_MIN_LINE_LEN,
            maxLineGap=LANE_HOUGH_MAX_LINE_GAP,
        )

        left_pts: List[Tuple[float, float]] = []
        right_pts: List[Tuple[float, float]] = []
        dbg_small: List[Tuple[int, int, int, int]] = []
        if segs is not None:
            for x1, y1, x2, y2 in segs[:, 0, :]:
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                if abs(dx) < 1e-3:
                    continue
                slope = dy / dx
                # Filter out near-horizontal
                if abs(slope) < 0.35:
                    continue
                # Also filter too vertical (often poles)
                if abs(slope) > 4.5:
                    continue
                dbg_small.append((x1, y1, x2, y2))
                # Classify by slope sign in image (y down): left lane tends to negative slope
                if slope < 0:
                    left_pts.append((x1, y1))
                    left_pts.append((x2, y2))
                else:
                    right_pts.append((x1, y1))
                    right_pts.append((x2, y2))

        lp = np.array(left_pts, dtype=np.float32) if left_pts else np.zeros((0, 2), dtype=np.float32)
        rp = np.array(right_pts, dtype=np.float32) if right_pts else np.zeros((0, 2), dtype=np.float32)

        left_c = _polyfit_x_of_y(lp) if lp.shape[0] else None
        right_c = _polyfit_x_of_y(rp) if rp.shape[0] else None

        # Update EMA state
        self._left_c = self._ema(self._left_c, left_c)
        self._right_c = self._ema(self._right_c, right_c)

        got = (left_c is not None) or (right_c is not None)
        if got:
            self._missing = 0
        else:
            self._missing += 1

        # Decide whether to reuse old state
        use_left = self._left_c if (self._left_c is not None and self._missing <= self.keep_frames) else None
        use_right = self._right_c if (self._right_c is not None and self._missing <= self.keep_frames) else None

        # Sample points in small image and scale back to original
        ys = np.linspace(rh - 1, y_top, num=18, dtype=np.float32)
        left_curve: List[Tuple[int, int]] = []
        right_curve: List[Tuple[int, int]] = []
        center_curve: List[Tuple[int, int]] = []

        if use_left is not None:
            xl = _poly_eval(use_left, ys)
            for x, y in zip(xl, ys):
                ox = int(round(x / scale))
                oy = int(round(y / scale))
                left_curve.append((ox, oy))
        if use_right is not None:
            xr = _poly_eval(use_right, ys)
            for x, y in zip(xr, ys):
                ox = int(round(x / scale))
                oy = int(round(y / scale))
                right_curve.append((ox, oy))

        # Center path: average left/right if both; else offset from visible lane toward center
        if left_curve and right_curve:
            for (xl, yl), (xr, yr) in zip(left_curve, right_curve):
                center_curve.append(((xl + xr) // 2, (yl + yr) // 2))
            conf = 1.0
        elif left_curve:
            for (xl, yl) in left_curve:
                center_curve.append((min(w - 1, xl + int(w * 0.25)), yl))
            conf = 0.55
        elif right_curve:
            for (xr, yr) in right_curve:
                center_curve.append((max(0, xr - int(w * 0.25)), yr))
            conf = 0.55
        else:
            conf = 0.0

        # Curvature direction from center path deviation at far vs near
        curve_raw = 0.0
        direction = CurveDir.STRAIGHT
        if len(center_curve) >= 6:
            near_x = float(center_curve[0][0])
            far_x = float(center_curve[-1][0])
            dx = (far_x - near_x) / max(w, 1)
            curve_raw = float(np.clip(dx * 2.45 * LANE_CURVE_GAIN, -1.0, 1.0))
            if curve_raw < -0.06:
                direction = CurveDir.LEFT
            elif curve_raw > 0.06:
                direction = CurveDir.RIGHT
            else:
                direction = CurveDir.STRAIGHT

        # Heavy EMA on curve strength + clamp effective curvature for downstream FSD
        ca = float(LANE_CURVE_SMOOTH_ALPHA)
        self._curve_smoothed = ca * curve_raw + (1.0 - ca) * float(self._curve_smoothed)
        max_mag = max(0.08, min(0.95, float(LANE_CURVE_MAX_OFFSET_RATIO) * 5.5))
        curve_strength = float(np.clip(self._curve_smoothed, -max_mag, max_mag))

        # Low confidence → drift curve toward straight (caller blends geometrically too)
        if float(conf) < float(LANE_CONFIDENCE_MIN_FOR_CURVE):
            curve_strength *= max(0.0, float(conf) / max(float(LANE_CONFIDENCE_MIN_FOR_CURVE), 1e-6))
            if abs(curve_strength) < 0.05:
                direction = CurveDir.STRAIGHT

        # Scale debug segments back to original
        dbg: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in dbg_small:
            dbg.append(
                (
                    int(round(x1 / scale)),
                    int(round(y1 / scale)),
                    int(round(x2 / scale)),
                    int(round(y2 / scale)),
                )
            )

        # Fallback straight lanes if no state
        if not left_curve and not right_curve:
            # just center path down the middle
            center_curve = [(w // 2, int(round(y / scale))) for y in ys]
            left_curve = [(int(w * 0.38), int(round(y / scale))) for y in ys]
            right_curve = [(int(w * 0.62), int(round(y / scale))) for y in ys]

        return LanePacket(
            left_pts=left_curve,
            right_pts=right_curve,
            center_pts=center_curve,
            confidence=float(conf),
            direction=direction,
            curve_strength=float(curve_strength),
            debug_segments=dbg,
        )

