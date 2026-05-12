"""Mutable helpers for manual dashcam lane calibration (frontend only)."""

from __future__ import annotations

from typing import List

MIN_BOUND_GAP = 0.015


def repair_monotonic(boundaries: List[float]) -> None:
    """Keep endpoints fixed and enforce strict increase with MIN_BOUND_GAP."""
    if len(boundaries) < 2:
        return
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    for _ in range(4):
        for i in range(1, len(boundaries) - 1):
            lo = boundaries[i - 1] + MIN_BOUND_GAP
            hi = boundaries[i + 1] - MIN_BOUND_GAP
            boundaries[i] = max(lo, min(hi, boundaries[i]))
        for i in range(len(boundaries) - 2, 0, -1):
            lo = boundaries[i - 1] + MIN_BOUND_GAP
            hi = boundaries[i + 1] - MIN_BOUND_GAP
            boundaries[i] = max(lo, min(hi, boundaries[i]))


def shift_inner_boundaries(boundaries: List[float], delta: float) -> None:
    """Shift interior lane cuts left (delta < 0) or right (delta > 0)."""
    for i in range(1, len(boundaries) - 1):
        boundaries[i] += delta
    repair_monotonic(boundaries)


def adjust_ego_lane_width(boundaries: List[float], widen: bool, step: float = 0.012) -> None:
    """
    Narrow/widen the ego lane strip.

    Assumes 6 boundaries / 5 lanes with ego between indices 2 and 3:
    far_left | left_lane | ego_lane | right_lane | far_right
    """
    if len(boundaries) < 5:
        return
    if widen:
        boundaries[2] -= step
        boundaries[3] += step
    else:
        boundaries[2] += step
        boundaries[3] -= step
    repair_monotonic(boundaries)


def format_boundaries_for_config(boundaries: List[float]) -> str:
    return "LANE_X_BOUNDARIES = " + repr(boundaries)
