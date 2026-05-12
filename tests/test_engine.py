"""Unit tests for geometry, risk utilities (no GUI, optional slow YOLO tests)."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from driving_scene.geometry import bbox_iou_xyxy
from driving_scene.risk import FrameRiskResult, RiskLevel, smooth_global_risk_display, Decision


class TestIoU(unittest.TestCase):
    def test_overlap(self) -> None:
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        iou = bbox_iou_xyxy(a, b)
        self.assertAlmostEqual(iou, 25.0 / 175.0, places=5)

    def test_disjoint(self) -> None:
        self.assertEqual(bbox_iou_xyxy((0, 0, 1, 1), (5, 5, 6, 6)), 0.0)


class TestRiskSmooth(unittest.TestCase):
    def test_ema_only_globals(self) -> None:
        raw = FrameRiskResult(
            per_object=[],
            global_risk_level=RiskLevel.LOW,
            global_decision=Decision.SAFE,
            global_score=0.0,
            primary_threat_id=None,
        )
        out, ema = smooth_global_risk_display(None, raw, 0.2)
        self.assertEqual(ema, 0.0)
        self.assertEqual(out.global_score, 0.0)


if __name__ == "__main__":
    unittest.main()
