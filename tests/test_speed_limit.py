"""
Unit tests for driving_scene/speed_limit.py.

Run from project root:

    python -m unittest tests.test_speed_limit -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.speed_limit import SpeedLimitTracker  # noqa: E402


def _det(cls, bbox):
    return SimpleNamespace(class_name=cls, bbox_xyxy=bbox)


class _StubTracker(SpeedLimitTracker):
    """SpeedLimitTracker variant whose classifier returns a fixed value
    so we can test the windowing / agreement logic without a real model."""

    def __init__(self, fake_value=50.0, fake_conf=0.9, **kwargs):
        super().__init__(**kwargs)
        self.fake_value = fake_value
        self.fake_conf = fake_conf

    def _classify_crop(self, crop_bgr):
        return (self.fake_value, self.fake_conf)


class TestSpeedLimitTracker(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_default_classifier_returns_none(self):
        slt = SpeedLimitTracker()
        for _ in range(5):
            slt.observe(detections=[_det("stop sign", (100, 100, 200, 200))], frame=self.frame)
        self.assertIsNone(slt.current())

    def test_needs_agreement_before_publishing(self):
        slt = _StubTracker(min_agree=3)
        slt.observe(detections=[_det("stop sign", (10, 10, 100, 100))], frame=self.frame)
        self.assertIsNone(slt.current())
        slt.observe(detections=[_det("stop sign", (10, 10, 100, 100))], frame=self.frame)
        self.assertIsNone(slt.current())
        slt.observe(detections=[_det("stop sign", (10, 10, 100, 100))], frame=self.frame)
        self.assertEqual(slt.current(), 50.0)

    def test_ignores_non_sign_classes(self):
        slt = _StubTracker(min_agree=1)
        slt.observe(detections=[_det("car", (10, 10, 100, 100))], frame=self.frame)
        self.assertIsNone(slt.current())

    def test_low_confidence_filtered(self):
        slt = _StubTracker(fake_conf=0.10, min_agree=1, min_confidence=0.5)
        slt.observe(detections=[_det("stop sign", (10, 10, 100, 100))], frame=self.frame)
        self.assertIsNone(slt.current())

    def test_handles_missing_inputs(self):
        slt = _StubTracker(min_agree=1)
        slt.observe(detections=None, frame=None)
        slt.observe(detections=[], frame=self.frame)
        slt.observe(detections=[_det("stop sign", None)], frame=self.frame)
        self.assertIsNone(slt.current())

    def test_stale_reading_returns_none(self):
        slt = _StubTracker(min_agree=1, stale_after=0.05)
        slt.observe(detections=[_det("stop sign", (10, 10, 100, 100))], frame=self.frame)
        self.assertEqual(slt.current(), 50.0)
        time.sleep(0.1)
        self.assertIsNone(slt.current())

    def test_reset(self):
        slt = _StubTracker(min_agree=1)
        slt.observe(detections=[_det("stop sign", (10, 10, 100, 100))], frame=self.frame)
        self.assertIsNotNone(slt.current())
        slt.reset()
        self.assertIsNone(slt.current())


if __name__ == "__main__":
    unittest.main()

