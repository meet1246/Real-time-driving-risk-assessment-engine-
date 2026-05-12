"""
Unit tests for driving_scene/scene_renderer.py + scene_mapper.py.

These tests are deliberately tolerant — the modules have fallbacks for
missing fields, so we mostly verify that:

  * SceneRenderer renders a frame of the right shape for every view mode.
  * It tolerates None telemetry, None risk_state, empty tracks, etc.
  * Per-track EMA smoothing in SceneMapper actually moves a point toward
    the target.
  * image_bbox_to_scene_plane returns sensible ranges.

Run from project root:

    python -m unittest tests.test_scene_renderer -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.scene_renderer import SceneRenderer  # noqa: E402
from driving_scene.scene_mapper import image_bbox_to_scene_plane  # noqa: E402


def _track(tid, cls, bbox, score=10.0, level="LOW"):
    return SimpleNamespace(
        track_id=tid,
        class_name=cls,
        bbox_xyxy=bbox,
        risk_score=score,
        risk_level=level,
    )


class TestBBoxMapping(unittest.TestCase):
    def test_centered_bbox_lateral_zero(self):
        lat, close, area = image_bbox_to_scene_plane((590, 350, 690, 450), 1280, 720)
        self.assertAlmostEqual(lat, 0.0, delta=0.1)
        self.assertGreater(close, 0.3)
        self.assertLess(close, 1.0)
        self.assertGreater(area, 0.0)

    def test_far_left_bbox_negative_lateral(self):
        lat, _, _ = image_bbox_to_scene_plane((10, 350, 90, 430), 1280, 720)
        self.assertLess(lat, -0.5)

    def test_far_right_bbox_positive_lateral(self):
        lat, _, _ = image_bbox_to_scene_plane((1180, 350, 1270, 430), 1280, 720)
        self.assertGreater(lat, 0.5)

    def test_near_bottom_higher_closeness(self):
        _, c_far, _ = image_bbox_to_scene_plane((600, 280, 680, 320), 1280, 720)
        _, c_near, _ = image_bbox_to_scene_plane((500, 600, 780, 700), 1280, 720)
        self.assertGreater(c_near, c_far)


class TestRender(unittest.TestCase):
    def setUp(self):
        self.r = SceneRenderer(width=640, height=360)
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def test_render_scene_no_inputs(self):
        out = self.r.render(self.frame, view_mode="scene")
        self.assertEqual(out.shape, (360, 640, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_render_dashcam_no_inputs(self):
        out = self.r.render(self.frame, view_mode="dashcam")
        self.assertEqual(out.shape, (360, 640, 3))

    def test_render_split_no_inputs(self):
        out = self.r.render(self.frame, view_mode="split")
        self.assertEqual(out.shape, (360, 640, 3))

    def test_render_with_tracks(self):
        tracks = [
            _track(1, "car", (300, 200, 400, 280), 70, "HIGH"),
            _track(2, "person", (500, 230, 540, 320), 20, "LOW"),
        ]
        risk_state = SimpleNamespace(global_score=70, global_level="HIGH", global_action="ALERT", primary_track_id=1)
        out = self.r.render(
            self.frame,
            tracks=tracks,
            risk_state=risk_state,
            view_mode="scene",
            ego_speed_kmh=42.0,
            speed_limit_kmh=50.0,
        )
        self.assertEqual(out.shape, (360, 640, 3))

    def test_render_handles_missing_fields(self):
        bare = SimpleNamespace(track_id=9, class_name="car", bbox_xyxy=(100, 100, 200, 180))
        out = self.r.render(self.frame, tracks=[bare], view_mode="scene")
        self.assertEqual(out.shape, (360, 640, 3))

    def test_ema_smoothing_moves_toward_target(self):
        # Two bboxes inside the SAME ego lane bucket (cx_norm both fall in
        # 0.43–0.58 → ego_lane) so lane stickiness doesn't fight the move.
        # We're verifying that EMA on (scene_x) moves rightward when the
        # bbox slides rightward within its lane.
        track = _track(1, "car", (280, 200, 360, 280), 50, "MEDIUM")  # cx_norm = 0.50
        self.r.render(self.frame, tracks=[track], view_mode="scene")
        first = self.r.mapper.track_points[1]
        track.bbox_xyxy = (320, 200, 400, 280)  # cx_norm = 0.5625 → still ego_lane
        self.r.render(self.frame, tracks=[track], view_mode="scene")
        second = self.r.mapper.track_points[1]
        self.assertGreater(second.x, first.x)

    def test_toggle_debug(self):
        self.assertFalse(self.r.state.debug)
        self.r.toggle_debug()
        self.assertTrue(self.r.state.debug)
        out = self.r.render(self.frame, view_mode="scene")
        self.assertEqual(out.shape, (360, 640, 3))

    def test_reset_clears_track_points(self):
        track = _track(1, "car", (300, 200, 400, 280), 50, "MEDIUM")
        self.r.render(self.frame, tracks=[track], view_mode="scene")
        self.assertIn(1, self.r.mapper.track_points)
        self.r.reset()
        self.assertEqual(self.r.mapper.track_points, {})


if __name__ == "__main__":
    unittest.main()

