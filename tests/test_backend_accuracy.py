"""
Tests for backend-accuracy upgrades:
  * Boundary-based lane assignment (LaneModel.assign_lane_from_boundaries)
  * LaneModel.assign_lane respects configured boundaries
  * Detector post-NMS deduplicates overlapping boxes
  * Track quality rises on hits and falls on misses
  * SceneMapper records hidden_reason when visibility gates fire
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.detector import Detection, post_nms  # noqa: E402
from driving_scene.lane_model import (  # noqa: E402
    LANE_EGO,
    LANE_ONCOMING_1,
    LANE_ONCOMING_2,
    LANE_RIGHT_1,
    LaneModel,
)
from driving_scene.scene_mapper import SceneMapper  # noqa: E402
from driving_scene.tracker import TrackedObject  # noqa: E402


class TestBoundaryLaneAssignment(unittest.TestCase):
    """Pure boundary buckets — no perspective math involved."""

    def setUp(self):
        self.lm = LaneModel(1280, 720)
        # Default boundaries from config: [0, 0.28, 0.43, 0.58, 0.73, 1.0]

    def test_centered_cx_is_ego_lane(self):
        lane_id, label, _c, _w = self.lm.assign_lane_from_boundaries(0.50)
        self.assertEqual(lane_id, LANE_EGO)
        self.assertEqual(label, "ego_lane")

    def test_right_of_center_is_right_lane(self):
        lane_id, label, _c, _w = self.lm.assign_lane_from_boundaries(0.65)
        self.assertEqual(lane_id, LANE_RIGHT_1)
        self.assertEqual(label, "right_lane")

    def test_left_of_center_is_left_lane(self):
        lane_id, label, _c, _w = self.lm.assign_lane_from_boundaries(0.35)
        self.assertEqual(lane_id, LANE_ONCOMING_1)
        self.assertEqual(label, "left_lane")

    def test_far_left_edge(self):
        lane_id, label, _c, _w = self.lm.assign_lane_from_boundaries(0.10)
        self.assertEqual(lane_id, LANE_ONCOMING_2)
        self.assertEqual(label, "far_left")

    def test_far_right_edge(self):
        lane_id, label, _c, _w = self.lm.assign_lane_from_boundaries(0.90)
        # far_right shares the LANE_RIGHT_1 slot (we only render one right lane).
        self.assertEqual(lane_id, LANE_RIGHT_1)
        self.assertEqual(label, "far_right")

    def test_lane_width_is_bucket_width(self):
        _id, _label, _c, w = self.lm.assign_lane_from_boundaries(0.50)
        # 0.58 - 0.43 = 0.15
        self.assertAlmostEqual(w, 0.15, places=3)


class TestAssignLaneUsesBoundaries(unittest.TestCase):
    """The full assign_lane(bbox, ...) entry point routes vehicles through boundary buckets."""

    def setUp(self):
        self.lm = LaneModel(1280, 720)

    def test_vehicle_on_right_assigns_to_right_lane(self):
        # A car bbox centered at x=900 in a 1280-wide frame → cx_norm = 0.70 → right_lane
        lane_id, offset = self.lm.assign_lane(
            bbox=(850, 350, 950, 430), frame_w=1280, frame_h=720, cls="car",
        )
        self.assertEqual(lane_id, LANE_RIGHT_1)
        # Offset stays inside ±0.35.
        self.assertLessEqual(abs(offset), 0.35 + 1e-6)

    def test_vehicle_on_left_assigns_to_left_lane(self):
        lane_id, _offset = self.lm.assign_lane(
            bbox=(400, 350, 500, 430), frame_w=1280, frame_h=720, cls="car",
        )
        self.assertEqual(lane_id, LANE_ONCOMING_1)

    def test_centered_vehicle_assigns_to_ego(self):
        lane_id, _offset = self.lm.assign_lane(
            bbox=(620, 350, 720, 430), frame_w=1280, frame_h=720, cls="car",
        )
        self.assertEqual(lane_id, LANE_EGO)


class TestPostNMS(unittest.TestCase):
    def _make(self, conf, x1, y1, x2, y2, cls="car"):
        return Detection(
            class_id=2 if cls == "car" else 7,
            class_name=cls,
            confidence=conf,
            bbox=(x1, y1, x2, y2),
            center=((x1 + x2) / 2, (y1 + y2) / 2),
            area=(x2 - x1) * (y2 - y1),
        )

    def test_keeps_disjoint_boxes(self):
        a = self._make(0.9, 100, 100, 200, 200)
        b = self._make(0.85, 400, 400, 500, 500)
        out = post_nms([a, b])
        self.assertEqual(len(out), 2)

    def test_drops_lower_conf_duplicate(self):
        high = self._make(0.92, 100, 100, 200, 200)
        low_dup = self._make(0.55, 105, 102, 198, 199)  # heavy overlap
        out = post_nms([low_dup, high])  # order shouldn't matter
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].confidence, 0.92, places=4)

    def test_drops_dual_class_hit(self):
        car = self._make(0.80, 100, 100, 200, 200, cls="car")
        truck = self._make(0.65, 102, 99, 201, 202, cls="truck")
        out = post_nms([car, truck])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].class_name, "car")


class TestTrackQuality(unittest.TestCase):
    def test_quality_rises_on_hits(self):
        from driving_scene.detector import Detection
        tr = TrackedObject(track_id=1, class_name="car", birth_frame=0)
        for i in range(5):
            tr.append_state(
                Detection(class_id=2, class_name="car", confidence=0.9,
                          bbox=(100, 100, 200, 200), center=(150, 150), area=10000),
                frame_idx=i,
            )
        self.assertGreater(tr.track_quality, 0.5)
        self.assertEqual(tr.hits, 5)

    def test_quality_falls_on_misses(self):
        from driving_scene.detector import Detection
        tr = TrackedObject(track_id=1, class_name="car", birth_frame=0)
        for i in range(5):
            tr.append_state(
                Detection(class_id=2, class_name="car", confidence=0.9,
                          bbox=(100, 100, 200, 200), center=(150, 150), area=10000),
                frame_idx=i,
            )
        q_before = tr.track_quality
        for _ in range(10):
            tr.register_miss()
        self.assertLess(tr.track_quality, q_before)
        self.assertEqual(tr.misses, 10)


class TestHiddenReason(unittest.TestCase):
    def test_tiny_box_recorded_with_reason(self):
        lm = LaneModel(1280, 720)
        mapper = SceneMapper(lm, min_height_ratio=0.05, min_area_ratio=0.001)
        from types import SimpleNamespace
        # A 5x4 px detection — well under any visibility threshold.
        tiny = SimpleNamespace(
            track_id=1, class_name="car",
            bbox=(100, 100, 105, 104),
            risk_score=10, risk_level="LOW",
        )
        items = mapper.collect_renderables([tiny], None, 1280, 720)
        self.assertEqual(items, [])
        self.assertEqual(len(mapper.hidden_items), 1)
        self.assertIn("height_ratio", mapper.hidden_items[0]["hidden_reason"])


if __name__ == "__main__":
    unittest.main()
