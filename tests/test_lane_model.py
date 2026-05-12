"""
Unit tests for driving_scene/lane_model.py.

Verifies the lane-assignment heuristics behave correctly on a range
of realistic dashcam bbox positions and motion patterns.

Run from project root:

    python -m unittest tests.test_lane_layout -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.lane_model import (  # noqa: E402
    LaneModel,
    LANE_SIDEWALK_LEFT,
    LANE_SIDEWALK_RIGHT,
    LANE_ONCOMING_1,
    LANE_ONCOMING_2,
    LANE_EGO,
    LANE_RIGHT_1,
)


class TestVehicleLaneAssignment(unittest.TestCase):
    """Vehicles should land in plausible NA-road lanes."""

    def setUp(self):
        self.ll = LaneModel(1280, 720)
        self.fw, self.fh = 1280, 720

    def _assign(self, x_center, y_center=400, w=120, h=100, cls="car", velocity=None, tid=None):
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        return self.ll.assign_lane(
            bbox=(x1, y1, x1 + w, y1 + h),
            frame_w=self.fw,
            frame_h=self.fh,
            cls=cls,
            velocity=velocity,
            track_id=tid,
        )

    def test_centered_car_goes_to_ego_lane(self):
        lane, _ = self._assign(x_center=640)
        self.assertEqual(lane, LANE_EGO)

    def test_far_left_car_goes_to_oncoming(self):
        lane, _ = self._assign(x_center=120)
        self.assertIn(lane, (LANE_ONCOMING_1, LANE_ONCOMING_2))

    def test_mild_left_car_goes_to_oncoming(self):
        lane, _ = self._assign(x_center=420)  # lateral ~ -0.34
        self.assertEqual(lane, LANE_ONCOMING_1)

    def test_mild_right_car_goes_to_right_lane(self):
        lane, _ = self._assign(x_center=860)  # lateral ~ +0.34
        self.assertEqual(lane, LANE_RIGHT_1)

    def test_far_right_car_still_right_lane(self):
        lane, _ = self._assign(x_center=1160)  # lateral ~ +0.81
        self.assertEqual(lane, LANE_RIGHT_1)

    def test_very_far_left_car_outer_oncoming(self):
        lane, _ = self._assign(x_center=30)  # lateral close to -1.0
        self.assertEqual(lane, LANE_ONCOMING_2)


class TestPedestrianAssignment(unittest.TestCase):
    """Pedestrians should stay on the sidewalk unless they step onto the road."""

    def setUp(self):
        self.ll = LaneModel(1280, 720)
        self.fw, self.fh = 1280, 720

    def _assign(self, x_center, y_center=440, w=40, h=120, cls="person", velocity=None, tid=None):
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        return self.ll.assign_lane(
            bbox=(x1, y1, x1 + w, y1 + h),
            frame_w=self.fw,
            frame_h=self.fh,
            cls=cls,
            velocity=velocity,
            track_id=tid,
        )

    def test_far_left_static_pedestrian_on_left_sidewalk(self):
        lane, _ = self._assign(x_center=120)  # lateral ~ -0.81
        self.assertEqual(lane, LANE_SIDEWALK_LEFT)

    def test_far_right_static_pedestrian_on_right_sidewalk(self):
        lane, _ = self._assign(x_center=1180)  # lateral ~ +0.84
        self.assertEqual(lane, LANE_SIDEWALK_RIGHT)

    def test_pedestrian_walking_along_sidewalk_stays_on_sidewalk(self):
        lane, _ = self._assign(x_center=120, velocity=(80, 0))
        self.assertEqual(lane, LANE_SIDEWALK_LEFT)

    def test_pedestrian_directly_in_front_close_in_road(self):
        lane, _ = self._assign(x_center=640, y_center=560, w=60, h=200)
        self.assertIn(lane, (LANE_EGO, LANE_RIGHT_1, LANE_ONCOMING_1))

    def test_pedestrian_stepping_off_curb_enters_road(self):
        lane, _ = self._assign(x_center=380, y_center=520, w=50, h=160, velocity=(120, 0))
        self.assertIn(lane, (LANE_ONCOMING_1, LANE_EGO))


class TestCyclistAssignment(unittest.TestCase):
    """Cyclists default to the right travel lane (bike lane)."""

    def setUp(self):
        self.ll = LaneModel(1280, 720)
        self.fw, self.fh = 1280, 720

    def _assign(self, x_center, y_center=460, w=70, h=140, tid=None):
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        return self.ll.assign_lane(
            bbox=(x1, y1, x1 + w, y1 + h),
            frame_w=self.fw,
            frame_h=self.fh,
            cls="bicycle",
            velocity=None,
            track_id=tid,
        )

    def test_right_side_cyclist_in_right_lane(self):
        lane, offset = self._assign(x_center=900)
        self.assertEqual(lane, LANE_RIGHT_1)
        self.assertGreater(offset, 0.3)

    def test_centered_cyclist_in_right_lane(self):
        lane, _ = self._assign(x_center=640)
        self.assertEqual(lane, LANE_RIGHT_1)

    def test_far_left_cyclist_on_left_sidewalk_if_far(self):
        lane, _ = self._assign(x_center=100, y_center=380, w=40, h=60)
        self.assertEqual(lane, LANE_SIDEWALK_LEFT)


class TestStickiness(unittest.TestCase):
    """Anti-flicker: tracks shouldn't bounce between lanes every frame."""

    def setUp(self):
        self.ll = LaneModel(1280, 720)
        self.fw, self.fh = 1280, 720

    def _bbox(self, x_center, y_center=400, w=120, h=100):
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        return (x1, y1, x1 + w, y1 + h)

    def test_track_in_ego_lane_doesnt_flicker_to_oncoming_on_one_bad_frame(self):
        bbox_good = self._bbox(640)
        bbox_jitter = self._bbox(440)
        for _ in range(5):
            lane, _ = self.ll.assign_lane(
                bbox=bbox_good,
                frame_w=self.fw,
                frame_h=self.fh,
                cls="car",
                velocity=None,
                track_id=99,
            )
            self.assertEqual(lane, LANE_EGO)
        lane, _ = self.ll.assign_lane(
            bbox=bbox_jitter,
            frame_w=self.fw,
            frame_h=self.fh,
            cls="car",
            velocity=None,
            track_id=99,
        )
        self.assertEqual(lane, LANE_EGO)

    def test_persistent_lane_change_eventually_sticks(self):
        bbox_old = self._bbox(640)
        bbox_new = self._bbox(380)
        for _ in range(3):
            self.ll.assign_lane(
                bbox=bbox_old,
                frame_w=self.fw,
                frame_h=self.fh,
                cls="car",
                velocity=None,
                track_id=88,
            )
        last_lane = None
        for _ in range(20):
            last_lane, _ = self.ll.assign_lane(
                bbox=bbox_new,
                frame_w=self.fw,
                frame_h=self.fh,
                cls="car",
                velocity=None,
                track_id=88,
            )
        self.assertEqual(last_lane, LANE_ONCOMING_1)

    def test_prune_clears_history(self):
        self.ll.assign_lane(
            bbox=self._bbox(640),
            frame_w=self.fw,
            frame_h=self.fh,
            cls="car",
            velocity=None,
            track_id=7,
        )
        self.assertIn(7, self.ll._history)
        self.ll.prune_tracks(active_ids=set())
        self.assertNotIn(7, self.ll._history)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.ll = LaneModel(1280, 720)

    def test_ego_is_right_of_canvas_center(self):
        cx, _ = self.ll.ego_position()
        self.assertGreater(cx, 1280 * 0.5)

    def test_lanes_dont_overlap_at_bottom(self):
        ey = self.ll.ego_y() + 40
        ego_right = self.ll.lane_center_x_at_y(LANE_EGO, ey) + self.ll.lane_half_width_at_y(LANE_EGO, ey)
        right1_left = self.ll.lane_center_x_at_y(LANE_RIGHT_1, ey) - self.ll.lane_half_width_at_y(LANE_RIGHT_1, ey)
        self.assertLess(ego_right, right1_left)

    def test_lanes_converge_at_horizon(self):
        hy = self.ll.horizon_y()
        ey = self.ll.ego_y() + 40
        ego_half_top = self.ll.lane_half_width_at_y(LANE_EGO, hy)
        ego_half_bot = self.ll.lane_half_width_at_y(LANE_EGO, ey)
        self.assertLess(ego_half_top, ego_half_bot)


if __name__ == "__main__":
    unittest.main()

