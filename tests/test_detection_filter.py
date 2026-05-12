"""
Unit tests for driving_scene/detection_filter.py.

Verifies that the filter rejects the kinds of false-positives that
show up in urban dashcam footage: low-confidence ghosts, tiny
detections on far pixels, road-paint hits, sky-band hits, and
single-frame flickers.

Run from project root:

    python -m unittest tests.test_detection_filter -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.detection_filter import DetectionFilter  # noqa: E402


def det(cls, bbox, conf=0.9):
    return SimpleNamespace(class_name=cls, bbox_xyxy=bbox, confidence=conf)


class TestClassFilter(unittest.TestCase):
    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720)

    def test_unknown_class_dropped(self):
        d = det("zebra", (100, 200, 200, 320), conf=0.9)
        self.assertEqual(self.f.apply([d]), [])
        self.assertEqual(self.f.stats()["drop_class"], 1)

    def test_known_class_passes(self):
        d = det("car", (400, 300, 600, 460), conf=0.9)
        self.assertEqual(len(self.f.apply([d])), 1)


class TestConfidenceFilter(unittest.TestCase):
    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720)

    def test_low_conf_car_dropped(self):
        d = det("car", (400, 300, 600, 460), conf=0.20)
        self.assertEqual(self.f.apply([d]), [])
        self.assertEqual(self.f.stats()["drop_conf"], 1)

    def test_low_conf_person_dropped(self):
        d = det("person", (500, 300, 540, 500), conf=0.30)
        self.assertEqual(self.f.apply([d]), [])

    def test_at_threshold_passes(self):
        d = det("car", (400, 300, 600, 460), conf=0.46)
        self.assertEqual(len(self.f.apply([d])), 1)


class TestSizeFilter(unittest.TestCase):
    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720)

    def test_tiny_person_dropped(self):
        # 12 pixel tall in a 720p frame â‰ˆ 0.017 fraction; min is 0.06.
        d = det("person", (640, 400, 660, 412), conf=0.9)
        self.assertEqual(self.f.apply([d]), [])
        self.assertEqual(self.f.stats()["drop_size"], 1)

    def test_tiny_car_dropped(self):
        # 5-pixel-tall car bbox â€” clearly noise even by new looser thresholds.
        d = det("car", (300, 400, 340, 405), conf=0.9)
        self.assertEqual(self.f.apply([d]), [])

    def test_huge_bbox_dropped(self):
        # >50% of frame area = bug, not a vehicle.
        d = det("car", (0, 0, 1280, 720), conf=0.9)
        self.assertEqual(self.f.apply([d]), [])


class TestAspectFilter(unittest.TestCase):
    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720)

    def test_wide_flat_car_dropped_as_paint(self):
        # 300 wide x 25 tall = aspect 12.0 â†’ road-paint range.
        d = det("car", (300, 600, 600, 625), conf=0.9)
        self.assertEqual(self.f.apply([d]), [])

    def test_tall_thin_person_passes(self):
        # 35w x 140h person â€” normal aspect 0.25.
        d = det("person", (640, 400, 675, 540), conf=0.9)
        self.assertEqual(len(self.f.apply([d])), 1)


class TestRoadPaintFilter(unittest.TestCase):
    """The headline scenario: wide-flat box below the horizon labeled 'car'."""

    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720)

    def test_crosswalk_stripe_rejected_even_as_car(self):
        # below horizon (y > 230 â‰ˆ 0.32*720), aspect 4.5
        d = det("car", (300, 580, 660, 660), conf=0.9)
        self.assertEqual(self.f.apply([d]), [])
        # Could be classified as paint or as aspect; both are correct rejections.
        s = self.f.stats()
        self.assertTrue(s["drop_paint"] >= 1 or s["drop_aspect"] >= 1)

    def test_crosswalk_arrow_low_in_frame_rejected(self):
        # Crosswalk-arrow pattern: bottom of bbox at y=660 (deep in hood
        # band), width 200px, height 40px, aspect 5.0 â†’ road paint.
        d = det("car", (480, 620, 680, 660), conf=0.55)
        self.assertEqual(self.f.apply([d]), [])
        # The aspect filter or the paint filter catches it; both are
        # valid rejections of the same false-positive pattern.
        s = self.f.stats()
        self.assertTrue(s["drop_paint"] >= 1 or s["drop_aspect"] >= 1)

    def test_real_distant_car_NOT_rejected_as_paint(self):
        # Real distant car at the vanishing point: ~50w x 30h, sitting
        # at yâ‰ˆ360 (just below horizon). This was the false-rejection
        # the previous (too-aggressive) filter caused. MUST be kept.
        d = det("car", (610, 360, 660, 390), conf=0.55)
        out = self.f.apply([d])
        self.assertEqual(len(out), 1)

    def test_real_distant_truck_kept(self):
        # Big truck silhouette near vanishing point, ~80w x 60h.
        d = det("truck", (600, 340, 680, 400), conf=0.40)
        out = self.f.apply([d])
        self.assertEqual(len(out), 1)

    def test_pedestrian_silhouette_kept_when_short_but_tall_aspect(self):
        # 25w x 60h person â€” fairly small but tall-thin shape preserves
        # the aspect signature. ~8% of 720 = 60px.
        d = det("person", (640, 380, 665, 460), conf=0.55)
        out = self.f.apply([d])
        self.assertEqual(len(out), 1)


class TestSkyBandFilter(unittest.TestCase):
    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720)

    def test_detection_entirely_in_sky_dropped(self):
        # bbox fully above 0.20*720 = 144 â†’ sky/building area.
        d = det("car", (200, 30, 320, 110), conf=0.9)
        self.assertEqual(self.f.apply([d]), [])
        self.assertEqual(self.f.stats()["drop_offroad"], 1)


class TestTemporalFilter(unittest.TestCase):
    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720, require_consecutive=3)

    def test_single_frame_ghost_rejected(self):
        d = det("car", (400, 300, 600, 460), conf=0.9)
        # First frame â€” should be held in pending.
        self.assertEqual(self.f.apply([d]), [])

    def test_three_consistent_frames_pass(self):
        d1 = det("car", (400, 300, 600, 460), conf=0.9)
        d2 = det("car", (405, 302, 605, 462), conf=0.9)
        d3 = det("car", (408, 305, 608, 465), conf=0.9)
        self.f.apply([d1])
        self.f.apply([d2])
        out = self.f.apply([d3])
        self.assertEqual(len(out), 1)

    def test_flickering_box_never_promoted(self):
        # Different bbox each frame â†’ no IoU match â†’ never reaches hits=3.
        for i in range(6):
            d = det("car", (100 + 200 * i, 300, 200 + 200 * i, 460), conf=0.9)
            self.assertEqual(self.f.apply([d]), [])


class TestRealisticBatch(unittest.TestCase):
    """A full frame's worth of detections, mix of real + false-positives."""

    def setUp(self):
        self.f = DetectionFilter(frame_w=1280, frame_h=720)

    def test_keeps_real_drops_fakes(self):
        dets = [
            # 3 real objects
            det("car", (560, 300, 760, 480), conf=0.85),
            det("person", (350, 350, 400, 520), conf=0.70),
            det("truck", (820, 280, 1080, 540), conf=0.80),
            # false positives we want gone
            det("car", (200, 600, 480, 625), conf=0.55),  # wide+flat = paint
            det("person", (640, 410, 655, 425), conf=0.65),  # 15px tall = tiny
            det("car", (100, 50, 200, 110), conf=0.85),  # sky band
            det("car", (400, 300, 600, 460), conf=0.18),  # low conf
            det("zebra", (400, 300, 600, 460), conf=0.95),  # wrong class
        ]
        out = self.f.apply(dets)
        self.assertEqual(len(out), 3)
        kept_classes = sorted([d.class_name for d in out])
        self.assertEqual(kept_classes, ["car", "person", "truck"])


class TestStatsAndReset(unittest.TestCase):
    def test_reset_clears_state(self):
        f = DetectionFilter(require_consecutive=2)
        f.apply([det("car", (400, 300, 600, 460), conf=0.9)])
        self.assertGreater(sum(f.stats().values()), 0)
        f.reset()
        self.assertEqual(sum(f.stats().values()), 0)


if __name__ == "__main__":
    unittest.main()

