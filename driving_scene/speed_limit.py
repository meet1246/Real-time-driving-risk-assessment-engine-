"""
speed_limit.py — speed-limit sign detection scaffold.

This is a stub-with-real-hooks module. Today it does two useful things:

  1. Provides a single `SpeedLimitTracker` that the renderer / main loop can
     query each frame for a "current posted speed limit" reading, with
     proper confidence + staleness logic. If nothing has been seen recently,
     it returns None and the HUD shows "--".

  2. Demonstrates the integration shape for plugging in a real classifier
     later. Two reasonable upgrade paths are sketched in `_classify_crop`:
       a) Fine-tuned YOLO/MobileNet head trained on GTSRB (German Traffic
          Sign Benchmark) or LISA (US) datasets, with classes for each
          common posted limit (20/30/40/50/60/70/80/100/120 km/h, plus
          25/35/45/55/65 mph for US).
       b) A two-stage approach: detect "sign-like" boxes with YOLO, then
          run an OCR pass (e.g. EasyOCR / Tesseract) on the cropped patch
          and parse the digits.

For now, `_classify_crop` returns None so the HUD shows "--". When you wire
in a real classifier, that's the one method to replace.

This module is intentionally decoupled from the rest of the risk engine —
the renderer only needs the float (km/h) or None. No other module depends
on it.

NOTE: This is *not* certified for real driving use. It's a portfolio /
educational hook. Real driver-assistance systems integrate map data, GPS
geofencing, and redundant sensor fusion to confirm speed-limit readings.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Optional, Tuple

import numpy as np


# Classes from COCO that YOLOv8 already knows about and that are *visually
# similar* to speed-limit signs. We don't get the value from these — we use
# them as "there's probably a sign here, run the classifier on the crop".
SIGN_LIKE_COCO_CLASSES = {"stop sign", "traffic light"}


@dataclass
class _Reading:
    value_kmh: float
    confidence: float
    timestamp: float


class SpeedLimitTracker:
    """
    Accumulates speed-limit readings over time and exposes a single
    `.current()` method returning the best estimate (or None if stale).

    Typical usage in the main loop:

        slt = SpeedLimitTracker()
        ...
        slt.observe(detections=last_detections, frame=frame_bgr)
        canvas = renderer.render(
            ...,
            speed_limit_kmh=slt.current(),
        )
    """

    def __init__(
        self,
        window_seconds: float = 6.0,
        min_confidence: float = 0.55,
        min_agree: int = 2,
        stale_after: float = 10.0,
    ):
        # How long readings stay in the rolling window.
        self.window_seconds = float(window_seconds)
        # Per-reading minimum confidence to be considered at all.
        self.min_confidence = float(min_confidence)
        # How many *agreeing* readings inside the window before we publish.
        self.min_agree = int(min_agree)
        # If no reading has been published for this long, .current() -> None.
        self.stale_after = float(stale_after)

        self._readings: Deque[_Reading] = deque()
        self._last_published: Optional[_Reading] = None

    # ---- public API ---------------------------------------------------- #

    def observe(
        self,
        detections: Optional[Iterable[Any]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Run one observation step. Safe to call every frame."""
        if frame is None or detections is None:
            self._gc()
            return

        for det in detections:
            cls = (getattr(det, "class_name", None) or getattr(det, "cls", None) or "").lower()
            if cls not in SIGN_LIKE_COCO_CLASSES:
                continue
            bbox = getattr(det, "bbox_xyxy", None) or getattr(det, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = (int(max(0, v)) for v in bbox)
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            result = self._classify_crop(crop)
            if result is None:
                continue
            value, conf = result
            if conf < self.min_confidence:
                continue
            self._readings.append(_Reading(value_kmh=float(value), confidence=float(conf), timestamp=time.time()))

        self._gc()
        self._maybe_publish()

    def current(self) -> Optional[float]:
        """The best current speed-limit estimate in km/h, or None."""
        if self._last_published is None:
            return None
        if time.time() - self._last_published.timestamp > self.stale_after:
            return None
        return self._last_published.value_kmh

    def reset(self) -> None:
        self._readings.clear()
        self._last_published = None

    # ---- override point: real classifier goes here -------------------- #

    def _classify_crop(self, crop_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Classify a sign crop. Return (speed_kmh, confidence in 0..1) or None.

        Replace this method to plug in a real model. Two recommended paths:

        --- Path A: dedicated classifier --------------------------------
        Train a small CNN (MobileNetV3 or a YOLO classification head) on
        GTSRB or LISA. Inference looks like:

            with torch.no_grad():
                probs = self._model(self._prep(crop_bgr))   # (n_classes,)
            cls_idx = int(probs.argmax())
            conf = float(probs[cls_idx])
            kmh = self.CLASS_TO_KMH.get(cls_idx)
            return (kmh, conf) if kmh is not None else None

        --- Path B: detection + OCR -------------------------------------
        Run OCR on the crop, parse digits:

            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            text_results = reader.readtext(crop_bgr, detail=1, paragraph=False)
            for _, txt, conf in text_results:
                digits = "".join(ch for ch in txt if ch.isdigit())
                if digits and 5 <= int(digits) <= 130:
                    return float(int(digits)), float(conf)
            return None

        Today: returns None so the HUD shows "--".
        """
        return None

    # ---- internals ---------------------------------------------------- #

    def _gc(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._readings and self._readings[0].timestamp < cutoff:
            self._readings.popleft()

    def _maybe_publish(self) -> None:
        if len(self._readings) < self.min_agree:
            return
        # Group recent readings by value; publish the most common.
        counts: dict = {}
        conf_sum: dict = {}
        for r in self._readings:
            counts[r.value_kmh] = counts.get(r.value_kmh, 0) + 1
            conf_sum[r.value_kmh] = conf_sum.get(r.value_kmh, 0.0) + r.confidence
        best_value = max(counts, key=lambda v: (counts[v], conf_sum[v]))
        if counts[best_value] < self.min_agree:
            return
        avg_conf = conf_sum[best_value] / counts[best_value]
        self._last_published = _Reading(value_kmh=best_value, confidence=avg_conf, timestamp=time.time())

