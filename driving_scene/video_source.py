"""Video capture from webcam index or file path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import cv2


@dataclass
class VideoSource:
    """Wraps cv2.VideoCapture with explicit source type."""

    source: Union[int, str]
    cap: cv2.VideoCapture

    @classmethod
    def open(cls, source: str | int) -> "VideoSource":
        if isinstance(source, str) and source.isdigit():
            idx = int(source)
            cap = cv2.VideoCapture(idx)
        elif isinstance(source, int):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")
        return cls(source=source, cap=cap)

    def read(self):
        return self.cap.read()

    @property
    def fps(self) -> float:
        v = self.cap.get(cv2.CAP_PROP_FPS)
        return float(v) if v and v > 1.0 else 30.0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self) -> None:
        self.cap.release()
