"""Simple centroid + class matching tracker (no DeepSORT)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    MAX_MATCH_DISTANCE_PX,
    MAX_MISSING_FRAMES,
    TRACK_HISTORY_LEN,
    TRACK_MATCH_DIST_WEIGHT,
    TRACK_MATCH_IOU_WEIGHT,
    TRACK_MATCH_MAX_COST,
)
from .detector import Detection
from .geometry import bbox_iou_xyxy


def _centroid_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float(np.hypot(dx, dy))


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    birth_frame: int = 0
    bbox_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    center_history: List[Tuple[float, float]] = field(default_factory=list)
    last_seen_frame: int = 0
    missing_frames: int = 0
    last_confidence: float = 0.0
    last_class_id: int = 0
    # Quality / stability instrumentation. Downstream code (SceneMapper /
    # debug overlay) reads track_quality to fade or hide flicker tracks.
    hits: int = 0                  # frames matched to a detection
    misses: int = 0                # cumulative missed frames over the lifetime
    track_quality: float = 0.0     # EMA in [0, 1]; rises with matches, falls with misses

    @property
    def age(self) -> int:
        """Frames since track was created (inclusive of birth frame)."""
        if self.last_seen_frame <= 0:
            return 0
        return max(0, self.last_seen_frame - self.birth_frame + 1)

    def append_state(
        self,
        det: Detection,
        frame_idx: int,
        history_len: int = TRACK_HISTORY_LEN,
    ) -> None:
        self.bbox_history.append(det.bbox)
        self.center_history.append(det.center)
        self.last_seen_frame = frame_idx
        self.missing_frames = 0
        self.last_confidence = det.confidence
        self.last_class_id = det.class_id
        self.class_name = det.class_name
        self.hits += 1
        # EMA toward 1.0 on each hit. Confidence weights the reward so a 0.9-
        # confidence detection bumps quality more than a 0.45 one.
        reward = 0.85 * max(0.0, min(1.0, det.confidence)) + 0.15
        self.track_quality = 0.85 * self.track_quality + 0.15 * reward
        if len(self.bbox_history) > history_len:
            self.bbox_history = self.bbox_history[-history_len:]
        if len(self.center_history) > history_len:
            self.center_history = self.center_history[-history_len:]

    def register_miss(self) -> None:
        """Called by the tracker for every frame this track was not matched."""
        self.missing_frames += 1
        self.misses += 1
        # Decay quality faster than it grew so flicker tracks fall below the
        # render threshold quickly.
        self.track_quality = 0.80 * self.track_quality

    @property
    def last_bbox(self) -> Tuple[float, float, float, float]:
        return self.bbox_history[-1] if self.bbox_history else (0.0, 0.0, 0.0, 0.0)

    @property
    def last_center(self) -> Tuple[float, float]:
        return self.center_history[-1] if self.center_history else (0.0, 0.0)


class CentroidTracker:
    def __init__(self) -> None:
        self._next_id = 1
        self._tracks: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Detection], frame_idx: int) -> Dict[int, TrackedObject]:
        """Assign stable IDs; increment age; mark missing."""
        used_track_ids: set = set()
        unmatched_dets = list(detections)

        # Greedy match: for each det, nearest track of same class within radius
        tracks_by_class: Dict[str, List[int]] = {}
        for tid, tr in self._tracks.items():
            tracks_by_class.setdefault(tr.class_name, []).append(tid)

        # Sort detections by confidence for stable matching
        unmatched_dets.sort(key=lambda d: -d.confidence)

        for det in list(unmatched_dets):
            best_tid: Optional[int] = None
            best_cost = TRACK_MATCH_MAX_COST + 0.01
            for tid in tracks_by_class.get(det.class_name, []):
                if tid in used_track_ids:
                    continue
                tr = self._tracks[tid]
                dist = _centroid_dist(det.center, tr.last_center)
                dist_n = min(1.0, dist / max(MAX_MATCH_DISTANCE_PX, 1e-6))
                iou = bbox_iou_xyxy(det.bbox, tr.last_bbox)
                cost = TRACK_MATCH_DIST_WEIGHT * dist_n + TRACK_MATCH_IOU_WEIGHT * (1.0 - iou)
                # Strong overlap can rescue a centroid jump (lane change, jitter)
                if iou >= 0.35:
                    cost = min(cost, TRACK_MATCH_DIST_WEIGHT * dist_n * 0.65 + TRACK_MATCH_IOU_WEIGHT * (1.0 - iou))
                if cost < best_cost:
                    best_cost = cost
                    best_tid = tid
            if best_tid is not None and best_cost <= TRACK_MATCH_MAX_COST:
                tr = self._tracks[best_tid]
                tr.append_state(det, frame_idx)
                used_track_ids.add(best_tid)
                unmatched_dets.remove(det)

        # New tracks for leftover detections
        for det in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            tr = TrackedObject(
                track_id=tid,
                class_name=det.class_name,
                birth_frame=frame_idx,
                last_seen_frame=frame_idx,
            )
            tr.append_state(det, frame_idx)
            self._tracks[tid] = tr
            used_track_ids.add(tid)

        # Missing tracks
        stale: List[int] = []
        for tid, tr in self._tracks.items():
            if tid in used_track_ids:
                continue
            if tr.last_seen_frame == frame_idx:
                continue
            tr.register_miss()
            if tr.missing_frames > MAX_MISSING_FRAMES:
                stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

        return dict(self._tracks)
