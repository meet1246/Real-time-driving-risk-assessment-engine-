"""
types.py — Tesla-Inspired Driving Scene Visualizer

Central type definitions. Most dataclasses already live next to the code that
produces them (Detection in detector.py, TrackedObject in tracker.py, etc.) —
this module simply re-exports them under one roof so callers can write
`from driving_scene.types import Detection, Track` without remembering which
internal module owns which struct.

Also adds visualization-layer dataclasses that don't have a natural home in
a detector/tracker module:

    LaneAssignment   the result of bbox → lane resolution
    SceneObject     an item ready to be drawn on the FSD-style canvas
    SceneFrame      a full frame's worth of SceneObjects + meta
    TelemetryState  what the HUD reads (alias of FrameTelemetry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Re-exports from existing modules — keep the canonical definitions there.
from .detector import Detection  # noqa: F401
from .tracker import TrackedObject as Track  # noqa: F401
from .motion import MotionState  # noqa: F401
from .telemetry import FrameTelemetry as TelemetryState  # noqa: F401


# ---------------------------------------------------------------------- #
# Visualization-layer types                                              #
# ---------------------------------------------------------------------- #

@dataclass
class LaneAssignment:
    """Result of resolving a detection / track to a lane on the FSD scene."""

    lane_id: int
    lane_label: str
    lane_offset: float            # in [-1, 1], position inside the lane
    lane_center_norm: float       # lane center as fraction of frame width
    lane_width_norm: float        # lane width as fraction of frame width


@dataclass
class SceneObject:
    """A single drawable object on the FSD-style canvas.

    The renderer consumes these. The scene_mapper produces them from tracks +
    detections.
    """

    track_id: Optional[int]
    class_name: str
    bbox: Tuple[float, float, float, float]
    scene_x: float
    scene_y: float
    icon_w: float
    icon_h: float
    closeness: float
    lateral: float
    lane: int
    velocity: Optional[Tuple[float, float]] = None
    score: Optional[float] = None
    level: Optional[str] = None


@dataclass
class SceneFrame:
    """One frame's worth of SceneObjects plus the meta the HUD needs."""

    objects: List[SceneObject] = field(default_factory=list)
    primary_threat_id: Optional[int] = None
    lane_curve: float = 0.0
    lane_confidence: float = 0.0
    speed_limit_kmh: Optional[float] = None
    ego_speed_kmh: Optional[float] = None
