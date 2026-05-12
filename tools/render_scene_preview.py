#!/usr/bin/env python3
"""Static preview image for Tesla-inspired scene UI (no YOLO / video)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from driving_scene.risk import Decision, FrameRiskResult, ObjectRisk, RiskComponents, RiskLevel
from driving_scene.scene_renderer import SceneSmoothState, render_tesla_scene
from driving_scene.tracker import TrackedObject


def _obj(
    tid: int,
    cls: str,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    level: RiskLevel,
) -> tuple[TrackedObject, ObjectRisk]:
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    tr = TrackedObject(track_id=tid, class_name=cls, birth_frame=1, last_seen_frame=1)
    tr.bbox_history = [(x1, y1, x2, y2)]
    tr.center_history = [(cx, cy)]
    tr.missing_frames = 0
    z = RiskComponents(0, 0, 0, 0, 0, 0)
    o = ObjectRisk(
        track_id=tid,
        class_name=cls,
        risk_level=level,
        risk_score=55.0,
        ttc_sec=6.0,
        decision=Decision.MONITOR,
        components=z,
        bbox=(x1, y1, x2, y2),
    )
    return tr, o


def main() -> None:
    W, H = 1280, 720
    frame_shape = (H, W)

    tracks: dict = {}
    per: list = []

    # Synthetic dashcam-style placements (image coords): far = small y; near = large y
    pairs = [
        (1, "car", 420.0, 210.0, 72.0, 46.0, RiskLevel.LOW),
        (2, "car", 680.0, 510.0, 210.0, 130.0, RiskLevel.HIGH),  # primary
        (3, "person", 920.0, 440.0, 38.0, 95.0, RiskLevel.LOW),
        (4, "truck", 240.0, 360.0, 130.0, 95.0, RiskLevel.MEDIUM),
        (5, "motorcycle", 1040.0, 490.0, 55.0, 85.0, RiskLevel.LOW),
    ]
    for tid, cls, cx, cy, bw, bh, lvl in pairs:
        tr, o = _obj(tid, cls, cx, cy, bw, bh, lvl)
        tracks[tid] = tr
        per.append(o)

    risk = FrameRiskResult(
        per_object=per,
        global_risk_level=RiskLevel.HIGH,
        global_decision=Decision.ALERT,
        global_score=70.0,
        primary_threat_id=2,
    )

    smooth = SceneSmoothState()
    img = render_tesla_scene(
        frame_shape,
        risk,
        tracks,
        smooth,
        58.0,
        "balanced",
        simulated_speed_kmh=None,
        debug=False,
    )

    out = ROOT / "outputs" / "preview_tesla_scene.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
