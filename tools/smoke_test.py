"""
Quick smoke test: imports, detector on synthetic frame, risk path â€” no GUI.

Run from repo root:
  python tools/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def main() -> None:
    from driving_scene.config import RuntimeConfig
    from driving_scene.detector import YoloDetector
    from driving_scene.motion import estimate_motion
    from driving_scene.risk import RiskLevel, compute_frame_risks
    from driving_scene.tracker import CentroidTracker

    cfg = RuntimeConfig(imgsz=320, half=False)
    det = YoloDetector(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets, ms = det.predict_resized_to_original(frame)
    print("YOLO OK ms=", round(ms * 1000, 1), "dets=", len(dets))

    trk = CentroidTracker()
    tracks = trk.update(dets, 0)
    motions = {tid: estimate_motion(t, 0.033) for tid, t in tracks.items() if t.missing_frames == 0}
    risk = compute_frame_risks(tracks, motions, frame.shape)
    assert risk.global_risk_level == RiskLevel.LOW
    print("Risk OK global=", risk.global_risk_level.value)
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
