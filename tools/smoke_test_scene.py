"""
Quick smoke test for the upgraded scene_view.SceneRenderer.

Run from the project root:

    python tools/smoke_test_scene.py

It synthesizes fake detections / tracks and renders a single frame to
outputs/scene_smoke.png so you can eyeball the new Tesla-style look
without needing a webcam or a video file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.scene_renderer import SceneRenderer  # noqa: E402


def make_fake_track(track_id, cls, x1, y1, x2, y2, score, level):
    return SimpleNamespace(
        track_id=track_id,
        class_name=cls,
        bbox_xyxy=(x1, y1, x2, y2),
        risk_score=score,
        risk_level=level,
    )


def main():
    W, H = 1280, 720
    fw, fh = 1280, 720
    fake_frame = np.full((fh, fw, 3), 30, dtype=np.uint8)

    tracks = [
        make_fake_track(1, "car", 540, 340, 720, 470, 78, "HIGH"),
        make_fake_track(2, "truck", 120, 300, 280, 470, 42, "MEDIUM"),
        make_fake_track(3, "person", 900, 430, 950, 560, 30, "LOW"),
        make_fake_track(4, "bicycle", 1050, 410, 1140, 520, 22, "LOW"),
        make_fake_track(5, "car", 770, 300, 860, 380, 60, "MEDIUM"),
        make_fake_track(6, "bus", 300, 280, 500, 430, 88, "CRITICAL"),
    ]
    telemetry = SimpleNamespace(
        display_fps=58.4,
        detection_fps=14.7,
        yolo_latency_ms=42.0,
        track_latency_ms=1.4,
        risk_latency_ms=0.7,
        dropped_detections=3,
    )
    risk_state = SimpleNamespace(
        global_score=72.0,
        global_level="HIGH",
        global_action="ALERT",
        primary_track_id=6,
    )

    r = SceneRenderer(width=W, height=H)
    out = None
    for _ in range(4):
        out = r.render(
            frame_bgr=fake_frame,
            tracks=tracks,
            telemetry=telemetry,
            risk_state=risk_state,
            ego_speed_kmh=58.0,
            view_mode="scene",
            speed_limit_kmh=60.0,
        )
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "scene_smoke.png"
    cv2.imwrite(str(out_path), out)
    print(f"[smoke] wrote {out_path}  shape={out.shape}")

    split = r.render(
        frame_bgr=fake_frame,
        tracks=tracks,
        telemetry=telemetry,
        risk_state=risk_state,
        ego_speed_kmh=58.0,
        view_mode="split",
        speed_limit_kmh=60.0,
    )
    cv2.imwrite(str(out_dir / "scene_split.png"), split)
    print(f"[smoke] wrote {out_dir / 'scene_split.png'}")


if __name__ == "__main__":
    main()

