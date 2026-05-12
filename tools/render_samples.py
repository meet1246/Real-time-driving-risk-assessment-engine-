"""Render a couple of edge-case scenes to verify visual robustness."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.scene_renderer import SceneRenderer


def trk(tid, cls, bbox, s, lvl):
    return SimpleNamespace(track_id=tid, class_name=cls, bbox_xyxy=bbox, risk_score=s, risk_level=lvl)


W, H = 1280, 720
fake = np.full((720, 1280, 3), 30, dtype=np.uint8)

# --- Scene 1: empty road (no objects) -----------------------------------
r = SceneRenderer(W, H)
out = r.render(fake, tracks=[], view_mode="scene", ego_speed_kmh=42)
cv2.imwrite(str(ROOT / "outputs" / "scene_empty.png"), out)
print("wrote scene_empty.png")

# --- Scene 2: only pedestrians + cyclist (urban) ------------------------
r = SceneRenderer(W, H)
peds = [
    trk(11, "person", (350, 420, 400, 540), 35, "MEDIUM"),
    trk(12, "person", (560, 460, 610, 600), 70, "HIGH"),
    trk(13, "bicycle", (820, 430, 920, 540), 50, "MEDIUM"),
    trk(14, "person", (180, 380, 220, 480), 18, "LOW"),
]
risk = SimpleNamespace(global_score=70, global_level="HIGH", global_action="MONITOR", primary_track_id=12)
for _ in range(3):
    out = r.render(fake, tracks=peds, risk_state=risk, view_mode="scene", ego_speed_kmh=22, speed_limit_kmh=30)
cv2.imwrite(str(ROOT / "outputs" / "scene_urban.png"), out)
print("wrote scene_urban.png")

# --- Scene 3: highway with a critical truck -----------------------------
r = SceneRenderer(W, H)
hwy = [
    trk(20, "car", (620, 360, 700, 430), 30, "LOW"),
    trk(21, "car", (480, 340, 560, 400), 25, "LOW"),
    trk(22, "truck", (600, 420, 760, 580), 95, "CRITICAL"),
    trk(23, "car", (900, 350, 970, 410), 35, "LOW"),
]
risk = SimpleNamespace(global_score=95, global_level="CRITICAL", global_action="BRAKE", primary_track_id=22)
for _ in range(3):
    out = r.render(fake, tracks=hwy, risk_state=risk, view_mode="scene", ego_speed_kmh=110, speed_limit_kmh=100)
cv2.imwrite(str(ROOT / "outputs" / "scene_highway.png"), out)
print("wrote scene_highway.png")

