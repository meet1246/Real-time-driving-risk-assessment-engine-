"""Render scenarios that exercise the new lane-aware placement.

These should reproduce the situations the user called out:
  1. Cyclist + delivery van in the user's actual Toronto frame:
     cyclist in the bike lane (right side), van in ego lane up ahead.
  2. Oncoming car: stays on the LEFT of the double-yellow, not in
     front of the ego.
  3. Pedestrian crossing left-to-right: starts on left sidewalk,
     migrates onto road only when actually crossing.
  4. Same-direction traffic mixed: car in ego lane + car in right
     lane, oncoming on the left.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene.scene_renderer import SceneRenderer


def trk(tid, cls, bbox, s, lvl, vel=None):
    return SimpleNamespace(track_id=tid, class_name=cls, bbox_xyxy=bbox, risk_score=s, risk_level=lvl, velocity=vel)


W, H = 1280, 720
fake = np.full((720, 1280, 3), 30, dtype=np.uint8)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# --- Scenario A: Toronto-style dashcam frame -----------------------------
r = SceneRenderer(W, H)
toronto = [
    trk(1, "truck", (560, 360, 760, 540), 35, "MEDIUM"),
    trk(2, "bicycle", (510, 460, 580, 600), 40, "MEDIUM"),
    trk(3, "car", (200, 380, 320, 470), 18, "LOW"),
]
risk = SimpleNamespace(global_score=40, global_level="MEDIUM", global_action="MONITOR", primary_track_id=1)
for _ in range(4):
    out = r.render(fake, tracks=toronto, risk_state=risk, view_mode="scene", ego_speed_kmh=18, speed_limit_kmh=40)
cv2.imwrite(str(OUT / "scene_toronto.png"), out)
print("wrote scene_toronto.png")


# --- Scenario B: oncoming car only --------------------------------------
r = SceneRenderer(W, H)
onc = [trk(10, "car", (280, 400, 380, 480), 22, "LOW", vel=(0, 25))]
for _ in range(4):
    out = r.render(fake, tracks=onc, view_mode="scene", ego_speed_kmh=55, speed_limit_kmh=60)
cv2.imwrite(str(OUT / "scene_oncoming.png"), out)
print("wrote scene_oncoming.png")


# --- Scenario C: pedestrian crossing -------------------------------------
r = SceneRenderer(W, H)
ped_start = [trk(20, "person", (140, 440, 180, 560), 8, "LOW", vel=(80, 0))]
ped_mid = [trk(21, "person", (560, 460, 600, 580), 78, "HIGH", vel=(80, 0))]
for _ in range(3):
    out = r.render(fake, tracks=ped_start, view_mode="scene", ego_speed_kmh=22, speed_limit_kmh=30)
cv2.imwrite(str(OUT / "scene_ped_sidewalk.png"), out)
print("wrote scene_ped_sidewalk.png")

risk = SimpleNamespace(global_score=78, global_level="HIGH", global_action="ALERT", primary_track_id=21)
r2 = SceneRenderer(W, H)
for _ in range(3):
    out = r2.render(fake, tracks=ped_mid, risk_state=risk, view_mode="scene", ego_speed_kmh=8, speed_limit_kmh=30)
cv2.imwrite(str(OUT / "scene_ped_crossing.png"), out)
print("wrote scene_ped_crossing.png")


# --- Scenario D: mixed traffic, two same-direction lanes + oncoming ------
r = SceneRenderer(W, H)
mixed = [
    trk(30, "car", (610, 360, 690, 430), 30, "LOW"),
    trk(31, "truck", (760, 380, 880, 500), 50, "MEDIUM"),
    trk(32, "car", (240, 400, 340, 470), 25, "LOW"),
    trk(33, "car", (140, 380, 220, 440), 18, "LOW"),
]
risk = SimpleNamespace(global_score=50, global_level="MEDIUM", global_action="MONITOR", primary_track_id=31)
for _ in range(3):
    out = r.render(fake, tracks=mixed, risk_state=risk, view_mode="scene", ego_speed_kmh=75, speed_limit_kmh=80)
cv2.imwrite(str(OUT / "scene_mixed.png"), out)
print("wrote scene_mixed.png")


# --- Scenario E: car directly in front of ego (BRAKE) -------------------
r = SceneRenderer(W, H)
front = [trk(40, "car", (570, 360, 720, 560), 92, "CRITICAL")]
risk = SimpleNamespace(global_score=92, global_level="CRITICAL", global_action="BRAKE", primary_track_id=40)
for _ in range(3):
    out = r.render(fake, tracks=front, risk_state=risk, view_mode="scene", ego_speed_kmh=70, speed_limit_kmh=80)
cv2.imwrite(str(OUT / "scene_brake.png"), out)
print("wrote scene_brake.png")

