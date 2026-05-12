"""
backend_probe.py — load a video, run the full backend pipeline (no GUI),
and write a per-frame CSV of detections / tracks / lane assignments.

Usage:
    python tools/backend_probe.py --source data/dashcam.mp4
    python tools/backend_probe.py --source data/dashcam.mp4 --max-frames 300
    python tools/backend_probe.py --source 0 --max-frames 200 --no-csv

CSV columns:
    frame, track_id, class, conf,
    cx_norm, by_norm, height_ratio, area_ratio,
    lane_id, lane_label, lane_center_norm, lane_width_norm,
    closeness, scene_x, scene_y, track_quality, hidden_reason

Use this to debug WHY a car ended up where it did on the scene without
launching a window. Open the CSV in any spreadsheet, group by track_id, and
you'll see exactly which lane each track was assigned to over time.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from driving_scene import config as cfg  # noqa: E402
from driving_scene import detection_filter as filt_mod  # noqa: E402
from driving_scene import detector as det_mod  # noqa: E402
from driving_scene import lane_model as _lm  # noqa: E402
from driving_scene import motion as motion_mod  # noqa: E402
from driving_scene import tracker as tracker_mod  # noqa: E402
from driving_scene.scene_mapper import SceneMapper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Backend pipeline probe (no GUI).")
    p.add_argument("--source", default=str(cfg.PREFERRED_DEFAULT_DRIVE_VIDEO),
                   help="Video file or webcam index.")
    p.add_argument("--performance", default="fast", choices=["fast", "balanced", "quality"])
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--max-frames", type=int, default=300, help="Stop after N frames.")
    p.add_argument("--csv", default="outputs/backend_probe.csv")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--summary-every", type=int, default=60)
    return p.parse_args()


def main():
    args = parse_args()

    src = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[probe] cannot open {src}")
        sys.exit(2)

    preset = cfg.PERFORMANCE_PRESETS[args.performance]
    runtime = det_mod.RuntimeConfig(
        model_path=preset.model_path, conf_threshold=args.conf, imgsz=preset.imgsz,
    )
    detector = det_mod.YoloDetector(runtime)
    # Probe uses SYNCHRONOUS detection — every frame gets inference. That's
    # slow but deterministic, which is the whole point of a backend probe.
    detect_every = int(preset.detect_every_n_frames)

    tracker = tracker_mod.CentroidTracker()
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    # Match the in-app defaults: prefer saved calibration if available.
    boundaries, labels = cfg.load_lane_calibration()
    if not boundaries:
        boundaries = list(cfg.LANE_X_BOUNDARIES)
        labels = list(cfg.LANE_LABELS)

    lanes = _lm.LaneModel(fw, fh)
    lanes.set_boundaries(boundaries, labels)
    mapper = SceneMapper(lanes)

    det_filter = filt_mod.DetectionFilter(
        frame_w=fw, frame_h=fh,
        min_confidence=filt_mod.DEFAULT_MIN_CONFIDENCE,
        min_box_area_frac=0.0006,
        require_consecutive=3,
    )

    csv_path = None
    csv_writer = None
    csv_file = None
    if not args.no_csv:
        csv_path = ROOT / args.csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "track_id", "class", "conf",
            "cx_norm", "by_norm", "height_ratio", "area_ratio",
            "lane_id", "lane_label", "lane_center_norm", "lane_width_norm",
            "closeness", "scene_x", "scene_y", "track_quality", "hidden_reason",
        ])

    last_detections = []
    frame_idx = 0
    started_at = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # Sync detection: run YOLO every Nth frame (matches preset).
            if (frame_idx % detect_every) == 0:
                try:
                    last_detections, _lat = detector.predict_resized_to_original(frame)
                except Exception as e:  # noqa: BLE001
                    print(f"[probe] inference failed: {e}")
                    last_detections = []

            det_filter.update_frame_size(frame.shape[1], frame.shape[0])
            last_detections = det_filter.apply(last_detections)
            tracks = tracker.update(last_detections, frame_idx)

            motions = {
                tid: motion_mod.estimate_motion(t, 1 / 30.0)
                for tid, t in tracks.items() if t.missing_frames == 0
            }

            tracks_with_vel = []
            for tid, tr in tracks.items():
                if tr.missing_frames > 0:
                    continue
                mo = motions.get(tid)
                vel = (float(mo.velocity_x), float(mo.velocity_y)) if mo else None
                tracks_with_vel.append(SimpleNamespace(
                    track_id=tid,
                    class_name=tr.class_name,
                    bbox=tr.last_bbox,
                    velocity=vel,
                    track_quality=tr.track_quality,
                ))

            items = mapper.collect_renderables(tracks_with_vel, last_detections, fw, fh)
            mapper.update_trails(items)
            mapper.prune_stale()

            # Write CSV rows for visible items + hidden items.
            if csv_writer is not None:
                for it in items + mapper.hidden_items:
                    csv_writer.writerow([
                        frame_idx,
                        it.get("track_id"),
                        it.get("class"),
                        f"{(it.get('score') if isinstance(it.get('score'), (int, float)) else 0):0.2f}",
                        f"{it.get('cx_norm', 0):0.3f}",
                        f"{it.get('by_norm', 0):0.3f}",
                        f"{it.get('height_ratio', 0):0.4f}",
                        f"{it.get('area_ratio', 0):0.5f}",
                        it.get("lane"),
                        it.get("lane_label"),
                        f"{it.get('lane_center_norm', 0):0.3f}",
                        f"{it.get('lane_width_norm', 0):0.3f}",
                        f"{it.get('closeness', 0):0.3f}",
                        f"{it.get('scene_x', 0):0.1f}",
                        f"{it.get('scene_y', 0):0.1f}",
                        f"{it.get('track_quality', 0):0.2f}",
                        it.get("hidden_reason") or "",
                    ])

            if frame_idx % args.summary_every == 0:
                avg_q = (
                    sum(t.track_quality for t in tracks.values()) / max(1, len(tracks))
                ) if tracks else 0.0
                elapsed = time.time() - started_at
                fps = frame_idx / max(1e-6, elapsed)
                print(
                    f"[probe] frame={frame_idx} dets={len(last_detections)} "
                    f"tracks={len(tracks)} visible={len(items)} hidden={len(mapper.hidden_items)} "
                    f"avg_quality={avg_q:0.2f} fps={fps:0.1f}"
                )
    finally:
        cap.release()
        if csv_file is not None:
            csv_file.close()
            print(f"[probe] wrote {csv_path}")
        print(f"[probe] done — {frame_idx} frames")


if __name__ == "__main__":
    main()
