"""YOLOv8 detector (ultralytics) filtered to road-relevant COCO classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import COCO_CLASS_IDS, COCO_ID_TO_NAME, RuntimeConfig
from .geometry import bbox_iou_xyxy


# Post-NMS sanity threshold. Ultralytics already runs NMS at inference, but
# we re-check per-class because:
#   * dual-class hits (a "car" + "truck" on the same vehicle from one frame)
#     sometimes survive Ultralytics' NMS;
#   * downscale → upscale rounding can produce slightly offset duplicates;
#   * dedup'ing here keeps the tracker from spawning two IDs for one object.
POST_NMS_IOU_THRESHOLD: float = 0.55


def post_nms(detections: List["Detection"], iou_threshold: float = POST_NMS_IOU_THRESHOLD) -> List["Detection"]:
    """Greedy NMS pass on detector outputs. Keeps the higher-confidence box of
    any pair with IoU > threshold, regardless of class — same object should
    not appear twice even under two class names.
    """
    if not detections:
        return detections
    ordered = sorted(detections, key=lambda d: -d.confidence)
    kept: List["Detection"] = []
    for d in ordered:
        drop = False
        for k in kept:
            if bbox_iou_xyxy(d.bbox, k.bbox) > iou_threshold:
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    area: float


class YoloDetector:
    def __init__(self, cfg: RuntimeConfig):
        from ultralytics import YOLO

        self._cfg = cfg
        self._model = YOLO(cfg.model_path)
        device = cfg.device
        if device is None:
            try:
                import torch  # noqa: WPS433
                if torch.cuda.is_available():
                    device = "cuda"
                    # FP16 is a big win on GPU and matches the half-precision flag.
                    cfg.half = True
                    print(f"[detector] CUDA detected ({torch.cuda.get_device_name(0)}) — using GPU + FP16.")
            except Exception:
                device = None
        if device:
            self._model.to(device)
            self._cfg.device = device

    def predict(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Run inference on BGR frame.
        Returns (detections, inference_seconds).
        """
        import time

        t0 = time.perf_counter()
        results = self._model.predict(
            source=frame_bgr,
            conf=self._cfg.conf_threshold,
            imgsz=self._cfg.imgsz,
            half=self._cfg.half,
            verbose=False,
            classes=list(COCO_CLASS_IDS),
        )
        t1 = time.perf_counter()
        latency = t1 - t0

        dets: List[Detection] = []
        if not results:
            return dets, latency
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return dets, latency

        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), cf, cid in zip(xyxy, confs, cls_ids):
            if cid not in COCO_CLASS_IDS:
                continue
            name = COCO_ID_TO_NAME.get(cid, str(cid))
            cx = (float(x1) + float(x2)) * 0.5
            cy = (float(y1) + float(y2)) * 0.5
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            area = w * h
            dets.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=float(cf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    center=(cx, cy),
                    area=area,
                )
            )
        return post_nms(dets), latency

    def predict_resized_to_original(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[List[Detection], float]:
        """
        Downscale frame so longest side <= imgsz (CPU-friendly), run YOLO, map boxes to full resolution.
        """
        import time

        h0, w0 = frame_bgr.shape[:2]
        max_side = max(h0, w0)
        scale = min(float(self._cfg.imgsz) / float(max_side), 1.0)
        nw = max(1, int(round(w0 * scale)))
        nh = max(1, int(round(h0 * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=interp)
        sx = float(w0) / float(nw)
        sy = float(h0) / float(nh)

        t0 = time.perf_counter()
        results = self._model.predict(
            source=small,
            conf=self._cfg.conf_threshold,
            imgsz=max(nw, nh),
            half=self._cfg.half,
            verbose=False,
            classes=list(COCO_CLASS_IDS),
        )
        t1 = time.perf_counter()
        latency = t1 - t0

        dets: List[Detection] = []
        if not results:
            return dets, latency
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return dets, latency

        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), cf, cid in zip(xyxy, confs, cls_ids):
            if cid not in COCO_CLASS_IDS:
                continue
            name = COCO_ID_TO_NAME.get(cid, str(cid))
            fx1, fy1, fx2, fy2 = (
                float(x1) * sx,
                float(y1) * sy,
                float(x2) * sx,
                float(y2) * sy,
            )
            cx = (fx1 + fx2) * 0.5
            cy = (fy1 + fy2) * 0.5
            bw = max(0.0, fx2 - fx1)
            bh = max(0.0, fy2 - fy1)
            area = bw * bh
            dets.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=float(cf),
                    bbox=(fx1, fy1, fx2, fy2),
                    center=(cx, cy),
                    area=area,
                )
            )
        return post_nms(dets), latency
