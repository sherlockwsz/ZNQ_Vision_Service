from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import cv2


@dataclass
class SlotInspection:
    # N: empty / skip; Y: occupied / continue original Screw algorithm.
    presence: str
    max_circle_confidence: float
    image: np.ndarray


class EmptySlotDetector:
    """Independent YOLO Circle model; does not modify ScrewSpringDetector."""
    def __init__(self, model_path: str | Path, threshold: float = 0.70):
        import torch
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.threshold = float(threshold)
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("empty_slot.threshold must be within (0,1]")
        names = self.model.names
        if isinstance(names, dict):
            class0 = names.get(0, names.get('0'))
        else:
            class0 = names[0]
        if str(class0).lower() != 'circle':
            raise RuntimeError(f"Empty-slot model class 0 must be Circle, got {names!r}")

    def process_frame(self, frame: np.ndarray) -> SlotInspection:
        if frame is None or not frame.size:
            raise ValueError("Empty slot frame")
        image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Unsupported frame shape {frame.shape}")
        # Default model inference confidence is below the 0.70 acceptance
        # threshold. Explicitly retain low-confidence detections for diagnostics.
        outputs = self.model(image, device=self.device, conf=0.01, verbose=False)
        max_conf = 0.0
        for output in outputs:
            boxes = getattr(output, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                if cls == 0 and math.isfinite(confidence):
                    max_conf = max(max_conf, confidence)
        # The agreed business mapping: detected Circle >= threshold => empty N.
        presence = 'N' if max_conf >= self.threshold else 'Y'
        return SlotInspection(presence=presence, max_circle_confidence=max_conf, image=image)


class SlotThenScrewDetector:
    """Same-frame cascade. Exactly one source.capture_single() per PLC request."""
    def __init__(self, empty_detector: EmptySlotDetector, screw_detector):
        self.empty_detector = empty_detector
        self.screw_detector = screw_detector

    def process_frame(self, frame: np.ndarray):
        first = self.empty_detector.process_frame(frame)
        if first.presence == 'N':
            # Never run screw model on an explicitly empty slot.
            return SlotScrewResult(first.presence, first.max_circle_confidence, None, first.image)
        second = self.screw_detector.process_frame(frame)
        return SlotScrewResult(first.presence, first.max_circle_confidence, second, second.annotated_image)


@dataclass
class SlotScrewResult:
    presence: str
    empty_confidence: float
    screw: object | None
    annotated_image: np.ndarray
