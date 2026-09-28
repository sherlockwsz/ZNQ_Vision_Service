from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


log = logging.getLogger(__name__)


@dataclass
class ScrewInspectionResult:
    detected: bool
    detected_angle: float
    confidence: float
    annotated_image: np.ndarray
    details: Dict


class ScrewSpringDetector:
    """YOLO screw detector using an undirected two-screw line angle."""

    def __init__(self, model_path: str | Path, *, mm_per_pixel: float,
                 min_screw_distance_mm: float, max_screw_distance_mm: float):
        import torch
        from ultralytics import YOLO

        self._torch = torch
        self.model = YOLO(str(model_path))
        # Keep the original device selection logic unchanged.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.mm_per_pixel = float(mm_per_pixel)
        self.min_screw_distance_mm = float(min_screw_distance_mm)
        self.max_screw_distance_mm = float(max_screw_distance_mm)
        if not math.isfinite(self.mm_per_pixel) or self.mm_per_pixel <= 0.0:
            raise ValueError("screw_algorithm.mm_per_pixel must be a finite value greater than zero")
        if (not math.isfinite(self.min_screw_distance_mm)
                or not math.isfinite(self.max_screw_distance_mm)
                or self.min_screw_distance_mm < 0.0
                or self.min_screw_distance_mm > self.max_screw_distance_mm):
            raise ValueError("screw_algorithm screw distance range is invalid")

        self.class_names = {0: "Screw", 1: "Spring"}
        self.colors = {
            "Screw": (0, 0, 255),
            "line": (255, 0, 0),
            "text": (255, 255, 255),
        }

    def calculate_center(self, box: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

    def filter_boxes(self, results) -> Dict:
        filtered_results = {
            "screw_boxes": [],
            "screw_centers": [],
        }

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls[0])
                if class_id != 0:
                    # The model may still report Spring, but Spring is outside
                    # the current two-Screw angle business logic.
                    continue
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                if len(bbox) != 4 or not np.all(np.isfinite(bbox)) or not math.isfinite(confidence):
                    continue
                filtered_results["screw_boxes"].append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "center": self.calculate_center(bbox),
                })

        #filtered_results["screw_centers"] = [box["center"] for box in filtered_results["screw_boxes"]]
        
        #return filtered_results

        # TEMP DEBUG:
        # 多个 Screw 时按置信度从高到低排列，
        # 后续临时取前两个计算角度。
        filtered_results["screw_boxes"].sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        filtered_results["screw_centers"] = [
            box["center"]
            for box in filtered_results["screw_boxes"]
        ]

        return filtered_results

    @staticmethod
    def _detected_line_angle(screw1: Tuple[float, float], screw2: Tuple[float, float]) -> float:
        """Angle of an undirected line in image coordinates, normalized to [0, 180)."""
        dx = screw2[0] - screw1[0]
        dy = screw2[1] - screw1[1]
        return math.degrees(math.atan2(dy, dx)) % 180.0
    '''
    def _evaluate_screw_geometry(self, screw_centers):
        """Return angle/distance for exactly two valid centers, or an invalid reason."""
        screw_count = len(screw_centers)
        if screw_count != 2:
            return None, None, None, f"wrong screw count: {screw_count}"

        screw1, screw2 = screw_centers
        distance_px = self.calculate_distance(screw1, screw2)
        distance_mm = distance_px * self.mm_per_pixel
        if not self.min_screw_distance_mm <= distance_mm <= self.max_screw_distance_mm:
            return None, distance_px, distance_mm, "screw distance invalid"

        detected_angle = self._detected_line_angle(screw1, screw2)
        return detected_angle, distance_px, distance_mm, ""
    '''    
    def _evaluate_screw_geometry(self, screw_centers):
        """TEMP DEBUG: use the first two screws and bypass distance validity check."""
        screw_count = len(screw_centers)

        # 至少必须存在两个 Screw，否则无法计算连线角度
        if screw_count < 2:
            return None, None, None, f"not enough screws: {screw_count}"

        # TEMP DEBUG:
        # screw_centers 已按 confidence 对应顺序排列，
        # 临时只取前两个 Screw 计算。
        screw1 = screw_centers[0]
        screw2 = screw_centers[1]

        # 中心距仍然计算，只是不再用于 InvalidResult 判定
        distance_px = self.calculate_distance(screw1, screw2)
        distance_mm = distance_px * self.mm_per_pixel

        # TEMP DEBUG:
        # 暂时旁路正式的 24~26 mm 中心距判定
        #
        # if not self.min_screw_distance_mm <= distance_mm <= self.max_screw_distance_mm:
        #     return None, distance_px, distance_mm, "screw distance invalid"

        detected_angle = self._detected_line_angle(
            screw1,
            screw2
        )

        return (
            detected_angle,
            distance_px,
            distance_mm,
            ""
        )

    def process_frame(self, frame: np.ndarray) -> ScrewInspectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Empty screw camera frame")

        # The supplied file used cv2.imread(), which yields a 3-channel BGR image.
        # The production camera is mono, so convert Mono8 -> BGR before YOLO to keep
        # the model input semantics equivalent to the original offline program.
        if frame.ndim == 2:
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            image = frame.copy()
        else:
            raise ValueError(f"Unsupported screw frame shape: {frame.shape}")

        results = self.model(image, device=self.device)
        filtered_results = self.filter_boxes(results)

        detected_angle, distance_px, distance_mm, invalid_reason = self._evaluate_screw_geometry(
            filtered_results["screw_centers"]
        )
        if distance_px is not None:
            filtered_results["distance_px"] = float(distance_px)
            filtered_results["distance_mm"] = float(distance_mm)
        if invalid_reason:
            filtered_results["invalid_reason"] = invalid_reason
            if invalid_reason.startswith("wrong screw count"):
                log.warning("Screw result invalid: %s", invalid_reason)
            else:
                log.warning("Screw result invalid: distance %.3f mm outside [%.3f, %.3f] mm",
                            distance_mm, self.min_screw_distance_mm, self.max_screw_distance_mm)
            return ScrewInspectionResult(
                detected=False,
                detected_angle=0.0,
                confidence=0.0,
                annotated_image=image,
                details=filtered_results,
            )

        # Drawing follows the supplied program and does not affect detection.
        for screw in filtered_results["screw_boxes"]:
            bbox = screw["bbox"].astype(int)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors["Screw"], thickness=4)
            center_x, center_y = map(int, screw["center"])
            cv2.circle(image, (center_x, center_y), 8, self.colors["Screw"], thickness=-1)
            cv2.circle(image, (center_x, center_y), 12, self.colors["Screw"], thickness=3)

        center1 = tuple(map(int, filtered_results["screw_centers"][0]))
        center2 = tuple(map(int, filtered_results["screw_centers"][1]))
        cv2.line(image, center1, center2, self.colors["line"], thickness=5)
        mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        cv2.circle(image, mid_point, 10, (0, 255, 255), thickness=-1)

        selected_confidences = [x["confidence"] for x in filtered_results["screw_boxes"][:2]]
        confidence = min(selected_confidences) if selected_confidences else 0.0

        text = f"Detected angle: {detected_angle:.2f} deg  Distance: {distance_mm:.2f} mm"
        cv2.rectangle(image, (10, 10), (900, 105), (0, 0, 0), -1)
        cv2.putText(image, text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, self.colors["text"], 4)

        return ScrewInspectionResult(
            detected=True,
            detected_angle=float(detected_angle),
            confidence=float(confidence),
            annotated_image=image,
            details=filtered_results,
        )
