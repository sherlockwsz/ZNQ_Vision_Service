from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class ScrewInspectionResult:
    detected: bool
    detected_angle: float
    correction_angle: float
    confidence: float
    annotated_image: np.ndarray
    details: Dict


class ScrewSpringDetector:
    """Production wrapper around the supplied Screw-Test algorithm.

    The detection/filter/angle algorithm is intentionally kept the same as the
    supplied file. Only the I/O shell changed from cv2.imread/cv2.imwrite and
    folder loops to an in-memory numpy frame supplied by Camera Manager.
    """

    def __init__(self, model_path: str | Path):
        import torch
        from ultralytics import YOLO

        self._torch = torch
        self.model = YOLO(str(model_path))
        # Keep the original device selection logic unchanged.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.class_names = {0: "Screw", 1: "Spring"}
        self.colors = {
            "Screw": (0, 0, 255),
            "Spring": (0, 255, 0),
            "line": (255, 0, 0),
            "text": (255, 255, 255),
        }

    def calculate_center(self, box: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def calculate_angle_for_rotation(
        self,
        screw1: Tuple[float, float],
        screw2: Tuple[float, float],
        spring: Tuple[float, float],
    ) -> float:
        # BEGIN supplied algorithm: unchanged
        if screw1[0] > screw2[0]:
            screw1, screw2 = screw2, screw1

        dx = screw2[0] - screw1[0]
        dy = screw2[1] - screw1[1]

        angle_rad = math.atan2(dy, dx)
        current_angle = math.degrees(angle_rad)

        if current_angle < 0:
            current_angle += 360

        mid_x = (screw1[0] + screw2[0]) / 2
        mid_y = (screw1[1] + screw2[1]) / 2

        perp_dx = dy
        perp_dy = -dx

        spring_vec_x = spring[0] - mid_x
        spring_vec_y = spring[1] - mid_y

        dot_product = spring_vec_x * perp_dx + spring_vec_y * perp_dy

        if dot_product < 0:
            target_angle = 180
            rotation_angle = target_angle - current_angle
        else:
            target_angle = 0
            rotation_angle = target_angle - current_angle

        if rotation_angle < 0:
            rotation_angle += 360

        angle_rad = math.radians(rotation_angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        rotated_mid_x = mid_x * cos_angle - mid_y * sin_angle
        rotated_mid_y = mid_x * sin_angle + mid_y * cos_angle

        rotated_spring_x = spring[0] * cos_angle - spring[1] * sin_angle
        rotated_spring_y = spring[0] * sin_angle + spring[1] * cos_angle

        y_diff = rotated_spring_y - rotated_mid_y

        if y_diff > 0:
            rotation_angle = (rotation_angle + 180) % 360

        return rotation_angle
        # END supplied algorithm

    def filter_boxes(self, results) -> Dict:
        # BEGIN supplied algorithm: unchanged
        filtered_results = {
            "screw_boxes": [],
            "spring_boxes": [],
            "screw_centers": [],
            "spring_center": None,
        }

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                if class_id == 0:
                    filtered_results["screw_boxes"].append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "center": self.calculate_center(bbox),
                    })
                elif class_id == 1:
                    filtered_results["spring_boxes"].append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "center": self.calculate_center(bbox),
                    })

        filtered_results["screw_boxes"].sort(key=lambda x: x["confidence"], reverse=True)
        filtered_results["screw_boxes"] = filtered_results["screw_boxes"][:2]
        filtered_results["screw_centers"] = [box["center"] for box in filtered_results["screw_boxes"]]

        if filtered_results["spring_boxes"] and len(filtered_results["screw_centers"]) == 2:
            screw_center1 = filtered_results["screw_centers"][0]
            screw_center2 = filtered_results["screw_centers"][1]

            min_distance_diff = float("inf")
            selected_spring = None

            for spring in filtered_results["spring_boxes"]:
                spring_center = spring["center"]
                dist1 = self.calculate_distance(spring_center, screw_center1)
                dist2 = self.calculate_distance(spring_center, screw_center2)
                distance_diff = abs(dist1 - dist2)

                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    selected_spring = spring

            if selected_spring:
                filtered_results["spring_center"] = selected_spring["center"]
                filtered_results["selected_spring"] = selected_spring

        return filtered_results
        # END supplied algorithm

    @staticmethod
    def _detected_line_angle(screw1: Tuple[float, float], screw2: Tuple[float, float]) -> float:
        """Diagnostic raw line angle; it does not participate in correction logic."""
        if screw1[0] > screw2[0]:
            screw1, screw2 = screw2, screw1
        angle = math.degrees(math.atan2(screw2[1] - screw1[1], screw2[0] - screw1[0]))
        return angle + 360.0 if angle < 0 else angle

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

        if len(filtered_results["screw_centers"]) < 2 or filtered_results["spring_center"] is None:
            return ScrewInspectionResult(
                detected=False,
                detected_angle=0.0,
                correction_angle=0.0,
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

        spring = filtered_results["selected_spring"]
        bbox = spring["bbox"].astype(int)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors["Spring"], thickness=3)
        center_x, center_y = map(int, spring["center"])
        cv2.circle(image, (center_x, center_y), 8, self.colors["Spring"], thickness=-1)

        center1 = tuple(map(int, filtered_results["screw_centers"][0]))
        center2 = tuple(map(int, filtered_results["screw_centers"][1]))
        cv2.line(image, center1, center2, self.colors["line"], thickness=5)
        mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
        cv2.circle(image, mid_point, 10, (0, 255, 255), thickness=-1)

        rotation_angle = self.calculate_angle_for_rotation(
            filtered_results["screw_centers"][0],
            filtered_results["screw_centers"][1],
            filtered_results["spring_center"],
        )
        detected_angle = self._detected_line_angle(
            filtered_results["screw_centers"][0],
            filtered_results["screw_centers"][1],
        )

        selected_confidences = [x["confidence"] for x in filtered_results["screw_boxes"]]
        selected_confidences.append(float(spring["confidence"]))
        confidence = min(selected_confidences) if selected_confidences else 0.0

        text = f"Rotation: {rotation_angle:.2f} deg"
        cv2.rectangle(image, (10, 10), (900, 105), (0, 0, 0), -1)
        cv2.putText(image, text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, self.colors["text"], 4)

        return ScrewInspectionResult(
            detected=True,
            detected_angle=float(detected_angle),
            correction_angle=float(rotation_angle),
            confidence=float(confidence),
            annotated_image=image,
            details=filtered_results,
        )
