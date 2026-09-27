from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CoaxInspectionResult:
    detection_ok: bool
    coaxiality_mm: float
    delta_x_mm: float
    delta_y_mm: float
    center_x: float
    center_y: float
    radius: float
    score: float
    used_fallback: bool
    detect_time_ms: float
    annotated_image: np.ndarray


class CoaxHalconDetector:
    """Runs the supplied circle algorithm through HALCON HDevEngine.

    The .hdvp procedure contains the same shape-model coarse location,
    metrology fine measurement, fallback and quality thresholds. It returns
    only the validated circle geometry; Phase 1 reference/scale conversion is
    deliberately performed here from the single config.toml parameter source.
    """

    def __init__(self, procedure_path: str | Path, *, reference_x_px: float,
                 reference_y_px: float, mm_per_pixel: float):
        try:
            import halcon as ha
        except ImportError as exc:
            raise RuntimeError(
                "HALCON/Python is not importable. Install the Python package matching HALCON 24.11 "
                "and make sure the HALCON runtime/license environment is available."
            ) from exc

        self.ha = ha
        self.procedure_path = Path(procedure_path).resolve()
        self.reference_x_px = float(reference_x_px)
        self.reference_y_px = float(reference_y_px)
        self.mm_per_pixel = float(mm_per_pixel)
        if not math.isfinite(self.mm_per_pixel) or self.mm_per_pixel <= 0.0:
            raise ValueError("coax_algorithm.mm_per_pixel must be a finite value greater than zero")
        for name, value in (("reference_x_px", self.reference_x_px),
                            ("reference_y_px", self.reference_y_px)):
            if not math.isfinite(value):
                raise ValueError(f"coax_algorithm.{name} must be finite")

        engine = ha.HDevEngine()
        engine.set_procedure_path(str(self.procedure_path.parent))
        self._engine = engine
        self._procedure = ha.HDevProcedure.load_external(self.procedure_path.stem)

    @staticmethod
    def _scalar(value, default=0.0):
        try:
            return value[0]
        except Exception:
            return value if value is not None else default

    def _get(self, call, name: str, default=0.0):
        return self._scalar(call.get_output_control_param_by_name(name), default)

    def process_frame(self, frame: np.ndarray) -> CoaxInspectionResult:
        if frame is None or frame.size == 0:
            raise ValueError("Empty coax camera frame")
        if frame.dtype != np.uint8:
            raise ValueError(f"HALCON production input must be Mono8, got {frame.dtype}")
        if frame.ndim != 2:
            raise ValueError(f"HALCON production input must be single-channel, got {frame.shape}")

        frame = np.ascontiguousarray(frame)
        height, width = frame.shape
        h_image = self.ha.gen_image1("byte", int(width), int(height), int(frame.ctypes.data))

        call = self.ha.HDevProcedureCall(self._procedure)
        call.set_input_iconic_param_by_name("Image", h_image)
        call.execute()

        ok = bool(int(self._get(call, "DetectionOK", 0)))
        center_x = float(self._get(call, "CenterColumn", -1.0))
        center_y = float(self._get(call, "CenterRow", -1.0))
        radius = float(self._get(call, "Radius", -1.0))
        score = float(self._get(call, "FinalScore", -1.0))
        fallback = bool(int(self._get(call, "UsedFineFallback", 0)))
        detect_ms = float(self._get(call, "DetectTimeMs", 0.0))
        if ok and (not all(math.isfinite(v) for v in (center_x, center_y, radius, score)) or radius <= 0.0):
            # Never allow malformed algorithm output to become NaN/Infinity in
            # the PLC payload. Treat it as a completed but invalid detection.
            ok = False

        # Negative reference values are an explicit temporary commissioning
        # default. Replace both values in config.toml with standard-part
        # calibration results before production use.
        ref_x = self.reference_x_px if self.reference_x_px >= 0.0 else (width - 1) / 2.0
        ref_y = self.reference_y_px if self.reference_y_px >= 0.0 else (height - 1) / 2.0
        if ok:
            # Phase 1 image coordinates: +X right, +Y down. No machine-axis
            # direction mapping or Adjustment compensation is applied here.
            delta_x = (center_x - ref_x) * self.mm_per_pixel
            delta_y = (center_y - ref_y) * self.mm_per_pixel
            coaxiality = math.hypot(delta_x, delta_y)
        else:
            delta_x = 0.0
            delta_y = 0.0
            coaxiality = 0.0

        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if ok:
            cv2.circle(annotated, (int(round(center_x)), int(round(center_y))), int(round(radius)), (0, 0, 255), 2)
            cv2.circle(annotated, (int(round(center_x)), int(round(center_y))), 3, (255, 0, 0), -1)
            cv2.circle(annotated, (int(round(ref_x)), int(round(ref_y))), 3, (0, 255, 0), -1)
            label = f"dX={delta_x:.4f}mm dY={delta_y:.4f}mm D={coaxiality:.4f}mm"
        else:
            label = "COAX NG: circle not validated"
        cv2.rectangle(annotated, (10, 10), (1450, 85), (0, 0, 0), -1)
        cv2.putText(annotated, label, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (255, 255, 255), 3)

        return CoaxInspectionResult(
            detection_ok=ok,
            coaxiality_mm=coaxiality,
            delta_x_mm=delta_x,
            delta_y_mm=delta_y,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            score=score,
            used_fallback=fallback,
            detect_time_ms=detect_ms,
            annotated_image=annotated,
        )
