from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageFileSource:
    """Reads one fresh Mono8 frame from a configured image file per request."""

    def __init__(self, image_path: str | Path):
        self.image_path = Path(image_path)

    def capture_single(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Unable to read image file: {self.image_path}")
        if image.dtype != np.uint8:
            raise RuntimeError(
                f"Image file must decode as uint8 Mono8, got dtype={image.dtype}: {self.image_path}"
            )
        if image.ndim != 2:
            raise RuntimeError(
                f"Image file must decode as a single-channel frame, got shape={image.shape}: {self.image_path}"
            )
        return np.ascontiguousarray(image)
