from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np


class ImageFileSource:
    """Reads one fresh Mono8 frame from a configured image file per request."""

    def __init__(self, image_path: str | Path):
        self.image_path = Path(image_path)
        self._preview_lock = threading.Lock()
        self._preview_mtime_ns: int | None = None
        self._preview_failed_mtime_ns: int | None = None
        self._preview_image: np.ndarray | None = None

    def capture_single(self) -> np.ndarray:
        """Always decode a fresh frame for a formal PLC transaction."""
        return self._read_fresh()

    @property
    def source_name(self) -> str:
        return self.image_path.name

    def capture_preview(self) -> np.ndarray:
        """Reuse a decoded preview until the configured file changes."""
        try:
            mtime_ns = self.image_path.stat().st_mtime_ns
        except OSError as exc:
            raise RuntimeError(f"Unable to stat image file: {self.image_path}") from exc

        with self._preview_lock:
            if self._preview_image is not None and self._preview_mtime_ns == mtime_ns:
                return self._preview_image

            if self._preview_failed_mtime_ns == mtime_ns:
                raise RuntimeError(
                    f"Image file remains unreadable: {self.image_path}"
                )

            try:
                image = self._read_fresh()
            except Exception:
                self._preview_failed_mtime_ns = mtime_ns
                raise
            self._preview_image = image
            self._preview_mtime_ns = mtime_ns
            self._preview_failed_mtime_ns = None
            return image

    def _read_fresh(self) -> np.ndarray:
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
