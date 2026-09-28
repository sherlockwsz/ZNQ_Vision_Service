from __future__ import annotations

import logging
import queue
import threading
import time
import traceback

from tcp_image_server import LatestImageStore

log = logging.getLogger(__name__)


class PipelineWorker(threading.Thread):
    """Owns one source and schedules formal detection ahead of preview."""

    def __init__(self, name, source, detector, result_queue: queue.Queue,
                 stop_event: threading.Event, *, source_mode: str,
                 preview_fps: float, preview_max_width: int,
                 preview_jpeg_quality: int):
        super().__init__(name=f"{name}-worker", daemon=True)
        self.channel = name
        self.source = source
        self.detector = detector
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.jobs: queue.Queue[int] = queue.Queue(maxsize=1)
        self.source_mode = source_mode
        self.preview_interval_s = 1.0 / max(0.1, float(preview_fps))
        self.preview_max_width = int(preview_max_width)
        self.preview_jpeg_quality = int(preview_jpeg_quality)
        self.image_store: LatestImageStore | None = None
        self.preview_frame_id = 0
        self._last_preview_error = ""

    def configure_image_store(self, image_store: LatestImageStore | None) -> None:
        self.image_store = image_store

    def submit(self, request_id: int) -> bool:
        if not self.is_alive():
            return False
        try:
            self.jobs.put_nowait(int(request_id))
            return True
        except queue.Full:
            return False

    def run(self) -> None:
        next_preview_at = time.monotonic()
        while not self.stop_event.is_set():
            try:
                request_id = self.jobs.get_nowait()
            except queue.Empty:
                request_id = None

            if request_id is not None:
                self._run_detection(request_id)
                next_preview_at = time.monotonic()
                continue

            now = time.monotonic()
            if self.image_store is not None and now >= next_preview_at:
                self._run_preview()
                next_preview_at = max(
                    next_preview_at + self.preview_interval_s,
                    time.monotonic(),
                )
                continue

            wait_s = 0.05
            if self.image_store is not None:
                wait_s = min(wait_s, max(0.001, next_preview_at - now))
            self.stop_event.wait(wait_s)

    def _run_detection(self, request_id: int) -> None:
        started = time.perf_counter()
        try:
            # File-mode formal detection deliberately bypasses preview cache.
            frame = self.source.capture_single()
            result = self.detector.process_frame(frame)
            self.result_queue.put(
                (self.channel, request_id, result, None,
                 time.perf_counter() - started)
            )
        except Exception as exc:
            self.result_queue.put(
                (self.channel, request_id, None,
                 (exc, traceback.format_exc()), time.perf_counter() - started)
            )
        finally:
            self.jobs.task_done()

    def _run_preview(self) -> None:
        if self.image_store is None:
            return
        preview_channel = f"{self.channel}_preview"
        try:
            capture = getattr(
                self.source,
                "capture_preview",
                self.source.capture_single,
            )
            frame = capture()
            self.preview_frame_id += 1
            metadata = {
                "frame_id": self.preview_frame_id,
                "source_mode": self.source_mode,
            }
            source_name = getattr(self.source, "source_name", "")
            if source_name:
                metadata["source_name"] = source_name
            self.image_store.update(
                preview_channel,
                frame,
                metadata,
                jpeg_quality=self.preview_jpeg_quality,
                max_width=self.preview_max_width,
            )
            if self._last_preview_error:
                log.info("%s preview recovered", self.channel.capitalize())
                self._last_preview_error = ""
        except Exception as exc:
            message = str(exc)
            self.image_store.mark_error(
                preview_channel,
                message,
                {
                    "source_mode": self.source_mode,
                    "source_name": getattr(self.source, "source_name", ""),
                },
            )
            if message != self._last_preview_error:
                log.warning(
                    "%s preview unavailable: %s",
                    self.channel.capitalize(),
                    message,
                )
                self._last_preview_error = message
