from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import traceback

try:
    import tomllib
except ImportError:  # Python 3.10 fallback
    import tomli as tomllib

from ads_bridge import AdsConfig, VisionAdsBridge
from camera_manager import CameraConfig, CameraManager
from coax_halcon import CoaxHalconDetector
from image_file_source import ImageFileSource
from pipeline_worker import PipelineWorker as PreviewPipelineWorker
from screw_ai import ScrewSpringDetector
from tcp_image_server import LatestImageStore, TcpImageServer

log = logging.getLogger("vision_service")


class PipelineWorker(threading.Thread):
    def __init__(self, name, source, detector, result_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(name=f"{name}-worker", daemon=True)
        self.channel = name
        self.source = source
        self.detector = detector
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.jobs: queue.Queue[tuple[int, int]] = queue.Queue(maxsize=1)

    def submit(self, request_id: int, ads_session_id: int) -> bool:
        if not self.is_alive():
            return False
        try:
            self.jobs.put_nowait((int(request_id), int(ads_session_id)))
            return True
        except queue.Full:
            return False

    def run(self):
        while not self.stop_event.is_set():
            try:
                request_id, ads_session_id = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                frame = self.source.capture_single()
                result = self.detector.process_frame(frame)
                self.result_queue.put((self.channel, request_id, ads_session_id, result, None, time.perf_counter() - started))
            except Exception as exc:
                self.result_queue.put((self.channel, request_id, ads_session_id, None, (exc, traceback.format_exc()), time.perf_counter() - started))
            finally:
                self.jobs.task_done()


def _camera_cfg(table: dict, name: str) -> CameraConfig:
    return CameraConfig(
        name=name,
        model_name=str(table["model_name"]),
        serial=str(table.get("serial", "")),
        exposure_us=float(table.get("exposure_us", 0.0)),
        gain_db=float(table.get("gain_db", -1.0)),
        timeout_ms=int(table.get("timeout_ms", 3000)),
    )


def _input_mode(cfg: dict, channel: str) -> str:
    mode = str(cfg.get(f"{channel}_input", {}).get("mode", "camera")).strip().lower()
    if mode not in ("camera", "file"):
        raise ValueError(f"{channel}_input.mode must be 'camera' or 'file', got {mode!r}")
    return mode


def _create_source(cfg: dict, base_dir: Path, channel: str, mode: str,
                   manager: CameraManager | None):
    if mode == "camera":
        if manager is None:
            raise RuntimeError(f"CameraManager is unavailable for {channel} camera input")
        return manager.open_camera(_camera_cfg(cfg[f"{channel}_camera"], channel))

    input_cfg = cfg.get(f"{channel}_input", {})
    configured_path = str(input_cfg.get("image_path", "")).strip()
    if not configured_path:
        raise ValueError(f"{channel}_input.image_path is required when mode='file'")
    image_path = Path(configured_path)
    if not image_path.is_absolute():
        image_path = base_dir / image_path
    return ImageFileSource(image_path)


def configure_logging(base_dir: Path, cfg: dict):
    level = getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO)
    log_dir = base_dir / str(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s - %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = RotatingFileHandler(log_dir / "vision_service.log", maxBytes=10_000_000, backupCount=10, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="TwinCAT Python Vision Service")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    exe_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    launch_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Prefer editable config beside the EXE/script; fall back to bundled config.
        p = launch_dir / config_path
        config_path = p if p.exists() else exe_dir / config_path
    cfg = load_config(config_path)
    base_dir = config_path.parent
    configure_logging(base_dir, cfg.get("service", {}))

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    manager = None
    ads = None
    tcp = None
    workers = {}
    results: queue.Queue = queue.Queue()
    latest_submitted = {"screw": None, "coax": None}

    try:
        screw_mode = _input_mode(cfg, "screw")
        coax_mode = _input_mode(cfg, "coax")
        image_cfg = cfg.get("tcp_image", {})
        preview_fps = float(image_cfg.get("preview_fps", 5.0))
        preview_max_width = int(image_cfg.get("preview_max_width", 1280))
        preview_jpeg_quality = int(image_cfg.get("preview_jpeg_quality", 80))
        if "camera" in (screw_mode, coax_mode):
            try:
                manager = CameraManager()
            except Exception:
                # A camera-side initialization fault must not prevent an
                # independently configured file pipeline from running.
                log.exception("Galaxy CameraManager initialization failed")

        # Initialize each pipeline independently so one camera/algorithm can remain
        # online while the other is being serviced. File and camera sources both
        # enter the same worker, detector, ADS and ResultId publication path.
        screw_ready = False
        coax_ready = False
        try:
            screw_source = _create_source(cfg, base_dir, "screw", screw_mode, manager)
            screw_cfg = cfg.get("screw_algorithm", {})
            model_path = base_dir / str(screw_cfg.get("model_path", "best.pt"))
            screw_detector = ScrewSpringDetector(
                model_path,
                mm_per_pixel=float(screw_cfg["mm_per_pixel"]),
                min_screw_distance_mm=float(screw_cfg["min_screw_distance_mm"]),
                max_screw_distance_mm=float(screw_cfg["max_screw_distance_mm"]),
            )
            workers["screw"] = PreviewPipelineWorker(
                "screw", screw_source, screw_detector, results, stop_event,
                source_mode=screw_mode, preview_fps=preview_fps,
                preview_max_width=preview_max_width,
                preview_jpeg_quality=preview_jpeg_quality)
            workers["screw"].start()
            screw_ready = True
            log.info("Screw pipeline ready (input=%s)", screw_mode)
        except Exception:
            log.exception("Screw pipeline initialization failed")

        try:
            coax_source = _create_source(cfg, base_dir, "coax", coax_mode, manager)
            coax_cfg = cfg.get("coax_algorithm", {})
            proc_path = base_dir / str(coax_cfg.get("procedure_path", "detect_coax.hdvp"))
            coax_detector = CoaxHalconDetector(
                proc_path,
                reference_x_px=float(coax_cfg.get("reference_x_px", -1.0)),
                reference_y_px=float(coax_cfg.get("reference_y_px", -1.0)),
                mm_per_pixel=float(coax_cfg.get("mm_per_pixel", 0.0025)),
            )
            workers["coax"] = PreviewPipelineWorker(
                "coax", coax_source, coax_detector, results, stop_event,
                source_mode=coax_mode, preview_fps=preview_fps,
                preview_max_width=preview_max_width,
                preview_jpeg_quality=preview_jpeg_quality)
            workers["coax"].start()
            coax_ready = True
            log.info("Coax pipeline ready (input=%s)", coax_mode)
        except Exception:
            log.exception("Coax pipeline initialization failed")

        if not screw_ready and not coax_ready:
            raise RuntimeError("Neither vision pipeline could initialize")

        # TCP/WPF is retained for future use but is not part of the Phase 1 ADS
        # measurement chain. Port binding failure must not stop PLC measurements.
        store = None
        try:
            store = LatestImageStore(jpeg_quality=int(image_cfg.get("jpeg_quality", 85)))
            tcp = TcpImageServer(str(image_cfg.get("host", "0.0.0.0")), int(image_cfg.get("port", 50010)), store)
            tcp.start()
        except Exception:
            tcp = None
            store = None
            log.exception("Optional TCP image server unavailable; ADS vision remains active")

        for worker in workers.values():
            worker.configure_image_store(store)

        ads_cfg = cfg.get("ads", {})
        ads = VisionAdsBridge(AdsConfig(
            ams_net_id=str(ads_cfg.get("ams_net_id", "")),
            ams_port=int(ads_cfg.get("ams_port", 851)),
            ip_address=str(ads_cfg.get("ip_address", "")),
        ))
        service_cfg = cfg.get("service", {})
        poll_s = max(0.005, float(service_cfg.get("ads_poll_interval_ms", 20)) / 1000.0)
        reconnect_s = max(
            0.1,
            float(service_cfg.get("ads_reconnect_interval_ms", 1000)) / 1000.0,
        )
        disconnect_threshold = max(
            1,
            int(service_cfg.get("ads_disconnect_failure_threshold", 5)),
        )
        ads_connected = False
        ads_session_id = 0
        ads_failure_count = 0
        next_reconnect_at = 0.0
        next_online_refresh_at = 0.0
        log.info("Vision service RUNNING (screw=%s coax=%s)", screw_ready, coax_ready)

        while not stop_event.is_set():
            now = time.monotonic()

            if not ads_connected:
                if now < next_reconnect_at:
                    stop_event.wait(min(poll_s, next_reconnect_at - now))
                    continue
                try:
                    ads.open()
                    ads.validate_connection()
                except Exception as exc:
                    ads.disconnect()
                    next_reconnect_at = time.monotonic() + reconnect_s
                    log.warning("ADS reconnect pending: %s", exc)
                    stop_event.wait(poll_s)
                    continue

                old_session_id = ads_session_id
                ads_session_id += 1
                latest_submitted["screw"] = None
                latest_submitted["coax"] = None
                ads_failure_count = 0
                ads_connected = True
                log.info("ADS communication recovered")
                log.info(
                    "ADS session changed: %d -> %d",
                    old_session_id,
                    ads_session_id,
                )
                try:
                    ads.set_online(screw_ready, coax_ready)
                    next_online_refresh_at = float("inf")
                except Exception as exc:
                    next_online_refresh_at = time.monotonic() + 1.0
                    log.warning("ADS online flag refresh pending: %s", exc)
                # Formal PLC requests are intentionally deferred to the next
                # complete poll after connection validation/session rollover.
                continue

            if now >= next_online_refresh_at:
                try:
                    ads.set_online(screw_ready, coax_ready)
                    next_online_refresh_at = float("inf")
                except Exception as exc:
                    next_online_refresh_at = time.monotonic() + 1.0
                    log.warning("ADS online flag refresh pending: %s", exc)

            round_had_ads_error = False
            last_ads_error = None

            # Poll PLC command structure. A new RequestId schedules exactly one
            # frame from the configured source for the corresponding pipeline.
            if screw_ready:
                try:
                    st = ads.read_screw_request()
                    if st.enabled and st.request and st.request_id != latest_submitted["screw"]:
                        # Claim before prepare/submit. A queue failure consumes this
                        # RequestId and publishes one terminal service-fault result.
                        latest_submitted["screw"] = st.request_id
                        try:
                            ads.prepare_screw_request()
                        except Exception as exc:
                            round_had_ads_error = True
                            last_ads_error = exc
                            log.error(
                                "Screw request transaction failed: request_id=%d "
                                "session_id=%d stage=prepare error=%s",
                                st.request_id,
                                ads_session_id,
                                exc,
                            )
                        else:
                            if workers["screw"].submit(st.request_id, ads_session_id):
                                log.info("Screw request accepted: %d", st.request_id)
                            else:
                                log.error(
                                    "Screw request %d failed: worker queue submit fault",
                                    st.request_id,
                                )
                                try:
                                    ads.publish_screw(
                                        st.request_id,
                                        invalid=True,
                                        service_fault=True,
                                    )
                                except Exception as exc:
                                    round_had_ads_error = True
                                    last_ads_error = exc
                                    log.error(
                                        "Screw request transaction failed: request_id=%d "
                                        "session_id=%d stage=publish error=%s",
                                        st.request_id,
                                        ads_session_id,
                                        exc,
                                    )
                except Exception as exc:
                    round_had_ads_error = True
                    last_ads_error = exc
                    log.warning("ADS Screw poll failed: %s", exc)

            if coax_ready:
                try:
                    st = ads.read_coax_request()
                    if st.enabled and st.request and st.request_id != latest_submitted["coax"]:
                        # Claim the ID before any queue/publish side effect. Even a
                        # queue failure therefore produces exactly one terminal
                        # result and can never re-enqueue the same RequestId.
                        latest_submitted["coax"] = st.request_id
                        try:
                            ads.prepare_coax_request()
                        except Exception as exc:
                            round_had_ads_error = True
                            last_ads_error = exc
                            log.error(
                                "Coax request transaction failed: request_id=%d "
                                "session_id=%d stage=prepare error=%s",
                                st.request_id,
                                ads_session_id,
                                exc,
                            )
                        else:
                            if workers["coax"].submit(st.request_id, ads_session_id):
                                log.info("Coax request accepted: %d", st.request_id)
                            else:
                                log.error(
                                    "Coax request %d failed: worker queue submit fault",
                                    st.request_id,
                                )
                                try:
                                    ads.publish_coax(
                                        st.request_id,
                                        invalid=True,
                                        service_fault=True,
                                    )
                                except Exception as exc:
                                    round_had_ads_error = True
                                    last_ads_error = exc
                                    log.error(
                                        "Coax request transaction failed: request_id=%d "
                                        "session_id=%d stage=publish error=%s",
                                        st.request_id,
                                        ads_session_id,
                                        exc,
                                    )
                except Exception as exc:
                    round_had_ads_error = True
                    last_ads_error = exc
                    log.warning("ADS Coax poll failed: %s", exc)

            # Once the current round reaches the configured consecutive-failure
            # threshold, do not attempt to publish any queued result through the
            # connection that is now classified as lost.
            if round_had_ads_error and ads_failure_count + 1 >= disconnect_threshold:
                ads_failure_count += 1
                log.error(
                    "ADS communication lost after %d consecutive failed rounds: %s",
                    ads_failure_count,
                    last_ads_error,
                )
                ads.disconnect()
                ads_connected = False
                next_reconnect_at = time.monotonic() + reconnect_s
                continue

            # Publish completed worker results from this single ADS-owning thread.
            while True:
                try:
                    channel, request_id, result_session_id, result, err, wall_s = results.get_nowait()
                except queue.Empty:
                    break
                try:
                    if result_session_id != ads_session_id:
                        log.warning(
                            "Discard stale %s result: request_id=%d "
                            "result_session=%d current_session=%d",
                            channel.capitalize(),
                            request_id,
                            result_session_id,
                            ads_session_id,
                        )
                        continue

                    if err is not None:
                        exc, tb = err
                        log.error("%s request %d failed after %.3fs: %s\n%s", channel, request_id, wall_s, exc, tb)
                        try:
                            if channel == "screw":
                                ads.publish_screw(request_id, invalid=True, service_fault=True)
                            else:
                                ads.publish_coax(request_id, invalid=True, service_fault=True)
                        except Exception as publish_exc:
                            round_had_ads_error = True
                            last_ads_error = publish_exc
                            log.error(
                                "%s result transaction failed: request_id=%d "
                                "session_id=%d stage=publish error=%s",
                                channel.capitalize(),
                                request_id,
                                ads_session_id,
                                publish_exc,
                            )
                        continue

                    if channel == "screw":
                        # Convert the measured undirected line angle into the PLC
                        # correction command using only configured machine mapping.
                        screw_mapping = cfg.get("screw_mapping", {})
                        correction_sign = float(screw_mapping["correction_sign"])
                        correction_offset_deg = float(screw_mapping["correction_offset_deg"])
                        if result.detected:
                            detected_angle = result.detected_angle

                            # Minimum signed rotation required to make the
                            # undirected screw line horizontal (0° / 180°).
                            #
                            # Assumption:
                            # DamperRotation positive direction increases
                            # the detected image angle.
                            correction_angle = (
                                (90.0 - detected_angle) % 180.0
                            ) - 90.0

                            plc_correction_angle = (
                                correction_angle * correction_sign
                                + correction_offset_deg
                            )
                        else:
                            plc_correction_angle = 0.0  
                                                  
                        try:
                            ads.publish_screw(
                                request_id,
                                detected_angle=result.detected_angle,
                                correction_angle=plc_correction_angle,
                                detected=result.detected,
                                confidence=result.confidence,
                                invalid=not result.detected,
                                service_fault=False,
                            )
                        except Exception as exc:
                            round_had_ads_error = True
                            last_ads_error = exc
                            log.error(
                                "Screw result transaction failed: request_id=%d "
                                "session_id=%d stage=publish error=%s",
                                request_id,
                                ads_session_id,
                                exc,
                            )
                            continue
                        if store is not None:
                            try:
                                store.update("screw", result.annotated_image, {
                                    "request_id": request_id,
                                    "source_mode": screw_mode,
                                    "source_name": getattr(screw_source, "source_name", ""),
                                    "detected": result.detected,
                                    "detected_angle": result.detected_angle,
                                    "correction_angle": plc_correction_angle,
                                    "confidence": result.confidence,
                                    "invalid_reason": result.details.get("invalid_reason", ""),
                                })
                            except Exception:
                                log.exception("Optional Screw JPEG update failed; ADS result was already committed")
                        log.info("Screw result %d: detected=%s detectedAngle=%.3f plcCorrection=%.3f confidence=%.3f reason=%s wall=%.3fs",
                                 request_id, result.detected, result.detected_angle, plc_correction_angle,
                                 result.confidence, result.details.get("invalid_reason", ""), wall_s)
                    else:
                        try:
                            ads.publish_coax(
                                request_id,
                                coaxiality=result.coaxiality_mm,
                                delta_x=result.delta_x_mm,
                                delta_y=result.delta_y_mm,
                                invalid=not result.detection_ok,
                                service_fault=False,
                            )
                        except Exception as exc:
                            round_had_ads_error = True
                            last_ads_error = exc
                            log.error(
                                "Coax result transaction failed: request_id=%d "
                                "session_id=%d stage=publish error=%s",
                                request_id,
                                ads_session_id,
                                exc,
                            )
                            continue
                        if store is not None:
                            try:
                                store.update("coax", result.annotated_image, {
                                    "request_id": request_id,
                                    "source_mode": coax_mode,
                                    "source_name": getattr(coax_source, "source_name", ""),
                                    "detection_ok": result.detection_ok,
                                    "coaxiality_mm": result.coaxiality_mm,
                                    "delta_x_mm": result.delta_x_mm,
                                    "delta_y_mm": result.delta_y_mm,
                                    "score": result.score,
                                    "used_fallback": result.used_fallback,
                                    "detect_time_ms": result.detect_time_ms,
                                })
                            except Exception:
                                log.exception("Optional Coax JPEG update failed; ADS result was already committed")
                        log.info("Coax result %d: ok=%s D=%.4f dX=%.4f dY=%.4f detect=%.1fms wall=%.3fs",
                                 request_id, result.detection_ok, result.coaxiality_mm,
                                 result.delta_x_mm, result.delta_y_mm, result.detect_time_ms, wall_s)
                finally:
                    results.task_done()

            if round_had_ads_error:
                ads_failure_count += 1
                if ads_failure_count >= disconnect_threshold:
                    log.error(
                        "ADS communication lost after %d consecutive failed rounds: %s",
                        ads_failure_count,
                        last_ads_error,
                    )
                    ads.disconnect()
                    ads_connected = False
                    next_reconnect_at = time.monotonic() + reconnect_s
            else:
                # Only a complete round without any ADS communication error
                # clears the consecutive-failure count.
                ads_failure_count = 0

            stop_event.wait(poll_s)

    except Exception:
        log.exception("Vision service terminated by fatal error")
        return 1
    finally:
        stop_event.set()
        if tcp is not None:
            try: tcp.close()
            except Exception: log.exception("TCP shutdown failed")
        # Camera handles are never closed while a worker can still access them.
        for worker in workers.values():
            worker.join()
        if manager is not None:
            try: manager.close()
            except Exception: log.exception("Camera shutdown failed")
        if ads is not None:
            try: ads.close()
            except Exception: log.exception("ADS shutdown failed")
        log.info("Vision service stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
