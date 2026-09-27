from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    name: str
    model_name: str
    serial: str = ""
    exposure_us: float = 0.0
    gain_db: float = -1.0
    timeout_ms: int = 3000


class DahengCamera:
    def __init__(self, gx_module, device_manager, info: Dict[str, Any], cfg: CameraConfig):
        self.gx = gx_module
        self.device_manager = device_manager
        self.info = info
        self.cfg = cfg
        self.cam = None
        self._remote = None
        self._streaming = False

    @property
    def serial(self) -> str:
        return str(self.info.get("sn", ""))

    def open(self) -> None:
        self.cam = self.device_manager.open_device_by_sn(self.serial)
        try:
            self._remote = self.cam.get_remote_device_feature_control()
        except Exception:
            self._remote = None

        self._set_enum("PixelFormat", "Mono8", legacy_value=getattr(self.gx.GxPixelFormatEntry, "MONO8", None))
        self._set_enum("TriggerMode", "On", legacy_value=self.gx.GxSwitchEntry.ON)
        self._set_enum("TriggerSource", "Software", legacy_value=self.gx.GxTriggerSourceEntry.SOFTWARE)

        if self.cfg.exposure_us > 0:
            self._set_float("ExposureTime", self.cfg.exposure_us)
        if self.cfg.gain_db >= 0:
            self._set_float("Gain", self.cfg.gain_db)

        self.cam.stream_on()
        self._streaming = True
        log.info("Camera %s opened: model=%s sn=%s ip=%s", self.cfg.name,
                 self.info.get("model_name"), self.serial, self.info.get("ip", ""))

    def _set_enum(self, name: str, string_value: str, legacy_value=None) -> None:
        # Newer Galaxy Python API
        if self._remote is not None:
            try:
                self._remote.get_enum_feature(name).set(string_value)
                return
            except Exception as exc:
                log.debug("Remote enum feature %s=%s failed: %s", name, string_value, exc)
        # Legacy gxipy API
        feature = getattr(self.cam, name, None)
        if feature is not None and legacy_value is not None:
            try:
                feature.set(legacy_value)
                return
            except Exception as exc:
                log.warning("Legacy enum feature %s failed: %s", name, exc)
        if name in ("TriggerMode", "TriggerSource"):
            raise RuntimeError(f"Camera does not allow required feature {name}={string_value}")
        log.warning("Optional camera feature not applied: %s=%s", name, string_value)

    def _set_float(self, name: str, value: float) -> None:
        if self._remote is not None:
            try:
                self._remote.get_float_feature(name).set(float(value))
                return
            except Exception as exc:
                log.debug("Remote float feature %s=%s failed: %s", name, value, exc)
        feature = getattr(self.cam, name, None)
        if feature is not None:
            try:
                feature.set(float(value))
                return
            except Exception as exc:
                log.warning("Camera float feature %s failed: %s", name, exc)
        log.warning("Optional camera float feature not applied: %s=%s", name, value)

    def capture_single(self) -> np.ndarray:
        if self.cam is None or not self._streaming:
            raise RuntimeError(f"Camera {self.cfg.name} is not open")

        # Software trigger. The camera remains streaming, but exactly one frame is
        # requested per PLC transaction.
        triggered = False
        if self._remote is not None:
            try:
                self._remote.get_command_feature("TriggerSoftware").send_command()
                triggered = True
            except Exception as exc:
                log.debug("Remote software trigger failed: %s", exc)
        if not triggered:
            self.cam.TriggerSoftware.send_command()

        try:
            raw = self.cam.data_stream[0].get_image(timeout=self.cfg.timeout_ms)
        except TypeError:
            raw = self.cam.data_stream[0].get_image()
        if raw is None:
            raise TimeoutError(f"No frame from {self.cfg.name} within {self.cfg.timeout_ms} ms")

        # Reject incomplete frames when the SDK exposes frame status.
        try:
            status = raw.get_status()
            success = getattr(self.gx.GxFrameStatusList, "SUCCESS", 0)
            if status != success:
                raise RuntimeError(f"Incomplete frame from {self.cfg.name}, status={status}")
        except AttributeError:
            pass

        arr = raw.get_numpy_array()
        if arr is None:
            raise RuntimeError(f"Galaxy SDK returned no numpy data for {self.cfg.name}")
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            raise RuntimeError(
                f"{self.cfg.name} delivered {arr.dtype}; set PixelFormat=Mono8 in Galaxy Viewer/SDK "
                "to preserve the supplied algorithm's 8-bit input semantics"
            )
        if arr.ndim != 2:
            raise RuntimeError(f"{self.cfg.name} expected monochrome frame, got shape={arr.shape}")
        return np.ascontiguousarray(arr.copy())

    def close(self) -> None:
        if self.cam is None:
            return
        try:
            if self._streaming:
                self.cam.stream_off()
        except Exception:
            log.exception("stream_off failed for %s", self.cfg.name)
        try:
            self.cam.close_device()
        except Exception:
            log.exception("close_device failed for %s", self.cfg.name)
        self._streaming = False
        self.cam = None


class CameraManager:
    """Owns Daheng SDK device discovery and all camera objects.

    Algorithm modules never open/close/configure cameras directly.
    """

    def __init__(self):
        try:
            import gxipy as gx
        except ImportError as exc:
            raise RuntimeError(
                "gxipy is not installed. Install Daheng Galaxy SDK and its Python package first."
            ) from exc
        self.gx = gx
        self.manager = gx.DeviceManager()
        self._infos = self._enumerate()
        self._opened: list[DahengCamera] = []

    def _enumerate(self):
        try:
            num, infos = self.manager.update_all_device_list(timeout=1500)
        except TypeError:
            try:
                num, infos = self.manager.update_all_device_list()
            except Exception:
                num, infos = self.manager.update_device_list()
        if num <= 0:
            raise RuntimeError("No Daheng cameras were enumerated by Galaxy SDK")
        log.info("Galaxy SDK enumerated %d camera(s): %s", num,
                 [{k: x.get(k) for k in ("model_name", "sn", "ip")} for x in infos])
        return infos

    def open_camera(self, cfg: CameraConfig) -> DahengCamera:
        candidates = self._infos
        if cfg.serial.strip():
            candidates = [x for x in candidates if str(x.get("sn", "")) == cfg.serial.strip()]
            criterion = f"SN={cfg.serial.strip()}"
        else:
            candidates = [x for x in candidates if str(x.get("model_name", "")).strip() == cfg.model_name.strip()]
            criterion = f"model={cfg.model_name}"

        if not candidates:
            raise RuntimeError(f"Camera {cfg.name}: no device matched {criterion}")
        if len(candidates) > 1:
            raise RuntimeError(
                f"Camera {cfg.name}: {len(candidates)} devices matched {criterion}; configure serial= explicitly"
            )
        info = candidates[0]
        if any(c.serial == str(info.get("sn", "")) for c in self._opened):
            raise RuntimeError(f"Camera {cfg.name}: selected camera is already assigned to another task")

        camera = DahengCamera(self.gx, self.manager, info, cfg)
        camera.open()
        self._opened.append(camera)
        return camera

    def close(self) -> None:
        for cam in reversed(self._opened):
            cam.close()
        self._opened.clear()
        try:
            if hasattr(self.manager, "del_manager"):
                self.manager.del_manager()
        except Exception:
            log.exception("Galaxy DeviceManager cleanup failed")
