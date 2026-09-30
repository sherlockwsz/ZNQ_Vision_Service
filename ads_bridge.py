from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pyads

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdsConfig:
    ams_net_id: str
    ams_port: int = 851
    ip_address: str = ""


@dataclass(frozen=True)
class RequestState:
    enabled: bool
    request: bool
    request_id: int


class VisionAdsBridge:
    SCREW = "GVL_ScrewVision.stInterface"
    COAX = "GVL_CoaxVision.stInterface"

    def __init__(self, cfg: AdsConfig):
        self.cfg = cfg
        self.conn: Optional[pyads.Connection] = None

    def open(self) -> None:
        # A reconnect always starts with a new pyads.Connection. Never reuse an
        # object that has already been classified as disconnected.
        self.disconnect()
        target = self.cfg.ams_net_id.strip()
        if not target:
            try:
                target = str(pyads.get_local_address().netid)
            except Exception as exc:
                raise RuntimeError(
                    "ads.ams_net_id is empty and local AMS Net ID could not be detected; set it in config.toml"
                ) from exc
        kwargs = {}
        if self.cfg.ip_address.strip():
            kwargs["ip_address"] = self.cfg.ip_address.strip()
        conn = pyads.Connection(target, int(self.cfg.ams_port), **kwargs)
        try:
            conn.open()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise
        self.conn = conn
        log.info("ADS transport opened to %s:%s", target, self.cfg.ams_port)

    def validate_connection(self) -> None:
        """Confirm that at least one known PLC vision symbol is readable."""
        errors = []
        for name in (
            f"{self.SCREW}.bUseScrewVision",
            f"{self.COAX}.bUseCoaxVision",
        ):
            try:
                self._read(name, pyads.PLCTYPE_BOOL)
                return
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError(
            "ADS opened but no known PLC vision symbol is readable ("
            + "; ".join(errors)
            + ")"
        )

    def _read(self, name, plc_type):
        if self.conn is None:
            raise RuntimeError("ADS not connected")
        return self.conn.read_by_name(name, plc_type)

    def _write(self, name, value, plc_type):
        if self.conn is None:
            raise RuntimeError("ADS not connected")
        self.conn.write_by_name(name, value, plc_type)

    def read_screw_request(self) -> RequestState:
        p = self.SCREW
        return RequestState(
            enabled=bool(self._read(f"{p}.bUseScrewVision", pyads.PLCTYPE_BOOL)),
            request=bool(self._read(f"{p}.bMeasureRequest", pyads.PLCTYPE_BOOL)),
            request_id=int(self._read(f"{p}.udiRequestId", pyads.PLCTYPE_UDINT)),
        )

    def read_coax_request(self) -> RequestState:
        p = self.COAX
        return RequestState(
            enabled=bool(self._read(f"{p}.bUseCoaxVision", pyads.PLCTYPE_BOOL)),
            request=bool(self._read(f"{p}.bMeasureRequest", pyads.PLCTYPE_BOOL)),
            request_id=int(self._read(f"{p}.udiRequestId", pyads.PLCTYPE_UDINT)),
        )

    def set_online(self, screw: bool, coax: bool) -> None:
        self._write(f"{self.SCREW}.bOnline", bool(screw), pyads.PLCTYPE_BOOL)
        self._write(f"{self.COAX}.bOnline", bool(coax), pyads.PLCTYPE_BOOL)

    def prepare_screw_request(self) -> None:
        p = self.SCREW
        self._write(f"{p}.bResultValid", False, pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bResultInvalid", False, pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bServiceFault", False, pyads.PLCTYPE_BOOL)

    def prepare_coax_request(self) -> None:
        p = self.COAX
        self._write(f"{p}.bResultValid", False, pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bResultInvalid", False, pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bServiceFault", False, pyads.PLCTYPE_BOOL)

    def publish_screw(self, request_id: int, *, detected_angle: float = 0.0,
                      correction_angle: float = 0.0, detected: bool = False,
                      confidence: float = 0.0, invalid: bool = False,
                      service_fault: bool = False) -> None:
        p = self.SCREW
        # Payload first; ResultValid last is the transaction commit flag.
        self._write(f"{p}.udiResultId", int(request_id), pyads.PLCTYPE_UDINT)
        self._write(f"{p}.fDetectedAngle", float(detected_angle), pyads.PLCTYPE_LREAL)
        self._write(f"{p}.fCorrectionAngle", float(correction_angle), pyads.PLCTYPE_LREAL)
        self._write(f"{p}.bDetected", bool(detected), pyads.PLCTYPE_BOOL)
        self._write(f"{p}.fConfidence", float(confidence), pyads.PLCTYPE_LREAL)
        self._write(f"{p}.bResultInvalid", bool(invalid), pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bServiceFault", bool(service_fault), pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bResultValid", True, pyads.PLCTYPE_BOOL)

    def publish_coax(self, request_id: int, *, coaxiality: float = 0.0,
                     delta_x: float = 0.0, delta_y: float = 0.0,
                     invalid: bool = False, service_fault: bool = False) -> None:
        p = self.COAX
        self._write(f"{p}.udiResultId", int(request_id), pyads.PLCTYPE_UDINT)
        self._write(f"{p}.fCoaxiality", float(coaxiality), pyads.PLCTYPE_LREAL)
        self._write(f"{p}.fDeltaX", float(delta_x), pyads.PLCTYPE_LREAL)
        self._write(f"{p}.fDeltaY", float(delta_y), pyads.PLCTYPE_LREAL)
        self._write(f"{p}.bResultInvalid", bool(invalid), pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bServiceFault", bool(service_fault), pyads.PLCTYPE_BOOL)
        self._write(f"{p}.bResultValid", True, pyads.PLCTYPE_BOOL)

    def close(self) -> None:
        if self.conn is None:
            return
        try:
            self.set_online(False, False)
        except Exception:
            log.exception("Failed to clear Vision online flags during shutdown")
        self.disconnect()

    def disconnect(self) -> None:
        """Close the current transport without performing any ADS writes."""
        conn = self.conn
        self.conn = None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            log.warning("Failed to close stale ADS connection", exc_info=True)
