from __future__ import annotations

import json
import logging
import socketserver
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ImagePacket:
    metadata: dict
    jpeg: bytes


class LatestImageStore:
    def __init__(self, jpeg_quality: int = 85):
        self.jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._packets: Dict[str, ImagePacket] = {}

    def update(self, channel: str, image: np.ndarray, metadata: dict) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        meta = dict(metadata)
        meta.update({"channel": channel, "timestamp_unix": time.time()})
        packet = ImagePacket(meta, encoded.tobytes())
        with self._lock:
            self._packets[channel] = packet

    def get(self, channel: str) -> Optional[ImagePacket]:
        with self._lock:
            return self._packets.get(channel)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        store: LatestImageStore = self.server.store  # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline(128)
            if not line:
                return
            cmd = line.decode("utf-8", errors="replace").strip().split()
            if not cmd:
                continue
            if cmd[0].upper() == "PING":
                self._send({"ok": True, "type": "pong"}, b"")
                continue
            if len(cmd) != 2 or cmd[0].upper() != "GET" or cmd[1].lower() not in ("screw", "coax"):
                self._send({"ok": False, "error": "use: GET screw | GET coax | PING"}, b"")
                continue
            packet = store.get(cmd[1].lower())
            if packet is None:
                self._send({"ok": False, "error": "no image available yet", "channel": cmd[1].lower()}, b"")
            else:
                meta = dict(packet.metadata)
                meta["ok"] = True
                self._send(meta, packet.jpeg)

    def _send(self, meta: dict, jpeg: bytes):
        header = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.wfile.write(struct.pack(">I", len(header)))
        self.wfile.write(header)
        self.wfile.write(struct.pack(">I", len(jpeg)))
        if jpeg:
            self.wfile.write(jpeg)
        self.wfile.flush()


class _ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TcpImageServer:
    def __init__(self, host: str, port: int, store: LatestImageStore):
        self.host = host
        self.port = int(port)
        self.store = store
        self._server = _ReusableThreadingTCPServer((host, self.port), _Handler)
        self._server.store = store  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, name="tcp-image-server", daemon=True)

    def start(self):
        self._thread.start()
        log.info("TCP image server listening on %s:%d", self.host, self.port)

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
