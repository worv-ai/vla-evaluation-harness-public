"""Tiny stdlib HTTP server for ``/health``.

A separate aiohttp would be overkill: the daemon's main async loop is busy
with WS traffic and disk writes; the health endpoint just needs to answer
GET /health with a 200 + a few counters from any thread.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vla_eval.recording_daemon.daemon import RecordingDaemon

logger = logging.getLogger(__name__)


class HealthServer:
    """Thread-hosted /health endpoint for the daemon."""

    def __init__(self, daemon: RecordingDaemon, port: int) -> None:
        self._daemon = daemon
        self._port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._port <= 0:
            return  # disabled

        daemon = self._daemon

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return  # silence noisy access log

            def do_GET(self) -> None:  # noqa: N802 (http.server contract)
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(
                    {
                        "buckets": daemon.bucket_count,
                        "evals": daemon.eval_count,
                        "uptime_s": round(daemon.uptime_sec, 2),
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        try:
            self._httpd = ThreadingHTTPServer(("0.0.0.0", self._port), Handler)
        except OSError as exc:
            logger.warning("Health endpoint bind failed on port %d (%s); disabled.", self._port, exc)
            self._httpd = None
            return
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="recording-health", daemon=True)
        self._thread.start()
        logger.info("Health endpoint at http://0.0.0.0:%d/health", self._port)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
