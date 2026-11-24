"""Lightweight live server for Clay run archives.

Streams runtime events (``events.jsonl``) to a browser frontend over
Server-Sent Events (SSE) and serves the static schematic viewer. Stdlib
only (``http.server``, ``threading``, ``pathlib``, ``json``, ``time``) so
it can run anywhere Clay runs.

Endpoints
---------
``GET /``
    Static directory (default index = ``living_console_preview.html``).
``GET /runs/<run_id>/run.manifest.json``
    The manifest for a recorded run.
``GET /runs/<run_id>/events.jsonl``
    The full events file (offline replay).
``GET /runs/<run_id>/events/stream``
    SSE stream. Emits ``event: clay_event\\ndata: <jsonl-line>\\n\\n`` for
    every new line appended to ``events.jsonl`` after the connection
    opens. Honours ``?since=<n>`` to first replay all existing lines from
    line index ``n`` (0-based) before tailing. Sends an
    ``event: heartbeat\\ndata: {}\\n\\n`` every ``heartbeat_interval``
    seconds so proxies do not idle-close the connection.
``GET /healthz``
    Returns ``200`` with ``{"status":"ok"}``.

Programmatic entry: :func:`run_server`.

CLI::

    python -m clay.runtime.live_server \
        --runs-dir tools/schematic_viewer/data/runs \
        --static-dir tools/schematic_viewer \
        --port 8765
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse


_DEFAULT_INDEX = "living_console_preview.html"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _safe_run_id(run_id: str) -> bool:
    return bool(_RUN_ID_RE.match(run_id)) and ".." not in run_id


def _resolve_within(root: Path, *parts: str) -> Optional[Path]:
    """Join ``parts`` under ``root`` and refuse escapes."""
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


class LiveServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with per-instance config bag."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        runs_dir: Path,
        static_dir: Path,
        *,
        heartbeat_interval: float = 15.0,
        tail_poll_interval: float = 0.05,
    ) -> None:
        self.runs_dir = Path(runs_dir).resolve()
        self.static_dir = Path(static_dir).resolve()
        self.heartbeat_interval = float(heartbeat_interval)
        self.tail_poll_interval = float(tail_poll_interval)
        super().__init__(server_address, LiveRequestHandler)


class LiveRequestHandler(BaseHTTPRequestHandler):
    """Routes static + manifest + SSE traffic."""

    server_version = "ClayLiveServer/1.0"

    # Quieter than the default stderr-per-request logger.
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    # ------------------------------------------------------------------ helpers
    @property
    def runs_dir(self) -> Path:
        return self.server.runs_dir  # type: ignore[attr-defined]

    @property
    def static_dir(self) -> Path:
        return self.server.static_dir  # type: ignore[attr-defined]

    @property
    def heartbeat_interval(self) -> float:
        return self.server.heartbeat_interval  # type: ignore[attr-defined]

    @property
    def tail_poll_interval(self) -> float:
        return self.server.tail_poll_interval  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict) -> None:
        body = (json.dumps(payload) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_text(self, status: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, path: Path, content_type: Optional[str] = None) -> None:
        if not path.is_file():
            self._send_text(HTTPStatus.NOT_FOUND, "not found\n")
            return
        if content_type is None:
            guess, _ = mimetypes.guess_type(path.name)
            content_type = guess or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ------------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler convention)
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        # /runs/<run_id>/...
        if path.startswith("/runs/"):
            rest = path[len("/runs/"):]
            parts = rest.split("/", 1)
            run_id = unquote(parts[0])
            tail = parts[1] if len(parts) > 1 else ""

            if not _safe_run_id(run_id):
                self._send_text(HTTPStatus.BAD_REQUEST, "bad run id\n")
                return

            if tail == "run.manifest.json":
                target = _resolve_within(self.runs_dir, run_id, "run.manifest.json")
                if target is None:
                    self._send_text(HTTPStatus.BAD_REQUEST, "bad path\n")
                    return
                self._send_file(target, "application/json; charset=utf-8")
                return

            if tail == "events.jsonl":
                target = _resolve_within(self.runs_dir, run_id, "events.jsonl")
                if target is None:
                    self._send_text(HTTPStatus.BAD_REQUEST, "bad path\n")
                    return
                self._send_file(target, "application/x-ndjson; charset=utf-8")
                return

            if tail == "events/stream":
                target = _resolve_within(self.runs_dir, run_id, "events.jsonl")
                if target is None:
                    self._send_text(HTTPStatus.BAD_REQUEST, "bad path\n")
                    return
                qs = parse_qs(parsed.query)
                since_raw = qs.get("since", ["-1"])[0]
                try:
                    since = int(since_raw)
                except ValueError:
                    since = -1
                self._stream_events(target, since=since)
                return

            # Allow direct artifact access under runs/<id>/...
            if tail:
                target = _resolve_within(self.runs_dir, run_id, *tail.split("/"))
                if target is None:
                    self._send_text(HTTPStatus.BAD_REQUEST, "bad path\n")
                    return
                self._send_file(target)
                return

            self._send_text(HTTPStatus.NOT_FOUND, "not found\n")
            return

        # Static directory (default index)
        rel = path.lstrip("/")
        if rel in ("", "/"):
            rel = _DEFAULT_INDEX
        target = _resolve_within(self.static_dir, *rel.split("/")) if rel else None
        if target is None:
            self._send_text(HTTPStatus.BAD_REQUEST, "bad path\n")
            return
        if target.is_dir():
            target = target / _DEFAULT_INDEX
        self._send_file(target)

    # ------------------------------------------------------------------- SSE
    def _stream_events(self, events_path: Path, *, since: int) -> None:
        """Stream events.jsonl over SSE.

        Behaviour:
            ``since == 0``: replay every existing line, then tail.
            ``since == n``: replay lines from index ``n`` (0-based), then tail.
            ``since < 0``  (default): no replay, only tail new lines.
        """
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # SSE retry hint + initial comment to flush headers through proxies.
        if not self._sse_write(b": clay-live-server\nretry: 2000\n\n"):
            return

        # Read current content. We tail by line index (count newlines).
        try:
            existing = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
        except OSError:
            existing = ""
        existing_lines = existing.splitlines()
        total_lines = len(existing_lines)

        if since >= 0:
            start = min(since, total_lines)
            for line in existing_lines[start:]:
                if line.strip():
                    if not self._sse_event("clay_event", line):
                        return
            cursor = total_lines
        else:
            cursor = total_lines

        # Tail loop.
        last_heartbeat = time.monotonic()
        last_size = events_path.stat().st_size if events_path.exists() else 0

        while True:
            try:
                if events_path.exists():
                    cur_size = events_path.stat().st_size
                    if cur_size < last_size:
                        # File was truncated/rotated; restart.
                        cursor = 0
                        last_size = 0
                    if cur_size > last_size:
                        text = events_path.read_text(encoding="utf-8")
                        lines = text.splitlines()
                        if len(lines) > cursor:
                            for line in lines[cursor:]:
                                if not line.strip():
                                    continue
                                if not self._sse_event("clay_event", line):
                                    return
                            cursor = len(lines)
                        last_size = cur_size
                # heartbeat
                now = time.monotonic()
                if now - last_heartbeat >= self.heartbeat_interval:
                    if not self._sse_event("heartbeat", "{}"):
                        return
                    last_heartbeat = now
                time.sleep(self.tail_poll_interval)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                # Never crash the handler thread on a transient IO hiccup.
                time.sleep(self.tail_poll_interval)

    def _sse_event(self, event: str, data: str) -> bool:
        # Each "data:" line is one logical event chunk; the JSONL line is
        # already a single line so a single "data:" frame is correct.
        frame = f"event: {event}\ndata: {data}\n\n".encode("utf-8")
        return self._sse_write(frame)

    def _sse_write(self, frame: bytes) -> bool:
        try:
            self.wfile.write(frame)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False


# ============================================================== programmatic
def run_server(
    runs_dir: str | Path,
    static_dir: str | Path,
    *,
    port: int = 8765,
    bind: str = "127.0.0.1",
    heartbeat_interval: float = 15.0,
    tail_poll_interval: float = 0.05,
) -> LiveServer:
    """Construct and start a :class:`LiveServer` and return it.

    The server is started in a background daemon thread so the caller
    can run code and later call :meth:`LiveServer.shutdown` to stop it.
    """
    server = LiveServer(
        (bind, int(port)),
        Path(runs_dir),
        Path(static_dir),
        heartbeat_interval=heartbeat_interval,
        tail_poll_interval=tail_poll_interval,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="clay-live-server")
    thread.start()
    server._thread = thread  # type: ignore[attr-defined]
    return server


# ===================================================================== CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clay.runtime.live_server",
        description="Live SSE bridge from Clay run archives to the schematic viewer frontend.",
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory containing run archives (e.g. tools/schematic_viewer/data/runs).",
    )
    parser.add_argument(
        "--static-dir",
        required=True,
        help="Directory served at /. Defaults its index to living_console_preview.html.",
    )
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default 8765).")
    parser.add_argument("--bind", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=15.0,
        help="Seconds between SSE heartbeats (default 15).",
    )
    parser.add_argument(
        "--tail-poll-interval",
        type=float,
        default=0.05,
        help="Seconds between tail polls (default 0.05).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    server = LiveServer(
        (args.bind, args.port),
        Path(args.runs_dir),
        Path(args.static_dir),
        heartbeat_interval=args.heartbeat_interval,
        tail_poll_interval=args.tail_poll_interval,
    )
    host, port = server.server_address[:2]
    print(f"clay live server: http://{host}:{port}/ "
          f"(runs={server.runs_dir}, static={server.static_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("clay live server: shutting down")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
