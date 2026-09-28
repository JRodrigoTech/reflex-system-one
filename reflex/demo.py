"""Line-oriented direct transport. This module never imports the HTTP server."""

from __future__ import annotations

import json
from typing import TextIO

from reflex.errors import ReflexError
from reflex.service import DecisionService


def _bounded_lines(source: TextIO, max_line_bytes: int):
    while True:
        parts: list[str] = []
        total_bytes = 0
        oversized = False
        invalid_unicode = False
        received = False
        while True:
            chunk = source.readline(4096)
            if chunk == "":
                break
            received = True
            try:
                chunk_bytes = len(chunk.encode("utf-8"))
            except UnicodeEncodeError:
                chunk_bytes = max_line_bytes + 1
                invalid_unicode = True
            total_bytes += chunk_bytes
            if total_bytes > max_line_bytes:
                oversized = True
            if not oversized and not invalid_unicode:
                parts.append(chunk)
            if chunk.endswith("\n"):
                break
        if not received:
            return
        yield (None if oversized or invalid_unicode else "".join(parts), oversized)


def run_demo(service: DecisionService, source: TextIO, sink: TextIO, *, max_line_bytes: int = 1_048_576) -> None:
    if type(max_line_bytes) is not int or max_line_bytes < 1:
        raise ValueError("max_line_bytes must be a positive integer")
    try:
        service.prepare()
        for line, oversized in _bounded_lines(source, max_line_bytes):
            if oversized:
                sink.write(json.dumps({"error": {"code": "request_too_large", "message": "Input exceeds the configured byte limit."}}, separators=(",", ":")) + "\n")
                sink.flush()
                continue
            if line is None:
                sink.write(json.dumps({"error": {"code": "invalid_json", "message": "Input must be one line of valid JSON."}}, separators=(",", ":")) + "\n")
                sink.flush()
                continue
            if not line.strip():
                continue
            if line.strip().casefold() == "exit":
                break
            try:
                request = json.loads(line)
                result = service.predict(request)
                sink.write(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            except (json.JSONDecodeError, RecursionError, UnicodeEncodeError):
                sink.write(json.dumps({"error": {"code": "invalid_json", "message": "Input must be one line of valid JSON."}}, separators=(",", ":")) + "\n")
            except ReflexError as exc:
                sink.write(json.dumps({"error": {"code": exc.code, "message": exc.message}}, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception:
                sink.write(json.dumps({"error": {"code": "inference_failed", "message": "The active model could not complete this request."}}, separators=(",", ":")) + "\n")
            sink.flush()
    finally:
        service.backend.close()
