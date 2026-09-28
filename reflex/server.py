"""Uvicorn lifecycle entry point."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any

from reflex.api import create_app
from reflex.operations import create_server_pid
from reflex.paths import repository_root
from reflex.service import DecisionService

SERVER_SHUTDOWN_GRACE_SECONDS = 5.0


def serve_service(service: DecisionService, config: dict[str, Any], root: Path | None = None) -> None:
    root = (root or repository_root()).resolve()
    external_lease = os.environ.get("REFLEX_EXTERNAL_SERVER_OWNER") == "1"
    level = {"QUIET": "WARNING", "NORMAL": "INFO", "DEBUG": "DEBUG"}[config["log_level"]]
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("reflex").info("MODEL_PREPARATION starting model=%s device=%s", service.active_model,
                                     getattr(service.backend, "device", "unknown"))
    try:
        service.prepare()
    except Exception:
        service.backend.close()
        raise
    logging.getLogger("reflex").info("MODEL_READY model=%s", service.active_model)
    import uvicorn

    app = create_app(
        service,
        api_key=config.get("api_key"),
        host=config["host"],
        admission_capacity=config["admission_capacity"],
        max_body_bytes=config["max_body_bytes"],
        shutdown_grace_seconds=SERVER_SHUTDOWN_GRACE_SECONDS,
    )
    pid_file = None if external_lease else create_server_pid(root)
    try:
        uvicorn.run(
            app,
            host=config["host"],
            port=config["port"],
            workers=1,
            log_level=level.lower(),
            access_log=False,
            timeout_graceful_shutdown=SERVER_SHUTDOWN_GRACE_SECONDS,
        )
    finally:
        if pid_file is not None:
            pid_file.unlink(missing_ok=True)
    if app.state.shutdown_grace_expired:
        # A Python worker thread cannot be safely interrupted. Once the finite
        # grace period is exceeded, terminate this server process after the PID
        # marker is removed instead of letting ThreadPoolExecutor keep it alive.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        logging.shutdown()
        os._exit(1)


def check_bind_config(config: dict[str, Any]) -> None:
    if config["host"] not in {"127.0.0.1", "localhost", "::1"} and not config.get("api_key"):
        raise ValueError("A bearer API key is required for non-loopback binding.")
