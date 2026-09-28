"""FastAPI transport. Model semantics remain in DecisionService."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from reflex.errors import ReflexError
from reflex.service import DecisionService

log = logging.getLogger("reflex")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")


class Admission:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.active = 0
        self.waiting = 0
        self.accepting = True
        self._lock = asyncio.Lock()
        self._gate = asyncio.Lock()

    async def acquire_admission(self) -> bool:
        async with self._lock:
            if not self.accepting or self.active >= self.capacity:
                return False
            self.active += 1
            return True

    async def release_admission(self) -> None:
        async with self._lock:
            self.active = max(0, self.active - 1)

    def snapshot(self) -> dict[str, int]:
        return {"active": self.active, "waiting": self.waiting, "capacity": self.capacity}


def _error_body(error: ReflexError, request_id: str) -> dict[str, Any]:
    return {"error": {"code": error.code, "message": error.message, "request_id": request_id}}


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id")
    if supplied:
        if not REQUEST_ID_RE.fullmatch(supplied):
            raise ReflexError("invalid_request_id", "X-Request-Id must contain 1 to 96 letters, numbers, underscores, or hyphens.", 400)
        return supplied
    return "req_" + uuid.uuid4().hex


def create_app(
    service: DecisionService,
    *,
    api_key: str | None = None,
    host: str = "127.0.0.1",
    admission_capacity: int = 16,
    max_body_bytes: int = 1048576,
    shutdown_grace_seconds: float = 5.0,
) -> FastAPI:
    if host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ValueError("Bearer authentication is required for non-loopback binding")
    admission = Admission(admission_capacity)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflex-inference")
    active_jobs: set[asyncio.Future[Any]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        admission.accepting = False
        service.ready = False
        jobs = tuple(active_jobs)
        if jobs:
            _, pending = await asyncio.wait(jobs, timeout=max(0.0, shutdown_grace_seconds))
        else:
            pending = set()
        executor.shutdown(wait=False, cancel_futures=True)
        if pending:
            # Do not close a backend while a synchronous inference still owns its tensors.
            app.state.shutdown_grace_expired = True
            log.error("SHUTDOWN_GRACE_EXPIRED active=1")
            return
        service.backend.close()

    app = FastAPI(title="Reflex", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.admission = admission
    app.state.service = service
    app.state.shutdown_grace_expired = False

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {**service.health_snapshot(), "queue": admission.snapshot()}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return service.models_payload()

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        started = time.perf_counter()
        try:
            req_id = _request_id(request)
        except ReflexError as exc:
            req_id = "req_" + uuid.uuid4().hex
            elapsed = (time.perf_counter() - started) * 1000.0
            return JSONResponse(_error_body(exc, req_id), status_code=exc.status_code, headers={
                "X-TypeSafe-Request-Id": req_id, "X-Reflex-Request-Id": req_id,
                "Server-Timing": f"inference;dur={elapsed:.2f}",
                "X-Inference-Time-Ms": f"{elapsed:.2f}",
            })
        owned = False
        waiting_for_gate = False
        gate_owned = False
        future: asyncio.Future[Any] | None = None
        inference_started: float | None = None
        inference_ended: float | None = None

        def headers(status_code: int) -> dict[str, str]:
            if inference_started is None:
                elapsed = 0.0
            else:
                elapsed = ((inference_ended or time.perf_counter()) - inference_started) * 1000.0
            return {
                "X-TypeSafe-Request-Id": req_id,
                "X-Reflex-Request-Id": req_id,
                "Server-Timing": f"inference;dur={elapsed:.2f}",
                "X-Inference-Time-Ms": f"{elapsed:.2f}",
            }

        try:
            if api_key is not None:
                authorization = request.headers.get("authorization", "")
                scheme, _, credential = authorization.partition(" ")
                if scheme.lower() != "bearer" or not credential or not hmac.compare_digest(credential, api_key):
                    raise ReflexError("unauthorized", "A valid bearer token is required.", 401)
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_body_bytes:
                        raise ReflexError("request_too_large", "Request body exceeds the configured byte limit.", 413)
                except ValueError:
                    raise ReflexError("invalid_content_length", "Content-Length is invalid.", 400)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > max_body_bytes:
                    raise ReflexError("request_too_large", "Request body exceeds the configured byte limit.", 413)
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                raise ReflexError("invalid_json", "Request body must be valid UTF-8 JSON.", 400)
            if not await admission.acquire_admission():
                raise ReflexError("server_busy", "Reflex is busy; retry the request shortly.", 503)
            owned = True
            admission.waiting += 1
            waiting_for_gate = True
            while True:
                if not admission.accepting:
                    raise ReflexError("server_shutting_down", "Reflex is shutting down and is not accepting work.", 503)
                if await request.is_disconnected():
                    raise asyncio.CancelledError()
                try:
                    await asyncio.wait_for(admission._gate.acquire(), timeout=0.05)
                    gate_owned = True
                    break
                except TimeoutError:
                    continue
            admission.waiting = max(0, admission.waiting - 1)
            waiting_for_gate = False
            inference_started = time.perf_counter()
            future = asyncio.ensure_future(asyncio.get_running_loop().run_in_executor(executor, service.predict, payload))
            active_jobs.add(future)
            future.add_done_callback(active_jobs.discard)
            try:
                result = await asyncio.shield(future)
            except asyncio.CancelledError:
                release_gate = gate_owned
                future_for_cleanup = future
                async def finish_detached() -> None:
                    try:
                        await future_for_cleanup
                    except BaseException:
                        pass
                    finally:
                        if release_gate:
                            admission._gate.release()
                        await admission.release_admission()
                asyncio.create_task(finish_detached())
                gate_owned = False
                owned = False
                raise
            finally:
                inference_ended = time.perf_counter()
            log.info("%s COMPLETE status=200", req_id)
            return JSONResponse(result, headers=headers(200))
        except asyncio.CancelledError:
            raise
        except ReflexError as exc:
            return JSONResponse(_error_body(exc, req_id), status_code=exc.status_code, headers=headers(exc.status_code))
        except Exception:
            # The exception may contain request data, local paths, or runtime secrets.
            log.error("%s INFER_ERROR", req_id)
            error = ReflexError("inference_failed", "The active model could not complete this request.", 500)
            return JSONResponse(_error_body(error, req_id), status_code=500, headers=headers(500))
        finally:
            if waiting_for_gate:
                admission.waiting = max(0, admission.waiting - 1)
            if gate_owned:
                admission._gate.release()
            if owned:
                await admission.release_admission()

    return app
