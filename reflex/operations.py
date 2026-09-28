"""Cross-process mutation lock and safe disk/path helpers."""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGING_MARGIN_BYTES = 2 * 1024**3


class OperationLock:
    def __init__(self, path: Path, operation: str) -> None:
        self.path = path
        self.operation = operation
        self.token = uuid.uuid4().hex

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except (OSError, ProcessLookupError):
            return False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "process_started": _process_start_identity(os.getpid()),
            "operation": self.operation,
            "token": self.token,
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                raise RuntimeError("An unreadable operation lock exists; inspect runtime/state/operation.lock before repair.")
            pid = old.get("pid")
            same_host = old.get("host") == socket.gethostname()
            if not same_host:
                raise RuntimeError("An operation lock belongs to another host and cannot be reclaimed automatically.")
            alive = isinstance(pid, int) and same_host and self._pid_alive(pid)
            current_start = _process_start_identity(pid) if alive else None
            old_start = old.get("process_started")
            same_process = (
                alive
                and old_start is not None
                and current_start is not None
                and old_start == current_start
            )
            if same_process:
                raise RuntimeError(f"Another Reflex operation is active: {old.get('operation', 'unknown')}.")
            # Reclaim only a lock whose owner is provably stale on this host.
            if same_host and isinstance(pid, int) and alive and (old_start is None or current_start is None):
                raise RuntimeError("Operation lock owner is alive but its start identity cannot be verified.")
            stale = self.path.with_name(self.path.name + f".stale-{uuid.uuid4().hex}")
            try:
                os.replace(self.path, stale)
                stale.unlink(missing_ok=True)
            except OSError:
                raise RuntimeError("A stale operation lock could not be reclaimed.")
            return self.acquire()
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream)
            stream.flush()
            os.fsync(stream.fileno())

    def release(self) -> None:
        try:
            old = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if old.get("token") == self.token:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "OperationLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _process_start_identity(pid: int) -> str | None:
    """Return a stable OS process start identity to guard against PID reuse."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return None
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            try:
                ok = kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user_time))
                if not ok:
                    return None
                return f"{creation.dwHighDateTime:08x}{creation.dwLowDateTime:08x}"
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21]
    except (OSError, IndexError):
        return None


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return __import__("shutil").disk_usage(path).free


def ensure_within(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError("Path escapes the Reflex-managed root")
    return resolved_path


def server_is_running(root: Path) -> bool:
    pid_file = root / "runtime" / "state" / "server.pid"
    try:
        metadata = json.loads(pid_file.read_text(encoding="utf-8"))
        pid = metadata.get("pid")
        started = metadata.get("process_started")
        if not isinstance(pid, int) or not OperationLock._pid_alive(pid):
            return False
        if metadata.get("host") not in {None, socket.gethostname()}:
            return True
        current = _process_start_identity(pid)
        return current is None or started is None or current == started
    except Exception:
        return False


def create_server_pid(root: Path) -> Path:
    path = root / "runtime" / "state" / "server.pid"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pid": os.getpid(),
        "process_started": _process_start_identity(os.getpid()),
        "host": socket.gethostname(),
        "token": uuid.uuid4().hex,
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if server_is_running(root):
            raise RuntimeError("A Reflex inference process is already running.") from None
        stale = path.with_name(path.name + f".stale-{uuid.uuid4().hex}")
        try:
            os.replace(path, stale)
            stale.unlink(missing_ok=True)
        except OSError:
            raise RuntimeError("A stale Reflex process marker could not be recovered.") from None
        return create_server_pid(root)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return path
