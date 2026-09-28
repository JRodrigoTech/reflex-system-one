"""Scoped reinstall operations with explicit preservation behavior."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from reflex.config import DEFAULT_CONFIG, save_config
from reflex.operations import OperationLock, ensure_within, server_is_running
from reflex.paths import model_root, runtime_root


class ReinstallError(RuntimeError):
    pass


_DEFERRED_SCOPES = {"runtime", "reset-settings", "full"}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _would_remove_running_python(scope: str, runtime: Path) -> bool:
    executable = Path(sys.executable)
    if scope in _DEFERRED_SCOPES:
        return _is_within(executable, runtime)
    if scope.startswith("family:"):
        family = scope.split(":", 1)[1]
        return _is_within(executable, runtime / "envs" / family)
    return False


def _schedule_deferred_reinstall(scope: str, root: Path, runtime: Path) -> None:
    if os.environ.get("REFLEX_BOOTSTRAPPED") != "1":
        raise ReinstallError("Close Reflex and run reflex.bat to reinstall its active managed Python runtime safely.")
    state = runtime / "state"
    state.mkdir(parents=True, exist_ok=True)
    request_path = state / "reinstall.request.json"
    if request_path.exists():
        raise ReinstallError("A deferred reinstall is already pending; run reflex.bat to resume it.")
    payload = {
        "schema_version": 1,
        "scope": scope,
        "confirmation": "DELETE ALL MODELS" if scope == "full" else "REINSTALL",
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix="reinstall.", suffix=".new", dir=state)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, request_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _remove(path: Path, root: Path) -> None:
    if path.exists():
        shutil.rmtree(ensure_within(path, root))


def clean_reinstall(scope: str, *, root: Path, confirmation: str | None = None) -> bool:
    """Apply a reinstall, returning False when bootstrap must finish it after exit."""
    root = root.resolve()
    runtime = runtime_root(root)
    models = model_root(root)
    if server_is_running(root):
        raise ReinstallError("Stop Reflex before reinstalling runtime or model data.")
    if scope not in {"runtime", "reset-settings", "family:laya", "family:decider", "family:decision-cuda", "full"}:
        raise ReinstallError("Reinstall scope is not recognized.")
    expected = "DELETE ALL MODELS" if scope == "full" else "REINSTALL"
    if confirmation != expected:
        raise ReinstallError(f"Type {expected} exactly to confirm this operation.")
    with OperationLock(runtime / "state" / "operation.lock", f"reinstall:{scope}"):
        if _would_remove_running_python(scope, runtime):
            _schedule_deferred_reinstall(scope, root, runtime)
            return False
        if scope.startswith("family:"):
            family = scope.split(":", 1)[1]
            _remove(runtime / "envs" / family, runtime)
            return True
        _remove(runtime / "python", runtime)
        _remove(runtime / "python.new", runtime)
        _remove(runtime / "python.old", runtime)
        _remove(runtime / "envs", runtime)
        # Clean up managed Linux/WSL state from older Reflex builds during a
        # full runtime reinstall. No current setup or execution path uses it.
        _remove(runtime / "wsl", runtime)
        _remove(runtime / "cache", runtime)
        _remove(runtime / "logs", runtime)
        if scope == "reset-settings":
            settings = runtime / "state"
            if settings.exists():
                for item in settings.iterdir():
                    if item.name != "operation.lock":
                        if item.is_dir():
                            _remove(item, runtime)
                        else:
                            item.unlink()
            save_config(DEFAULT_CONFIG, root)
            (settings / "state.json").write_text(json.dumps({"schema_version": 1, "setup_complete": False}) + "\n", encoding="utf-8")
        elif scope == "full":
            _remove(models, root)
            settings = runtime / "state"
            if settings.exists():
                for item in settings.iterdir():
                    if item.name != "operation.lock":
                        if item.is_dir():
                            _remove(item, runtime)
                        else:
                            item.unlink()
            save_config(DEFAULT_CONFIG, root)
    return True
