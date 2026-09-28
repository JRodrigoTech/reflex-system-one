"""Versioned Reflex configuration and state."""

from __future__ import annotations

import json
import ipaddress
import os
import tempfile
from pathlib import Path
from typing import Any

from reflex.errors import ReflexError
from reflex.paths import config_path, state_path
from reflex.registry import MODEL_IDS

CONFIG_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
VALID_LAYA_VARIANTS = {"multilingual", "english", "typed-decisions"}
DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": CONFIG_SCHEMA_VERSION,
    "active_model": "laya",
    "laya_variant": "multilingual",
    "device_policy": "auto",
    "host": "127.0.0.1",
    "port": 1919,
    "api_key": None,
    "log_level": "NORMAL",
    "admission_capacity": 16,
    "max_body_bytes": 1048576,
}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".new", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def validate_config(value: dict[str, Any]) -> dict[str, Any]:
    result = {**DEFAULT_CONFIG, **value}
    if result.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ReflexError("config_version_unsupported", "Configuration version is not supported.", 500)
    if result["active_model"] not in MODEL_IDS:
        raise ReflexError("config_invalid", "Configured model is not in the Reflex registry.", 500)
    if result["laya_variant"] not in VALID_LAYA_VARIANTS:
        raise ReflexError("config_invalid", "Configured Laya variant is invalid.", 500)
    if not isinstance(result["device_policy"], str) or result["device_policy"] not in {"auto", "cpu", "cuda"}:
        raise ReflexError("config_invalid", "Configured device policy must be auto, cpu, or cuda.", 500)
    if type(result["port"]) is not int or not 1 <= result["port"] <= 65535:
        raise ReflexError("config_invalid", "Configured port is invalid.", 500)
    if type(result["admission_capacity"]) is not int or not 1 <= result["admission_capacity"] <= 1024:
        raise ReflexError("config_invalid", "Configured admission capacity is invalid.", 500)
    if type(result["max_body_bytes"]) is not int or not 1024 <= result["max_body_bytes"] <= 16777216:
        raise ReflexError("config_invalid", "Configured body limit is invalid.", 500)
    if result["log_level"] not in {"QUIET", "NORMAL", "DEBUG"}:
        raise ReflexError("config_invalid", "Configured log level is invalid.", 500)
    host = result.get("host")
    if not isinstance(host, str) or not host:
        raise ReflexError("config_invalid", "Configured bind host must be a literal IP address.", 500)
    try:
        bind_ip = ipaddress.ip_address(host)
    except ValueError:
        raise ReflexError("config_invalid", "Configured bind host must be a literal IP address.", 500)
    api_key = result.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or len(api_key) < 32 or any(ch in api_key for ch in "\r\n\x00")):
        raise ReflexError("config_invalid", "Configured bearer key must contain at least 32 safe characters.", 500)
    if not bind_ip.is_loopback and not api_key:
        raise ReflexError("config_invalid", "Non-loopback binding requires a bearer API key.", 500)
    return result


def load_config(root: Path | None = None) -> dict[str, Any]:
    path = config_path(root)
    if not path.exists():
        save_config(DEFAULT_CONFIG, root)
        return dict(DEFAULT_CONFIG)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ReflexError("config_unreadable", "Configuration file is unreadable. Run Doctor and repair configuration.", 500)
    if not isinstance(value, dict):
        raise ReflexError("config_invalid", "Configuration file must contain a JSON object.", 500)
    return validate_config(value)


def save_config(value: dict[str, Any], root: Path | None = None) -> None:
    _atomic_json(config_path(root), validate_config(value))


def load_state(root: Path | None = None) -> dict[str, Any]:
    path = state_path(root)
    if not path.exists():
        value = {"schema_version": STATE_SCHEMA_VERSION, "setup_complete": False}
        _atomic_json(path, value)
        return value
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ReflexError("state_unreadable", "Reflex state file is unreadable.", 500)
    if not isinstance(value, dict) or value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ReflexError("state_invalid", "Reflex state file has an unsupported format.", 500)
    return value
