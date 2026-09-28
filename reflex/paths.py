"""Repository-owned data paths."""

from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    override = os.environ.get("REFLEX_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[1]


def runtime_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "runtime"


def model_root(root: Path | None = None) -> Path:
    return (root or repository_root()) / "models"


def config_path(root: Path | None = None) -> Path:
    return runtime_root(root) / "state" / "config.json"


def state_path(root: Path | None = None) -> Path:
    return runtime_root(root) / "state" / "state.json"


def model_path(model_id: str, variant: str | None = None, root: Path | None = None) -> Path:
    if not model_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in model_id):
        raise ValueError("Invalid model identifier")
    suffix = f"-{variant}" if variant else ""
    if variant and any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in variant):
        raise ValueError("Invalid model variant")
    return model_root(root) / f"{model_id}{suffix}"
