"""Construct the one backend selected for a server process."""

from __future__ import annotations

from pathlib import Path

from reflex.backends.decider import DeciderBackend
from reflex.backends.laya import LayaBackend
from reflex.backends.lux import LuxBackend
from reflex.backends.semantic_router import SemanticRouterBackend
from reflex.devices import resolve_device
from reflex.errors import ReflexError
from reflex.model_manager import ModelManager
from reflex.paths import repository_root
from reflex.qualification import qualified_devices as model_qualified_devices
from reflex.qualification import qualification_path, qualification_record
from reflex.registry import ModelSpec, load_registry
from reflex.service import DecisionService


def load_service(
    root: Path | None = None,
    device: str | None = None,
    *,
    allow_experimental: bool = False,
) -> DecisionService:
    from reflex.config import load_config

    root = (root or repository_root()).resolve()
    config = load_config(root)
    registry = load_registry(root)
    spec: ModelSpec = registry[config["active_model"]]
    variant = config["laya_variant"] if spec.model_id == "laya" else None
    if spec.support_status(variant) != "SUPPORTED" and not allow_experimental:
        raise ReflexError("model_experimental", f"The active model or checkpoint variant is experimental and is not available for normal inference.", 503)
    manager = ModelManager(root, registry)
    requested_device = device or config.get("device_policy", "auto")
    if isinstance(requested_device, str) and requested_device.startswith("cuda"):
        requested_device = "cuda"
    elif isinstance(requested_device, str) and requested_device.startswith("cpu"):
        requested_device = "cpu"
    passing_devices = set(model_qualified_devices(root, spec, variant))
    execution_device = resolve_device(
        spec,
        requested_device,
        qualified_devices=passing_devices if requested_device == "auto" else None,
    )
    if not allow_experimental and qualification_record(root, spec, variant, execution_device) is None:
        raise ReflexError(
            "model_qualification_missing",
            f"The active checkpoint has no passing {execution_device.upper()} qualification for its pinned revision and environment.",
            503,
        )
    if not manager.verify(spec.model_id, variant):
        raise ReflexError("model_not_ready", "The selected model is not installed and verified. Run Setup and Doctor.", 503)
    model_dir = manager.destination(spec.model_id, variant)
    if spec.backend == "laya":
        backend = LayaBackend.load(spec, model_dir, variant or "multilingual", device=execution_device)
    elif spec.backend == "decider":
        backend = DeciderBackend.load(spec, model_dir, device=execution_device)
    elif spec.backend == "lux":
        backend = LuxBackend.load(
            spec,
            model_dir,
            qualification_path(root, spec.model_id, device=execution_device),
            device=execution_device,
            required_vram_bytes=spec.estimated_bytes.get("default"),
        )
    elif spec.backend == "semantic-router":
        backend = SemanticRouterBackend.load(
            spec,
            model_dir,
            qualification_path(root, spec.model_id, device=execution_device),
            device=execution_device,
            allow_unqualified=allow_experimental,
        )
    else:
        raise ReflexError("backend_unknown", "The selected model backend is not available.", 503)
    return DecisionService(registry, spec.model_id, backend, variant)
