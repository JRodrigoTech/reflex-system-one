"""Resolve the configured execution device against a model's declared support."""

from __future__ import annotations

from reflex.errors import ReflexError
from reflex.registry import ModelSpec


DEVICE_POLICIES = frozenset({"auto", "cpu", "cuda"})


def resolve_device(
    spec: ModelSpec,
    policy: str,
    *,
    cuda_available: bool | None = None,
    qualified_devices: set[str] | frozenset[str] | None = None,
) -> str:
    """Return the canonical device for one model or raise an owned error.

    CUDA is preferred by ``auto``. CPU is selected only for models whose
    static registry entry explicitly declares CPU execution support.
    """
    if not isinstance(policy, str) or policy not in DEVICE_POLICIES:
        raise ReflexError("device_policy_invalid", "Device policy must be auto, cpu, or cuda.", 500)
    if cuda_available is None:
        try:
            import torch
        except ImportError:
            raise ReflexError("environment_missing", "The isolated backend environment is missing PyTorch.", 503) from None
        cuda_available = bool(torch.cuda.is_available())

    if policy == "cpu":
        if "cpu" not in spec.devices:
            raise ReflexError("device_unsupported", f"{spec.model_id} does not support CPU inference.", 503)
        return "cpu"

    if policy == "cuda":
        if "cuda" not in spec.devices:
            raise ReflexError("device_unsupported", f"{spec.model_id} does not support CUDA inference.", 503)
        if not cuda_available:
            raise ReflexError("cuda_unavailable", "CUDA was selected, but no usable CUDA device is available in this backend environment.", 503)
        return "cuda"

    if ("cuda" in spec.devices and cuda_available
            and (qualified_devices is None or "cuda" in qualified_devices)):
        return "cuda"
    if "cpu" in spec.devices and (qualified_devices is None or "cpu" in qualified_devices):
        return "cpu"
    if "cuda" in spec.devices and cuda_available and qualified_devices is None:
        return "cuda"
    if qualified_devices is not None:
        raise ReflexError("device_qualification_missing", f"{spec.model_id} has no currently available device with a passing qualification.", 503)
    raise ReflexError("cuda_unavailable", f"{spec.model_id} requires a compatible CUDA device; automatic CPU fallback is not supported.", 503)
