"""Qualification records bound to static model and environment revisions."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from reflex.model_manager import ModelManager
from reflex.lock_hash import lock_hash_matches, lock_sha256
from reflex.paths import repository_root
from reflex.registry import MODEL_IDS, ModelSpec, load_registry


def qualification_path(
    root: Path,
    model_id: str,
    variant: str | None = None,
    device: str | None = None,
) -> Path:
    if model_id not in MODEL_IDS:
        raise ValueError("Qualification model ID must be in the static registry.")
    if device not in {None, "cpu", "cuda"}:
        raise ValueError("Qualification device must be cpu or cuda.")
    suffix = f"-{variant}" if model_id == "laya" and variant else ""
    if device == "cpu":
        suffix += "-cpu"
    return root / "manifests" / "qualifications" / f"{model_id}{suffix}.json"


def qualification_evidence_exists(
    root: Path,
    spec: ModelSpec,
    variant: str | None = None,
    device: str | None = None,
) -> bool:
    """Whether a passing record exists for this exact checkpoint and variant.

    This deliberately ignores the current environment-lock hash. The Wizard
    uses it to distinguish an absent qualification from a valid qualification
    that became unusable after the environment lock changed.
    """
    try:
        value = json.loads(qualification_path(root, spec.model_id, variant, device).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == 1
        and value.get("model_id") == spec.model_id
        and value.get("revision") == spec.revision
        and value.get("variant") == variant
        and value.get("device", "cuda") == (device or "cuda")
        and value.get("passed") is True
        and isinstance(value.get("hardware"), dict)
        and isinstance(value.get("environment"), dict)
    )


def qualification_record(
    root: Path,
    spec: ModelSpec,
    variant: str | None = None,
    device: str | None = None,
) -> dict[str, Any] | None:
    # A variant that is not release-supported cannot become normal inference
    # evidence, even if a record is copied into the qualifications directory.
    if spec.support_status(variant if spec.model_id == "laya" else None) != "SUPPORTED":
        return None
    try:
        value = json.loads(qualification_path(root, spec.model_id, variant, device).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("model_id") != spec.model_id or value.get("revision") != spec.revision
            or value.get("variant") != variant or value.get("passed") is not True
            or value.get("device", "cuda") != (device or "cuda")
            or not isinstance(value.get("hardware"), dict)
            or not isinstance(value.get("environment"), dict)):
        return None
    lock = root / "requirements" / f"{spec.environment}.lock"
    marker = root / "runtime" / "envs" / spec.environment / "reflex-lock.json"
    try:
        environment = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        if not lock_hash_matches(value["environment"].get("lock_sha256"), lock):
            return None
        if not lock_hash_matches(environment.get("sha256", environment.get("lock_sha256")), lock):
            return None
    except OSError:
        return None
    return value


def qualified_devices(root: Path, spec: ModelSpec, variant: str | None = None) -> tuple[str, ...]:
    """List device-specific passing records for the selected checkpoint."""
    return tuple(
        device for device in spec.devices
        if qualification_record(root, spec, variant, device) is not None
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _driver_metadata(device_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"device": device_name}
    executable = "nvidia-smi.exe" if os.name == "nt" else "nvidia-smi"
    try:
        query = subprocess.run(
            [executable, "--query-gpu=driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if query.returncode == 0:
            fields = [field.strip() for field in query.stdout.splitlines()[0].split(",")]
            if fields:
                result["driver"] = fields[0]
            if len(fields) > 1:
                result["vram_mib"] = int(fields[1])
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        pass
    return result


def write_qualification_record(
    model_id: str,
    *,
    root: Path | None = None,
    variant: str | None = None,
    backend: Any,
    fixtures: list[dict[str, Any]],
) -> Path:
    """Write a record only after the caller's real golden assertions pass."""
    root = (root or repository_root()).resolve()
    registry = load_registry(root)
    spec = registry.get(model_id)
    if spec is None or not spec.pinned:
        raise ValueError("Qualification requires a pinned static model revision.")
    if model_id == "laya" and variant not in spec.variants:
        raise ValueError("Laya qualification requires a registered checkpoint variant.")
    if model_id == "laya" and spec.support_status(variant) != "SUPPORTED":
        raise ValueError("This Laya variant is not release-supported and cannot write a qualification record.")
    if model_id != "laya" and variant is not None:
        raise ValueError("Only Laya has checkpoint variants.")
    if not fixtures or any(item.get("passed") is not True for item in fixtures):
        raise ValueError("Qualification fixtures must all pass before recording.")
    manager = ModelManager(root, registry)
    if not manager.verify(model_id, variant):
        raise ValueError("The exact pinned checkpoint must pass full file verification before qualification.")

    backend_device = str(getattr(backend, "device", ""))
    if backend_device.startswith("cuda"):
        execution_device = "cuda"
    elif backend_device == "cpu" or backend_device.startswith("cpu:"):
        execution_device = "cpu"
    else:
        raise ValueError("Qualification requires an explicitly supported CPU or CUDA backend device.")
    if execution_device not in spec.devices:
        raise ValueError(f"{model_id} does not declare {execution_device} execution support.")

    torch_version = cuda_version = None
    hardware: dict[str, Any] = {"os": platform.platform(), "python": platform.python_version()}
    precision = "unknown"
    if execution_device == "cuda":
        import torch

        props = torch.cuda.get_device_properties(backend.device)
        hardware.update(_driver_metadata(str(props.name)))
        hardware["compute_capability"] = f"{props.major}.{props.minor}"
        hardware["vram_bytes"] = int(props.total_memory)
        hardware["peak_allocated_vram_bytes"] = int(torch.cuda.max_memory_allocated(backend.device))
        hardware["peak_reserved_vram_bytes"] = int(torch.cuda.max_memory_reserved(backend.device))
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        model = getattr(backend, "model", None)
        if model is None:
            model = getattr(getattr(backend, "engine", None), "m", None)
        parameter_dtype = "unknown"
        try:
            parameter_dtype = str(next(model.parameters()).dtype)
        except (AttributeError, StopIteration, TypeError):
            nested_model = getattr(model, "model", None)
            try:
                parameter_dtype = str(next(nested_model.parameters()).dtype)
            except (AttributeError, StopIteration, TypeError):
                pass
        if getattr(backend, "name", None) == "decider":
            inference_mode = "eager; CUDA graphs disabled"
        elif getattr(backend, "name", None) == "semantic-router":
            inference_mode = "BF16 CUDA backbone; FP32 CandidateHead; SDPA; one-question batches; FLA Gated DeltaNet"
        elif getattr(backend, "name", None) == "lux":
            inference_mode = "BF16 CUDA backbone; FP32 CandidateHead; SDPA; one-question batches; FLA Gated DeltaNet"
        else:
            inference_mode = f"autocast={getattr(model, 'dtype', None)}"
        precision = f"parameters={parameter_dtype}; {inference_mode}"
    else:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        hardware["processor"] = platform.processor() or platform.machine()
        hardware["logical_cpu_count"] = os.cpu_count()
        hardware["torch_cpu_threads"] = int(torch.get_num_threads())
        model = getattr(backend, "model", None)
        if model is None:
            model = getattr(getattr(backend, "engine", None), "m", None)
        parameter_dtype = "unknown"
        try:
            parameter_dtype = str(next(model.parameters()).dtype)
        except (AttributeError, StopIteration, TypeError):
            nested_model = getattr(model, "model", None)
            try:
                parameter_dtype = str(next(nested_model.parameters()).dtype)
            except (AttributeError, StopIteration, TypeError):
                pass
        precision = f"parameters={parameter_dtype}; CPU inference"

    family = spec.environment
    lock_path = root / "requirements" / f"{family}.lock"
    lock_hash = lock_sha256(lock_path) if lock_path.is_file() else None
    marker_path = root / "runtime" / "envs" / family / "reflex-lock.json"
    try:
        installed_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("The exact dependency lock must be installed in its isolated environment before qualification.") from None
    if not lock_hash_matches(installed_marker.get("sha256", installed_marker.get("lock_sha256")), lock_path):
        raise ValueError("The installed backend environment does not match the checked-in lock.")
    record = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": spec.revision,
        "variant": variant,
        "device": execution_device,
        "passed": True,
        "qualified_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "backend": spec.backend,
        "precision": precision,
        "hardware": hardware,
        "runtime": {
            "torch": torch_version,
            "cuda": cuda_version,
            "transformers": _package_version("transformers"),
            "triton": (
                _package_version("triton-windows")
                if spec.environment == "decision-cuda"
                else _package_version("triton")
            ),
            "triton_distribution": "triton-windows" if spec.environment == "decision-cuda" else "triton",
            "flash_linear_attention": _package_version("flash-linear-attention"),
            "fla_core": _package_version("fla-core"),
            "tokenizers": _package_version("tokenizers"),
            "safetensors": _package_version("safetensors"),
            "backend_package": (
                _package_version("laya") if spec.backend == "laya" else
                "bundled-model-source" if spec.backend in {"decider", "lux"} else
                _package_version("flash-linear-attention") if spec.backend == "semantic-router" else
                None
            ),
            "backend_source_revision": spec.revision if spec.backend in {"decider", "semantic-router", "lux"} else None,
        },
        "environment": {"family": family, "lock_sha256": lock_hash},
        "fixtures": fixtures,
    }
    if model_id == "laya" and variant == "typed-decisions":
        record["limitations"] = [
            "Runtime and output-contract checks do not evaluate statistical probability calibration; no validation dataset was supplied."
        ]
    destination = qualification_path(root, model_id, variant, execution_device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".new")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination
