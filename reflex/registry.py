"""Static, trusted model registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from reflex.paths import repository_root

MODEL_IDS = ("laya", "decider-2b", "decider-4b", "sol-2b", "nox-4b", "lux-9b")
SUPPORT_STATES = {"SUPPORTED", "EXPERIMENTAL", "UNAVAILABLE"}


@dataclass(frozen=True, slots=True)
class RuntimeRequirements:
    accelerator: str
    minimum_compute_capability: tuple[int, int]
    precision: str
    minimum_free_vram_bytes: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    repository: str
    source_url: str
    revision: str | None
    variants: tuple[str, ...]
    default_variant: str | None
    variant_subfolders: dict[str, str]
    estimated_bytes: dict[str, int]
    backend: str
    environment: str
    status: str
    variant_status: dict[str, str]
    license: str
    required_files: tuple[str, ...]
    devices: tuple[str, ...]
    runtime_requirements: RuntimeRequirements | None

    @property
    def pinned(self) -> bool:
        return bool(self.revision and len(self.revision) == 40 and all(c in "0123456789abcdef" for c in self.revision.lower()))

    def support_status(self, variant: str | None = None) -> str:
        return self.variant_status.get(variant or "", self.status)


def validate_model_manifest(value: object) -> dict[str, ModelSpec]:
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("models"), dict):
        raise ValueError("Model manifest must have schema_version 1 and a models object")
    raw_models = value["models"]
    if set(raw_models) != set(MODEL_IDS):
        raise ValueError("Model manifest must contain exactly the six public Reflex IDs")
    specs: dict[str, ModelSpec] = {}
    for model_id, raw in raw_models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Model {model_id} must be an object")
        if not isinstance(raw.get("source_url"), str) or not raw["source_url"].startswith("https://"):
            raise ValueError(f"Model {model_id} must have a trusted HTTPS source URL")
        revision = raw.get("revision")
        if revision is not None and not (
            isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower())
        ):
            raise ValueError(f"Model {model_id} revision must be a full commit SHA or null")
        variants = raw.get("variants", [])
        variant_subfolders = raw.get("variant_subfolders", {})
        estimated_bytes = raw.get("estimated_bytes", {})
        required_files = raw.get("required_files", [])
        devices = raw.get("devices")
        if not isinstance(variants, list) or not all(isinstance(x, str) for x in variants):
            raise ValueError(f"Model {model_id} variants must be strings")
        if not isinstance(variant_subfolders, dict) or set(variant_subfolders) != set(variants) or not all(
            isinstance(k, str) and isinstance(v, str) and (not v or all(part not in {"", ".", ".."} for part in v.split("/")))
            for k, v in variant_subfolders.items()
        ):
            raise ValueError(f"Model {model_id} variant_subfolders must map each variant to a safe source subfolder")
        if not isinstance(estimated_bytes, dict) or not estimated_bytes or not all(
            isinstance(k, str) and type(v) is int and v > 0 for k, v in estimated_bytes.items()
        ):
            raise ValueError(f"Model {model_id} must include positive pinned size estimates")
        if not isinstance(required_files, list) or not all(isinstance(x, str) and x and not Path(x).is_absolute() and ".." not in Path(x).parts for x in required_files):
            raise ValueError(f"Model {model_id} required_files must be safe relative paths")
        if raw.get("status") not in SUPPORT_STATES:
            raise ValueError(f"Model {model_id} has an invalid support status")
        variant_status = raw.get("variant_status", {})
        if (not isinstance(variant_status, dict)
                or not set(variant_status).issubset(set(variants))
                or not all(isinstance(key, str) and state in SUPPORT_STATES
                           for key, state in variant_status.items())):
            raise ValueError(f"Model {model_id} variant_status must map registered variants to valid support states")
        if not isinstance(devices, list) or not devices or not all(
            isinstance(device, str) and device in {"cpu", "cuda"} for device in devices
        ):
            raise ValueError(f"Model {model_id} must declare supported cpu/cuda execution devices")
        if len(set(devices)) != len(devices):
            raise ValueError(f"Model {model_id} execution devices must be unique")
        default_variant = raw.get("default_variant")
        if default_variant is not None and default_variant not in variants:
            raise ValueError(f"Model {model_id} default_variant must be listed in variants")
        backend = raw.get("backend")
        if not isinstance(backend, str) or not backend:
            raise ValueError(f"Model {model_id} backend must be a non-empty string")
        expected_devices = {"cpu", "cuda"} if backend in {"laya", "decider"} else {"cuda"}
        if set(devices) != expected_devices:
            raise ValueError(f"Model {model_id} device support does not match its reviewed backend family")
        requirements_raw = raw.get("runtime_requirements")
        runtime_requirements = None
        if backend == "semantic-router":
            if not isinstance(requirements_raw, dict) or set(requirements_raw) != {
                "accelerator",
                "minimum_compute_capability",
                "precision",
                "minimum_free_vram_bytes",
            }:
                raise ValueError(f"Model {model_id} must declare its NVIDIA runtime requirements")
            capability = requirements_raw.get("minimum_compute_capability")
            minimum_free_vram = requirements_raw.get("minimum_free_vram_bytes")
            if (
                requirements_raw.get("accelerator") != "nvidia-cuda"
                or not isinstance(capability, list)
                or len(capability) != 2
                or any(type(part) is not int or part < 0 for part in capability)
                or (capability[0], capability[1]) < (8, 6)
                or requirements_raw.get("precision") != "bfloat16"
                or type(minimum_free_vram) is not int
                or minimum_free_vram <= 0
            ):
                raise ValueError(f"Model {model_id} has invalid NVIDIA runtime requirements")
            runtime_requirements = RuntimeRequirements(
                accelerator="nvidia-cuda",
                minimum_compute_capability=(capability[0], capability[1]),
                precision="bfloat16",
                minimum_free_vram_bytes=minimum_free_vram,
            )
        elif requirements_raw is not None:
            raise ValueError(f"Model {model_id} cannot declare semantic-router runtime requirements")
        specs[model_id] = ModelSpec(
            model_id=model_id,
            repository=str(raw["repository"]),
            source_url=str(raw["source_url"]),
            revision=revision,
            variants=tuple(variants),
            default_variant=default_variant,
            variant_subfolders=dict(variant_subfolders),
            estimated_bytes=dict(estimated_bytes),
            backend=str(backend),
            environment=str(raw["environment"]),
            status=str(raw["status"]),
            variant_status=dict(variant_status),
            license=str(raw["license"]),
            required_files=tuple(required_files),
            devices=tuple(devices),
            runtime_requirements=runtime_requirements,
        )
    return specs


def load_registry(root: Path | None = None) -> dict[str, ModelSpec]:
    path = (root or repository_root()) / "manifests" / "models.json"
    return validate_model_manifest(json.loads(path.read_text(encoding="utf-8")))
