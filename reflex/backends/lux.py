"""Pinned Lux candidate-head adapter with an explicit runtime qualification gate.

Lux is a custom candidate readout, not a language-model next-token head. This
adapter uses the checkpoint's FP32 CandidateHead over BF16 backbone states and
the upstream segmented renderer. It refuses devices that do not match the
checkpoint's pinned normalization/runtime profile.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from reflex.backends.base import BackendResult
from reflex.errors import ReflexError
from reflex.registry import ModelSpec
from reflex.schemas import SystemOneRequest

MAX_LUX_TOKENS = 16384
MAX_LUX_OPTIONS = 255
LUX_PROMPT_VERSION = "structured-segmented-candidate-endpoints-global-query-v2"


def validate_lux_token_count(token_count: int) -> None:
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token_count must be a non-negative integer")
    if token_count > MAX_LUX_TOKENS:
        raise ReflexError("context_too_long", "The complete request exceeds the Lux 16,384-token limit.", 413)


def normalized_candidate_probabilities(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not logits or isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("logits must be non-empty and temperature must be finite and positive")
    scaled = [float(value) / float(temperature) for value in logits]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("logits must be finite")
    peak = max(scaled)
    exps = [math.exp(value - peak) for value in scaled]
    total = math.fsum(exps)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("logits do not define a finite probability distribution")
    return [value / total for value in exps]


def _confidence(probabilities: Sequence[float]) -> float:
    count = len(probabilities)
    if count < 2:
        raise ValueError("A Lux decision distribution must have at least two candidates")
    return max(0.0, min(1.0, (count * max(probabilities) - 1.0) / (count - 1.0)))


def validate_lux_windows_cuda(
    profile: dict[str, Any],
    properties: Any,
    *,
    os_name: str,
    cuda_available: bool,
    cuda_version: str | None,
    hip_version: str | None,
    bf16_supported: bool,
    free_vram_bytes: int,
    required_vram_bytes: int,
) -> str:
    """Check the native Windows CUDA port's minimum device requirements.

    The upstream runtime profile is AMD-only. Reflex's CUDA adapter imports
    the pinned decision model directly and does not install or bind that AMD
    profile. This gate checks the CUDA/BF16 capabilities needed by the
    unqualified Windows port; the exact device still needs its own goldens.
    """
    if os_name != "nt":
        raise ReflexError("lux_windows_only", "Lux runs only in a native Windows CUDA environment.", 503)
    if profile.get("kind") != "decision-fla-l2norm-profile-v1" or profile.get("validated_arch") != "gfx942":
        raise ReflexError("model_experimental", "The pinned Lux runtime profile is unknown or changed.", 503)
    if not cuda_available or not cuda_version or hip_version is not None:
        raise ReflexError("lux_cuda_unavailable", "Lux requires the isolated native Windows NVIDIA CUDA runtime.", 503)
    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    if type(major) is not int or type(minor) is not int:
        raise ReflexError("lux_gpu_arch_unqualified", "The selected Windows CUDA GPU architecture is not recognized.", 503)
    actual = f"sm_{major}{minor}"
    if (major, minor) < (8, 0) or not bf16_supported:
        raise ReflexError(
            "lux_gpu_arch_unqualified",
            "The native Windows Lux port requires an NVIDIA BF16-capable GPU with compute capability 8.0 or newer.",
            503,
        )
    if type(required_vram_bytes) is not int or required_vram_bytes <= 0:
        raise ReflexError("model_experimental", "The pinned Lux checkpoint has no valid BF16 weight-size estimate.", 503)
    if type(free_vram_bytes) is not int or free_vram_bytes < 0:
        raise ReflexError("gpu_memory_unavailable", "Available NVIDIA GPU memory could not be checked safely.", 503)
    if free_vram_bytes < required_vram_bytes:
        raise ReflexError(
            "gpu_memory_insufficient",
            "The native Windows GPU has less free VRAM than the pinned Lux BF16 checkpoint weights require before loading.",
            503,
        )
    return actual


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Pinned Lux source file cannot be imported.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LuxBackend:
    name = "lux"

    def __init__(self, model: Any, tokenizer: Any, model_module: ModuleType, api_module: ModuleType,
                 device: str, temperatures: dict[str, float], model_id: str = "lux-9b") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_module = model_module
        self.api_module = api_module
        self.device = device
        self.temperatures = temperatures
        self.model_id = model_id
        self.loaded = True

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        model_dir: Path,
        qualification_file: Path,
        *,
        device: str | None = None,
        allow_unqualified: bool = False,
        required_vram_bytes: int | None = None,
    ) -> "LuxBackend":
        """Load the registered checkpoint, bypassing qualification only for qualification runs."""
        if spec.model_id != "lux-9b" or spec.backend != "lux" or spec.environment != "decision-cuda":
            raise ReflexError("model_experimental", "The native Lux adapter received an incompatible registry entry.", 503)
        if os.name != "nt":
            raise ReflexError("lux_windows_only", "Lux runs only in a native Windows CUDA environment.", 503)
        try:
            manifest = json.loads((model_dir / "reflex-model.json").read_text(encoding="utf-8"))
            metadata = json.loads((model_dir / "decision_config.json").read_text(encoding="utf-8"))
            runtime = json.loads((model_dir / "runtime.json").read_text(encoding="utf-8"))
            calibration = json.loads((model_dir / "temperature.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ReflexError("model_experimental", "Lux remains experimental until its pinned runtime passes real hardware qualification.", 503)
        if not all(isinstance(value, dict) for value in (manifest, metadata, runtime, calibration)):
            raise ReflexError("model_experimental", "The pinned Lux checkpoint metadata is invalid.", 503)
        try:
            record = json.loads(qualification_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = None
        if manifest.get("model_id") != spec.model_id or manifest.get("revision") != spec.revision:
            raise ReflexError("model_experimental", "The local checkpoint is not the pinned Lux model.", 503)
        if (
            metadata.get("prompt_version") != LUX_PROMPT_VERSION
            or metadata.get("max_options") != MAX_LUX_OPTIONS
            or metadata.get("head_precision") != "float32-outside-autocast"
        ):
            raise ReflexError("model_experimental", "The pinned Lux renderer or CandidateHead metadata changed.", 503)
        if runtime.get("gated_delta") != "fla.ops.gated_delta_rule.chunk":
            raise ReflexError("model_experimental", "The pinned Lux checkpoint does not declare the reviewed Gated DeltaNet runtime.", 503)
        if not allow_unqualified and (
            not isinstance(record, dict)
            or record.get("schema_version") != 1
            or record.get("model_id") != "lux-9b"
            or record.get("revision") != spec.revision
            or record.get("passed") is not True
            or not isinstance(record.get("hardware"), dict)
            or not isinstance(record.get("environment"), dict)
            or record["environment"].get("family") != spec.environment
        ):
            raise ReflexError("model_experimental", "Lux has no passing qualification for this exact checkpoint revision.", 503)

        try:
            import torch

            if not torch.cuda.is_available():
                raise ReflexError("lux_cuda_unavailable", "Lux requires the isolated native Windows NVIDIA CUDA runtime.", 503)
            device = device or "cuda:0"
            if not str(device).startswith("cuda"):
                raise ReflexError("lux_cuda_unavailable", "Lux requires the isolated native Windows NVIDIA CUDA runtime.", 503)
            props = torch.cuda.get_device_properties(device)
            try:
                free_vram_bytes, _ = torch.cuda.mem_get_info(device)
            except Exception:
                raise ReflexError("gpu_memory_unavailable", "Available NVIDIA GPU memory could not be checked safely.", 503) from None
            profile = runtime.get("normalization_profile")
            if not isinstance(profile, dict):
                raise ReflexError("model_experimental", "The pinned Lux checkpoint has no runtime profile.", 503)
            minimum_weight_bytes = spec.estimated_bytes.get("default")
            if type(minimum_weight_bytes) is not int or minimum_weight_bytes <= 0:
                raise ReflexError("model_experimental", "The pinned Lux checkpoint has no valid BF16 weight-size estimate.", 503)
            if required_vram_bytes is not None and (type(required_vram_bytes) is not int or required_vram_bytes <= 0):
                raise ReflexError("model_experimental", "The pinned Lux checkpoint has no valid BF16 weight-size estimate.", 503)
            required_vram_bytes = max(minimum_weight_bytes, required_vram_bytes or 0)
            validate_lux_windows_cuda(
                profile,
                props,
                os_name=os.name,
                cuda_available=bool(torch.cuda.is_available()),
                cuda_version=getattr(torch.version, "cuda", None),
                hip_version=getattr(torch.version, "hip", None),
                bf16_supported=bool(torch.cuda.is_bf16_supported()),
                free_vram_bytes=free_vram_bytes,
                required_vram_bytes=required_vram_bytes,
            )
            if not allow_unqualified:
                qualified_hardware = record["hardware"]
                if not str(qualified_hardware.get("os", "")).startswith("Windows"):
                    raise ReflexError("model_experimental", "Lux qualification was not performed on native Windows CUDA.", 503)
            temperature_map = calibration.get("temperatures")
            if not isinstance(temperature_map, dict) or not all(
                kind in temperature_map for kind in ("choice", "noul", "score")
            ):
                raise ReflexError("model_experimental", "The pinned Lux task calibration is incomplete.", 503)
            temperatures = {
                kind: float(temperature_map[kind])
                for kind in ("choice", "noul", "score")
            }
            if not all(math.isfinite(value) and value > 0 for value in temperatures.values()):
                raise ValueError("Invalid Lux calibration temperatures")
            model_module = _load_module("reflex_pinned_lux_model", model_dir / "code" / "decision_model.py")
            api_module = _load_module("reflex_pinned_lux_api", model_dir / "code" / "decision_api.py")
            if (
                getattr(model_module, "PROMPT_VERSION", None) != LUX_PROMPT_VERSION
                or getattr(model_module, "MAX_OPTIONS", None) != MAX_LUX_OPTIONS
            ):
                raise ValueError("Pinned Lux model source does not match its declared renderer metadata")
            model, tokenizer = model_module.DecisionModel.from_checkpoint(
                model_dir, dtype=torch.bfloat16, attention="sdpa"
            )
            model = model.to(device).eval()
            return cls(model, tokenizer, model_module, api_module, str(device), temperatures)
        except ReflexError:
            raise
        except ImportError:
            raise ReflexError("environment_missing", "The isolated Windows CUDA runtime is missing its pinned model packages.", 503)
        except Exception:
            raise ReflexError("model_load_failed", "The pinned Lux candidate-head checkpoint could not be loaded.", 503)

    def _rows(self, request: SystemOneRequest) -> list[dict[str, Any]]:
        rows = []
        for question_id, question in request.questions.items():
            item: dict[str, Any] = {
                "type": question.kind,
                "instructions": question.instructions
                if question.instructions is not None
                else "Choose the answer best supported by the state.",
            }
            if question.criteria is not None:
                item["criteria"] = question.criteria
            rows.append(self.api_module.question_row(request.state, question_id, item))
        return rows

    def validate_request(self, request: SystemOneRequest) -> None:
        try:
            rows = self._rows(request)
            for row in rows:
                item = self.model_module.encode(row, self.tokenizer, max_length=MAX_LUX_TOKENS)
                validate_lux_token_count(len(item["ids"]))
        except ReflexError:
            raise
        except ValueError as exc:
            if "exceeds max_length" in str(exc):
                raise ReflexError("context_too_long", "The complete request exceeds the Lux 16,384-token limit.", 413) from None
            raise ReflexError("backend_capability_unavailable", "The pinned Lux renderer could not validate this request.", 503) from None
        except Exception:
            raise ReflexError("backend_capability_unavailable", "The pinned Lux renderer could not validate this request.", 503) from None

    def predict(self, request: SystemOneRequest) -> BackendResult:
        try:
            import torch
            rows = self._rows(request)
            pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            distributions: dict[str, list[float]] = {}
            confidences: dict[str, float] = {}
            input_tokens = 0
            with torch.inference_mode():
                for row in rows:
                    encoded = self.model_module.encode(row, self.tokenizer, max_length=MAX_LUX_TOKENS)
                    input_tokens += len(encoded["ids"])
                    batch = self.model_module.collate([encoded], pad)
                    inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = self.model(**inputs)[0, :len(row["options"])].float().tolist()
                    normalized = normalized_candidate_probabilities(
                        logits,
                        self.temperatures[row["task_type"]],
                    )
                    if len(normalized) != len(row["options"]):
                        raise ValueError("Lux CandidateHead returned a different candidate count")
                    distributions[row["id"]] = normalized
                    confidences[row["id"]] = _confidence(normalized)
            if set(distributions) != set(request.questions):
                raise ValueError("Lux CandidateHead returned a different question set")
            return BackendResult(distributions, confidences, input_tokens, 0, self.model_id)
        except ReflexError:
            raise
        except Exception:
            raise ReflexError("backend_failure", "The active Lux candidate-head model could not complete this request.", 500)

    def close(self) -> None:
        if not self.loaded:
            return
        self.model = None
        self.tokenizer = None
        if str(self.device).startswith("cuda"):
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.loaded = False
