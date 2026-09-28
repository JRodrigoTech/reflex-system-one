"""Pinned Sol/Nox CandidateHead adapter with an explicit CUDA qualification gate."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from reflex.backends.base import BackendResult
from reflex.errors import ReflexError
from reflex.registry import ModelSpec
from reflex.schemas import SystemOneRequest

MAX_CANDIDATE_TOKENS = 16384
MAX_CANDIDATE_OPTIONS = 255
QUESTION_BATCH_SIZE = 1


def validate_candidate_token_count(token_count: int) -> None:
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("token_count must be a non-negative integer")
    if token_count > MAX_CANDIDATE_TOKENS:
        raise ReflexError(
            "context_too_long",
            "The complete request exceeds the pinned 16,384-token decision-model limit.",
            413,
        )


def normalized_candidate_probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    if not logits or isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("logits must be non-empty and temperature must be finite and positive")
    scaled = [float(value) / float(temperature) for value in logits]
    if not all(math.isfinite(value) for value in scaled):
        raise ValueError("logits must be finite")
    peak = max(scaled)
    exponentials = [math.exp(value - peak) for value in scaled]
    total = math.fsum(exponentials)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("logits do not define a finite probability distribution")
    return [value / total for value in exponentials]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Pinned CandidateHead source file cannot be imported.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _confidence(probabilities: Sequence[float]) -> float:
    count = len(probabilities)
    if count < 2:
        raise ValueError("A decision distribution must have at least two options")
    return max(0.0, min(1.0, (count * max(probabilities) - 1.0) / (count - 1.0)))


class SemanticRouterBackend:
    """Direct typed CandidateHead inference for one pinned Sol or Nox snapshot.

    This adapter intentionally imports the snapshot's reviewed renderer/model
    source, not its AMD-only runtime-profile loader. A passing model/runtime
    qualification is separate from device eligibility: any native Windows
    NVIDIA device meeting the static compute, BF16, and free-VRAM requirements
    may use the same pinned backend stack.
    """

    name = "semantic-router"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_module: ModuleType,
        api_module: ModuleType,
        model_id: str,
        device: str,
        temperatures: dict[str, float],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_module = model_module
        self.api_module = api_module
        self.model_id = model_id
        self.device = device
        self.temperatures = temperatures
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
    ) -> "SemanticRouterBackend":
        try:
            import torch

            if spec.model_id not in {"sol-2b", "nox-4b"} or spec.backend != "semantic-router":
                raise ValueError("The semantic-router adapter only accepts registered Sol and Nox checkpoints")
            manifest = json.loads((model_dir / "reflex-model.json").read_text(encoding="utf-8"))
            metadata = json.loads((model_dir / "decision_config.json").read_text(encoding="utf-8"))
            calibration = json.loads((model_dir / "temperature.json").read_text(encoding="utf-8"))
            upstream_runtime = json.loads((model_dir / "runtime.json").read_text(encoding="utf-8"))
            if manifest.get("model_id") != spec.model_id or manifest.get("revision") != spec.revision:
                raise ValueError("Pinned model metadata mismatch")
            if metadata.get("prompt_version") != "structured-segmented-candidate-endpoints-global-query-v2":
                raise ValueError("The pinned candidate renderer is not the reviewed v2 format")
            if metadata.get("max_options") != MAX_CANDIDATE_OPTIONS:
                raise ValueError("The pinned candidate option limit changed")
            if metadata.get("head_precision") != "float32-outside-autocast":
                raise ValueError("The candidate head precision boundary changed")

            try:
                record = json.loads(qualification_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                record = None
            if record is None and not allow_unqualified:
                raise ReflexError(
                    "model_qualification_missing",
                    "This checkpoint has no passing NVIDIA qualification for its pinned revision and runtime.",
                    503,
                )
            if record is not None:
                if (
                    record.get("schema_version") != 1
                    or record.get("model_id") != spec.model_id
                    or record.get("revision") != spec.revision
                    or record.get("passed") is not True
                    or not isinstance(record.get("hardware"), dict)
                    or not isinstance(record.get("environment"), dict)
                    or record["environment"].get("family") != spec.environment
                ):
                    raise ReflexError(
                        "model_qualification_mismatch",
                        "The passing qualification does not match this pinned checkpoint and backend family.",
                        503,
                    )

            requirements = spec.runtime_requirements
            if requirements is None or requirements.accelerator != "nvidia-cuda":
                raise ValueError("The pinned CandidateHead model has no NVIDIA runtime requirements")
            if not torch.cuda.is_available():
                raise ReflexError("gpu_unavailable", "Sol and Nox require a native NVIDIA CUDA runtime.", 503)
            torch_version = getattr(torch, "version", None)
            if getattr(torch_version, "cuda", None) is None or getattr(torch_version, "hip", None) is not None:
                raise ReflexError("gpu_unavailable", "Sol and Nox require the pinned NVIDIA CUDA build, not ROCm.", 503)
            device = device or "cuda:0"
            props = torch.cuda.get_device_properties(device)
            capability = (int(props.major), int(props.minor))
            if capability < requirements.minimum_compute_capability:
                required = ".".join(str(part) for part in requirements.minimum_compute_capability)
                actual = ".".join(str(part) for part in capability)
                raise ReflexError(
                    "gpu_arch_unsupported",
                    f"This model requires NVIDIA compute capability {required} or newer; this GPU reports {actual}.",
                    503,
                )
            if not torch.cuda.is_bf16_supported():
                raise ReflexError("gpu_unavailable", "The selected NVIDIA GPU does not support BF16 inference.", 503)
            try:
                free_vram_bytes, _ = torch.cuda.mem_get_info(device)
            except Exception:
                raise ReflexError("gpu_memory_unavailable", "Available NVIDIA GPU memory could not be checked safely.", 503) from None
            if type(free_vram_bytes) is not int or free_vram_bytes < requirements.minimum_free_vram_bytes:
                required_gib = requirements.minimum_free_vram_bytes / (1024**3)
                available_gib = max(0, int(free_vram_bytes)) / (1024**3) if type(free_vram_bytes) is int else 0.0
                raise ReflexError(
                    "gpu_memory_insufficient",
                    f"{spec.model_id} needs at least {required_gib:.2f} GiB of free NVIDIA VRAM before loading; {available_gib:.2f} GiB is available.",
                    503,
                )

            temperature_map = calibration.get("temperatures")
            if not isinstance(temperature_map, dict):
                raise ValueError("Pinned CandidateHead calibration is invalid")
            temperatures = {kind: float(temperature_map[kind]) for kind in ("choice", "noul", "score")}
            if not all(math.isfinite(value) and value > 0 for value in temperatures.values()):
                raise ValueError("Pinned CandidateHead calibration temperatures are invalid")
            declared_runtime = upstream_runtime.get("gated_delta")
            if declared_runtime != "fla.ops.gated_delta_rule.chunk":
                raise ValueError("The checkpoint does not declare the reviewed FLA Gated DeltaNet implementation")

            suffix = spec.model_id.replace("-", "_")
            model_module = _load_module(
                f"reflex_pinned_{suffix}_model", model_dir / "code" / "decision_model.py"
            )
            api_module = _load_module(
                f"reflex_pinned_{suffix}_api", model_dir / "code" / "decision_api.py"
            )
            if getattr(model_module, "MAX_OPTIONS", None) != MAX_CANDIDATE_OPTIONS:
                raise ValueError("Pinned CandidateHead source option limit does not match the Reflex contract")
            model, tokenizer = model_module.DecisionModel.from_checkpoint(
                model_dir, dtype=torch.bfloat16, attention="sdpa"
            )
            model = model.to(device).eval()
            return cls(model, tokenizer, model_module, api_module, spec.model_id, str(device), temperatures)
        except ReflexError:
            raise
        except ImportError:
            raise ReflexError(
                "environment_missing",
                "The isolated Decision CUDA environment is missing its pinned model packages.",
                503,
            ) from None
        except Exception:
            raise ReflexError("model_load_failed", "The pinned CandidateHead checkpoint could not be loaded.", 503) from None

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
                encoded = self.model_module.encode(row, self.tokenizer, max_length=MAX_CANDIDATE_TOKENS)
                validate_candidate_token_count(len(encoded["ids"]))
        except ReflexError:
            raise
        except ValueError as exc:
            if "exceeds max_length" in str(exc):
                raise ReflexError(
                    "context_too_long",
                    "The complete request exceeds the pinned 16,384-token decision-model limit.",
                    413,
                ) from None
            raise ReflexError(
                "backend_capability_unavailable",
                "The pinned CandidateHead renderer could not validate this request.",
                503,
            ) from None
        except Exception:
            raise ReflexError(
                "backend_capability_unavailable",
                "The pinned CandidateHead renderer could not validate this request.",
                503,
            ) from None

    def predict(self, request: SystemOneRequest) -> BackendResult:
        try:
            import torch

            rows = self._rows(request)
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id
            distributions: dict[str, list[float]] = {}
            confidences: dict[str, float] = {}
            input_tokens = 0
            with torch.inference_mode():
                for start in range(0, len(rows), QUESTION_BATCH_SIZE):
                    chunk = rows[start : start + QUESTION_BATCH_SIZE]
                    encoded = [
                        self.model_module.encode(row, self.tokenizer, max_length=MAX_CANDIDATE_TOKENS)
                        for row in chunk
                    ]
                    input_tokens += sum(len(item["ids"]) for item in encoded)
                    batch = self.model_module.collate(encoded, pad)
                    inputs = {
                        key: value.to(self.device) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits_batch = self.model(**inputs).float()
                    for index, (row, item) in enumerate(zip(chunk, encoded, strict=True)):
                        count = item["nopts"]
                        logits = logits_batch[index, :count].tolist()
                        probabilities = normalized_candidate_probabilities(
                            logits, self.temperatures[row["task_type"]]
                        )
                        distributions[row["id"]] = probabilities
                        confidences[row["id"]] = _confidence(probabilities)
            if set(distributions) != set(request.questions):
                raise ValueError("Pinned CandidateHead returned a different question set")
            return BackendResult(distributions, confidences, input_tokens, 0, self.model_id)
        except ReflexError:
            raise
        except Exception:
            raise ReflexError(
                "backend_failure",
                "The active CandidateHead model could not complete this request.",
                500,
            ) from None

    def close(self) -> None:
        if not self.loaded:
            return
        self.model = None
        self.tokenizer = None
        if self.device.startswith("cuda"):
            try:
                import gc
                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.loaded = False
