"""Model-agnostic request resolution and public response normalization."""

from __future__ import annotations

import json
import math
from typing import Any

from reflex.backends.base import BackendResult, DecisionBackend
from reflex.errors import ReflexError
from reflex.registry import ModelSpec
from reflex.schemas import SystemOneRequest, parse_request

ALIASES = {"reflex", "jev-latest"}


def _usage_count(value: int | None, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return fallback
    return value


def _probabilities(values: list[float], expected: int) -> list[float]:
    if not isinstance(values, list) or len(values) != expected:
        raise ReflexError("backend_invalid_output", "The active model returned an invalid answer distribution.", 500)
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ReflexError("backend_invalid_output", "The active model returned an invalid answer distribution.", 500)
        normalized.append(float(value))
    total = math.fsum(normalized)
    if total <= 0 or not math.isfinite(total):
        raise ReflexError("backend_invalid_output", "The active model returned an invalid answer distribution.", 500)
    return [value / total for value in normalized]


def _confidence(result: BackendResult, question_id: str, probabilities: list[float]) -> float:
    value = result.confidences.get(question_id)
    if value is None:
        return max(probabilities)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= value <= 1:
        raise ReflexError("backend_invalid_output", "The active model returned invalid confidence metadata.", 500)
    return float(value)


class DecisionService:
    def __init__(
        self,
        registry: dict[str, ModelSpec],
        active_model: str,
        backend: DecisionBackend,
        variant: str | None = None,
    ) -> None:
        self.registry = registry
        self.active_model = active_model
        self.backend = backend
        self.variant = variant
        self.ready = bool(getattr(backend, "loaded", False))
        self.prepared = False

    def prepare(self) -> None:
        """Run a fixed private inference so lazy kernels initialize before serving."""
        if self.prepared:
            return
        if not self.ready:
            raise ReflexError("backend_unavailable", "The active model is not ready.", 503)
        request = {
            "state": "Reflex startup readiness check.",
            "questions": {
                "reflex_startup_noul": {"type": "noul", "instructions": "Is this a startup readiness check?"},
                "reflex_startup_choice": {
                    "type": "choice", "instructions": "Choose the startup readiness result.",
                    "criteria": {"ready": "The local model is ready.", "unavailable": "The local model is unavailable."},
                },
                "reflex_startup_score": {"type": "score", "instructions": "Rate readiness.",
                                          "criteria": ["not ready", "ready"]},
            },
        }
        try:
            self.predict(request)
        except Exception:
            self.ready = False
            raise ReflexError("startup_preparation_failed", "The active model could not finish its startup readiness check.", 503) from None
        self.prepared = True

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ready else "starting",
            "ready": self.ready,
            "backend": self.registry[self.active_model].backend,
            "model": self.active_model,
            "device": getattr(self.backend, "device", "unknown"),
        }

    def models_payload(self) -> dict[str, Any]:
        spec = self.registry[self.active_model]
        date = "2026-09-25"
        return {
            "models": [
                {"name": "reflex", "description": "Alias for the active local Reflex decision model.", "release_date": date},
                {"name": "jev-latest", "description": "Jev compatibility alias for the active local Reflex decision model.", "release_date": date},
                {"name": spec.model_id, "description": f"Active local {spec.model_id} decision model.", "release_date": date},
            ]
        }

    def _resolve(self, requested: str | None) -> ModelSpec:
        if not self.ready:
            raise ReflexError("backend_unavailable", "The active model is not ready.", 503)
        if requested is None or requested in ALIASES:
            return self.registry[self.active_model]
        if requested not in self.registry:
            raise ReflexError("model_not_found", "Requested model is not in the Reflex registry.", 404)
        if requested != self.active_model:
            raise ReflexError("model_not_active", "Requested model is known but is not the active Reflex model.", 409)
        return self.registry[requested]

    def predict(self, value: Any) -> dict[str, Any]:
        request = value if isinstance(value, SystemOneRequest) else parse_request(value)
        spec = self._resolve(request.model)
        validate_request = getattr(self.backend, "validate_request", None)
        if callable(validate_request):
            validate_request(request)
        result = self.backend.predict(request)
        if not isinstance(result, BackendResult):
            raise ReflexError("backend_invalid_output", "The active model returned an invalid response.", 500)
        if set(result.distributions) != set(request.questions):
            raise ReflexError("backend_invalid_output", "The active model returned an invalid response.", 500)
        answers: dict[str, Any] = {}
        for qid, question in request.questions.items():
            p = _probabilities(result.distributions[qid], len(question.option_keys))
            confidence = _confidence(result, qid, p)
            if question.kind == "noul":
                answers[qid] = {"type": "noul", "noul": p[1]}
            elif question.kind == "choice":
                assert isinstance(question.criteria, dict)
                selected = max(range(len(p)), key=p.__getitem__)
                answers[qid] = {
                    "type": "choice",
                    "choice": question.option_keys[selected],
                    "confidence": confidence,
                    "probabilities": {key: p[i] for i, key in enumerate(question.option_keys)},
                }
            else:
                score = math.fsum(i * probability for i, probability in enumerate(p))
                answers[qid] = {
                    "type": "score",
                    "score": score,
                    "confidence": confidence,
                    "legend": question.legend,
                    "probabilities": {str(i): probability for i, probability in enumerate(p)},
                }
        state_chars = len(json.dumps(request.state, ensure_ascii=False, separators=(",", ":")))
        question_chars = len(json.dumps(value.get("questions", {}), ensure_ascii=False, separators=(",", ":"))) if isinstance(value, dict) else 0
        input_fallback = (state_chars + question_chars + 3) // 4
        return {
            "model": spec.model_id,
            "answers": answers,
            "usage": {
                "input_tokens": _usage_count(result.input_tokens, input_fallback),
                "output_tokens": _usage_count(result.output_tokens, 0),
            },
        }
