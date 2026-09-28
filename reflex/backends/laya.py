"""Laya direct single-checkpoint adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reflex.backends.base import BackendResult
from reflex.errors import ReflexError
from reflex.registry import ModelSpec
from reflex.schemas import SystemOneRequest


class LayaBackend:
    name = "laya"

    def __init__(self, model: Any, model_id: str = "laya", device: str = "cuda") -> None:
        self.model = model
        self.model_id = model_id
        self.device = device
        self.loaded = True

    @classmethod
    def load(cls, spec: ModelSpec, model_dir: Path, variant: str, device: str | None = None) -> "LayaBackend":
        if variant not in spec.variants:
            raise ReflexError("variant_invalid", "Selected Laya variant is not in the registry.", 422)
        try:
            import laya
        except ImportError:
            raise ReflexError("environment_missing", "The isolated Laya environment is not installed.", 503)
        try:
            # The model manager materializes only the chosen upstream subfolder.
            agent = laya.load(str(model_dir), device=device)
        except Exception:
            raise ReflexError("model_load_failed", "The selected Laya checkpoint could not be loaded.", 503)
        actual_device = str(getattr(agent, "device", device or "unknown"))
        return cls(agent, device=actual_device)

    @staticmethod
    def _upstream_questions(request: SystemOneRequest) -> dict[str, dict[str, Any]]:
        questions = {}
        for key, q in request.questions.items():
            item: dict[str, Any] = {
                "type": q.kind,
                "instructions": q.instructions if q.instructions is not None else "Choose the answer best supported by the state.",
            }
            if q.criteria is not None:
                item["criteria"] = q.criteria
            questions[key] = item
        return questions

    def validate_request(self, request: SystemOneRequest) -> None:
        """Use Laya's renderer to reject a request before upstream truncation."""
        model = self.model
        if not all(callable(getattr(model, name, None)) for name in ("_to_internal", "_encode_state")):
            raise ReflexError("backend_capability_unavailable", "The active Laya runtime cannot validate its context limits.", 503)
        config = getattr(model, "cfg", {})
        max_len = int(config.get("max_len", 512))
        head_max_len = int(config.get("head_max_len", 192))
        questions = self._upstream_questions(request)
        try:
            internal = {qid: model._to_internal(qdef) for qid, qdef in questions.items()}
            rendered = model._encode_state(request.state, list(internal), internal, max_len=10_000_000, head_max_len=head_max_len)
        except Exception:
            raise ReflexError("request_exceeds_backend_limits", "The complete request exceeds the selected Laya checkpoint limits.", 413)
        if any(len(item.get("ids", ())) > max_len for item in rendered):
            raise ReflexError("request_exceeds_backend_limits", "The complete request exceeds the selected Laya checkpoint context limit.", 413)

    def predict(self, request: SystemOneRequest) -> BackendResult:
        questions = self._upstream_questions(request)
        try:
            raw = self.model.predict(request.state, questions)
            answers = raw["answers"]
            if not isinstance(answers, dict) or set(answers) != set(request.questions):
                raise ValueError("unexpected answer IDs")
            output: dict[str, list[float]] = {}
            confidences: dict[str, float] = {}
            for qid, q in request.questions.items():
                answer = answers[qid]
                if q.kind == "noul":
                    positive = float(answer["noul"])
                    output[qid] = [1.0 - positive, positive]
                elif q.kind == "choice":
                    probs = answer.get("probabilities") or answer.get("probabilities_by_option")
                    if isinstance(probs, dict):
                        output[qid] = [float(probs[k]) for k in q.option_keys]
                    else:
                        chosen = answer["choice"]
                        confidence = float(answer.get("confidence", 1.0))
                        remainder = (1.0 - confidence) / max(1, len(q.option_keys) - 1)
                        output[qid] = [confidence if k == chosen else remainder for k in q.option_keys]
                    if "confidence" in answer:
                        confidences[qid] = float(answer["confidence"])
                else:
                    probs = answer.get("probabilities")
                    if not isinstance(probs, dict):
                        raise KeyError("probabilities")
                    output[qid] = [float(probs[str(i)]) for i in range(len(q.option_keys))]
                    if "confidence" in answer:
                        confidences[qid] = float(answer["confidence"])
            usage = raw.get("usage", {})
            return BackendResult(output, confidences, usage.get("input_tokens"), usage.get("output_tokens"), self.model_id)
        except Exception:
            raise ReflexError("backend_failure", "The active Laya model could not complete this request.", 500)

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
        self.loaded = False
