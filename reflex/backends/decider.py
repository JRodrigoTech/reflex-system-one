"""Pinned-source eager Decider adapter.

The model snapshot carries the reviewed ``decider`` implementation at its
exact registry revision. Reflex imports that local package rather than an
unrelated globally installed package or mutable ``trust_remote_code`` branch.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from reflex.backends.base import BackendResult
from reflex.errors import ReflexError
from reflex.registry import ModelSpec
from reflex.schemas import SystemOneRequest

MAX_DECIDER_OPTIONS = 255


def _load_pinned_decider_package(source_root: Path) -> Any:
    package_init = source_root / "decider" / "__init__.py"
    if not package_init.is_file():
        raise ImportError("Pinned Decider package files are missing.")
    # The process serves one model family. Clear only modules with the exact
    # top-level package name so a foreign `decider` install cannot win import.
    for name in tuple(sys.modules):
        if name == "decider" or name.startswith("decider."):
            del sys.modules[name]
    package_spec = importlib.util.spec_from_file_location(
        "decider", package_init, submodule_search_locations=[str(package_init.parent)]
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError("Pinned Decider package could not be loaded.")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules["decider"] = package
    package_spec.loader.exec_module(package)
    return importlib.import_module("decider.infer").Decider


class DeciderBackend:
    name = "decider"

    def __init__(self, engine: Any, model_id: str, device: str = "cuda", context_limit: int = 32768) -> None:
        self.engine = engine
        self.model_id = model_id
        self.device = device
        self.context_limit = context_limit
        self.loaded = True

    @classmethod
    def load(cls, spec: ModelSpec, model_dir: Path, device: str | None = None) -> "DeciderBackend":
        try:
            import torch
            metadata = json.loads((model_dir / "decider_config.json").read_text(encoding="utf-8"))
            manifest = json.loads((model_dir / "reflex-model.json").read_text(encoding="utf-8"))
            if manifest.get("model_id") != spec.model_id or manifest.get("revision") != spec.revision:
                raise ValueError("Pinned model metadata mismatch")
            Decider = _load_pinned_decider_package(model_dir)
            max_options = importlib.import_module("decider.prompt").MAX_OPTIONS
            if max_options != MAX_DECIDER_OPTIONS:
                raise ValueError("Pinned Decider option limit changed; update the Reflex protocol gate first")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            context_limit = metadata.get("max_state_tokens", 32768)
            if type(context_limit) is not int or context_limit < 1:
                raise ValueError("Invalid Decider context limit")
            # Eager slot-logit inference is the correctness baseline. CUDA
            # graphs are deliberately disabled until separately qualified.
            engine = Decider(str(model_dir), device=device, use_graphs=False)
            if Path(engine.m.lm.config.name_or_path).is_absolute():
                resolved = Path(engine.m.lm.config.name_or_path).resolve()
                if resolved != model_dir.resolve():
                    raise ValueError("Decider attempted to load weights outside its verified checkpoint")
            return cls(engine, spec.model_id, str(engine.dev), context_limit)
        except ImportError:
            raise ReflexError("environment_missing", "The isolated Decider environment is missing its pinned runtime packages.", 503)
        except ReflexError:
            raise
        except Exception:
            raise ReflexError("model_load_failed", "The selected pinned Decider checkpoint could not be loaded.", 503)

    @staticmethod
    def _upstream_questions(request: SystemOneRequest) -> dict[str, Any]:
        questions: dict[str, Any] = {}
        for question_id, question in request.questions.items():
            item: dict[str, Any] = {
                "type": question.kind,
                "instructions": question.instructions if question.instructions is not None else "Choose the answer best supported by the state.",
            }
            if question.criteria is not None:
                item["criteria"] = question.criteria
            questions[question_id] = item
        return questions

    def validate_request(self, request: SystemOneRequest) -> None:
        try:
            if any(len(question.option_keys) > MAX_DECIDER_OPTIONS for question in request.questions.values()):
                raise ReflexError("request_exceeds_backend_limits", "A Choice question exceeds the selected Decider option limit.", 413)
            # Render at most one token past the configured limit. The pinned
            # renderer may truncate to its cap, so limit+1 lets Reflex detect
            # overflow without allocating an unbounded token sequence.
            _, _, items = self.engine._system_one_items(
                request.state,
                self._upstream_questions(request),
                independent=True,
                max_state_tokens=self.context_limit + 1,
                layout="state_first",
            )
            if any(len(item.get("ids", ())) > self.context_limit for item in items):
                raise ReflexError("context_too_long", "The complete request exceeds the pinned Decider context limit.", 413)
        except ReflexError:
            raise
        except Exception:
            raise ReflexError("backend_capability_unavailable", "The pinned Decider renderer could not validate this request.", 503)

    def predict(self, request: SystemOneRequest) -> BackendResult:
        try:
            raw = self.engine.system_one(
                request.state,
                self._upstream_questions(request),
                independent=True,
                max_state_tokens=self.context_limit,
                layout="state_first",
            )
            answers = raw["answers"]
            if not isinstance(answers, dict) or set(answers) != set(request.questions):
                raise ValueError("unexpected answer IDs")
            distributions: dict[str, list[float]] = {}
            confidences: dict[str, float] = {}
            for question_id, question in request.questions.items():
                answer = answers[question_id]
                probs = answer.get("probabilities") or answer.get("probs")
                if question.kind == "noul" and probs is None:
                    positive = float(answer["noul"])
                    distributions[question_id] = [1.0 - positive, positive]
                elif question.kind == "choice" and isinstance(probs, dict):
                    distributions[question_id] = [float(probs[key]) for key in question.option_keys]
                elif question.kind == "score" and isinstance(probs, dict):
                    distributions[question_id] = [float(probs[str(index)]) for index in range(len(question.option_keys))]
                else:
                    raise ValueError("missing typed probabilities")
                if "confidence" in answer:
                    confidences[question_id] = float(answer["confidence"])
            usage = raw.get("usage", {})
            return BackendResult(
                distributions,
                confidences,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                self.model_id,
            )
        except ReflexError:
            raise
        except Exception:
            raise ReflexError("backend_failure", "The active Decider model could not complete this request.", 500)

    def close(self) -> None:
        if not self.loaded:
            return
        try:
            close = getattr(self.engine, "close", None)
            if callable(close):
                close()
            device = self.device
            self.engine = None
            if device.startswith("cuda"):
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.loaded = False
