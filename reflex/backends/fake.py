"""Deterministic backend for software tests and transport development."""

from __future__ import annotations

from reflex.backends.base import BackendResult
from reflex.schemas import SystemOneRequest


class FakeBackend:
    name = "fake"
    device = "cpu"
    loaded = True

    def __init__(self, distributions: dict[str, list[float]] | None = None) -> None:
        self.distributions = distributions or {}
        self.calls = 0

    def predict(self, request: SystemOneRequest) -> BackendResult:
        self.calls += 1
        result: dict[str, list[float]] = {}
        for question_id, question in request.questions.items():
            values = self.distributions.get(question_id)
            if values is None:
                values = [1.0 / len(question.option_keys)] * len(question.option_keys)
            result[question_id] = list(values)
        return BackendResult(distributions=result, model_name="fake")

    def close(self) -> None:
        self.loaded = False
