"""Backend protocol and normalized inference result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from reflex.schemas import SystemOneRequest


@dataclass(slots=True)
class BackendResult:
    distributions: dict[str, list[float]]
    confidences: dict[str, float] = field(default_factory=dict)
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_name: str | None = None


class DecisionBackend(Protocol):
    name: str
    device: str
    loaded: bool

    def predict(self, request: SystemOneRequest) -> BackendResult: ...

    def close(self) -> None: ...
