"""Owned System One request validation and normalization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from reflex.errors import invalid_request

MAX_QUESTIONS = 64
MAX_TOTAL_OPTIONS = 1024
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
_JSON_SCALARS = (str, int, float, bool, type(None))


def _json_value(value: Any, label: str) -> None:
    if isinstance(value, _JSON_SCALARS):
        if isinstance(value, float) and not math.isfinite(value):
            raise invalid_request(f"{label} must contain finite JSON values.")
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, label)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values():
            _json_value(item, label)
        return
    raise invalid_request(f"{label} must be a JSON string, object, or array.")


def canonical_legend_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    kind: str
    instructions: str | dict[str, Any] | list[Any] | None
    criteria: dict[str, Any] | list[Any] | None

    @property
    def option_keys(self) -> tuple[str, ...]:
        if self.kind == "noul":
            return ("false", "true")
        if self.kind == "choice":
            assert isinstance(self.criteria, dict)
            return tuple(self.criteria.keys())
        assert isinstance(self.criteria, list)
        return tuple(str(i) for i in range(len(self.criteria)))

    @property
    def legend(self) -> dict[str, str]:
        if self.kind == "score":
            assert isinstance(self.criteria, list)
            return {str(i): canonical_legend_value(item) for i, item in enumerate(self.criteria)}
        return {}


@dataclass(frozen=True, slots=True)
class SystemOneRequest:
    model: str | None
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question]


class NoulAnswer(TypedDict):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(TypedDict):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(TypedDict):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(TypedDict):
    model: str
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    usage: Usage


class ErrorDetail(TypedDict):
    code: str
    message: str
    request_id: NotRequired[str]


class ErrorResponse(TypedDict):
    error: ErrorDetail


def parse_request(value: Any) -> SystemOneRequest:
    if not isinstance(value, dict):
        raise invalid_request("Request body must be a JSON object.")
    state = value.get("state", ...)
    if state is ... or state is None or not isinstance(state, (str, dict, list)):
        raise invalid_request("state is required and must be a string, object, or array.")
    _json_value(state, "state")
    raw_questions = value.get("questions")
    if not isinstance(raw_questions, dict) or not raw_questions:
        raise invalid_request("questions must be a non-empty object.")
    if len(raw_questions) > MAX_QUESTIONS:
        raise invalid_request("Request contains too many questions.")
    questions: dict[str, Question] = {}
    total_options = 0
    for question_id, raw in raw_questions.items():
        if not isinstance(question_id, str) or not question_id or len(question_id) > 256:
            raise invalid_request("Question IDs must be non-empty strings of at most 256 characters.")
        if not isinstance(raw, dict):
            raise invalid_request(f"Question {question_id!r} must be an object.")
        kind = raw.get("type")
        if kind not in {"noul", "choice", "score"}:
            raise invalid_request(f"Question {question_id!r} has an unsupported type.")
        instructions = raw.get("instructions")
        if instructions is not None:
            if not isinstance(instructions, (str, dict, list)):
                raise invalid_request(f"Question {question_id!r} instructions must be a string, object, or array.")
            _json_value(instructions, f"Question {question_id!r} instructions")
        criteria = raw.get("criteria")
        if kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or not set(criteria).issubset({"true", "false"}):
                    raise invalid_request(f"Question {question_id!r} Noul criteria may contain only true and false.")
                for description in criteria.values():
                    if not isinstance(description, (str, dict, list)):
                        raise invalid_request(f"Question {question_id!r} Noul criteria values must be strings, objects, or arrays.")
                    _json_value(description, f"Question {question_id!r} criteria")
            option_count = 2
        elif kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise invalid_request(f"Question {question_id!r} Choice criteria must contain at least two options.")
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise invalid_request(f"Question {question_id!r} has too many Choice options.")
            for key, description in criteria.items():
                if not isinstance(key, str) or not key:
                    raise invalid_request(f"Question {question_id!r} Choice keys must be non-empty strings.")
                if description is not None and not isinstance(description, (str, dict, list)):
                    raise invalid_request(f"Question {question_id!r} Choice criteria values must be strings, objects, arrays, or null.")
                _json_value(description, f"Question {question_id!r} criteria")
            option_count = len(criteria)
        else:
            if not isinstance(criteria, list) or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
                raise invalid_request(f"Question {question_id!r} Score criteria must contain 2 to 10 ordered entries.")
            for description in criteria:
                if not isinstance(description, (str, dict, list)):
                    raise invalid_request(f"Question {question_id!r} Score criteria values must be strings, objects, or arrays.")
                _json_value(description, f"Question {question_id!r} criteria")
            option_count = len(criteria)
        total_options += option_count
        questions[question_id] = Question(question_id, kind, instructions, criteria)
    if total_options > MAX_TOTAL_OPTIONS:
        raise invalid_request("Request contains too many total options.")
    model = value.get("model")
    if model is not None and (not isinstance(model, str) or not model or len(model) > 128):
        raise invalid_request("model must be a non-empty string when provided.")
    return SystemOneRequest(model, state, questions)
