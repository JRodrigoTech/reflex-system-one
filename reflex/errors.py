"""Application errors with safe public messages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReflexError(Exception):
    code: str
    message: str
    status_code: int = 422

    def __str__(self) -> str:
        return self.message


def invalid_request(message: str) -> ReflexError:
    return ReflexError("invalid_request", message, 422)
