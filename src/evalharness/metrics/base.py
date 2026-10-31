from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from evalharness.cases import Case
from evalharness.llm.base import Completion


class Output(BaseModel):
    text: str
    data: dict | None = None
    completion: Completion | None = None


class Score(BaseModel):
    value: float
    detail: str = ""


class Metric(Protocol):
    name: str

    def score(self, case: Case, output: Output) -> Score: ...


class JudgeError(ValueError):
    """The judge's reply could not be read, or its score is outside the scale. The runner records an error row."""
