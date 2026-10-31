"""rubric_judge: an LLM grades an output against a rubric, on an integer scale."""
from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from evalharness.cases import Case
from evalharness.llm.base import Completion, LLMClient, Message, extract_json
from evalharness.metrics.base import JudgeError, Output, Score

SYSTEM = "You are a strict, consistent grader. Follow the rubric exactly and reply only with the requested JSON."


class JudgeVerdict(BaseModel):
    score: int
    reasoning: str = ""


def _reference_text(case: Case) -> str:
    ref = case.reference
    if ref is None:
        return "(none)"
    return ref if isinstance(ref, str) else json.dumps(ref, ensure_ascii=False)


class _RubricJudge:
    name = "rubric judge"

    def __init__(self, client: LLMClient, model: str, rubric: str, scale: tuple[int, int]):
        self.client, self.model, self.rubric, self.scale = client, model, rubric, scale
        self.completions: dict[str, Completion] = {}  # the judge's call for each case id, for latency and cost

    def _prompt(self, case: Case, output: Output) -> str:
        lo, hi = self.scale
        return (
            f"Rubric:\n{self.rubric}\n\n"
            f"Input: {json.dumps(case.input, ensure_ascii=False)}\n"
            f"Reference answer: {_reference_text(case)}\n"
            f"Answer to grade: {output.text}\n\n"
            f'Reply with JSON: {{"score": <integer from {lo} to {hi}>, "reasoning": "<one or two sentences>"}}')

    def score(self, case: Case, output: Output) -> Score:
        # max_tokens is generous: reasoning models spend a few hundred tokens before the JSON.
        completion = self.client.complete(
            [Message(role="system", content=SYSTEM), Message(role="user", content=self._prompt(case, output))],
            model=self.model, temperature=0.0, max_tokens=1024, response_schema=JudgeVerdict, seed=0)
        self.completions[case.id] = completion
        try:
            payload = completion.data if completion.data is not None else extract_json(completion.text)
            verdict = JudgeVerdict.model_validate(payload)
        except (ValueError, ValidationError) as e:
            raise JudgeError(f"could not parse the judge's reply: {completion.text[:120]!r} ({e})") from e
        lo, hi = self.scale
        if not lo <= verdict.score <= hi:
            raise JudgeError(f"judge score {verdict.score} is outside the scale {lo} to {hi}")
        return Score(value=float(verdict.score), detail=verdict.reasoning)


def rubric_judge(client: LLMClient, model: str, rubric: str, scale: tuple[int, int] = (1, 5)) -> _RubricJudge:
    return _RubricJudge(client, model, rubric, scale)
