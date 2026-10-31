"""Exact match and token F1 with SQuAD answer normalisation."""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

from evalharness.cases import Case
from evalharness.metrics.base import Output, Score

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)  # ASCII only: accents and emoji are kept


def normalize_answer(text: str) -> str:
    """Lowercase, strip ASCII punctuation, drop the articles a, an and the, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _references(case: Case) -> list[str]:
    ref = case.reference
    if isinstance(ref, str):
        return [ref]
    if isinstance(ref, dict):
        if isinstance(ref.get("answers"), list):
            return [str(a) for a in ref["answers"]]
        if "answer" in ref:
            return [str(ref["answer"])]
    raise ValueError(f"case {case.id} has no usable reference (a string, or a dict with 'answer' or 'answers')")


def _f1(prediction: str, reference: str) -> float:
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


class _ExactMatch:
    name = "exact match"

    def score(self, case: Case, output: Output) -> Score:
        got = normalize_answer(output.text)
        return Score(value=float(any(got == normalize_answer(r) for r in _references(case))))


class _TokenF1:
    name = "token F1"

    def score(self, case: Case, output: Output) -> Score:
        return Score(value=max(_f1(output.text, r) for r in _references(case)))


def exact_match() -> _ExactMatch:
    return _ExactMatch()


def token_f1() -> _TokenF1:
    return _TokenF1()
