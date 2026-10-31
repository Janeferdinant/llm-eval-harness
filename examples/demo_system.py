"""A demo system for the demo suite. It replays recorded answers, so it needs no model and no network:

    evalh run suites/demo --system examples.demo_system:answer --metrics exact_match,token_f1 --out runs/
"""
from pathlib import Path

from evalharness.cases import Case
from evalharness.llm.base import Message
from evalharness.llm.mock import MockClient
from evalharness.metrics import Output

MODEL = "demo-model"
_client = MockClient(Path(__file__).parent / "demo_fixtures")


def answer(case: Case) -> Output:
    completion = _client.complete([Message(role="user", content=case.input["question"])], model=MODEL, max_tokens=64)
    return Output(text=completion.text, completion=completion)
