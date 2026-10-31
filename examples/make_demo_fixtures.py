"""Write the recorded answers that examples/demo_system.py replays. Some answers are deliberately imperfect.

    uv run python examples/make_demo_fixtures.py
"""
from pathlib import Path

from evalharness.cases import load_suite
from evalharness.llm.base import Completion, Message
from evalharness.llm.mock import MockClient

ROOT = Path(__file__).resolve().parents[1]
SCRIPTED = {"q05": "Shakespeare", "q07": "nine", "q10": "The sky is blue"}


class Scripted:
    """Answers with the reference, except for the scripted imperfect answers."""

    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def complete(self, messages: list[Message], **kw) -> Completion:
        text = self.answers[messages[-1].content]
        return Completion(text=text, input_tokens=len(messages[-1].content.split()), output_tokens=len(text.split()),
                          latency_s=0.2, model=kw["model"], model_digest=None, cost_usd=0.0, billing="local")


def main() -> None:
    suite = load_suite(ROOT / "suites" / "demo")
    answers = {c.input["question"]: SCRIPTED.get(c.id, c.reference) for c in suite.cases}
    client = MockClient(ROOT / "examples" / "demo_fixtures", record_from=Scripted(answers))
    for case in suite.cases:
        client.complete([Message(role="user", content=case.input["question"])], model="demo-model", max_tokens=64)
    print("wrote", len(suite.cases), "fixtures")


if __name__ == "__main__":
    main()
