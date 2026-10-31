import json

import pytest
from pydantic import BaseModel

from evalharness.cases import Case
from evalharness.llm.base import Completion
from evalharness.metrics import (
    JudgeError,
    Output,
    exact_match,
    json_schema_valid,
    rubric_judge,
    token_f1,
)


def case(reference="x", **input_):
    return Case(id="c1", input=input_ or {"question": "q?"}, reference=reference)


def out(text="", data=None):
    return Output(text=text, data=data)


# ---- token F1 and exact match ------------------------------------------------------------------

def test_token_f1_squad_example():
    assert token_f1().score(case("cat sat on"), out("the cat sat")).value == pytest.approx(0.8)


def test_exact_match_ignores_articles_and_case():
    m = exact_match()
    assert m.score(case("The Eiffel Tower"), out("eiffel tower")).value == 1.0
    assert m.score(case("an apple"), out("A  APPLE!")).value == 1.0
    assert m.score(case("apple"), out("pear")).value == 0.0


def test_token_f1_empty_cases():
    f1 = token_f1()
    assert f1.score(case(""), out("")).value == 1.0
    assert f1.score(case("something"), out("")).value == 0.0
    assert f1.score(case(""), out("something")).value == 0.0
    assert f1.score(case("the"), out("a")).value == 1.0  # both are empty after removing articles


def test_token_f1_non_ascii_unchanged():
    assert token_f1().score(case("café 🙂"), out("café 🙂")).value == 1.0
    assert token_f1().score(case("café"), out("cafe")).value == 0.0
    assert exact_match().score(case("Zoë"), out("zoë")).value == 1.0


def test_token_f1_does_not_strip_articles_inside_words():
    assert exact_match().score(case("theatre"), out("theatre")).value == 1.0
    assert exact_match().score(case("banana"), out("banana")).value == 1.0


def test_text_metrics_take_the_best_of_several_references():
    ref = {"answers": ["Paris", "City of Paris"]}
    assert exact_match().score(case(ref), out("city of paris")).value == 1.0
    assert token_f1().score(case({"answer": "Paris"}), out("Paris")).value == 1.0


def test_text_metric_without_reference_is_an_error():
    with pytest.raises(ValueError, match="reference"):
        exact_match().score(Case(id="c", input={}), out("x"))


def test_metric_names():
    assert exact_match().name == "exact match" and token_f1().name == "token F1"


# ---- JSON schema validity ----------------------------------------------------------------------

class Person(BaseModel):
    name: str
    age: int


def test_json_schema_valid_reports_first_error():
    m = json_schema_valid(Person)
    ok = m.score(case(), out('{"name": "A", "age": 3}'))
    assert ok.value == 1.0
    bad = m.score(case(), out('{"name": "A", "age": "old"}'))
    assert bad.value == 0.0 and "age" in bad.detail


def test_json_schema_valid_uses_data_then_text_and_handles_garbage():
    m = json_schema_valid(Person)
    assert m.score(case(), out("ignored", data={"name": "A", "age": 3})).value == 1.0
    assert m.score(case(), out('```json\n{"name": "A", "age": 3}\n```')).value == 1.0
    none = m.score(case(), out("no json here"))
    assert none.value == 0.0 and "JSON" in none.detail


# ---- rubric judge ------------------------------------------------------------------------------

class StubJudge:
    def __init__(self, text, data=None):
        self.text, self.data, self.requests = text, data, []

    def complete(self, messages, **kw):
        self.requests.append((messages, kw))
        return Completion(text=self.text, data=self.data, input_tokens=5, output_tokens=5, latency_s=0.0,
                          model="judge", billing="local")


def judged(text, data=None, scale=(1, 5)):
    stub = StubJudge(text, data)
    metric = rubric_judge(stub, "judge-model", "Is the answer correct?", scale)
    return metric, stub


def test_rubric_judge_parses_structured_score():
    metric, stub = judged("", data={"score": 4, "reasoning": "close"})
    s = metric.score(case("Paris", question="Capital of France?"), out("Paris, France"))
    assert s.value == 4.0 and "close" in s.detail
    messages, kw = stub.requests[0]
    prompt = "\n".join(m.content for m in messages)
    for piece in ("Is the answer correct?", "Capital of France?", "Paris", "Paris, France", "1", "5"):
        assert piece in prompt
    assert kw["temperature"] == 0.0 and kw["seed"] == 0 and kw["response_schema"].__name__ == "JudgeVerdict"
    assert kw["model"] == "judge-model" and kw["max_tokens"] >= 1024


def test_rubric_judge_out_of_scale_is_error():
    metric, _ = judged('{"score": 7, "reasoning": "x"}')
    with pytest.raises(JudgeError, match="7"):
        metric.score(case(), out("a"))
    low, _ = judged('{"score": 0, "reasoning": "x"}')
    with pytest.raises(JudgeError):
        low.score(case(), out("a"))


def test_rubric_judge_parses_json_after_think_block_and_fences():
    reply = '<think>let me see</think>\n```json\n{"score": 3, "reasoning": "x"}\n```'
    metric, _ = judged(reply)
    assert metric.score(case(), out("a")).value == 3.0


def test_rubric_judge_unparseable_reply_is_a_judge_error():
    metric, _ = judged("I cannot decide.")
    with pytest.raises(JudgeError, match="parse"):
        metric.score(case(), out("a"))


def test_rubric_judge_scale_zero_to_two():
    metric, _ = judged('{"score": 0, "reasoning": "wrong"}', scale=(0, 2))
    assert metric.score(case(), out("a")).value == 0.0


def test_rubric_judge_without_reference_still_runs():
    metric, stub = judged('{"score": 5, "reasoning": "ok"}')
    metric.score(Case(id="c", input={"question": "q"}), out("a"))
    assert "(none)" in "\n".join(m.content for m in stub.requests[0][0])


def test_rubric_judge_prompt_is_deterministic():
    metric, stub = judged('{"score": 5, "reasoning": "ok"}')
    metric.score(case(), out("a"))
    metric.score(case(), out("a"))
    assert json.dumps([m.model_dump() for m in stub.requests[0][0]]) == \
        json.dumps([m.model_dump() for m in stub.requests[1][0]])


def test_rubric_judge_keeps_each_completion_for_latency_and_cost():
    metric, _ = judged('{"score": 4, "reasoning": "x"}')
    metric.score(case("a"), out("a"))
    other = Case(id="c2", input={"q": 1}, reference="b")
    metric.score(other, out("b"))
    assert set(metric.completions) == {"c1", "c2"}
    assert metric.completions["c1"].model == "judge" and metric.completions["c2"].latency_s == 0.0
