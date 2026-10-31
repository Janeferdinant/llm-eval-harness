import json
import random
import threading
import time

import pytest

from evalharness import runlog
from evalharness.cases import Case, Suite
from evalharness.llm.base import Completion, Message, UsageLimitError
from evalharness.llm.cache import CachingClient
from evalharness.metrics import JudgeError, Output, Score, exact_match, token_f1
from evalharness.runlog import BusyMachineError, MachineLoad, SummaryRow
from evalharness.runner import run


def make_suite(n=5):
    return Suite(name="demo", version="1",
                 cases=[Case(id=f"c{i}", input={"q": i}, reference=str(i)) for i in range(n)])


def echo(case):
    return Output(text=str(case.input["q"]))


def rows_of(result):
    return [json.loads(line) for line in (result.path / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def test_run_writes_summary_and_log(tmp_path):
    result = run(make_suite(), echo, [exact_match(), token_f1()], out_dir=tmp_path, title="Demo run")
    summary = json.loads((result.path / "summary.json").read_text(encoding="utf-8"))
    names = [r["name"] for r in summary["rows"]]
    assert names == ["exact match", "token F1", "error rate"]
    for row in summary["rows"]:
        SummaryRow(**row)
    em = summary["rows"][0]
    assert em["value"] == 1.0 and em["n"] == 5 and em["ci_low"] == 1.0 and em["ci_high"] == 1.0
    assert summary["status"] == "Verified" and summary["title"] == "Demo run"
    log = (result.path / "run_log.md").read_text(encoding="utf-8")
    assert "demo" in log and "5 cases" in log
    assert result.n_errors == 0 and result.scores["exact match"] == [1.0] * 5


def test_status_reported_is_passed_through(tmp_path):
    result = run(make_suite(2), echo, [exact_match()], out_dir=tmp_path, title="t", status="Reported")
    assert result.summary["status"] == "Reported"


def test_rerun_uses_cache_no_new_calls(tmp_path):
    class Inner:
        calls = 0

        def complete(self, messages, **kw):
            Inner.calls += 1
            return Completion(text=messages[-1].content, input_tokens=1, output_tokens=1, latency_s=0.0,
                              model="m", billing="local")

    client = CachingClient(Inner(), tmp_path / ".cache")

    def system(case):
        c = client.complete([Message(role="user", content=str(case.input["q"]))], model="m")
        return Output(text=c.text, completion=c)

    run(make_suite(), system, [exact_match()], out_dir=tmp_path / "runs", title="first")
    assert (client.hits, client.misses, Inner.calls) == (0, 5, 5)
    second = run(make_suite(), system, [exact_match()], out_dir=tmp_path / "runs", title="first")
    assert (client.hits, client.misses, Inner.calls) == (5, 5, 5) and second.scores["exact match"] == [1.0] * 5


def test_failed_case_recorded_not_dropped(tmp_path):
    def flaky(case):
        if case.id == "c2":
            raise ValueError("boom")
        return echo(case)

    result = run(make_suite(), flaky, [exact_match()], out_dir=tmp_path, title="t")
    rows = rows_of(result)
    assert len(rows) == 5 and result.n_errors == 1
    bad = next(r for r in rows if r["case_id"] == "c2")
    assert "boom" in bad["error"] and bad["scores"]["exact match"] is None
    summary = {r["name"]: r for r in result.summary["rows"]}
    assert summary["exact match"]["n"] == 4 and summary["error rate"]["value"] == pytest.approx(0.2)
    assert summary["error rate"]["n"] == 5
    assert result.scores["exact match"][2] is None


def test_metric_exception_records_error_and_keeps_the_other_metrics(tmp_path):
    class Picky:
        name = "picky"

        def score(self, case, output):
            if case.id == "c1":
                raise JudgeError("cannot parse")
            return Score(value=1.0)

    result = run(make_suite(3), echo, [exact_match(), Picky()], out_dir=tmp_path, title="t")
    rows = {r["case_id"]: r for r in rows_of(result)}
    assert rows["c1"]["scores"]["exact match"] == 1.0 and rows["c1"]["scores"]["picky"] is None
    assert "cannot parse" in rows["c1"]["error"] and result.n_errors == 1


def test_all_cases_failing_still_writes_a_summary(tmp_path):
    def always(case):
        raise RuntimeError("down")

    result = run(make_suite(3), always, [exact_match()], out_dir=tmp_path, title="t")
    rows = {r["name"]: r for r in result.summary["rows"]}
    assert rows["error rate"]["value"] == 1.0 and "exact match" not in rows


def test_results_jsonl_follows_suite_order_under_concurrency(tmp_path):
    rng = random.Random(1)
    delays = {f"c{i}": rng.random() * 0.03 for i in range(20)}

    def slow(case):
        time.sleep(delays[case.id])
        return echo(case)

    result = run(make_suite(20), slow, [exact_match()], out_dir=tmp_path, title="t", concurrency=8)
    assert [r["case_id"] for r in rows_of(result)] == [f"c{i}" for i in range(20)]


def test_concurrency_limit_is_respected(tmp_path):
    active, peak, lock = 0, 0, threading.Lock()

    def tracked(case):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)  # long enough that workers overlap even on a busy machine
        with lock:
            active -= 1
        return echo(case)

    run(make_suite(12), tracked, [exact_match()], out_dir=tmp_path, title="t", concurrency=3)
    assert 1 < peak <= 3


def test_usage_limit_in_run_saves_progress_and_raises(tmp_path):
    def limited(case):
        if case.id == "c3":
            raise UsageLimitError("claude_code")
        return echo(case)

    with pytest.raises(UsageLimitError):
        run(make_suite(6), limited, [exact_match()], out_dir=tmp_path, title="t", concurrency=1)
    folder = next(tmp_path.iterdir())
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["case_id"] for r in rows] == ["c0", "c1", "c2"] and all(r["error"] is None for r in rows)
    assert "usage limit" in (folder / "run_log.md").read_text(encoding="utf-8")
    assert not (folder / "summary.json").exists()


def test_repeats_multiply_rows_but_n_counts_cases(tmp_path):
    result = run(make_suite(4), echo, [exact_match()], out_dir=tmp_path, title="t", repeats=3)
    rows = rows_of(result)
    assert len(rows) == 12 and sorted({r["repeat"] for r in rows}) == [0, 1, 2]
    assert result.summary["rows"][0]["n"] == 4


def test_repeats_are_averaged_within_a_case(tmp_path):
    seen = {}

    def alternating(case):
        seen[case.id] = seen.get(case.id, 0) + 1
        return Output(text=str(case.input["q"]) if seen[case.id] % 2 else "wrong")

    result = run(make_suite(3), alternating, [exact_match()], out_dir=tmp_path, title="t", repeats=2, concurrency=1)
    assert result.scores["exact match"] == [0.5, 0.5, 0.5]


def test_summary_includes_latency_and_cost_when_completions_are_present(tmp_path):
    def with_completion(case):
        c = Completion(text=str(case.input["q"]), input_tokens=10, output_tokens=5, latency_s=0.5, model="m",
                       model_digest="abc123", cost_usd=0.01, billing="subscription")
        return Output(text=c.text, completion=c)

    result = run(make_suite(4), with_completion, [exact_match()], out_dir=tmp_path, title="t")
    rows = {r["name"]: r for r in result.summary["rows"]}
    assert rows["latency (s)"]["value"] == pytest.approx(0.5) and rows["latency (s)"]["unit"] == "s"
    assert rows["cost per case, API-equivalent (USD)"]["value"] == pytest.approx(0.01)
    log = (result.path / "run_log.md").read_text(encoding="utf-8")
    assert "abc123" in log and "subscription" in log


def test_cost_row_is_left_out_when_any_cost_is_unknown(tmp_path):
    def unknown_cost(case):
        c = Completion(text=str(case.input["q"]), input_tokens=1, output_tokens=1, latency_s=0.1, model="m",
                       cost_usd=None, billing="api")
        return Output(text=c.text, completion=c)

    result = run(make_suite(3), unknown_cost, [exact_match()], out_dir=tmp_path, title="t")
    names = [r["name"] for r in result.summary["rows"]]
    assert "latency (s)" in names and not any("cost" in n for n in names)


def test_duplicate_metric_names_rejected(tmp_path):
    with pytest.raises(ValueError, match="exact match"):
        run(make_suite(2), echo, [exact_match(), exact_match()], out_dir=tmp_path, title="t")


def test_run_refuses_a_busy_machine_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "machine_load", lambda sample_s=5.0: MachineLoad(cpu_pct=80.0, gpu_pct=None))
    with pytest.raises(BusyMachineError):
        run(make_suite(2), echo, [exact_match()], out_dir=tmp_path, title="t")
    result = run(make_suite(2), echo, [exact_match()], out_dir=tmp_path, title="t", allow_busy=True)
    assert "busy machine" in result.summary["title"]


def test_a_metrics_own_model_calls_are_recorded_per_row(tmp_path):
    class Judged:
        name = "judged"

        def __init__(self):
            self.completions = {}

        def score(self, case, output):
            self.completions[case.id] = Completion(text="x", input_tokens=7, output_tokens=3, latency_s=1.25,
                                                    model="j", model_digest="v1", cost_usd=0.002,
                                                    billing="subscription")
            return Score(value=1.0)

    result = run(make_suite(2), echo, [Judged()], out_dir=tmp_path, title="t")
    row = rows_of(result)[0]
    call = row["metric_calls"]["judged"]
    assert call == {"latency_s": 1.25, "cost_usd": 0.002, "input_tokens": 7, "output_tokens": 3, "model": "j",
                    "model_digest": "v1", "billing": "subscription"}
    assert all(r["metric_calls"]["judged"] for r in rows_of(result))
