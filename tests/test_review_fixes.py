"""Tests added after the whole-branch review (each one failed before its fix)."""
import json

import pytest
from pydantic import BaseModel

from evalharness.gate import gate
from evalharness.llm.cache import CachingClient
from evalharness.llm.cli import BillingError, strict_schema
from evalharness.llm.cli.claude_code import ClaudeCodeClient
from evalharness.llm.cli.codex import CodexClient
from evalharness.metrics import Output, exact_match
from evalharness.metrics.judge import JudgeVerdict
from evalharness.runner import run
from test_cache_helpers import Counting
from test_gate import summary
from test_llm_cli import FIX, MSGS, FakeRunner, Verdict, stdout_of
from test_runner import echo, make_suite


@pytest.fixture(autouse=True)
def fake_clis_on_path(monkeypatch):
    """The tests never need the real claude or codex: pretend both are on PATH."""
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")


# ---- the cache key knows who answered -----------------------------------------------------------

def test_cache_key_includes_the_provider(tmp_path):
    class A(Counting):
        provider = "claude_code"

    class B(Counting):
        provider = "codex_cli"

    a, b = A(), B()
    CachingClient(a, tmp_path).complete(MSGS, model="default")
    CachingClient(b, tmp_path).complete(MSGS, model="default")  # same messages and model, another provider
    assert (a.calls, b.calls) == (1, 1)


def test_cache_key_uses_the_clients_own_identity(tmp_path):
    class Versioned(Counting):
        provider = "p"

        def __init__(self, version):
            super().__init__()
            self.version = version

        def identity(self, model):
            return self.version

    old, new = Versioned("1.0"), Versioned("2.0")
    CachingClient(old, tmp_path).complete(MSGS, model="m")
    CachingClient(new, tmp_path).complete(MSGS, model="m")
    again = Versioned("1.0")
    CachingClient(again, tmp_path).complete(MSGS, model="m")  # same version: a hit
    assert (old.calls, new.calls, again.calls) == (1, 1, 0)


def test_codex_identity_is_the_cli_version():
    assert CodexClient(runner=FakeRunner("")).identity("default") == "codex-cli 0.166.0"


# ---- the gate cannot pass by accident -----------------------------------------------------------

def test_gate_fails_when_the_baseline_has_only_an_error_rate(tmp_path):
    base = summary(tmp_path, "base", [("error rate", 1.0, "")])
    cand = summary(tmp_path, "cand", [("error rate", 1.0, "")])
    result = gate(base, cand, max_drop=0.1)
    assert not result.passed and "nothing to compare" in result.details[0]


def _edit_row(folder, **changes):
    data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    data["rows"][0].update(changes)
    (folder / "summary.json").write_text(json.dumps(data), encoding="utf-8")


def test_gate_fails_when_the_samples_differ_in_size(tmp_path):
    base = summary(tmp_path, "base", [("m", 0.8, "")])
    cand = summary(tmp_path, "cand", [("m", 0.8, "")])
    _edit_row(cand, n=1)
    result = gate(base, cand, max_drop=0.5)
    assert not result.passed and any("different n" in d for d in result.details)


def test_gate_fails_cleanly_on_a_null_value(tmp_path):
    base = summary(tmp_path, "base", [("m", 0.8, "")])
    cand = summary(tmp_path, "cand", [("m", 0.8, "")])
    _edit_row(cand, value=None)
    result = gate(base, cand, max_drop=0.5)
    assert not result.passed and any("not a number" in d for d in result.details)


# ---- strict schemas for the CLIs' structured output ---------------------------------------------

class Nested(BaseModel):
    note: str = "x"


class WithDefaults(BaseModel):
    score: int
    reasoning: str = ""
    title: str = "a property that is named title"
    inner: Nested | None = None


def test_strict_schema_requires_every_property_and_closes_objects():
    schema = strict_schema(WithDefaults.model_json_schema())
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["score", "reasoning", "title", "inner"]
    assert "title" in schema["properties"] and '"default"' not in json.dumps(schema)
    nested = schema["$defs"]["Nested"]
    assert nested["additionalProperties"] is False and nested["required"] == ["note"]


def test_cli_structured_output_sends_the_strict_schema_for_the_exact_model():
    expected = strict_schema(Verdict.model_json_schema())
    codex = FakeRunner(stdout_of("codex_structured"))
    CodexClient(runner=codex).complete(MSGS, model="default", response_schema=Verdict)
    assert codex.calls[0]["schema"] == expected
    claude = FakeRunner(stdout_of("claude_structured"))
    ClaudeCodeClient(runner=claude).complete(MSGS, model="default", response_schema=Verdict)
    argv = claude.calls[0]["argv"]
    assert json.loads(argv[argv.index("--json-schema") + 1]) == expected


def test_recorded_structured_fixtures_were_made_with_the_schema_the_judge_sends():
    expected = strict_schema(JudgeVerdict.model_json_schema())
    for name in ("claude_structured", "codex_structured"):
        recorded = json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))
        assert recorded["schema"] == expected, name


# ---- billing and routing guards -----------------------------------------------------------------

def test_cli_environment_drops_every_billing_or_routing_variable(monkeypatch):
    names = ["CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID",
             "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL", "OPENAI_BASE_URL", "GEMINI_API_KEY", "ANTHROPIC_AUTH_TOKEN"]
    for n in names:
        monkeypatch.setenv(n, "x")
    monkeypatch.setenv("CODEX_HOME", "C:/keep")
    runner = FakeRunner(stdout_of("claude_text"))
    ClaudeCodeClient(runner=runner).complete(MSGS, model="default")
    env = runner.calls[0]["env"]
    assert not [n for n in names if n in env] and env.get("CODEX_HOME") == "C:/keep"


def test_claude_run_with_an_api_key_source_is_refused_as_pay_per_token():
    lines = stdout_of("claude_text").splitlines()
    init = {**json.loads(lines[0]), "apiKeySource": "ANTHROPIC_API_KEY"}
    runner = FakeRunner("\n".join([json.dumps(init), *lines[1:]]))
    with pytest.raises(BillingError, match="ANTHROPIC_API_KEY"):
        ClaudeCodeClient(runner=runner).complete(MSGS, model="default")


def test_claude_run_using_overage_is_refused_as_paid():
    lines = stdout_of("claude_text").splitlines()
    event = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "isUsingOverage": True}}
    runner = FakeRunner("\n".join([lines[0], json.dumps(event), *lines[1:]]))
    with pytest.raises(BillingError, match="overage"):
        ClaudeCodeClient(runner=runner).complete(MSGS, model="default")


# ---- repeats and run configuration --------------------------------------------------------------

def test_system_taking_a_second_argument_gets_the_repeat_index(tmp_path):
    seen = []

    def with_repeat(case, repeat):
        seen.append((case.id, repeat))
        return Output(text=str(case.input["q"]))

    run(make_suite(2), with_repeat, [exact_match()], out_dir=tmp_path, title="t", repeats=3, concurrency=1)
    assert sorted(seen) == [(f"c{i}", r) for i in range(2) for r in range(3)]


def test_one_argument_system_with_repeats_is_flagged_in_the_log(tmp_path):
    result = run(make_suite(2), echo, [exact_match()], out_dir=tmp_path, title="t", repeats=2)
    assert "identical requests" in (result.path / "run_log.md").read_text(encoding="utf-8")
    single = run(make_suite(2), echo, [exact_match()], out_dir=tmp_path, title="t2")
    assert "identical requests" not in (single.path / "run_log.md").read_text(encoding="utf-8")


def test_run_config_notes_are_written_to_the_log(tmp_path):
    result = run(make_suite(2), echo, [exact_match()], out_dir=tmp_path, title="t",
                 notes={"temperature": 0, "seed": 0, "num_ctx": 8192})
    log = (result.path / "run_log.md").read_text(encoding="utf-8")
    assert "config: temperature = 0" in log and "config: num_ctx = 8192" in log


def test_ollama_identity_is_the_model_digest():
    import httpx
    import respx

    from evalharness.llm.ollama import OllamaClient
    with respx.mock(base_url="http://localhost:11434") as m:
        m.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": [
            {"name": "qwen3:1.7b", "digest": "abc"}]}))
        assert OllamaClient().identity("qwen3:1.7b") == "abc"
        assert OllamaClient().identity("not-pulled:1b") == ""
