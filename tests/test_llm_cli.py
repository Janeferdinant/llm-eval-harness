import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from evalharness.llm import cli as _cli_pkg  # noqa: F401  (package must exist)
from evalharness.llm._proc import ProcResult
from evalharness.llm.base import (
    ArmUnavailableError,
    Message,
    ToolSpec,
    UsageLimitError,
    get_client,
)
from evalharness.llm.cache import CachingClient
from evalharness.llm.cli import UnexpectedToolsError
from evalharness.llm.cli.agent import AgentTrace, run_cli_agent
from evalharness.llm.cli.claude_code import ClaudeCodeClient
from evalharness.llm.cli.codex import CodexClient
from evalharness.llm.cli.gemini_cli import GeminiCLIClient
from evalharness.llm.pricing import cost_usd

FIX = Path(__file__).parent / "fixtures" / "cli"
ECHO = str(FIX / "echo_mcp.py")
MSGS = [Message(role="user", content="Reply with the single word OK")]


def stdout_of(name: str) -> str:
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))["stdout"]


@pytest.fixture(autouse=True)
def fake_clis_on_path(monkeypatch):
    """The tests never need the real claude or codex: pretend both are on PATH."""
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")


class FakeRunner:
    """Replays recorded CLI output and remembers how it was called."""

    def __init__(self, *outputs: str, returncode: int = 0):
        self.outputs = list(outputs)
        self.returncode = returncode
        self.calls: list[dict] = []

    def __call__(self, argv, stdin, *, cwd, env, timeout_s):
        if argv[-1] == "--version":
            return ProcResult(stdout="codex-cli 0.166.0", stderr="", returncode=0)
        schema = None
        if "--output-schema" in argv:  # the schema file is deleted after the call, so read it now
            schema = json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text(encoding="utf-8"))
        self.calls.append({"argv": argv, "stdin": stdin, "cwd": cwd, "env": env, "schema": schema,
                           "cwd_files": sorted(os.listdir(cwd))})
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return ProcResult(stdout=out, stderr="", returncode=self.returncode)


class Verdict(BaseModel):
    score: int


# ---- command line and environment -------------------------------------------------------------

def test_cli_command_line_has_no_tools_and_empty_workdir():
    claude, codex = FakeRunner(stdout_of("claude_text")), FakeRunner(stdout_of("codex_text"))
    ClaudeCodeClient(runner=claude).complete(MSGS, model="default")
    CodexClient(runner=codex).complete(MSGS, model="default")
    c = claude.calls[0]
    assert c["cwd_files"] == [] and "-p" in c["argv"]
    assert c["argv"][c["argv"].index("--tools") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in c["argv"]
    assert c["argv"][c["argv"].index("--setting-sources") + 1] == ""
    x = codex.calls[0]
    assert x["cwd_files"] == [] and x["argv"][1] == "exec"
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--json"):
        assert flag in x["argv"]
    assert x["argv"][x["argv"].index("-s") + 1] == "read-only"


def test_cli_prompt_goes_through_stdin():
    long_prompt = "x" * 60_000
    for prompt in (long_prompt, "--help me please"):
        runner = FakeRunner(stdout_of("claude_text"))
        ClaudeCodeClient(runner=runner).complete([Message(role="user", content=prompt)], model="default")
        call = runner.calls[0]
        assert call["stdin"] == prompt and all(prompt not in a for a in call["argv"])
        runner = FakeRunner(stdout_of("codex_text"))
        CodexClient(runner=runner).complete([Message(role="user", content=prompt)], model="default")
        assert runner.calls[0]["stdin"] == prompt and runner.calls[0]["argv"][-1] == "-"


def test_cli_system_message_is_appended_not_replacing_the_cli_prompt():
    runner = FakeRunner(stdout_of("claude_text"))
    ClaudeCodeClient(runner=runner).complete([Message(role="system", content="Be brief."), *MSGS], model="default")
    argv = runner.calls[0]["argv"]
    assert argv[argv.index("--append-system-prompt") + 1] == "Be brief." and "--system-prompt" not in argv


def test_cli_resolved_with_shutil_which(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"C:/fake/bin/{name}.cmd")
    runner = FakeRunner(stdout_of("codex_text"))
    CodexClient(runner=runner).complete(MSGS, model="default")
    assert runner.calls[0]["argv"][0] == "C:/fake/bin/codex.cmd"


def test_cli_missing_executable_is_a_clear_error(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="codex"):
        CodexClient(runner=FakeRunner("")).complete(MSGS, model="default")


def test_cli_environment_has_no_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x-not-real")
    monkeypatch.setenv("CODEX_API_KEY", "x-not-real")
    monkeypatch.setenv("KEEP_ME", "1")
    runner = FakeRunner(stdout_of("claude_text"))
    ClaudeCodeClient(runner=runner).complete(MSGS, model="default")
    env = runner.calls[0]["env"]
    assert not any(k.endswith("API_KEY") for k in env) and env.get("KEEP_ME") == "1"


def test_cli_tools_argument_is_rejected_by_complete():
    tool = ToolSpec(name="t", description="d", parameters={"type": "object"})
    for client in (ClaudeCodeClient(runner=FakeRunner("")), CodexClient(runner=FakeRunner(""))):
        with pytest.raises(ValueError, match="run_cli_agent"):
            client.complete(MSGS, model="default", tools=[tool])


# ---- parsing recorded output -------------------------------------------------------------------

def test_cli_json_output_parsed_to_completion_claude():
    out = ClaudeCodeClient(runner=FakeRunner(stdout_of("claude_text"))).complete(MSGS, model="default")
    assert out.text == "OK" and out.billing == "subscription" and out.model.startswith("claude-")
    assert out.input_tokens > 0 and out.output_tokens > 0 and out.latency_s >= 0
    assert out.model_digest and out.model_digest.startswith("claude-code ")


def test_cli_json_output_parsed_to_completion_codex():
    out = CodexClient(runner=FakeRunner(stdout_of("codex_text"))).complete(MSGS, model="default")
    events = [json.loads(line) for line in stdout_of("codex_text").splitlines()]
    usage = next(e for e in events if e["type"] == "turn.completed")["usage"]
    assert out.text == "OK" and out.billing == "subscription"
    assert (out.input_tokens, out.output_tokens) == (usage["input_tokens"], usage["output_tokens"])
    assert out.model_digest and out.model_digest.startswith("codex-cli ")


def test_cli_structured_output_claude_uses_json_schema_flag():
    runner = FakeRunner(stdout_of("claude_structured"))
    out = ClaudeCodeClient(runner=runner).complete(MSGS, model="default", response_schema=Verdict)
    argv = runner.calls[0]["argv"]
    sent = json.loads(argv[argv.index("--json-schema") + 1])
    assert sent["additionalProperties"] is False and sent["required"] == ["score"]
    assert out.data == {"score": 3}


def test_cli_structured_output_codex_uses_output_schema_file_outside_workdir():
    runner = FakeRunner(stdout_of("codex_structured"))
    out = CodexClient(runner=runner).complete(MSGS, model="default", response_schema=Verdict)
    call = runner.calls[0]
    assert "--output-schema" in call["argv"] and call["cwd_files"] == [] and out.data == {"score": 3}


def test_cli_structured_output_retries_once_then_raises():
    runner = FakeRunner(stdout_of("codex_text"))  # replies "OK", not the schema
    with pytest.raises(ValueError, match="response_schema"):
        CodexClient(runner=runner).complete(MSGS, model="default", response_schema=Verdict)
    assert len(runner.calls) == 2


def test_subscription_cost_marked_api_equivalent():
    claude = ClaudeCodeClient(runner=FakeRunner(stdout_of("claude_text"))).complete(MSGS, model="default")
    reported = [json.loads(line) for line in stdout_of("claude_text").splitlines()][-1]["total_cost_usd"]
    assert claude.billing == "subscription" and claude.cost_usd == pytest.approx(reported)
    codex = CodexClient(runner=FakeRunner(stdout_of("codex_text"))).complete(MSGS, model="gpt-6.1-sol")
    assert codex.cost_usd == cost_usd("gpt-6.1-sol", codex.input_tokens, codex.output_tokens, billing="subscription")
    assert codex.cost_usd > 0
    unknown = CodexClient(runner=FakeRunner(stdout_of("codex_text"))).complete(MSGS, model="default")
    assert unknown.cost_usd is None and unknown.model.startswith("codex-default")


# ---- guards ------------------------------------------------------------------------------------

def test_cli_arm_rejects_unexpected_tools_or_mcp():
    lines = stdout_of("claude_text").splitlines()
    init = json.loads(lines[0])
    assert init["subtype"] == "init"
    with_tool = [json.dumps({**init, "tools": ["Bash"]}), *lines[1:]]
    with pytest.raises(UnexpectedToolsError, match="Bash"):
        ClaudeCodeClient(runner=FakeRunner("\n".join(with_tool))).complete(MSGS, model="default")
    with_mcp = [json.dumps({**init, "mcp_servers": [{"name": "evil", "status": "connected"}]}), *lines[1:]]
    with pytest.raises(UnexpectedToolsError, match="evil"):
        ClaudeCodeClient(runner=FakeRunner("\n".join(with_mcp))).complete(MSGS, model="default")
    with_skill = [json.dumps({**init, "skills": ["commit"]}), *lines[1:]]
    with pytest.raises(UnexpectedToolsError, match="commit"):
        ClaudeCodeClient(runner=FakeRunner("\n".join(with_skill))).complete(MSGS, model="default")


def test_codex_arm_rejects_builtin_tool_use():
    events = [{"type": "thread.started", "thread_id": "x"},
              {"type": "item.completed", "item": {"id": "i", "type": "command_execution", "command": "ls"}},
              {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}]
    with pytest.raises(UnexpectedToolsError, match="command_execution"):
        CodexClient(runner=FakeRunner("\n".join(json.dumps(e) for e in events))).complete(MSGS, model="default")


def test_cli_nonzero_exit_without_events_is_an_error_with_stderr():
    runner = FakeRunner("", returncode=1)
    with pytest.raises(RuntimeError, match="exit code 1"):
        ClaudeCodeClient(runner=runner).complete(MSGS, model="default")


# ---- usage limits ------------------------------------------------------------------------------

@pytest.mark.parametrize("name,client_cls", [("claude_usage_limit", ClaudeCodeClient),
                                              ("codex_usage_limit", CodexClient)])
def test_usage_limit_raises_usage_limit_error(name, client_cls):
    with pytest.raises(UsageLimitError) as e:
        client_cls(runner=FakeRunner(stdout_of(name))).complete(MSGS, model="default")
    assert e.value.provider in {"claude_code", "codex_cli"}


def test_claude_usage_limit_reports_reset_time():
    with pytest.raises(UsageLimitError) as e:
        ClaudeCodeClient(runner=FakeRunner(stdout_of("claude_usage_limit"))).complete(MSGS, model="default")
    assert e.value.retry_after is not None and e.value.retry_after.year >= 2026


def test_usage_limit_stops_and_resumes(tmp_path):
    ok, limit = stdout_of("claude_text"), stdout_of("claude_usage_limit")
    runner = FakeRunner(ok, ok, ok, limit)  # three answers, then the limit repeats
    client = CachingClient(ClaudeCodeClient(runner=runner), tmp_path, salt="claude-code 2.1.195")
    prompts = [[Message(role="user", content=f"question {i}")] for i in range(6)]
    done = []
    with pytest.raises(UsageLimitError):
        for p in prompts:
            done.append(client.complete(p, model="default"))
    assert len(done) == 3 and len(runner.calls) == 4

    resumed = FakeRunner(ok)  # window is fresh
    client = CachingClient(ClaudeCodeClient(runner=resumed), tmp_path, salt="claude-code 2.1.195")
    for p in prompts:
        client.complete(p, model="default")
    assert len(resumed.calls) == 3 and client.hits == 3 and client.misses == 3


# ---- agent route -------------------------------------------------------------------------------

def test_cli_agent_trace_lists_mcp_tool_calls_claude():
    runner = FakeRunner(stdout_of("claude_agent"))
    trace = run_cli_agent("claude_code", "call echo", mcp_server=[sys.executable, ECHO], allowed_tools=["echo"],
                          runner=runner)
    assert isinstance(trace, AgentTrace)
    assert [(t.name, t.arguments) for t in trace.tool_calls] == [("echo", {"text": "ping"})]
    assert trace.final_text == "echo: ping" and trace.turns == 2 and trace.cli_version.startswith("claude-code")
    argv = runner.calls[0]["argv"]
    assert argv[argv.index("--allowedTools") + 1] == "mcp__tools__echo" and "--strict-mcp-config" in argv
    assert json.loads(argv[argv.index("--mcp-config") + 1])["mcpServers"]["tools"]["command"] == sys.executable
    assert runner.calls[0]["env"]["ENABLE_TOOL_SEARCH"] == "false"


def test_cli_agent_trace_lists_mcp_tool_calls_codex():
    runner = FakeRunner(stdout_of("codex_agent"))
    trace = run_cli_agent("codex_cli", "call echo", mcp_server=[sys.executable, ECHO], allowed_tools=["echo"],
                          runner=runner)
    assert [(t.name, t.arguments) for t in trace.tool_calls] == [("echo", {"text": "ping"})]
    assert trace.final_text == "echo: ping" and trace.turns == 2 and trace.input_tokens > 0
    argv = " ".join(runner.calls[0]["argv"])
    assert "mcp_servers.tools.enabled_tools" in argv and "default_tools_approval_mode" in argv


def test_cli_agent_rejects_tool_outside_allowed_list():
    with pytest.raises(UnexpectedToolsError, match="echo"):
        run_cli_agent("claude_code", "x", mcp_server=[sys.executable, ECHO], allowed_tools=["other"],
                      runner=FakeRunner(stdout_of("claude_agent")))
    with pytest.raises(UnexpectedToolsError, match="echo"):
        run_cli_agent("codex_cli", "x", mcp_server=[sys.executable, ECHO], allowed_tools=["other"],
                      runner=FakeRunner(stdout_of("codex_agent")))


# ---- the dropped arm and the registry ----------------------------------------------------------

def test_gemini_cli_arm_is_unavailable_with_the_reason():
    with pytest.raises(ArmUnavailableError, match="Antigravity"):
        GeminiCLIClient()
    with pytest.raises(ArmUnavailableError, match="Antigravity"):
        get_client("gemini_cli")


def test_get_client_builds_the_listed_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    for provider in ("ollama", "mock", "openai_compat", "openai", "anthropic", "gemini", "claude_code", "codex_cli"):
        kw = {"fixture_dir": "."} if provider == "mock" else {"base_url": "http://localhost:8000/v1"} \
            if provider == "openai_compat" else {}
        assert get_client(provider, **kw).complete is not None
    with pytest.raises(ValueError, match="unknown provider"):
        get_client("nope")
