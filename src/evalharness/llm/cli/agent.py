"""Agent route: let a subscription CLI call tools from one MCP server, and report what it called."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from evalharness.llm._proc import Runner, clean_env, resolve_exe, run_cli, run_in_empty_dir
from evalharness.llm.base import ToolCall
from evalharness.llm.cli import UnexpectedToolsError
from evalharness.llm.cli.claude_code import CLEAN_FLAGS as CLAUDE_FLAGS
from evalharness.llm.cli.claude_code import parse_stream
from evalharness.llm.cli.codex import CLEAN_FLAGS as CODEX_FLAGS
from evalharness.llm.cli.codex import parse_events

SERVER = "tools"  # the name the one MCP server is registered under


class AgentTrace(BaseModel):
    tool_calls: list[ToolCall]
    final_text: str
    turns: int
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    cli_version: str


def run_cli_agent(cli: Literal["claude_code", "codex_cli", "gemini_cli"], prompt: str, *, mcp_server: list[str],
                  allowed_tools: list[str], max_turns: int = 6, workdir: Path | None = None,
                  runner: Runner = run_cli, timeout_s: float = 900) -> AgentTrace:
    """Register `mcp_server` (a command and its arguments, stdio transport) for one run and allow only the named
    tools. For Claude Code the built-in tools are off and the start-up event is checked before anything runs.
    Codex has no switch for its built-in tools: its event stream is checked after the process ends, and any
    built-in tool use or unlisted MCP tool raises UnexpectedToolsError (the tool may have run, read-only).

    Claude Code reports its turn count. Codex does not, so `turns` there is the number of tool calls plus one,
    and `max_turns` is not enforced for Codex."""
    if cli == "gemini_cli":
        from evalharness.llm.cli.gemini_cli import GeminiCLIClient
        GeminiCLIClient()
    if cli == "claude_code":
        return _claude(prompt, mcp_server, allowed_tools, max_turns, workdir, runner, timeout_s)
    if cli == "codex_cli":
        return _codex(prompt, mcp_server, allowed_tools, workdir, runner, timeout_s)
    raise ValueError(f"unknown cli {cli!r}")


def _claude(prompt, mcp_server, allowed_tools, max_turns, workdir, runner, timeout_s) -> AgentTrace:
    full = {f"mcp__{SERVER}__{t}" for t in allowed_tools}
    config = {"mcpServers": {SERVER: {"command": mcp_server[0], "args": mcp_server[1:]}}}
    argv = [resolve_exe("claude"), "-p", *CLAUDE_FLAGS, "--mcp-config", json.dumps(config),
            "--allowedTools", ",".join(sorted(full)), "--max-turns", str(max_turns)]
    env = clean_env({"ENABLE_TOOL_SEARCH": "false"})  # otherwise MCP tools are not loaded in the first turn
    start = time.perf_counter()
    proc = run_in_empty_dir(runner, argv, prompt, env=env, timeout_s=timeout_s, workdir=workdir)
    latency = time.perf_counter() - start
    run = parse_stream(proc.stdout, returncode=proc.returncode, stderr=proc.stderr, allowed_tools=full,
                       allowed_mcp={SERVER})
    prefix = f"mcp__{SERVER}__"
    for call in run.tool_calls:
        if call.name not in full:
            raise UnexpectedToolsError(
                f"Claude Code called {call.name!r}, which is not in allowed_tools {allowed_tools}")
    calls = [ToolCall(id=c.id, name=c.name.removeprefix(prefix), arguments=c.arguments) for c in run.tool_calls]
    return AgentTrace(tool_calls=calls, final_text=run.text, turns=run.turns, input_tokens=run.input_tokens,
                      output_tokens=run.output_tokens, latency_s=latency, model=run.model, cli_version=run.version)


def _codex(prompt, mcp_server, allowed_tools, workdir, runner, timeout_s) -> AgentTrace:
    from evalharness.llm.cli.codex import DEFAULT_MODEL_LABEL
    exe = resolve_exe("codex")
    version = run_in_empty_dir(runner, [exe, "--version"], "", env=clean_env(), timeout_s=30).stdout.strip()
    argv = [exe, *CODEX_FLAGS,
            "-c", f"mcp_servers.{SERVER}.command={json.dumps(mcp_server[0])}",
            "-c", f"mcp_servers.{SERVER}.args={json.dumps(mcp_server[1:])}",
            "-c", f'mcp_servers.{SERVER}.default_tools_approval_mode="approve"',
            "-c", f"mcp_servers.{SERVER}.enabled_tools={json.dumps(list(allowed_tools))}", "-"]
    start = time.perf_counter()
    proc = run_in_empty_dir(runner, argv, prompt, env=clean_env(), timeout_s=timeout_s, workdir=workdir)
    latency = time.perf_counter() - start
    run = parse_events(proc.stdout, returncode=proc.returncode, stderr=proc.stderr,
                       allowed_mcp={SERVER: set(allowed_tools)})
    return AgentTrace(tool_calls=run.tool_calls, final_text=run.text, turns=len(run.tool_calls) + 1,
                      input_tokens=run.input_tokens, output_tokens=run.output_tokens, latency_s=latency,
                      model=DEFAULT_MODEL_LABEL, cli_version=version or "codex-cli unknown")
