"""ClaudeCodeClient: `claude -p` with JSON event output, run clean (no tools, settings, plugins or MCP servers)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel

from evalharness.llm._proc import Runner, clean_env, render_prompt, resolve_exe, run_cli, run_in_empty_dir
from evalharness.llm.base import Completion, Message, ToolCall, ToolSpec, UsageLimitError, with_structured_retry
from evalharness.llm.cli import LIMIT_PATTERN, BillingError, UnexpectedToolsError, parse_jsonl, strict_schema

CLEAN_FLAGS = ["--output-format", "stream-json", "--verbose", "--tools", "", "--strict-mcp-config",
               "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence"]
PROVIDER = "claude_code"


@dataclass
class ClaudeRun:
    model: str
    version: str
    text: str
    structured: dict | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 1
    cost_usd: float | None = None


def parse_stream(stdout: str, *, returncode: int = 0, stderr: str = "", allowed_tools: set[str],
                 allowed_mcp: set[str]) -> ClaudeRun:
    """Check the start-up event against what the run allows, then read the result."""
    events = parse_jsonl(stdout)
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    if init is not None:
        source = init.get("apiKeySource")
        if source not in (None, "none"):
            raise BillingError(
                f"Claude Code is using an API key ({source}), which bills per token, not the subscription")
        extra_tools = sorted(set(init.get("tools") or []) - allowed_tools)
        extra_mcp = sorted({s["name"] for s in init.get("mcp_servers") or []} - allowed_mcp)
        extras = {"tools": extra_tools, "mcp servers": extra_mcp, "skills": init.get("skills") or [],
                  "plugins": [p if isinstance(p, str) else json.dumps(p) for p in init.get("plugins") or []]}
        found = {k: v for k, v in extras.items() if v}
        if found:
            raise UnexpectedToolsError(f"Claude Code started with items this run does not allow: {found}")
    for e in events:
        info = e.get("rate_limit_info") if e.get("type") == "rate_limit_event" else None
        if info and info.get("isUsingOverage"):
            raise BillingError("Claude Code is using overage (paid usage beyond the plan), so this run is not free")
        if info and info.get("status") == "rejected":
            reset = info.get("resetsAt")
            raise UsageLimitError(PROVIDER, datetime.fromtimestamp(reset, UTC) if reset else None)
    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if result is None:
        raise RuntimeError(f"claude produced no result event (exit code {returncode}): {stderr.strip()[:300]}")
    if result.get("is_error"):
        text = str(result.get("result", ""))
        if result.get("api_error_status") == 429 or LIMIT_PATTERN.search(text):
            raise UsageLimitError(PROVIDER, None, text)
        raise RuntimeError(f"claude reported an error: {text[:300]}")
    calls = []
    for e in events:
        if e.get("type") == "assistant":
            for block in e["message"]["content"]:
                if block.get("type") == "tool_use" and block["name"] != "StructuredOutput":
                    calls.append(ToolCall(id=block["id"], name=block["name"], arguments=dict(block.get("input", {}))))
    usage = result.get("usage", {})
    n_in = (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0))
    version = (init or {}).get("claude_code_version", "unknown")
    return ClaudeRun(model=(init or {}).get("model", "unknown"), version=f"claude-code {version}",
                     text=str(result.get("result", "")), structured=result.get("structured_output"),
                     tool_calls=calls, input_tokens=n_in, output_tokens=usage.get("output_tokens", 0),
                     turns=result.get("num_turns", 1), cost_usd=result.get("total_cost_usd"))


class ClaudeCodeClient:
    """Billing is the subscription: `cost_usd` is the API-equivalent cost Claude Code itself reports
    (`total_cost_usd`, which prices cache reads and writes), not money paid. `temperature`, `seed` and
    `max_tokens` are accepted and ignored because the CLI has no such options."""

    provider = PROVIDER

    def __init__(self, *, runner: Runner = run_cli, timeout_s: float = 600):
        self._runner = runner
        self._timeout_s = timeout_s

    def complete(self, messages: list[Message], *, model: str = "default", temperature: float = 0.0,
                 max_tokens: int = 512, tools: list[ToolSpec] | None = None,
                 response_schema: type[BaseModel] | None = None, seed: int | None = 0) -> Completion:
        if tools:
            raise ValueError("tools= is not supported by complete(); use run_cli_agent for tool use")
        system, prompt = render_prompt(messages)
        argv = [resolve_exe("claude"), "-p", *CLEAN_FLAGS]
        if model not in ("default", ""):
            argv += ["--model", model]
        if system:
            argv += ["--append-system-prompt", system]
        if response_schema is not None:
            argv += ["--json-schema", json.dumps(strict_schema(response_schema.model_json_schema()))]
        allowed = {"StructuredOutput"} if response_schema is not None else set()

        def once() -> Completion:
            start = time.perf_counter()
            proc = run_in_empty_dir(self._runner, argv, prompt, env=clean_env(), timeout_s=self._timeout_s)
            latency = time.perf_counter() - start
            run = parse_stream(proc.stdout, returncode=proc.returncode, stderr=proc.stderr, allowed_tools=allowed,
                               allowed_mcp=set())
            if response_schema is not None and run.structured is not None:
                run.text = json.dumps(run.structured)  # the validated structured_output field is the answer
            return Completion(text=run.text, tool_calls=[], data=None,
                              input_tokens=run.input_tokens, output_tokens=run.output_tokens, latency_s=latency,
                              ttft_s=None, model=run.model, model_digest=run.version, cost_usd=run.cost_usd,
                              billing="subscription")

        return with_structured_retry(once, response_schema)
