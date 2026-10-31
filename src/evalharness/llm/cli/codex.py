"""CodexClient: `codex exec --json`, run clean (read-only sandbox, no user config or rules, empty folder)."""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from evalharness.llm._proc import Runner, clean_env, render_prompt, resolve_exe, run_cli, run_in_empty_dir
from evalharness.llm.base import Completion, Message, ToolCall, ToolSpec, UsageLimitError, with_structured_retry
from evalharness.llm.cli import LIMIT_PATTERN, UnexpectedToolsError, parse_jsonl, strict_schema
from evalharness.llm.pricing import cost_usd

CLEAN_FLAGS = ["exec", "--json", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
               "-s", "read-only"]
PROVIDER = "codex_cli"
BUILTIN_TOOL_ITEMS = {"command_execution", "file_change", "web_search"}
DEFAULT_MODEL_LABEL = "codex-default (the CLI does not report the model)"


@dataclass
class CodexRun:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    messages: int = 1


def parse_events(stdout: str, *, returncode: int = 0, stderr: str = "",
                 allowed_mcp: dict[str, set[str]] | None = None) -> CodexRun:
    """Read the JSONL stream. Built-in tool use, or an MCP call outside `allowed_mcp`, fails the run."""
    allowed_mcp = allowed_mcp or {}
    events = parse_jsonl(stdout)
    texts: list[str] = []
    calls: list[ToolCall] = []
    n_in = n_out = 0
    completed = False
    errors: list[str] = []
    for e in events:
        kind = e.get("type")
        if kind in ("item.started", "item.completed"):
            item = e["item"]
            itype = item.get("type")
            if itype in BUILTIN_TOOL_ITEMS:
                raise UnexpectedToolsError(f"Codex used a built-in tool this run does not allow: {itype}")
            if itype == "mcp_tool_call":
                server, tool = item.get("server"), item.get("tool")
                if tool not in allowed_mcp.get(server, set()):
                    raise UnexpectedToolsError(f"Codex called MCP tool {server}/{tool}, which this run does not allow")
                if kind == "item.completed" and item.get("status") == "completed":
                    calls.append(ToolCall(id=item["id"], name=tool, arguments=dict(item.get("arguments") or {})))
            elif itype == "agent_message" and kind == "item.completed":
                texts.append(item.get("text", ""))
        elif kind == "turn.completed":
            completed = True
            usage = e.get("usage", {})
            n_in += usage.get("input_tokens", 0)
            n_out += usage.get("output_tokens", 0)
        elif kind == "error":
            errors.append(str(e.get("message", "")))
        elif kind == "turn.failed":
            errors.append(str((e.get("error") or {}).get("message", "")))
    if errors and not completed:
        message = errors[-1]
        if LIMIT_PATTERN.search(message):
            raise UsageLimitError(PROVIDER, None, message)
        raise RuntimeError(f"codex reported an error: {message[:300]}")
    if not completed:
        raise RuntimeError(f"codex produced no completed turn (exit code {returncode}): {stderr.strip()[:300]}")
    return CodexRun(text=texts[-1] if texts else "", tool_calls=calls, input_tokens=n_in, output_tokens=n_out,
                    messages=len(texts))


class CodexClient:
    """`cost_usd` is the API-equivalent from pricing.yaml (None when the model has no entry), not money paid.
    The CLI reports neither the model nor a seed or temperature: with `model="default"` the model is whatever the
    CLI picks, labelled as such; pass an explicit model to pin it."""

    provider = PROVIDER

    def __init__(self, *, runner: Runner = run_cli, timeout_s: float = 600):
        self._runner = runner
        self._timeout_s = timeout_s
        self._version: str | None = None

    def version(self) -> str:
        if self._version is None:
            exe = resolve_exe("codex")
            proc = run_in_empty_dir(self._runner, [exe, "--version"], "", env=clean_env(), timeout_s=30)
            self._version = proc.stdout.strip() or "codex-cli unknown"
        return self._version

    def identity(self, model: str) -> str:
        """Part of the response-cache key: a new CLI version is a new cache namespace."""
        return self.version()

    def complete(self, messages: list[Message], *, model: str = "default", temperature: float = 0.0,
                 max_tokens: int = 512, tools: list[ToolSpec] | None = None,
                 response_schema: type[BaseModel] | None = None, seed: int | None = 0) -> Completion:
        if tools:
            raise ValueError("tools= is not supported by complete(); use run_cli_agent for tool use")
        system, prompt = render_prompt(messages)
        if system:
            prompt = f"{system}\n\n{prompt}"
        exe = resolve_exe("codex")
        version = self.version()

        def once() -> Completion:
            argv = [exe, *CLEAN_FLAGS]
            if model not in ("default", ""):
                argv += ["-m", model]
            # the schema file goes in its own folder, not in the empty working folder
            with tempfile.TemporaryDirectory(prefix="evalh-schema-", ignore_cleanup_errors=True) as schema_dir:
                if response_schema is not None:
                    schema_path = Path(schema_dir) / "schema.json"
                    schema_path.write_text(json.dumps(strict_schema(response_schema.model_json_schema())),
                                           encoding="utf-8")
                    argv += ["--output-schema", str(schema_path)]
                argv.append("-")
                start = time.perf_counter()
                proc = run_in_empty_dir(self._runner, argv, prompt, env=clean_env(), timeout_s=self._timeout_s)
                latency = time.perf_counter() - start
            run = parse_events(proc.stdout, returncode=proc.returncode, stderr=proc.stderr)
            used = DEFAULT_MODEL_LABEL if model in ("default", "") else model
            return Completion(text=run.text, tool_calls=[], data=None, input_tokens=run.input_tokens,
                              output_tokens=run.output_tokens, latency_s=latency, ttft_s=None, model=used,
                              model_digest=version,
                              cost_usd=cost_usd(used, run.input_tokens, run.output_tokens, billing="subscription"),
                              billing="subscription")

        return with_structured_retry(once, response_schema)
