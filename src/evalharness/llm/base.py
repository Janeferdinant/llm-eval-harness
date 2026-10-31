"""Shared types for every LLM client, the request hash, and helpers for structured output."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class Completion(BaseModel):
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    data: dict | None = None  # parsed structured output
    input_tokens: int
    output_tokens: int
    latency_s: float
    ttft_s: float | None = None
    model: str
    model_digest: str | None = None
    cost_usd: float | None = None
    billing: Literal["local", "subscription", "api"]  # subscription: cost_usd is API-equivalent, not paid


class LLMClient(Protocol):
    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 tools: list[ToolSpec] | None = None, response_schema: type[BaseModel] | None = None,
                 seed: int | None = 0) -> Completion: ...


class MissingFixtureError(Exception):
    def __init__(self, request_hash: str):
        super().__init__(f"no recorded fixture for request {request_hash}")
        self.request_hash = request_hash


class UsageLimitError(Exception):
    def __init__(self, provider: str, retry_after: datetime | None = None, message: str = ""):
        super().__init__(message or f"{provider}: usage limit reached"
                         + (f", retry after {retry_after.isoformat()}" if retry_after else ""))
        self.provider = provider
        self.retry_after = retry_after


class ArmUnavailableError(Exception):
    """A provider arm that was deliberately not built, with the reason."""


def request_hash(messages: list[Message], *, model: str, temperature: float, max_tokens: int,
                 tools: list[ToolSpec] | None, response_schema: type[BaseModel] | None,
                 seed: int | None, salt: str = "") -> str:
    """sha256 of the canonical JSON of everything that can change a model's answer."""
    canonical = {
        "messages": [m.model_dump() for m in messages],
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": [t.model_dump() for t in tools] if tools else None,
        "response_schema": response_schema.model_json_schema() if response_schema else None,
        "seed": seed,
        "salt": salt,
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove reasoning blocks. Some models emit the reasoning without an opening tag."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> dict:
    """First JSON object in a reply, ignoring reasoning blocks, code fences and surrounding prose."""
    text = strip_think(text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"no JSON object found in reply: {text[:80]!r}")


def parse_structured(text: str, schema: type[BaseModel]) -> dict:
    """Validated structured output, or ValueError mentioning response_schema."""
    try:
        return schema.model_validate(extract_json(text)).model_dump()
    except (ValueError, ValidationError) as e:
        raise ValueError(f"reply does not match response_schema {schema.__name__}: {e}") from e


def with_structured_retry(call: Callable[[], Completion], schema: type[BaseModel] | None) -> Completion:
    """Run `call`; when a schema is given, validate the reply into `data`, with one retry on a bad reply."""
    attempts = 2 if schema is not None else 1
    for attempt in range(attempts):
        out = call()
        if schema is None:
            return out
        try:
            out.data = parse_structured(out.text, schema)
            return out
        except ValueError:
            if attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")


def get_client(provider: str, **kw: Any) -> LLMClient:
    """Build a client by provider name. Imports are lazy so unused SDKs are never loaded."""
    if provider == "mock":
        from evalharness.llm.mock import MockClient
        return MockClient(**kw)
    if provider == "ollama":
        from evalharness.llm.ollama import OllamaClient
        return OllamaClient(**kw)
    if provider == "openai_compat":
        from evalharness.llm.openai_compat import OpenAICompatClient
        return OpenAICompatClient(**kw)
    if provider == "openai":
        from evalharness.llm.openai import OpenAIClient
        return OpenAIClient(**kw)
    if provider == "anthropic":
        from evalharness.llm.anthropic import AnthropicClient
        return AnthropicClient(**kw)
    if provider == "gemini":
        from evalharness.llm.gemini import GeminiClient
        return GeminiClient(**kw)
    if provider == "claude_code":
        from evalharness.llm.cli.claude_code import ClaudeCodeClient
        return ClaudeCodeClient(**kw)
    if provider == "codex_cli":
        from evalharness.llm.cli.codex import CodexClient
        return CodexClient(**kw)
    if provider == "gemini_cli":
        from evalharness.llm.cli.gemini_cli import GeminiCLIClient
        return GeminiCLIClient(**kw)
    raise ValueError(f"unknown provider {provider!r}")
