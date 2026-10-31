"""AnthropicClient: the anthropic SDK's Messages API. Pay-per-token; mock-tested only, never run live here.

anthropic 2.3.0's `messages.create` has no `temperature` or `seed` parameter, so neither is sent. The values
are accepted to keep the LLMClient signature and are ignored.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable

import anthropic
from pydantic import BaseModel

from evalharness.llm.base import Completion, Message, ToolCall, ToolSpec, with_structured_retry
from evalharness.llm.pricing import cost_usd
from evalharness.llm.retry import call_with_retry, require_key


class AnthropicClient:
    billing = "api"
    provider = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep, http_client=None):
        self._sleep = sleep
        self._http_client = http_client  # an httpx2.Client; tests inject a mock transport
        self._sdk: anthropic.Anthropic | None = None

    def _client(self) -> anthropic.Anthropic:
        if self._sdk is None:
            self._sdk = anthropic.Anthropic(api_key=require_key(self.api_key_env), max_retries=0,
                                          http_client=self._http_client)
        return self._sdk

    @staticmethod
    def _split(messages: list[Message]) -> tuple[str | None, list[dict]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id or "",
                                                         "content": m.content}]})
            else:
                out.append({"role": m.role, "content": m.content})
        return system, out

    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 tools: list[ToolSpec] | None = None, response_schema: type[BaseModel] | None = None,
                 seed: int | None = 0) -> Completion:
        client = self._client()
        system, turns = self._split(messages)
        kwargs: dict = {"model": model, "messages": turns, "max_tokens": max_tokens}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                               for t in tools]
        if response_schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": response_schema.model_json_schema()}}
        secret = os.environ.get(self.api_key_env)

        def once() -> Completion:
            start = time.perf_counter()
            resp = call_with_retry(lambda: client.messages.create(**kwargs), provider=self.provider, secret=secret,
                                   sleep=self._sleep)
            latency = time.perf_counter() - start
            text = "".join(b.text for b in resp.content if b.type == "text")
            calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input)) for b in resp.content
                     if b.type == "tool_use"]
            n_in, n_out = resp.usage.input_tokens, resp.usage.output_tokens
            return Completion(text=text, tool_calls=calls, data=None, input_tokens=n_in, output_tokens=n_out,
                              latency_s=latency, ttft_s=None, model=resp.model, model_digest=None,
                              cost_usd=cost_usd(resp.model, n_in, n_out, billing=self.billing), billing=self.billing)

        return with_structured_retry(once, response_schema)
