"""OpenAIClient: the openai SDK's chat completions. Pay-per-token; mock-tested only, never run live here."""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable

import openai
from pydantic import BaseModel

from evalharness.llm.base import Completion, Message, ToolCall, ToolSpec, with_structured_retry
from evalharness.llm.pricing import cost_usd
from evalharness.llm.retry import call_with_retry, require_key


class OpenAIClient:
    billing = "api"
    provider = "openai"

    def __init__(self, *, base_url: str | None = None, api_key_env: str | None = "OPENAI_API_KEY",
                 sleep: Callable[[float], None] = time.sleep, http_client=None):
        self.base_url = base_url
        self._http_client = http_client  # an httpx2.Client; tests inject a mock transport
        self.api_key_env = api_key_env
        self._sleep = sleep
        self._sdk: openai.OpenAI | None = None

    def _client(self) -> openai.OpenAI:
        if self._sdk is None:
            key = require_key(self.api_key_env) if self.api_key_env else "EMPTY"
            self._sdk = openai.OpenAI(api_key=key, base_url=self.base_url, max_retries=0,
                                    http_client=self._http_client)
        return self._sdk

    @staticmethod
    def _messages(messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            item: dict = {"role": m.role, "content": m.content}
            if m.role == "tool":
                item["tool_call_id"] = m.tool_call_id or ""
            out.append(item)
        return out

    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 tools: list[ToolSpec] | None = None, response_schema: type[BaseModel] | None = None,
                 seed: int | None = 0) -> Completion:
        client = self._client()  # raises MissingKeyError before any network call
        kwargs: dict = {"model": model, "messages": self._messages(messages), "temperature": temperature,
                        "max_completion_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                                   "parameters": t.parameters}} for t in tools]
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": response_schema.__name__, "schema": response_schema.model_json_schema()}}
        secret = os.environ.get(self.api_key_env or "")

        def once() -> Completion:
            start = time.perf_counter()
            resp = call_with_retry(lambda: client.chat.completions.create(**kwargs), provider=self.provider,
                                   secret=secret, sleep=self._sleep)
            latency = time.perf_counter() - start
            msg = resp.choices[0].message
            calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
                     for tc in (msg.tool_calls or []) if tc.type == "function"]
            usage = resp.usage
            n_in, n_out = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
            return Completion(text=msg.content or "", tool_calls=calls, data=None, input_tokens=n_in,
                              output_tokens=n_out, latency_s=latency, ttft_s=None, model=resp.model,
                              model_digest=None, cost_usd=cost_usd(resp.model, n_in, n_out, billing=self.billing),
                              billing=self.billing)

        return with_structured_retry(once, response_schema)
