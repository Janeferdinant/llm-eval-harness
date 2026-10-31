"""GeminiClient: the google-genai SDK. Pay-per-token; mock-tested only, never run live here.

Tool results (role "tool") are sent back as plain user text, because a Message carries no function name.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable

from google import genai
from google.genai import types
from pydantic import BaseModel

from evalharness.llm.base import Completion, Message, ToolCall, ToolSpec, with_structured_retry
from evalharness.llm.pricing import cost_usd
from evalharness.llm.retry import call_with_retry, require_key


class GeminiClient:
    billing = "api"
    provider = "gemini"
    api_key_env = "GEMINI_API_KEY"

    def __init__(self, *, sleep: Callable[[float], None] = time.sleep, http_client=None):
        self._sleep = sleep
        self._http_client = http_client  # an httpx2.Client; tests inject a mock transport
        self._sdk: genai.Client | None = None

    def _client(self) -> genai.Client:
        if self._sdk is None:
            self._sdk = genai.Client(
                api_key=require_key(self.api_key_env),
                http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1),
                                              httpx_client=self._http_client))
        return self._sdk

    @staticmethod
    def _contents(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        contents = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            text = f"[tool result {m.tool_call_id}] {m.content}" if m.role == "tool" else m.content
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        return system, contents

    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 tools: list[ToolSpec] | None = None, response_schema: type[BaseModel] | None = None,
                 seed: int | None = 0) -> Completion:
        client = self._client()
        system, contents = self._contents(messages)
        cfg: dict = {"temperature": temperature, "max_output_tokens": max_tokens, "seed": seed,
                     "system_instruction": system}
        if tools:
            cfg["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in tools])]
            cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        if response_schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = response_schema.model_json_schema()
        config = types.GenerateContentConfig(**cfg)
        secret = os.environ.get(self.api_key_env)

        def once() -> Completion:
            start = time.perf_counter()
            resp = call_with_retry(lambda: client.models.generate_content(model=model, contents=contents,
                                                                          config=config),
                                   provider=self.provider, secret=secret, sleep=self._sleep)
            latency = time.perf_counter() - start
            parts = resp.candidates[0].content.parts if resp.candidates and resp.candidates[0].content else []
            text = "".join(p.text for p in parts if p.text and not p.thought)
            calls = [ToolCall(id=f"call_{i}", name=p.function_call.name, arguments=dict(p.function_call.args or {}))
                     for i, p in enumerate(p for p in parts if p.function_call)]
            u = resp.usage_metadata
            n_in = (u.prompt_token_count or 0) if u else 0
            n_out = ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0
            used = resp.model_version or model
            return Completion(text=text, tool_calls=calls, data=None, input_tokens=n_in, output_tokens=n_out,
                              latency_s=latency, ttft_s=None, model=used, model_digest=None,
                              cost_usd=cost_usd(used, n_in, n_out, billing=self.billing), billing=self.billing)

        return with_structured_retry(once, response_schema)
