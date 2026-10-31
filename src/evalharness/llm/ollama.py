"""OllamaClient: /api/chat with streaming (for time to first token), tools and JSON-schema output."""
from __future__ import annotations

import json
import time

import httpx
from pydantic import BaseModel

from evalharness.llm.base import Completion, Message, ToolCall, ToolSpec, strip_think, with_structured_retry


class OllamaClient:
    """`num_ctx` is always sent: Ollama's own default context can be 256K, which does not fit a 10 GB GPU.

    `think=False` by default so hybrid models answer directly; models that cannot switch reasoning off
    (qwen3:4b) still put it in the reply, and it is stripped here. Reasoning tokens stay in `output_tokens`.
    """

    def __init__(self, base_url: str = "http://localhost:11434", *, num_ctx: int = 8192,
                 think: bool | None = False, timeout_s: float = 600):
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.think = think
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout_s)
        self._digests: dict[str, str] | None = None

    def _digest(self, model: str) -> str | None:
        # /api/show has no digest field (Ollama 0.40.13); /api/tags does.
        if self._digests is None:
            try:
                r = self._http.get("/api/tags")
                r.raise_for_status()
                self._digests = {m["name"]: m.get("digest") for m in r.json().get("models", [])}
            except (httpx.HTTPError, ValueError):
                self._digests = {}
        return self._digests.get(model) or self._digests.get(f"{model}:latest")

    def identity(self, model: str) -> str:
        """Part of the response-cache key: a re-pulled model with a new digest is a new cache namespace."""
        return self._digest(model) or ""

    def _payload(self, messages, model, temperature, max_tokens, tools, response_schema, seed) -> dict:
        options = {"num_ctx": self.num_ctx, "temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        payload: dict = {"model": model, "messages": [{"role": m.role, "content": m.content} for m in messages],
                         "stream": True, "options": options}
        if self.think is not None:
            payload["think"] = self.think
        if tools:
            payload["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                                    "parameters": t.parameters}} for t in tools]
        if response_schema is not None:
            payload["format"] = response_schema.model_json_schema()
        return payload

    def _once(self, payload: dict, model: str) -> Completion:
        start = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        final: dict = {}
        with self._http.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code >= 400:
                r.read()
                detail = r.text
                try:
                    detail = r.json().get("error", detail)
                except ValueError:
                    pass
                hint = ""
                if "memory" in detail.lower():
                    hint = f" (the model did not fit; lower num_ctx, currently {self.num_ctx})"
                raise RuntimeError(f"Ollama returned {r.status_code}: {detail}{hint}")
            for line in r.iter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                if msg.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    text_parts.append(msg["content"])
                for tc in msg.get("tool_calls") or []:
                    fn = tc["function"]
                    args = fn.get("arguments", {})
                    tool_calls.append(ToolCall(id=tc.get("id") or f"call_{len(tool_calls)}", name=fn["name"],
                                               arguments=json.loads(args) if isinstance(args, str) else args))
                if chunk.get("done"):
                    final = chunk
        latency = time.perf_counter() - start
        return Completion(text=strip_think("".join(text_parts)), tool_calls=tool_calls, data=None,
                          input_tokens=final.get("prompt_eval_count", 0), output_tokens=final.get("eval_count", 0),
                          latency_s=latency, ttft_s=ttft if ttft is not None else latency, model=model,
                          model_digest=self._digest(model), cost_usd=0.0, billing="local")

    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 tools: list[ToolSpec] | None = None, response_schema: type[BaseModel] | None = None,
                 seed: int | None = 0) -> Completion:
        payload = self._payload(messages, model, temperature, max_tokens, tools, response_schema, seed)
        return with_structured_retry(lambda: self._once(payload, model), response_schema)
