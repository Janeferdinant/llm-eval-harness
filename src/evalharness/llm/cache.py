"""CachingClient: a response cache keyed by the request hash, so reruns and resumed runs cost nothing."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from pydantic import BaseModel

from evalharness.llm.base import Completion, LLMClient, Message, ToolSpec, request_hash


class CachingClient:
    def __init__(self, inner: LLMClient, cache_dir: str | Path, *, salt: str = ""):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.salt = salt
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def _salt_for(self, model: str) -> str:
        """Who answered goes into the key: the provider, the client's own identity (model digest, CLI version)
        when it has one, and the salt the caller passed. Two judges asked the same question never share an entry."""
        provider = getattr(self.inner, "provider", type(self.inner).__name__)
        identity = getattr(self.inner, "identity", None)
        return f"{provider}|{identity(model) if callable(identity) else ''}|{self.salt}"

    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 tools: list[ToolSpec] | None = None, response_schema: type[BaseModel] | None = None,
                 seed: int | None = 0) -> Completion:
        h = request_hash(messages, model=model, temperature=temperature, max_tokens=max_tokens, tools=tools,
                         response_schema=response_schema, seed=seed, salt=self._salt_for(model))
        path = self.cache_dir / f"{h}.json"
        try:
            cached = Completion.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            cached = None
        if cached is not None:
            with self._lock:
                self.hits += 1
            return cached
        result = self.inner.complete(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                                     tools=tools, response_schema=response_schema, seed=seed)
        with self._lock:
            self.misses += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{h}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(result.model_dump_json(), encoding="utf-8")
        _publish(tmp, path)
        return result


def _publish(tmp: Path, path: Path) -> None:
    """Atomic move into place. On Windows os.replace raises PermissionError while another thread is replacing
    the same file; the content for one request hash is the same, so keep whichever writer got there first."""
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if path.exists():
                break
            time.sleep(0.01 * (attempt + 1))
    tmp.unlink(missing_ok=True)
