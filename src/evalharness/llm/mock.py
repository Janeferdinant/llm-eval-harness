"""MockClient: replays recorded completions, keyed by the request hash."""
from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from evalharness.llm.base import Completion, LLMClient, Message, MissingFixtureError, ToolSpec, request_hash

STALE_AFTER_DAYS = 90


class MockClient:
    def __init__(self, fixture_dir: str | Path, *, record_from: LLMClient | None = None):
        self.fixture_dir = Path(fixture_dir)
        self.record_from = record_from

    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 tools: list[ToolSpec] | None = None, response_schema: type[BaseModel] | None = None,
                 seed: int | None = 0) -> Completion:
        h = request_hash(messages, model=model, temperature=temperature, max_tokens=max_tokens, tools=tools,
                         response_schema=response_schema, seed=seed)
        path = self.fixture_dir / f"{h}.json"
        if not path.exists():
            if self.record_from is None:
                raise MissingFixtureError(h)
            real = self.record_from.complete(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                                             tools=tools, response_schema=response_schema, seed=seed)
            self.fixture_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"request_hash": h, "recorded_at": date.today().isoformat(),
                                        "completion": real.model_dump()}, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            return real
        fixture = json.loads(path.read_text(encoding="utf-8"))
        age = (date.today() - date.fromisoformat(fixture["recorded_at"])).days
        if age > STALE_AFTER_DAYS:
            warnings.warn(f"fixture {path.name} was recorded {age} days ago; re-record it against the provider",
                          UserWarning, stacklevel=2)
        return Completion.model_validate(fixture["completion"])
