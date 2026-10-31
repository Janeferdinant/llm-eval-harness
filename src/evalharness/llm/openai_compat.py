"""OpenAICompatClient: any server that speaks the OpenAI chat API (vLLM, llama.cpp server). Local, so cost 0."""
from __future__ import annotations

import time
from collections.abc import Callable

from evalharness.llm.openai import OpenAIClient


class OpenAICompatClient(OpenAIClient):
    billing = "local"
    provider = "openai_compat"

    def __init__(self, base_url: str, *, api_key_env: str | None = None,
                 sleep: Callable[[float], None] = time.sleep, http_client=None):
        super().__init__(base_url=base_url, api_key_env=api_key_env, sleep=sleep, http_client=http_client)
