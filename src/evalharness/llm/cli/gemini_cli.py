"""The Gemini command line arm was not built. The reason is checked in at NOTICE.md."""
from __future__ import annotations

from evalharness.llm.base import ArmUnavailableError

REASON = (
    "The Gemini subscription arm was not built. Gemini CLI stopped serving Google AI Pro and individual accounts "
    "on 2026-06-18, and its replacement, the Antigravity CLI, is covered by terms that call using third-party "
    "software or tools to access the Service a breach that can lead to account suspension "
    "(https://antigravity.google/terms, read 2026-10-30). Use the pay-per-token GeminiClient with your own key, "
    "or a local model."
)


class GeminiCLIClient:
    def __init__(self, *args, **kwargs):
        raise ArmUnavailableError(REASON)
