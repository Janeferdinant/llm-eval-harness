"""Build the SDK response fixtures from the installed SDKs' own response types.

    uv run python tests/fixtures/make_sdk_fixtures.py

These are NOT recordings of live API calls (the pay-per-token APIs are never called). Each body is validated
through the SDK's response class, so an SDK upgrade that changes the types fails
`test_fixtures_validate_against_sdk_types`.
"""
import json
from datetime import date
from pathlib import Path

from anthropic.types import Message
from google.genai.types import GenerateContentResponse
from openai.types.chat import ChatCompletion

OUT = Path(__file__).parent / "llm"
SOURCE = "constructed from the installed SDK response types; not recorded from the live API"


def write(provider: str, name: str, model_cls, body: dict) -> None:
    validated = model_cls.model_validate(body)
    dumped = validated.model_dump(mode="json", exclude_none=True, by_alias=True)
    path = OUT / provider / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"recorded_at": date.today().isoformat(), "source": SOURCE, "body": dumped},
                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def openai_bodies() -> None:
    base = {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "gpt-6.1-sol"}
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    write("openai", "text", ChatCompletion, {**base, "usage": usage, "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}]})
    write("openai", "tool_call", ChatCompletion, {**base, "usage": usage, "choices": [
        {"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_test", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"}}]}}]})


def anthropic_bodies() -> None:
    base = {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 2}}
    write("anthropic", "text", Message, {**base, "content": [{"type": "text", "text": "OK"}]})
    write("anthropic", "tool_use", Message, {**base, "stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "toolu_test", "name": "lookup", "input": {"q": "x"}}]})


def gemini_bodies() -> None:
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}
    ok = {"role": "model", "parts": [{"text": "OK"}]}
    write("gemini", "text", GenerateContentResponse, {"modelVersion": "gemini-4-argon", "usageMetadata": usage,
          "candidates": [{"index": 0, "finishReason": "STOP", "content": ok}]})
    write("gemini", "function_call", GenerateContentResponse, {"modelVersion": "gemini-4-argon",
          "usageMetadata": usage, "candidates": [{"index": 0, "finishReason": "STOP", "content": {
              "role": "model", "parts": [{"functionCall": {"name": "lookup", "args": {"q": "x"}}}]}}]})


if __name__ == "__main__":
    openai_bodies()
    anthropic_bodies()
    gemini_bodies()
    print("wrote fixtures under", OUT)
