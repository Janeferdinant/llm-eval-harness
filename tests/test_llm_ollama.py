import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from evalharness.llm.base import Message, ToolSpec
from evalharness.llm.ollama import OllamaClient

BASE = "http://localhost:11434"
MSGS = [Message(role="user", content="Reply with the single word OK")]
DIGEST = "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7"


def ndjson(*chunks):
    return "\n".join(json.dumps(c) for c in chunks) + "\n"


def stream(text="OK", *, thinking=None, eval_count=2, prompt=17, tool_calls=None):
    msg = {"role": "assistant", "content": "" if thinking else text, "thinking": thinking or ""}
    first = {"message": msg, "done": False}
    chunks = [first]
    if thinking:
        chunks.append({"message": {"role": "assistant", "content": text}, "done": False})
    if tool_calls:
        chunks.append({"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}, "done": False})
    chunks.append({"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": prompt,
                   "eval_count": eval_count})
    return ndjson(*chunks)


@pytest.fixture
def server():
    with respx.mock(base_url=BASE, assert_all_called=False) as m:
        m.get("/api/tags").mock(return_value=httpx.Response(200, json={"models": [
            {"name": "qwen3:1.7b", "model": "qwen3:1.7b", "digest": DIGEST}]}))
        yield m


def last_payload(server):
    return json.loads(server.calls.last.request.content)


def test_ollama_payload_includes_tools_and_schema(server):
    class Out(BaseModel):
        answer: str

    route = server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream('{"answer": "OK"}')))
    tool = ToolSpec(name="lookup", description="find", parameters={"type": "object", "properties": {}})
    out = OllamaClient().complete(MSGS, model="qwen3:1.7b", temperature=0.0, seed=7, max_tokens=64, tools=[tool],
                                  response_schema=Out)
    p = json.loads(route.calls.last.request.content)
    assert p["tools"][0]["function"]["name"] == "lookup"
    assert p["format"] == Out.model_json_schema()
    assert p["options"]["num_ctx"] == 8192 and p["options"]["temperature"] == 0.0
    assert p["options"]["seed"] == 7 and p["options"]["num_predict"] == 64 and p["stream"] is True
    assert out.data == {"answer": "OK"}


def test_ollama_always_sends_num_ctx_and_think_false_by_default(server):
    route = server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream()))
    OllamaClient().complete(MSGS, model="qwen3:1.7b")
    p = json.loads(route.calls.last.request.content)
    assert p["options"]["num_ctx"] == 8192 and p["think"] is False
    OllamaClient(num_ctx=4096, think=None).complete(MSGS, model="qwen3:1.7b")
    p = json.loads(route.calls.last.request.content)
    assert p["options"]["num_ctx"] == 4096 and "think" not in p


def test_completion_records_model_digest(server):
    server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream()))
    out = OllamaClient().complete(MSGS, model="qwen3:1.7b")
    assert out.model_digest == DIGEST and out.model == "qwen3:1.7b" and out.billing == "local" and out.cost_usd == 0.0


def test_unknown_model_digest_is_none(server):
    server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream()))
    assert OllamaClient().complete(MSGS, model="not-pulled:1b").model_digest is None


def test_ollama_strips_think_block_and_counts_its_tokens(server):
    server.post("/api/chat").mock(return_value=httpx.Response(
        200, content=stream("<think>hmm</think>\n\nOK", eval_count=151)))
    out = OllamaClient().complete(MSGS, model="qwen3:1.7b")
    assert out.text == "OK" and out.output_tokens == 151 and out.input_tokens == 17


def test_ollama_separate_thinking_field_is_not_part_of_text(server):
    server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream("OK", thinking="Okay, so...")))
    assert OllamaClient().complete(MSGS, model="qwen3:1.7b").text == "OK"


def test_ollama_ttft_is_time_to_first_content_chunk(server):
    server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream("OK", thinking="x")))
    out = OllamaClient().complete(MSGS, model="qwen3:1.7b")
    assert out.ttft_s is not None and 0 <= out.ttft_s <= out.latency_s


def test_ollama_tool_calls_mapped(server):
    calls = [{"function": {"name": "lookup", "arguments": {"q": "x"}}}]
    server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream("", tool_calls=calls)))
    out = OllamaClient().complete(MSGS, model="qwen3:1.7b")
    assert [(t.name, t.arguments) for t in out.tool_calls] == [("lookup", {"q": "x"})] and out.tool_calls[0].id


def test_ollama_out_of_memory_error_is_actionable(server):
    server.post("/api/chat").mock(return_value=httpx.Response(500, json={"error": "memory layout cannot be allocated"}))
    with pytest.raises(RuntimeError, match="num_ctx"):
        OllamaClient().complete(MSGS, model="qwen3:1.7b")


def test_ollama_other_http_error_includes_server_message(server):
    server.post("/api/chat").mock(return_value=httpx.Response(404, json={"error": "model 'x' not found"}))
    with pytest.raises(RuntimeError, match="not found"):
        OllamaClient().complete(MSGS, model="x")


def test_structured_output_retries_once_then_raises(server):
    class Out(BaseModel):
        n: int

    route = server.post("/api/chat").mock(return_value=httpx.Response(200, content=stream("not json")))
    with pytest.raises(ValueError, match="response_schema"):
        OllamaClient().complete(MSGS, model="qwen3:1.7b", response_schema=Out)
    assert route.call_count == 2
