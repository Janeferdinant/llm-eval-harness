import json
import socket
from pathlib import Path

import pytest
from anthropic.types import Message as AnthropicMessage
from google.genai.types import GenerateContentResponse
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from evalharness.llm.anthropic import AnthropicClient
from evalharness.llm.base import Message, ToolCall, ToolSpec
from evalharness.llm.gemini import GeminiClient
from evalharness.llm.openai import OpenAIClient
from evalharness.llm.openai_compat import OpenAICompatClient
from evalharness.llm.retry import MissingKeyError, ProviderError
from httpmock import FakeApi, Reply

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
MSGS = [Message(role="system", content="Be brief."), Message(role="user", content="Say OK")]
TOOL = ToolSpec(name="lookup", description="find",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}})
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
URL = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent",
}
MODEL = {"openai": "gpt-6.1-sol", "anthropic": "claude-sonnet-5-5", "gemini": "gemini-4-argon"}
PROVIDERS = ["openai", "anthropic", "gemini"]


def fixture(provider, name) -> dict:
    return json.loads((FIXTURES / provider / f"{name}.json").read_text(encoding="utf-8"))["body"]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for env in KEY_ENV.values():
        monkeypatch.setenv(env, "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def api():
    return FakeApi()


def make(provider, api, sleeps):
    cls = {"openai": OpenAIClient, "anthropic": AnthropicClient, "gemini": GeminiClient}[provider]
    return cls(sleep=sleeps.append, http_client=api.client())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_missing_key_error_names_env_var_not_value(provider, api, sleeps, monkeypatch):
    monkeypatch.delenv(KEY_ENV[provider])
    monkeypatch.setenv("DECOY_API_KEY", "decoy-value-123456")
    with pytest.raises(MissingKeyError) as e:
        make(provider, api, sleeps).complete(MSGS, model=MODEL[provider])
    assert KEY_ENV[provider] in str(e.value) and "decoy-value-123456" not in str(e.value)
    assert api.requests == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_text_reply_mapped_with_tokens_and_api_billing(provider, api, sleeps):
    api.reply(URL[provider], fixture(provider, "text"))
    out = make(provider, api, sleeps).complete(MSGS, model=MODEL[provider], max_tokens=32)
    assert out.text == "OK" and (out.input_tokens, out.output_tokens) == (10, 2)
    assert out.billing == "api" and out.model_digest is None and out.latency_s >= 0
    assert out.cost_usd == pytest.approx(10 * 2.0 / 1e6 + 2 * 10.0 / 1e6)  # all three priced $2 in, $10 out


def test_tool_call_mapping_openai(api, sleeps):
    api.reply(URL["openai"], fixture("openai", "tool_call"))
    out = make("openai", api, sleeps).complete(MSGS, model=MODEL["openai"], tools=[TOOL])
    assert out.tool_calls == [ToolCall(id="call_test", name="lookup", arguments={"q": "x"})]
    sent = api.last_json(URL["openai"])
    assert sent["tools"][0]["function"]["name"] == "lookup" and sent["seed"] == 0 and sent["temperature"] == 0.0


def test_tool_call_mapping_anthropic(api, sleeps):
    api.reply(URL["anthropic"], fixture("anthropic", "tool_use"))
    out = make("anthropic", api, sleeps).complete(MSGS, model=MODEL["anthropic"], tools=[TOOL])
    assert out.tool_calls == [ToolCall(id="toolu_test", name="lookup", arguments={"q": "x"})]
    sent = api.last_json(URL["anthropic"])
    assert sent["tools"][0]["name"] == "lookup" and sent["tools"][0]["input_schema"]["type"] == "object"


def test_tool_call_mapping_gemini(api, sleeps):
    api.reply(URL["gemini"], fixture("gemini", "function_call"))
    out = make("gemini", api, sleeps).complete(MSGS, model=MODEL["gemini"], tools=[TOOL])
    assert [(t.name, t.arguments) for t in out.tool_calls] == [("lookup", {"q": "x"})] and out.tool_calls[0].id
    sent = api.last_json(URL["gemini"])
    assert sent["tools"][0]["functionDeclarations"][0]["name"] == "lookup"


def test_anthropic_request_puts_system_aside_and_omits_sampling_parameters(api, sleeps):
    """anthropic 2.3.0's messages.create has no temperature or seed parameter, so none is sent."""
    api.reply(URL["anthropic"], fixture("anthropic", "text"))
    make("anthropic", api, sleeps).complete(MSGS, model=MODEL["anthropic"], temperature=0.0, seed=5, max_tokens=33)
    sent = api.last_json(URL["anthropic"])
    assert sent["system"] == "Be brief." and sent["max_tokens"] == 33
    assert [x["role"] for x in sent["messages"]] == ["user"]
    assert "temperature" not in sent and "seed" not in sent


def test_gemini_request_maps_generation_config(api, sleeps):
    api.reply(URL["gemini"], fixture("gemini", "text"))
    make("gemini", api, sleeps).complete(MSGS, model=MODEL["gemini"], temperature=0.0, seed=5, max_tokens=33)
    sent = api.last_json(URL["gemini"])
    cfg = sent["generationConfig"]
    assert cfg["temperature"] == 0.0 and cfg["seed"] == 5 and cfg["maxOutputTokens"] == 33
    assert sent["systemInstruction"]["parts"][0]["text"] == "Be brief."


def test_openai_compat_uses_base_url_needs_no_key_and_is_local_billing(api, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    url = "http://localhost:8000/v1/chat/completions"
    api.reply(url, fixture("openai", "text"))
    client = OpenAICompatClient(base_url="http://localhost:8000/v1", http_client=api.client())
    out = client.complete(MSGS, model="qwen3-8b")
    assert api.count(url) == 1 and out.billing == "local" and out.cost_usd == 0.0


class Verdict(BaseModel):
    score: int


@pytest.mark.parametrize("provider", PROVIDERS)
def test_structured_output_validates_against_response_schema(provider, api, sleeps):
    api.reply(URL[provider], fixture(provider, "text"))  # replies "OK", which is not the schema
    with pytest.raises(ValueError, match="response_schema"):
        make(provider, api, sleeps).complete(MSGS, model=MODEL[provider], response_schema=Verdict)
    assert api.count(URL[provider]) == 2


def test_structured_output_success_sets_data(api, sleeps):
    body = fixture("openai", "text")
    body["choices"][0]["message"]["content"] = '{"score": 4}'
    api.reply(URL["openai"], body)
    out = make("openai", api, sleeps).complete(MSGS, model=MODEL["openai"], response_schema=Verdict)
    assert out.data == {"score": 4}
    assert api.last_json(URL["openai"])["response_format"]["type"] == "json_schema"


def test_structured_output_request_shape_anthropic_and_gemini(api, sleeps):
    api.reply(URL["anthropic"], fixture("anthropic", "text"))
    api.reply(URL["gemini"], fixture("gemini", "text"))
    for provider in ("anthropic", "gemini"):
        with pytest.raises(ValueError):
            make(provider, api, sleeps).complete(MSGS, model=MODEL[provider], response_schema=Verdict)
    assert api.last_json(URL["anthropic"])["output_config"]["format"]["type"] == "json_schema"
    assert api.last_json(URL["gemini"])["generationConfig"]["responseMimeType"] == "application/json"


def test_retry_stops_after_five_tries_and_honours_retry_after(api, sleeps):
    api.reply(URL["openai"], Reply(429, {"error": {"message": "slow down"}}, {"retry-after": "2"}))
    with pytest.raises(ProviderError, match="429"):
        make("openai", api, sleeps).complete(MSGS, model=MODEL["openai"])
    assert api.count(URL["openai"]) == 5 and len(sleeps) == 4 and sleeps[0] == 2


def test_retry_backs_off_exponentially_without_retry_after(api, sleeps):
    api.reply(URL["openai"], Reply(503, {"error": {"message": "unavailable"}}))
    with pytest.raises(ProviderError):
        make("openai", api, sleeps).complete(MSGS, model=MODEL["openai"])
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_retry_applies_to_every_provider(provider, api, sleeps):
    api.reply(URL[provider], Reply(500, {"error": {"message": "boom"}}), fixture(provider, "text"))
    out = make(provider, api, sleeps).complete(MSGS, model=MODEL[provider])
    assert out.text == "OK" and api.count(URL[provider]) == 2 and len(sleeps) == 1


def test_retry_recovers_after_a_500(api, sleeps):
    api.reply(URL["openai"], Reply(500, {"error": {"message": "boom"}}), fixture("openai", "text"))
    out = make("openai", api, sleeps).complete(MSGS, model=MODEL["openai"])
    assert out.text == "OK" and api.count(URL["openai"]) == 2 and len(sleeps) == 1


def test_client_error_is_not_retried(api, sleeps):
    api.reply(URL["openai"], Reply(400, {"error": {"message": "bad request body"}}))
    with pytest.raises(ProviderError, match="400"):
        make("openai", api, sleeps).complete(MSGS, model=MODEL["openai"])
    assert api.count(URL["openai"]) == 1 and sleeps == []


def test_auth_error_does_not_echo_the_key(api, sleeps, monkeypatch):
    secret = "sk-test-secretvalue123456"  # prepublish-allow  # gitleaks:allow, fake value
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    api.reply(URL["openai"], Reply(401, {"error": {"message": f"bad key {secret}"}}))
    with pytest.raises(ProviderError) as e:
        make("openai", api, sleeps).complete(MSGS, model=MODEL["openai"])
    assert secret not in str(e.value) and "[REDACTED]" in str(e.value)


def test_fixtures_validate_against_sdk_types():
    pairs = [("openai", ["text", "tool_call"], ChatCompletion),
             ("anthropic", ["text", "tool_use"], AnthropicMessage),
             ("gemini", ["text", "function_call"], GenerateContentResponse)]
    for provider, names, cls in pairs:
        for name in names:
            raw = json.loads((FIXTURES / provider / f"{name}.json").read_text(encoding="utf-8"))
            assert "recorded_at" in raw and "source" in raw
            cls.model_validate(raw["body"])


def test_mock_fixtures_contain_no_keys():
    root = Path(__file__).parent / "fixtures"
    needles = ["sk-", "sk-ant-", "AIza"]
    bad = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".txt", ".md"}
           and any(n in p.read_text(encoding="utf-8", errors="ignore") for n in needles)]
    assert bad == []


def test_fake_api_refuses_unregistered_urls(api):
    client = api.client()
    with pytest.raises(AssertionError, match="unexpected request"):
        client.post("https://example.com/other")


def test_tests_cannot_open_outside_connections():
    s = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="may not open network"):
            s.connect(("93.184.216.34", 80))
    finally:
        s.close()
