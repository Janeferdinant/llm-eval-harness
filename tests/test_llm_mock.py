import json
import warnings
from datetime import date, timedelta

import pytest
from pydantic import BaseModel

from evalharness.llm.base import (
    Completion,
    Message,
    MissingFixtureError,
    extract_json,
    request_hash,
    strip_think,
)
from evalharness.llm.mock import MockClient

MSGS = [Message(role="user", content="Reply with OK")]
KW = dict(model="m", temperature=0.0, max_tokens=16, tools=None, response_schema=None, seed=0)


def completion(text="OK"):
    return Completion(text=text, tool_calls=[], data=None, input_tokens=3, output_tokens=1, latency_s=0.1,
                      ttft_s=0.05, model="m", model_digest="abc", cost_usd=0.0, billing="local")


def write_fixture(folder, *, recorded_at=None, text="OK"):
    folder.mkdir(parents=True, exist_ok=True)
    h = request_hash(MSGS, **KW)
    (folder / f"{h}.json").write_text(json.dumps({
        "request_hash": h, "recorded_at": (recorded_at or date.today()).isoformat(),
        "completion": completion(text).model_dump()}), encoding="utf-8")


def test_mock_replays_recorded_response(tmp_path):
    write_fixture(tmp_path, text="recorded")
    out = MockClient(tmp_path).complete(MSGS, model="m", max_tokens=16)
    assert out.text == "recorded" and out.model_digest == "abc"


def test_mock_missing_fixture_raises(tmp_path):
    with pytest.raises(MissingFixtureError) as e:
        MockClient(tmp_path).complete(MSGS, model="m", max_tokens=16)
    assert e.value.request_hash == request_hash(MSGS, **KW)


def test_mock_record_mode_writes_fixture_with_recorded_at(tmp_path):
    class Real:
        def complete(self, messages, **kw):
            return completion("fresh")
    client = MockClient(tmp_path, record_from=Real())
    assert client.complete(MSGS, model="m", max_tokens=16).text == "fresh"
    files = list(tmp_path.glob("*.json"))
    recorded = json.loads(files[0].read_text(encoding="utf-8"))["recorded_at"]
    assert len(files) == 1 and recorded == date.today().isoformat()
    assert MockClient(tmp_path).complete(MSGS, model="m", max_tokens=16).text == "fresh"


def test_fixture_older_than_90_days_warns(tmp_path):
    write_fixture(tmp_path, recorded_at=date.today() - timedelta(days=91))
    with pytest.warns(UserWarning, match="91 days"):
        MockClient(tmp_path).complete(MSGS, model="m", max_tokens=16)


def test_fresh_fixture_does_not_warn(tmp_path):
    write_fixture(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MockClient(tmp_path).complete(MSGS, model="m", max_tokens=16)


class Schema(BaseModel):
    score: int


def test_request_hash_changes_with_each_request_field():
    base = request_hash(MSGS, **KW)
    variants = [
        request_hash(MSGS, **{**KW, "temperature": 0.7}),
        request_hash(MSGS, **{**KW, "seed": 1}),
        request_hash(MSGS, **{**KW, "seed": None}),
        request_hash(MSGS, **{**KW, "model": "other"}),
        request_hash(MSGS, **{**KW, "max_tokens": 17}),
        request_hash([Message(role="user", content="Reply with OK.")], **KW),
        request_hash([Message(role="system", content="Reply with OK")], **KW),
        request_hash(MSGS, **{**KW, "response_schema": Schema}),
        request_hash(MSGS, **KW, salt="digest-2"),
    ]
    assert len({base, *variants}) == len(variants) + 1


def test_request_hash_is_stable():
    assert request_hash(MSGS, **KW) == request_hash(list(MSGS), **KW)
    assert len(request_hash(MSGS, **KW)) == 64


def test_strip_think_removes_blocks_and_leading_text_before_close_tag():
    assert strip_think("<think>hmm</think>\n\nOK") == "OK"
    assert strip_think("I wonder...\n</think>\n\nOK") == "OK"
    assert strip_think("no thinking here") == "no thinking here"


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"score": 3}\n```') == {"score": 3}
    assert extract_json('Sure! Here it is: {"score": 4, "nested": {"a": 1}} done') == {"score": 4, "nested": {"a": 1}}
    assert extract_json('<think>{"score": 1}</think>{"score": 2}') == {"score": 2}
    with pytest.raises(ValueError):
        extract_json("no json at all")
