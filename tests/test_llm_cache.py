from concurrent.futures import ThreadPoolExecutor

from evalharness.llm.base import Completion, Message
from evalharness.llm.cache import CachingClient

MSGS = [Message(role="user", content="hi")]


class Counting:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, model, temperature=0.0, max_tokens=512, tools=None, response_schema=None, seed=0):
        self.calls += 1
        return Completion(text=f"call {self.calls}", tool_calls=[], data=None, input_tokens=1, output_tokens=1,
                          latency_s=0.0, ttft_s=None, model=model, model_digest=None, cost_usd=0.0, billing="local")


def test_cache_hit_makes_no_inner_call(tmp_path):
    inner = Counting()
    c = CachingClient(inner, tmp_path)
    a = c.complete(MSGS, model="m")
    b = c.complete(MSGS, model="m")
    assert a == b and inner.calls == 1 and (c.hits, c.misses) == (1, 1)


def test_cache_persists_across_instances(tmp_path):
    inner = Counting()
    CachingClient(inner, tmp_path).complete(MSGS, model="m")
    again = CachingClient(inner, tmp_path)
    again.complete(MSGS, model="m")
    assert inner.calls == 1 and again.hits == 1


def test_cache_salt_separates_entries(tmp_path):
    inner = Counting()
    CachingClient(inner, tmp_path, salt="digest-1").complete(MSGS, model="m")
    CachingClient(inner, tmp_path, salt="digest-2").complete(MSGS, model="m")
    assert inner.calls == 2 and len(list(tmp_path.glob("*.json"))) == 2


def test_cache_key_changes_with_each_request_field(tmp_path):
    inner = Counting()
    c = CachingClient(inner, tmp_path)
    c.complete(MSGS, model="m")
    c.complete(MSGS, model="m", temperature=0.5)
    c.complete(MSGS, model="m", seed=1)
    c.complete(MSGS, model="other")
    c.complete(MSGS, model="m", max_tokens=7)
    assert inner.calls == 5 and c.hits == 0


def test_cache_is_safe_under_threads(tmp_path):
    inner = Counting()
    c = CachingClient(inner, tmp_path)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: c.complete([Message(role="user", content=f"q{i % 4}")], model="m"), range(32)))
    assert c.hits + c.misses == 32 and len(list(tmp_path.glob("*.json"))) == 4


def test_corrupt_cache_file_is_a_miss_not_a_crash(tmp_path):
    inner = Counting()
    c = CachingClient(inner, tmp_path)
    c.complete(MSGS, model="m")
    next(tmp_path.glob("*.json")).write_text("{truncated", encoding="utf-8")
    c.complete(MSGS, model="m")
    assert inner.calls == 2
