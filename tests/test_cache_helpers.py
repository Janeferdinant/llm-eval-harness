from evalharness.llm.base import Completion


class Counting:
    """An inner client that counts its calls."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, model, temperature=0.0, max_tokens=512, tools=None, response_schema=None, seed=0):
        self.calls += 1
        return Completion(text=f"call {self.calls}", tool_calls=[], data=None, input_tokens=1, output_tokens=1,
                          latency_s=0.0, ttft_s=None, model=model, model_digest=None, cost_usd=0.0, billing="local")
