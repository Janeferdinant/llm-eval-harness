"""A tiny fake HTTP API for the SDK clients.

The current openai, anthropic and google-genai SDKs use `httpx2`, a separate namespace from `httpx`, so respx
cannot intercept them. This helper builds an `httpx2.Client` on a MockTransport; a request to a URL nobody
registered raises, so a test can never reach the real network by accident.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx2


@dataclass
class Reply:
    status: int = 200
    body: dict | None = None
    headers: dict = field(default_factory=dict)


class FakeApi:
    def __init__(self) -> None:
        self._routes: dict[str, list[Reply]] = {}
        self.requests: list[httpx2.Request] = []

    def reply(self, url: str, *replies: Reply | dict) -> None:
        """Register replies for `url`. They are used in order, and the last one repeats."""
        self._routes[url] = [r if isinstance(r, Reply) else Reply(200, r) for r in replies]

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url).split("?")[0]
        if url not in self._routes:
            raise AssertionError(f"unexpected request to {url}")
        self.requests.append(request)
        queue = self._routes[url]
        r = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx2.Response(r.status, json=r.body, headers=r.headers)

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle))

    def count(self, url: str) -> int:
        return sum(1 for r in self.requests if str(r.url).split("?")[0] == url)

    def last_json(self, url: str) -> dict:
        matching = [r for r in self.requests if str(r.url).split("?")[0] == url]
        return json.loads(matching[-1].content)
