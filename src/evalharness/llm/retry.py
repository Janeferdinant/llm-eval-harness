"""Shared plumbing for the SDK clients: key lookup, bounded retry, and secret-free errors."""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")
MAX_TRIES = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 30.0


class MissingKeyError(RuntimeError):
    """The API key environment variable is not set. The message names the variable, never a value."""


class ProviderError(RuntimeError):
    def __init__(self, provider: str, status: int | None, message: str):
        super().__init__(f"{provider} request failed" + (f" ({status})" if status else "") + f": {message}")
        self.provider = provider
        self.status = status


def require_key(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise MissingKeyError(f"set the {env_name} environment variable to use this client")
    return value


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers is not None and headers.get("retry-after") else None
    except (TypeError, ValueError):
        return None


def _retryable(exc: BaseException) -> bool:
    status = _status(exc)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError) or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}


def call_with_retry(fn: Callable[[], T], *, provider: str, secret: str | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> T:
    """Call `fn`, retrying 429, 5xx and connection errors with exponential backoff, at most MAX_TRIES calls.
    The final error is re-raised as ProviderError with `secret` removed from its message."""
    for attempt in range(MAX_TRIES):
        try:
            return fn()
        except MissingKeyError:
            raise
        except Exception as exc:  # SDK exception classes differ per provider
            if _retryable(exc) and attempt < MAX_TRIES - 1:
                sleep(_retry_after(exc) or min(BASE_DELAY_S * 2**attempt, MAX_DELAY_S))
                continue
            message = str(exc)
            if secret:
                message = message.replace(secret, "[REDACTED]")
            raise ProviderError(provider, _status(exc), message) from None
    raise AssertionError("unreachable")
