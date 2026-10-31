"""Reading the usage meters of the subscription CLIs, and turning a measured change into calls per window.

Claude Code prints its meters for `/usage` (also non-interactively: `claude -p /usage`). Codex writes its rate
limits into the session file of every run (`rate_limits` in the `token_count` events); no command prints them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

_SESSION = re.compile(r"Current session:\s*([\d.]+)%\s*used\s*[·\-]\s*resets\s*([^\n]+)")
_WEEK = re.compile(r"Current week \(all models\):\s*([\d.]+)%\s*used\s*[·\-]\s*resets\s*([^\n]+)")


@dataclass
class ClaudeUsage:
    session_pct: float
    session_resets: str
    week_pct: float
    week_resets: str


@dataclass
class CodexLimits:
    window_pct: float  # the 5-hour window
    week_pct: float
    plan: str | None
    window_resets_at: int | None
    week_resets_at: int | None


@dataclass
class Estimate:
    pct_per_call: float | None
    calls_per_window: float | None
    min_calls_per_window: int | None
    note: str


def parse_claude_usage(text: str) -> ClaudeUsage:
    s, w = _SESSION.search(text), _WEEK.search(text)
    if not s or not w:
        raise ValueError(f"could not find the session and week meters in the usage text: {text[:80]!r}")
    return ClaudeUsage(session_pct=float(s.group(1)), session_resets=s.group(2).strip(), week_pct=float(w.group(1)),
                       week_resets=w.group(2).strip())


def parse_codex_rate_limits(rollout_text: str) -> CodexLimits:
    """The last `rate_limits` reading in a Codex session file."""
    last = None
    for line in rollout_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload", event) if isinstance(event, dict) else {}
        if isinstance(payload, dict) and isinstance(payload.get("rate_limits"), dict):
            last = payload["rate_limits"]
    if last is None or not last.get("primary"):
        raise ValueError("no rate_limits reading in the session file")
    primary, secondary = last["primary"], last.get("secondary") or {}
    return CodexLimits(window_pct=float(primary["used_percent"]), week_pct=float(secondary.get("used_percent", 0.0)),
                       plan=last.get("plan_type"), window_resets_at=primary.get("resets_at"),
                       week_resets_at=secondary.get("resets_at"))


def estimate_calls(*, calls: int, delta_pct: float, step_pct: float = 1.0) -> Estimate:
    """Calls per 5-hour window from `calls` calls that moved the meter by `delta_pct`. A meter that did not move
    only bounds the cost: each call used less than `step_pct / calls` of the window."""
    if delta_pct <= 0:
        return Estimate(pct_per_call=None, calls_per_window=None, min_calls_per_window=int(calls * 100 // step_pct),
                        note=f"the meter did not move: each call used below {step_pct / calls:.3f}% "
                             f"(the meter resolution is {step_pct}%)")
    per_call = delta_pct / calls
    return Estimate(pct_per_call=per_call, calls_per_window=100 / per_call, min_calls_per_window=None,
                    note=f"{delta_pct}% over {calls} calls")
