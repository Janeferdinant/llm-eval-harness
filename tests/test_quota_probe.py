import json

import pytest

from evalharness.quota import estimate_calls, parse_claude_usage, parse_codex_rate_limits

CLAUDE_USAGE = (
    "You are currently using your subscription to power your Claude Code usage\n\n"
    "Current session: 7% used · resets Oct 31, 2:30am (America/Toronto)\n"
    "Current week (all models): 66% used · resets Nov 3, 9pm (America/Toronto)\n\n"
    "What's contributing to your limits usage?\n"
)


def test_parse_claude_usage_reads_both_meters():
    u = parse_claude_usage(CLAUDE_USAGE)
    assert u.session_pct == 7.0 and u.week_pct == 66.0
    assert "Oct 31" in u.session_resets and "Nov 3" in u.week_resets


def test_parse_claude_usage_accepts_decimals_and_rejects_other_text():
    assert parse_claude_usage("Current session: 7.5% used · resets x\nCurrent week (all models): 1% used · resets y\n"
                              ).session_pct == 7.5
    with pytest.raises(ValueError, match="usage"):
        parse_claude_usage("Unknown command: /usage")


def event(used_5h, used_week, kind="token_count"):
    return json.dumps({"type": "event_msg", "payload": {"type": kind, "rate_limits": {
        "primary": {"used_percent": used_5h, "window_minutes": 300, "resets_at": 1793428254},
        "secondary": {"used_percent": used_week, "window_minutes": 10080, "resets_at": 1793797018},
        "plan_type": "plus"}}})


def test_parse_codex_rate_limits_takes_the_last_reading():
    text = "\n".join([event(0.0, 50.0), json.dumps({"type": "other"}), event(1.0, 50.0)])
    r = parse_codex_rate_limits(text)
    assert (r.window_pct, r.week_pct, r.plan) == (1.0, 50.0, "plus")


def test_parse_codex_rate_limits_without_a_reading_is_an_error():
    with pytest.raises(ValueError, match="rate_limits"):
        parse_codex_rate_limits(json.dumps({"type": "other"}) + "\nnot json\n")


def test_estimate_calls_from_a_measured_delta():
    e = estimate_calls(calls=20, delta_pct=2.0)
    assert e.pct_per_call == pytest.approx(0.1) and e.calls_per_window == 1000


def test_estimate_calls_with_no_measurable_change_gives_only_a_bound():
    e = estimate_calls(calls=20, delta_pct=0.0, step_pct=1.0)
    assert e.pct_per_call is None and e.min_calls_per_window == 20 * 100 // 1 and "below" in e.note
