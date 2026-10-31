"""Quota probe for the two subscription CLIs (run from the repo root, ideally at the start of a fresh window).

    uv run python calibration/quota_probe.py claude codex

For each CLI: read the usage meter, make 20 short calls from an empty folder with the clean-run flags, read the
meter again, and turn the change into an estimate of calls per 5-hour window. The meters only have whole-percent
steps, so 20 short calls may not move them; then the result is a bound and the full run is the real measurement.
Nothing is written to the repo: results go to calibration/work/quota_probe.json (git-ignored) and stdout.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from evalharness.llm._proc import clean_env, resolve_exe, run_cli, run_in_empty_dir
from evalharness.llm.base import Message
from evalharness.llm.cli.claude_code import ClaudeCodeClient
from evalharness.llm.cli.codex import CLEAN_FLAGS as CODEX_FLAGS
from evalharness.quota import estimate_calls, parse_claude_usage, parse_codex_rate_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "calibration" / "work" / "quota_probe.json"
CALLS = 20
CODEX_MODEL = "gpt-6.1-sol"


def claude_meter():
    r = run_in_empty_dir(run_cli, [resolve_exe("claude"), "-p", "/usage", "--output-format", "json"], "",
                         env=clean_env(), timeout_s=120)
    return parse_claude_usage(json.loads(r.stdout)["result"])


def probe_claude() -> dict:
    before = claude_meter()
    client = ClaudeCodeClient()
    tokens_in = tokens_out = 0
    start = time.perf_counter()
    for i in range(CALLS):
        c = client.complete([Message(role="user", content=f"Reply with the single word OK. (probe {i})")],
                            model="default")
        tokens_in += c.input_tokens
        tokens_out += c.output_tokens
    elapsed = time.perf_counter() - start
    after = claude_meter()
    d_session, d_week = after.session_pct - before.session_pct, after.week_pct - before.week_pct
    return {"cli": "claude_code", "calls": CALLS, "seconds": round(elapsed, 1), "input_tokens": tokens_in,
            "output_tokens": tokens_out, "session_before": before.session_pct, "session_after": after.session_pct,
            "week_before": before.week_pct, "week_after": after.week_pct, "session_resets": before.session_resets,
            "week_resets": before.week_resets, "estimate_window": estimate_calls(calls=CALLS, delta_pct=d_session),
            "estimate_week": estimate_calls(calls=CALLS, delta_pct=d_week)}


def newest_rollout() -> Path:
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    files = sorted(home.glob("sessions/*/*/*/rollout-*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("no Codex session file found; run one `codex exec` without --ephemeral first")
    return files[-1]


def probe_codex() -> dict:
    before = parse_codex_rate_limits(newest_rollout().read_text(encoding="utf-8"))
    exe = resolve_exe("codex")
    argv = [exe, *[f for f in CODEX_FLAGS if f != "--ephemeral"], "-m", CODEX_MODEL, "-"]  # keep the session file
    tokens_in = tokens_out = 0
    start = time.perf_counter()
    for i in range(CALLS):
        r = run_in_empty_dir(run_cli, argv, f"Reply with the single word OK. (probe {i})", env=clean_env(),
                             timeout_s=300)
        for line in r.stdout.splitlines():
            if line.startswith("{") and '"turn.completed"' in line:
                usage = json.loads(line)["usage"]
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
    elapsed = time.perf_counter() - start
    after = parse_codex_rate_limits(newest_rollout().read_text(encoding="utf-8"))
    return {"cli": "codex_cli", "calls": CALLS, "seconds": round(elapsed, 1), "input_tokens": tokens_in,
            "output_tokens": tokens_out, "window_before": before.window_pct, "window_after": after.window_pct,
            "week_before": before.week_pct, "week_after": after.week_pct, "plan": before.plan,
            "estimate_window": estimate_calls(calls=CALLS, delta_pct=after.window_pct - before.window_pct),
            "estimate_week": estimate_calls(calls=CALLS, delta_pct=after.week_pct - before.week_pct)}


def main(which: list[str]) -> None:
    results = []
    for name in which:
        results.append({"claude": probe_claude, "codex": probe_codex}[name]())
        r = results[-1]
        print(json.dumps({k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in r.items()}, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([{k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in r.items()}
                               for r in results], indent=1), encoding="utf-8")


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if a in ("claude", "codex")]
    if not wanted:
        sys.exit(__doc__)
    main(wanted)
