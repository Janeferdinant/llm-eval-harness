"""The regression gate: fail when a quality metric drops, or the error rate rises, by more than `max_drop`."""
from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel

from evalharness.runner import ERROR_ROW

LOWER_IS_BETTER = {ERROR_ROW}
NOT_GATED_UNITS = {"s", "USD"}  # latency and cost are reported, not gated: they vary run to run
TOLERANCE = 1e-9


class GateResult(BaseModel):
    passed: bool
    details: list[str]


def load_summary(path: str | Path) -> dict:
    p = Path(path)
    file = p / "summary.json" if p.is_dir() else p
    if not file.exists():
        raise FileNotFoundError(f"no summary.json at {p}")
    return json.loads(file.read_text(encoding="utf-8"))


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def gate(baseline: Path, candidate: Path, *, max_drop: float) -> GateResult:
    base_rows = {r["name"]: r for r in load_summary(baseline)["rows"]}
    cand_rows = {r["name"]: r for r in load_summary(candidate)["rows"]}
    gated = {n: r for n, r in base_rows.items() if r.get("unit", "") not in NOT_GATED_UNITS}
    if not any(n not in LOWER_IS_BETTER for n in gated):  # an error rate alone says nothing about quality
        return GateResult(passed=False, details=["nothing to compare: the baseline has no quality metric"])
    details: list[str] = []
    passed = True
    for name, base in gated.items():
        cand = cand_rows.get(name)
        if cand is None:
            details.append(f"{name}: missing from the candidate FAIL")
            passed = False
            continue
        b, c = _number(base.get("value")), _number(cand.get("value"))
        if math.isnan(c) or math.isnan(b):
            details.append(f"{name}: {b:.3f} -> {c:.3f} is not a number FAIL")
            passed = False
            continue
        if base.get("n") != cand.get("n"):
            details.append(f"{name}: different n (baseline {base.get('n')}, candidate {cand.get('n')}): "
                           "not the same sample FAIL")
            passed = False
            continue
        worse = (c - b) if name in LOWER_IS_BETTER else (b - c)
        failed = worse > max_drop + TOLERANCE
        passed = passed and not failed
        word = "rise" if name in LOWER_IS_BETTER else "drop"
        details.append(f"{name}: {b:.3f} -> {c:.3f} ({word} {max(worse, 0.0):.3f}, allowed {max_drop:.3f}) "
                       + ("FAIL" if failed else "ok"))
    return GateResult(passed=passed, details=details)
