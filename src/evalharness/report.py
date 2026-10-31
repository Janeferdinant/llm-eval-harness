"""Plain-text reports: a run summary, and a comparison of two runs with paired bootstrap deltas."""
from __future__ import annotations

import json
from pathlib import Path

from evalharness.gate import load_summary
from evalharness.stats import paired_bootstrap_delta


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def format_summary(summary: dict) -> str:
    lines = [f"{summary['title']}  [{summary['status']} {summary['date']}]  run {summary['run_id']}",
             f"{'metric':<42}{'value':>9}{'95% CI':>20}{'n':>7}"]
    for r in summary["rows"]:
        ci = "n/a" if r["ci_low"] is None else f"{_fmt(r['ci_low'])} to {_fmt(r['ci_high'])}"
        unit = f" {r['unit']}" if r.get("unit") else ""
        lines.append(f"{r['name']:<42}{_fmt(r['value']) + unit:>9}{ci:>20}{r['n']:>7}")
    return "\n".join(lines)


def case_scores(run: str | Path) -> dict[str, dict[str, float]]:
    """Per metric, the mean score of each case (over its repeats) from a run folder's results.jsonl."""
    path = Path(run) / "results.jsonl"
    if not path.exists():
        return {}
    sums: dict[str, dict[str, list[float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for metric, value in row["scores"].items():
            if value is not None:
                sums.setdefault(metric, {}).setdefault(row["case_id"], []).append(value)
    return {m: {cid: sum(v) / len(v) for cid, v in cases.items()} for m, cases in sums.items()}


def compare(a: str | Path, b: str | Path) -> str:
    """Both runs side by side. When both have per-case results, the delta (b minus a) gets a paired bootstrap
    interval over the cases they share."""
    sa, sb = load_summary(a), load_summary(b)
    rows_b = {r["name"]: r for r in sb["rows"]}
    ca, cb = case_scores(a), case_scores(b)
    lines = [f"A: {sa['title']} ({sa['run_id']})", f"B: {sb['title']} ({sb['run_id']})",
             f"{'metric':<42}{'A':>9}{'B':>9}{'delta (B-A)':>14}{'95% CI of delta':>22}{'paired n':>10}"]
    for r in sa["rows"]:
        other = rows_b.get(r["name"])
        if other is None:
            lines.append(f"{r['name']:<42}{_fmt(r['value']):>9}{'missing':>9}")
            continue
        delta = other["value"] - r["value"]
        ci, n = "n/a", ""
        common = sorted(set(ca.get(r["name"], {})) & set(cb.get(r["name"], {})))
        if len(common) >= 2:
            _, low, high = paired_bootstrap_delta([ca[r["name"]][c] for c in common],
                                                  [cb[r["name"]][c] for c in common])
            ci, n = f"{_fmt(low)} to {_fmt(high)}", str(len(common))
        lines.append(f"{r['name']:<42}{_fmt(r['value']):>9}{_fmt(other['value']):>9}{delta:>+14.3f}{ci:>22}{n:>10}")
    return "\n".join(lines)
