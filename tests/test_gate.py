import json
import math

import pytest

from evalharness.cli import main
from evalharness.gate import gate


def summary(tmp_path, name, rows):
    folder = tmp_path / name
    folder.mkdir()
    full = [{"name": n, "value": v, "ci_low": None, "ci_high": None, "n": 10, "unit": u} for n, v, u in rows]
    (folder / "summary.json").write_text(json.dumps({"run_id": name, "rows": full}), encoding="utf-8")
    return folder


def test_gate_fails_on_drop(tmp_path):
    base = summary(tmp_path, "base", [("token F1", 0.80, "")])
    cand = summary(tmp_path, "cand", [("token F1", 0.75, "")])
    result = gate(base, cand, max_drop=0.02)
    assert not result.passed and "token F1" in result.details[0] and "0.800" in result.details[0]
    assert main(["gate", "--baseline", str(base), "--candidate", str(cand), "--max-drop", "0.02"]) == 1


def test_gate_passes_within_tolerance(tmp_path):
    base = summary(tmp_path, "base", [("token F1", 0.80, "")])
    cand = summary(tmp_path, "cand", [("token F1", 0.79, "")])
    assert gate(base, cand, max_drop=0.02).passed
    assert main(["gate", "--baseline", str(base), "--candidate", str(cand), "--max-drop", "0.02"]) == 0


def test_gate_boundary_exactly_max_drop_passes(tmp_path):
    base = summary(tmp_path, "base", [("m", 0.80, "")])
    cand = summary(tmp_path, "cand", [("m", 0.78, "")])
    assert gate(base, cand, max_drop=0.02).passed


def test_gate_improvement_passes(tmp_path):
    base = summary(tmp_path, "base", [("m", 0.5, "")])
    assert gate(base, summary(tmp_path, "cand", [("m", 0.9, "")]), max_drop=0.0).passed


@pytest.mark.parametrize("candidate_rows", [[], [("other", 0.9, "")], [("m", math.nan, "")]])
def test_gate_fails_when_metric_missing_or_nan(tmp_path, candidate_rows):
    base = summary(tmp_path, "base", [("m", 0.8, "")])
    cand = summary(tmp_path, "cand", candidate_rows)
    result = gate(base, cand, max_drop=0.5)
    assert not result.passed and any("m" in d for d in result.details)


def test_gate_fails_when_baseline_has_nothing_to_compare(tmp_path):
    base = summary(tmp_path, "base", [("latency (s)", 1.0, "s")])
    result = gate(base, summary(tmp_path, "cand", [("latency (s)", 1.0, "s")]), max_drop=0.1)
    assert not result.passed and "nothing to compare" in result.details[0]


def test_gate_error_rate_rise_fails_and_latency_and_cost_are_not_gated(tmp_path):
    base = summary(tmp_path, "base", [("m", 0.8, ""), ("error rate", 0.0, ""), ("latency (s)", 1.0, "s"),
                                      ("cost per case, API-equivalent (USD)", 0.01, "USD")])
    slow = summary(tmp_path, "slow", [("m", 0.8, ""), ("error rate", 0.0, ""), ("latency (s)", 9.0, "s"),
                                      ("cost per case, API-equivalent (USD)", 0.5, "USD")])
    assert gate(base, slow, max_drop=0.02).passed
    flaky = summary(tmp_path, "flaky", [("m", 0.8, ""), ("error rate", 0.2, "")])
    result = gate(base, flaky, max_drop=0.02)
    assert not result.passed and any("error rate" in d for d in result.details)


def test_gate_accepts_summary_json_paths_directly(tmp_path):
    base = summary(tmp_path, "base", [("m", 0.8, "")]) / "summary.json"
    cand = summary(tmp_path, "cand", [("m", 0.8, "")]) / "summary.json"
    assert gate(base, cand, max_drop=0.01).passed


def test_gate_missing_path_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="summary"):
        gate(tmp_path / "nope", tmp_path / "nope2", max_drop=0.1)
