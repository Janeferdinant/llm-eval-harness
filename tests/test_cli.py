import json
from pathlib import Path

import pytest

from evalharness.cli import main

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO)


def run_demo(out, *extra):
    return main(["run", "suites/demo", "--system", "examples.demo_system:answer", "--metrics",
                 "exact_match,token_f1", "--out", str(out), "--title", "Demo run", *extra])


def only_run(out):
    folders = [p for p in Path(out).iterdir() if p.is_dir()]
    assert len(folders) == 1
    return folders[0]


def test_cli_run_end_to_end_with_mock(tmp_path, capsys):
    assert run_demo(tmp_path) == 0
    folder = only_run(tmp_path)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in summary["rows"]}
    assert rows["exact match"]["value"] == pytest.approx(0.7) and rows["exact match"]["n"] == 10
    assert 0.8 < rows["token F1"]["value"] < 0.95 and rows["error rate"]["value"] == 0.0
    assert "latency (s)" in rows
    out = capsys.readouterr().out
    assert "exact match" in out and str(folder.name) in out


def test_demo_run_matches_the_committed_baseline_so_the_gate_passes(tmp_path):
    assert run_demo(tmp_path) == 0
    baseline = REPO / "tests" / "fixtures" / "demo_baseline"
    assert main(["gate", "--baseline", str(baseline), "--candidate", str(only_run(tmp_path)),
                 "--max-drop", "0.001"]) == 0


def test_cli_compare_prints_delta_and_ci(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    assert run_demo(a) == 0
    assert main(["run", "suites/demo", "--system", "examples.demo_system:answer", "--metrics", "exact_match",
                 "--out", str(b), "--title", "Second"]) == 0
    capsys.readouterr()
    assert main(["compare", str(only_run(a)), str(only_run(b))]) == 0
    out = capsys.readouterr().out
    assert "exact match" in out and "delta" in out.lower() and "0.700" in out


def test_cli_smoke_requires_live_flag(capsys):
    assert main(["smoke", "--provider", "ollama"]) == 2
    assert "--live" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
def test_cli_smoke_refuses_pay_per_token_providers(provider, capsys):
    assert main(["smoke", "--live", "--provider", provider]) == 2
    assert "pay-per-token" in capsys.readouterr().err


def test_cli_smoke_unavailable_arm_explains(capsys):
    assert main(["smoke", "--live", "--provider", "gemini_cli"]) == 2
    assert "Antigravity" in capsys.readouterr().err


def test_cli_unknown_system_path_gives_clear_error(tmp_path, capsys):
    code = main(["run", "suites/demo", "--system", "no_such_module:fn", "--metrics", "exact_match",
                 "--out", str(tmp_path)])
    assert code == 2 and "no_such_module" in capsys.readouterr().err


def test_cli_system_without_colon_is_rejected(tmp_path, capsys):
    assert main(["run", "suites/demo", "--system", "examples.demo_system", "--metrics", "exact_match",
                 "--out", str(tmp_path)]) == 2
    assert "module:function" in capsys.readouterr().err


def test_cli_unknown_metric_lists_the_choices(tmp_path, capsys):
    code = main(["run", "suites/demo", "--system", "examples.demo_system:answer", "--metrics", "bleu",
                 "--out", str(tmp_path)])
    assert code == 2 and "exact_match" in capsys.readouterr().err


def test_cli_missing_suite_is_a_clear_error(tmp_path, capsys):
    assert main(["run", "suites/none", "--system", "examples.demo_system:answer", "--metrics", "exact_match",
                 "--out", str(tmp_path)]) == 2
    assert "suite" in capsys.readouterr().err


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "evalh" in capsys.readouterr().out
