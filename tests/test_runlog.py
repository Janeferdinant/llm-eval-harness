import json
import subprocess

import pytest
from pydantic import ValidationError

from evalharness import runlog
from evalharness.runlog import BusyMachineError, MachineLoad, SummaryRow, start_run

REAL_MACHINE_LOAD = runlog.machine_load


@pytest.fixture(autouse=True)
def idle_machine(monkeypatch):
    """Skip the real 5 s load sample; tests that need a load patch it themselves."""
    monkeypatch.setattr(runlog, "machine_load", lambda sample_s=5.0: MachineLoad(cpu_pct=1.0, gpu_pct=None))


def test_runlog_redacts_env_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123456")  # prepublish-allow
    ctx = start_run("t", out_root=tmp_path, allow_busy=True)
    ctx.log("request used sk-test-123456")  # prepublish-allow
    assert "sk-test-123456" not in (ctx.path / "run_log.md").read_text(encoding="utf-8")  # prepublish-allow


def test_runlog_redaction_also_covers_summary_title(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abcdef123456")  # gitleaks:allow, fake value
    ctx = start_run("t", out_root=tmp_path, allow_busy=True)
    p = ctx.write_summary([SummaryRow(name="acc", value=1.0, ci_low=1.0, ci_high=1.0, n=1)],
                          status="Reported", title="run with hf_abcdef123456")
    assert "hf_abcdef123456" not in p.read_text(encoding="utf-8")


def test_busy_machine_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "machine_load", lambda sample_s=5.0: MachineLoad(cpu_pct=55.0, gpu_pct=None))
    with pytest.raises(BusyMachineError):
        start_run("t", out_root=tmp_path)
    ctx = start_run("t", out_root=tmp_path, allow_busy=True)
    assert "busy machine" in (ctx.path / "run_log.md").read_text(encoding="utf-8")


def test_busy_gpu_also_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "machine_load", lambda sample_s=5.0: MachineLoad(cpu_pct=2.0, gpu_pct=30.0))
    with pytest.raises(BusyMachineError):
        start_run("t", out_root=tmp_path)


def test_idle_machine_log_does_not_say_busy(tmp_path):
    ctx = start_run("t", out_root=tmp_path)
    assert "busy machine" not in (ctx.path / "run_log.md").read_text(encoding="utf-8")


def test_summary_title_marks_busy_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "machine_load", lambda sample_s=5.0: MachineLoad(cpu_pct=55.0, gpu_pct=None))
    ctx = start_run("t", out_root=tmp_path, allow_busy=True)
    p = ctx.write_summary([SummaryRow(name="acc", value=0.5, ci_low=None, ci_high=None, n=3)],
                          status="Reported", title="speed test")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "busy machine" in data["title"] and data["machine"]["busy"] is True


def test_summary_rows_have_n(tmp_path):
    ctx = start_run("t", out_root=tmp_path, allow_busy=True)
    with pytest.raises(ValidationError):
        SummaryRow(name="acc", value=0.9, ci_low=None, ci_high=None)          # n missing
    p = ctx.write_summary([SummaryRow(name="acc", value=0.9, ci_low=0.88, ci_high=0.92, n=800)],
                          status="Verified", title="t")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "Verified" and data["rows"][0]["n"] == 800 and len(data["git_sha"]) >= 7


def test_summary_matches_master_schema(tmp_path):
    ctx = start_run("squad-qwen3-8b", out_root=tmp_path, allow_busy=True)
    p = ctx.write_summary([SummaryRow(name="token F1", value=0.81, ci_low=0.79, ci_high=0.83, n=800)],
                          status="Verified", title="SQuAD answers, qwen3:8b")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert set(data) == {"run_id", "date", "git_sha", "dirty", "status", "title", "machine", "rows"}
    assert set(data["machine"]) == {"cpu", "gpu", "busy"}
    assert set(data["rows"][0]) == {"name", "value", "ci_low", "ci_high", "n", "unit"}
    assert data["run_id"].endswith("-squad-qwen3-8b") and p.parent == ctx.path


def test_machine_load_reports_cpu_percent():
    got = REAL_MACHINE_LOAD(sample_s=0.2)
    assert isinstance(got, MachineLoad) and 0.0 <= got.cpu_pct <= 100.0


def test_gpu_missing_is_none(monkeypatch):
    def no_smi(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", no_smi)
    assert runlog._gpu_utilization() is None


def test_two_runs_with_the_same_name_in_one_second_get_distinct_folders(tmp_path):
    first = start_run("same", out_root=tmp_path)
    second = start_run("same", out_root=tmp_path)
    assert first.path != second.path and first.run_id != second.run_id
    assert first.path.is_dir() and second.path.is_dir() and second.run_id.startswith(first.run_id[:19])
