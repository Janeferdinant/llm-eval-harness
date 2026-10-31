import json
import subprocess

from evalharness.runlog import SummaryRow, start_run


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd, check=True,
                   capture_output=True)


def summary_of(ctx):
    p = ctx.write_summary([SummaryRow(name="m", value=1.0, ci_low=None, ci_high=None, n=1)], status="Reported",
                          title="t")
    return json.loads(p.read_text(encoding="utf-8"))


def test_a_runs_own_output_does_not_make_the_tree_dirty(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    ctx = start_run("clean", out_root=tmp_path / "runs")
    ctx.log("some line")
    assert summary_of(ctx)["dirty"] is False


def test_other_uncommitted_changes_still_make_the_tree_dirty(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")  # an edit nobody committed
    ctx = start_run("dirty", out_root=tmp_path / "runs")
    assert summary_of(ctx)["dirty"] is True
    assert "(dirty)" in (ctx.path / "run_log.md").read_text(encoding="utf-8")


def test_earlier_uncommitted_runs_in_the_same_output_folder_do_not_count(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.chdir(tmp_path)
    start_run("first", out_root=tmp_path / "runs").log("x")
    second = start_run("second", out_root=tmp_path / "runs")
    assert summary_of(second)["dirty"] is False  # run outputs are not code
    (tmp_path / "elsewhere.txt").write_text("not a run", encoding="utf-8")
    assert summary_of(start_run("third", out_root=tmp_path / "runs"))["dirty"] is True
