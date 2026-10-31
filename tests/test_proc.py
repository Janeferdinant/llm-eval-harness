import sys
import time

import pytest

from evalharness.llm._proc import run_cli


def test_run_cli_returns_output_and_exit_code():
    code = "import sys; print(sys.stdin.read().upper())"
    r = run_cli([sys.executable, "-c", code], "hello", cwd=".", env={}, timeout_s=30)
    assert r.stdout.strip() == "HELLO" and r.returncode == 0


def test_run_cli_timeout_kills_the_whole_process_tree():
    """A grandchild that keeps the pipes open must not make the timeout hang (codex.CMD runs node under cmd.exe)."""
    code = ("import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)")
    start = time.perf_counter()
    with pytest.raises(TimeoutError, match="within 1 s"):
        run_cli([sys.executable, "-c", code], "", cwd=".", env={}, timeout_s=1)
    assert time.perf_counter() - start < 20


def test_a_process_still_holding_the_working_folder_does_not_crash_cleanup():
    """On Windows a folder cannot be deleted while a process (or an open file) is still using it."""
    from evalharness.llm._proc import ProcResult, run_in_empty_dir
    handles = []

    def runner(argv, stdin, *, cwd, env, timeout_s):
        handles.append(open(f"{cwd}/held.txt", "w"))  # still open when the folder is cleaned up
        return ProcResult(stdout="ok", stderr="", returncode=0)

    try:
        assert run_in_empty_dir(runner, ["x"], "", env={}, timeout_s=1).stdout == "ok"
    finally:
        for h in handles:
            h.close()
