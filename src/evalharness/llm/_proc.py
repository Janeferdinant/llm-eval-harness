"""Running a command line tool the clean way: stdin prompt, empty working folder, no billing variables."""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from evalharness.llm.base import Message

# Anything that could move a run onto a paid or different endpoint: API keys and tokens, base URLs and model
# overrides, and the cloud-provider switches of Claude Code. CODEX_HOME and CLAUDE_CONFIG_DIR are kept: they say
# where the subscription login lives.
_DROP = re.compile(r"(API_KEY|AUTH_TOKEN|BASE_URL)$|^(ANTHROPIC|OPENAI|GEMINI|AWS|AZURE)_|^CLAUDE_CODE_USE_|"
                   r"^GOOGLE_(API_KEY|GENAI)", re.IGNORECASE)


@dataclass
class ProcResult:
    stdout: str
    stderr: str
    returncode: int


Runner = Callable[..., ProcResult]  # (argv, stdin, *, cwd, env, timeout_s) -> ProcResult


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The current environment without billing or routing variables, so a stray key or gateway cannot switch a
    CLI to pay-per-token billing or to another endpoint."""
    env = {k: v for k, v in os.environ.items() if not _DROP.search(k)}
    env.update(extra or {})
    return env


def resolve_exe(name: str) -> str:
    """Full path of a command, found with shutil.which so Windows shims such as codex.cmd work."""
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"command line tool not found on PATH: {name}")
    return path


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and everything it started. On Windows a .cmd shim runs node under cmd.exe, and killing
    only cmd.exe leaves node holding the pipes, so a plain timeout would wait for it forever."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()


def run_cli(argv: list[str], stdin: str, *, cwd: str | Path, env: dict[str, str], timeout_s: float) -> ProcResult:
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", cwd=cwd, env=env, **group)
    try:
        out, err = proc.communicate(stdin, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        raise TimeoutError(f"{argv[0]} did not finish within {timeout_s:.0f} s") from e
    return ProcResult(stdout=out or "", stderr=err or "", returncode=proc.returncode)


def run_in_empty_dir(runner: Runner, argv: list[str], stdin: str, *, env: dict[str, str], timeout_s: float,
                     workdir: Path | None = None) -> ProcResult:
    """Run from a fresh empty temp folder (or the given `workdir`)."""
    if workdir is not None:
        return runner(argv, stdin, cwd=workdir, env=env, timeout_s=timeout_s)
    # ignore_cleanup_errors: on Windows a folder a child process still uses cannot be deleted yet; the leftover
    # empty folder in the temp directory is harmless, a crash after a successful (and paid) call is not.
    with tempfile.TemporaryDirectory(prefix="evalh-", ignore_cleanup_errors=True) as empty:
        return runner(argv, stdin, cwd=empty, env=env, timeout_s=timeout_s)


def render_prompt(messages: list[Message]) -> tuple[str | None, str]:
    """System text (kept apart) and the prompt. One user message is sent as it is; a longer conversation
    becomes a labelled transcript."""
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    turns = [m for m in messages if m.role != "system"]
    if len(turns) == 1 and turns[0].role == "user":
        return system, turns[0].content
    label = {"user": "User", "assistant": "Assistant", "tool": "Tool result"}
    return system, "\n\n".join(f"{label[m.role]}: {m.content}" for m in turns)
