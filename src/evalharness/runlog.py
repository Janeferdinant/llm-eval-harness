"""Run logs: one folder per run with run_log.md, and a summary.json in the shared schema."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel

CPU_BUSY_PCT = 20.0
GPU_BUSY_PCT = 10.0
_SECRET_NAME = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD)$|^(HF_TOKEN|GH_TOKEN|GITHUB_TOKEN)$")
_MIN_SECRET_LEN = 6


class BusyMachineError(RuntimeError):
    """Raised when a timed run is started on a machine that is not idle."""


@dataclass(frozen=True)
class MachineLoad:
    cpu_pct: float
    gpu_pct: float | None


class SummaryRow(BaseModel):
    name: str
    value: float
    ci_low: float | None
    ci_high: float | None
    n: int
    unit: str = ""


def _gpu_utilization() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return max(float(line) for line in out.split() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return out.strip().splitlines()[0] or None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def machine_load(sample_s: float = 5.0) -> MachineLoad:
    """Average CPU use over `sample_s` seconds, and GPU use at the end (None without nvidia-smi)."""
    cpu = psutil.cpu_percent(interval=sample_s)
    return MachineLoad(cpu_pct=cpu, gpu_pct=_gpu_utilization())


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _git_sha() -> str:
    return _git("rev-parse", "--short=7", "HEAD") or "unknown"


def _git_dirty(exclude: Path | None = None) -> bool:
    """Uncommitted changes, not counting the run output folder `exclude` (outputs are not code)."""
    status = _git("status", "--porcelain", "--untracked-files=all")
    if not status:
        return False
    prefix = None
    top = _git("rev-parse", "--show-toplevel")
    if exclude is not None and top:
        try:
            prefix = Path(exclude).resolve().relative_to(Path(top).resolve()).as_posix() + "/"
        except ValueError:
            prefix = None  # the run folder is outside the repo
    paths = [line[3:].strip('"').replace("\\", "/") for line in status.splitlines()]
    return any(not (prefix and p.startswith(prefix)) for p in paths)


def _redact(text: str) -> str:
    """Replace the values of secret-looking environment variables, read at call time."""
    for name, value in os.environ.items():
        if _SECRET_NAME.search(name) and len(value) >= _MIN_SECRET_LEN and value in text:
            text = text.replace(value, f"[REDACTED:{name}]")
    return text


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "run"


class RunContext:
    def __init__(self, path: Path, run_id: str, *, busy: bool, seed: int):
        self.path = path
        self.run_id = run_id
        self.busy = busy
        self.seed = seed
        self._log_file = path / "run_log.md"

    def log(self, msg: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%SZ")
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(f"- {stamp} {_redact(msg)}\n")

    def write_summary(self, rows: list[SummaryRow], *, status: Literal["Verified", "Reported"],
                      title: str) -> Path:
        if self.busy:
            title = f"{title} (busy machine)"
        summary = {
            "run_id": self.run_id,
            "date": self.run_id[:10],
            "git_sha": _git_sha(),
            "dirty": _git_dirty(self.path.parent),  # the output folder: run outputs are not code
            "status": status,
            "title": title,
            "machine": {"cpu": platform.processor() or platform.machine(), "gpu": _gpu_name(),
                        "busy": self.busy},
            "rows": [r.model_dump() for r in rows],
        }
        path = self.path / "summary.json"
        path.write_text(_redact(json.dumps(summary, indent=2)) + "\n", encoding="utf-8")
        self.log(f"wrote summary.json ({len(rows)} rows, status {status})")
        return path


def start_run(name: str, *, out_root: Path = Path("runs"), allow_busy: bool = False,
              seed: int = 0) -> RunContext:
    load = machine_load()
    busy = load.cpu_pct > CPU_BUSY_PCT or (load.gpu_pct is not None and load.gpu_pct > GPU_BUSY_PCT)
    if busy and not allow_busy:
        raise BusyMachineError(
            f"machine is busy (cpu {load.cpu_pct:.0f}%, gpu {load.gpu_pct}%); close programs or pass allow_busy=True")
    started = datetime.now(UTC)
    base_id = f"{started.strftime('%Y-%m-%dT%H-%M-%SZ')}-{_slug(name)}"
    run_id, n = base_id, 1
    while True:  # two runs with one name in one second get -2, -3, ...
        path = Path(out_root) / run_id
        try:
            path.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            n += 1
            run_id = f"{base_id}-{n}"
    ctx = RunContext(path, run_id, busy=busy, seed=seed)
    header = [
        f"# Run {run_id}", "",
        f"- name: {name}",
        f"- started: {started.isoformat(timespec='seconds')}",
        f"- git: {_git_sha()}{' (dirty)' if _git_dirty(path.parent) else ''}",
        f"- seed: {seed}",
        f"- python: {platform.python_version()} on {platform.platform()}",
        f"- load before start: cpu {load.cpu_pct:.0f}%, gpu {load.gpu_pct if load.gpu_pct is not None else 'n/a'}",
    ]
    if busy:
        header.append("- **busy machine**: timings are not clean (started with allow_busy)")
    header += ["", "## Log", ""]
    ctx._log_file.write_text("\n".join(header) + "\n", encoding="utf-8")
    return ctx
