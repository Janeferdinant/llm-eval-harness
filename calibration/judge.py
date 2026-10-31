"""Judge the 100 labelled answers and compare each judge with human labels (run from the repo root).

    uv run python calibration/judge.py local              qwen3:4b and qwen3:8b through Ollama
    uv run python calibration/judge.py baseline           token F1 as a judge that needs no LLM
    uv run python calibration/judge.py report             rebuild every judge's calibration summary (no model calls)
    (add --allow-busy when the desktop keeps the machine above the idle limits; the run is then labelled busy)
    uv run python calibration/judge.py claude --confirm   Claude Code (a subscription arm; see below)
    uv run python calibration/judge.py codex --confirm    Codex CLI, pinned to gpt-6.1-sol (a subscription arm)

The judges grade the answers stored in the labelling sheet, not freshly generated ones (local decoding is not
bit-reproducible). `--confirm` is required for the hosted arms: they use up subscription usage, should start in a
fresh 5-hour window at 0% usage, and the Claude arm runs on a Team plan (see NOTICE.md).
Each judge writes two runs: the judging run itself and a calibration summary (weighted kappa, length bias, time, cost).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evalharness import calibration as cal
from evalharness.cases import Case, Suite
from evalharness.llm.base import LLMClient
from evalharness.llm.cache import CachingClient
from evalharness.llm.cli.claude_code import ClaudeCodeClient
from evalharness.llm.cli.codex import CodexClient
from evalharness.llm.ollama import OllamaClient
from evalharness.metrics import Output, rubric_judge
from evalharness.runlog import start_run
from evalharness.runner import run

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "calibration"
SHEET = CAL / "labels.csv"
PUBLIC = CAL / "labels_public.csv"
DEV = CAL / "work" / "dev-v2.0.json"
SCALE = (0, 2)
RUBRIC = (
    "Grade the answer against the reference answer on a scale from 0 to 2.\n"
    "2 = correct: the answer says what the reference says. Extra words are fine if nothing in them is wrong.\n"
    "1 = partly correct: the right idea, but incomplete, too vague, or with a wrong extra detail.\n"
    "0 = wrong: a different answer, or no real answer.\n"
    "If the reference is '(unanswerable)', the answer is correct (2) only if it says the question cannot be answered, "
    "and wrong (0) if it gives an answer. Judge only against the reference, not against what you know.")


def load_inputs():
    rows = cal._read_rows(SHEET)
    labels = cal.load_labels(PUBLIC)
    answers = {r["id"]: r["model_answer"] for r in rows}
    cases = [Case(id=r["id"], input={"question": r["question"]}, reference=r["gold"]) for r in rows]
    return Suite(name="squad2-calibration-labelled", version="1", cases=cases), labels, answers


def write_calibration_summary(name: str, results_path: Path, source_busy: bool, judging_run: str) -> None:
    """Compare a finished judging run with human labels and write the calibration summary run (no model calls).
    The "busy machine" label follows the judging run, because that is when the latencies were measured."""
    _, labels, answers = load_inputs()
    got = cal.judge_inputs_from_run(results_path)
    report = cal.analyze_judge({i: labels[i] for i in got.scores}, got.scores, answers=answers,
                               latencies=got.latencies, costs=got.costs)
    ctx = start_run(f"Judge calibration vs labels, {name}", out_root=ROOT / "runs", allow_busy=True)
    ctx.busy = source_busy
    ctx.log(f"judging run: {judging_run}")
    ctx.log(f"items judged: {report.n} of {len(labels)}" + (f"; no score for {got.failed}" if got.failed else ""))
    ctx.log("label counts: " + ", ".join(f"{cal.LABEL_NAMES[k]} {sum(1 for v in labels.values() if v == k)}"
                                          for k in (2, 1, 0)))
    ctx.log("confusion matrix (rows: human label wrong/partly/correct; columns: judge score 0/1/2): "
            + str(report.confusion))
    ctx.log(f"weighted kappa (linear) {report.kappa:.3f}, 95% bootstrap CI {report.kappa_low:.3f} to "
            f"{report.kappa_high:.3f}; length vs judge {report.length_vs_judge:.3f}, length vs labels "
            f"{report.length_vs_labels:.3f}")
    path = ctx.write_summary(cal.summary_rows(name, report), status="Verified",
                             title=f"Judge calibration vs human labels: {name}")
    print(f"{name}: kappa {report.kappa:.3f} ({report.kappa_low:.3f} to {report.kappa_high:.3f}), "
          f"n {report.n}, confusion {report.confusion}\n  runs: {judging_run}, {path.parent.name}")


def judge_one(name: str, client: LLMClient, model: str, *, concurrency: int, notes: dict) -> None:
    suite, _, answers = load_inputs()
    metric = rubric_judge(client, model, RUBRIC, SCALE)
    result = run(suite, lambda case: Output(text=answers[case.id]), [metric], out_dir=ROOT / "runs",
                 concurrency=concurrency, allow_busy="--allow-busy" in sys.argv,
                 title=f"Judge: {name}, 100 labelled answers",
                 notes={"judge": name, "model": model, "scale": f"{SCALE[0]} to {SCALE[1]}", "temperature": 0.0,
                        "seed": 0, "max_tokens": 1024, **notes})
    write_calibration_summary(name, result.path / "results.jsonl", result.summary["machine"]["busy"],
                              result.path.name)


def report() -> None:
    """Rebuild the calibration summary of the newest judging run of every judge (no model calls)."""
    newest: dict[str, Path] = {}
    for folder in sorted((ROOT / "runs").glob("*-judge-*-100-labelled-answers")):
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        name = cal.judge_name_from_title(summary["title"])
        newest[name] = folder
    for name, folder in newest.items():
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        write_calibration_summary(name, folder / "results.jsonl", summary["machine"]["busy"], folder.name)


def local() -> None:
    llm = OllamaClient()
    for model in ("qwen3:4b", "qwen3:8b"):
        digest = llm._digest(model)
        if digest is None:
            sys.exit(f"{model} is not pulled in Ollama")
        client = CachingClient(llm, ROOT / ".cache" / f"judge-{model.replace(':', '-')}", salt=f"{model}@{digest}")
        judge_one(model, client, model, concurrency=1, notes={"model digest": digest, "num_ctx": llm.num_ctx,
                                                              "think": llm.think})


def baseline() -> None:
    rows = cal._read_rows(SHEET)
    labels = cal.load_labels(PUBLIC)
    wanted = {r["id"] for r in rows}
    items = [i for i in cal.load_squad(DEV) if i["id"] in wanted]
    answers = {r["id"]: r["model_answer"] for r in rows}
    report = cal.analyze_baseline(items, answers, labels)
    ctx = start_run("Token F1 baseline vs labels", out_root=ROOT / "runs", allow_busy=True)
    ctx.log(f"token F1 mapped to 0/1/2 with fixed thresholds {cal.F1_THRESHOLDS}; no LLM involved")
    path = ctx.write_summary(cal.baseline_summary_rows(report), status="Verified",
                             title="Token F1 as a judge, against human labels")
    print(f"token F1 baseline: Spearman {report.spearman:.3f}, weighted kappa {report.kappa:.3f}, n {report.n}\n"
          f"  run: {path.parent.name}")


def hosted(arm: str) -> None:
    if "--confirm" not in sys.argv:
        sys.exit(f"{arm} uses subscription usage: start in a fresh window at 0% usage, then pass --confirm "
                 "(Claude Code is on a Team plan; see NOTICE.md)")
    if arm == "claude":
        client: LLMClient = ClaudeCodeClient()
        name, model = "Claude Code (claude-sonnet-4-6, Team plan)", "default"
    else:
        client = CodexClient()
        name, model = "Codex CLI (gpt-6.1-sol)", "gpt-6.1-sol"
    cached = CachingClient(client, ROOT / ".cache" / f"judge-{arm}", salt="calibration")
    judge_one(name, cached, model, concurrency=2, notes={"arm": arm})


if __name__ == "__main__":
    commands = {"local": local, "baseline": baseline, "report": report, "claude": lambda: hosted("claude"),
                "codex": lambda: hosted("codex")}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
