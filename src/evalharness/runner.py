"""The runner: run a system over a suite, score it, and write run_log.md, results.jsonl and summary.json."""
from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evalharness.cases import Case, Suite
from evalharness.llm.base import UsageLimitError
from evalharness.metrics.base import Metric
from evalharness.runlog import RunContext, SummaryRow, start_run
from evalharness.stats import bootstrap_ci

LATENCY_ROW = "latency (s)"
COST_ROW = "cost per case, API-equivalent (USD)"
ERROR_ROW = "error rate"


@dataclass
class RunResult:
    path: Path
    summary: dict
    n_errors: int
    scores: dict[str, list[float | None]]  # per metric, one entry per case (repeats averaged), None if all failed

    @property
    def run_id(self) -> str:
        return self.summary["run_id"]


def _takes_repeat(system: Callable) -> bool:
    """True when `system` accepts a second positional argument, which is then the repeat index."""
    try:
        params = list(inspect.signature(system).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params)


def _score_case(case: Case, repeat: int, system: Callable, metrics: list[Metric], stop: threading.Event,
                takes_repeat: bool) -> dict | None:
    if stop.is_set():
        return None
    row: dict = {"case_id": case.id, "repeat": repeat, "output": None, "scores": {m.name: None for m in metrics},
                 "details": {}, "error": None, "latency_s": None, "input_tokens": None, "output_tokens": None,
                 "cost_usd": None, "model": None, "model_digest": None, "billing": None, "metric_calls": {}}
    try:
        output = system(case, repeat) if takes_repeat else system(case)
    except UsageLimitError:
        stop.set()
        raise
    except Exception as e:  # a failing case is recorded, never dropped
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    row["output"] = output.text
    c = output.completion
    if c is not None:
        row.update(latency_s=c.latency_s, input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                   cost_usd=c.cost_usd, model=c.model, model_digest=c.model_digest, billing=c.billing)
    errors = []
    for m in metrics:
        try:
            s = m.score(case, output)
            row["scores"][m.name] = s.value
            if s.detail:
                row["details"][m.name] = s.detail
        except UsageLimitError:
            stop.set()
            raise
        except Exception as e:
            errors.append(f"{m.name}: {type(e).__name__}: {e}")
        call = getattr(m, "completions", {}).get(case.id)  # a metric that calls a model (a judge) exposes its call
        if call is not None:
            row["metric_calls"][m.name] = {
                "latency_s": call.latency_s, "cost_usd": call.cost_usd, "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens, "model": call.model, "model_digest": call.model_digest,
                "billing": call.billing}
    row["error"] = "; ".join(errors) or None
    return row


def _case_means(rows: list[dict], n_cases: int, repeats: int,
                pick: Callable[[dict], float | None]) -> list[float | None]:
    """One value per case: the mean over its repeats of `pick(row)`, ignoring repeats where it is None."""
    out: list[float | None] = []
    for ci in range(n_cases):
        values = [pick(r) for r in rows[ci * repeats:(ci + 1) * repeats]]
        values = [v for v in values if v is not None]
        out.append(sum(values) / len(values) if values else None)
    return out


def run(suite: Suite, system: Callable, metrics: list[Metric], *, out_dir: Path,
        concurrency: int = 4, repeats: int = 1, status: Literal["Verified", "Reported"] = "Verified",
        title: str, allow_busy: bool = False, seed: int = 0, notes: dict | None = None) -> RunResult:
    """Score `system` on every case of `suite`, `repeats` times each, `concurrency` at a time.

    `n` in the summary counts cases, not repeats: repeats of one case are averaged first, and the bootstrap
    resamples cases. Failed cases stay in results.jsonl with an error and count in the `error rate` row.
    A UsageLimitError saves the finished rows and is re-raised; rerun with the same cache to resume.

    `system` may be `f(case)` or `f(case, repeat)`. With repeats above 1, only the second form can vary its
    request per repeat; a response cache would otherwise answer every repeat from the first one.
    `notes` (temperature, seed, context size, ...) are written to run_log.md as `config:` lines."""
    names = [m.name for m in metrics]
    for name in names:
        if names.count(name) > 1:
            raise ValueError(f"duplicate metric name: {name}")
    ctx = start_run(title, out_root=out_dir, allow_busy=allow_busy, seed=seed)
    n_cases = len(suite.cases)
    ctx.log(f"suite {suite.name} v{suite.version}: {n_cases} cases, {repeats} repeat(s), concurrency {concurrency}")
    ctx.log(f"metrics: {', '.join(names)}")
    for key, value in (notes or {}).items():
        ctx.log(f"config: {key} = {value}")
    takes_repeat = _takes_repeat(system)
    if repeats > 1 and not takes_repeat:
        ctx.log("note: repeats > 1 but the system takes no repeat argument, so repeats send identical requests; "
                "behind a response cache they are copies of the first answer")

    order = [(ci, r) for ci in range(n_cases) for r in range(repeats)]
    results: dict[tuple[int, int], dict] = {}
    stop = threading.Event()
    limit: UsageLimitError | None = None
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_score_case, suite.cases[ci], r, system, metrics, stop, takes_repeat): (ci, r)
                   for ci, r in order}
        for future in as_completed(futures):
            try:
                row = future.result()
            except UsageLimitError as e:
                limit = limit or e
                for pending in futures:
                    pending.cancel()
                continue
            except Exception:  # cancelled futures
                continue
            if row is not None:
                results[futures[future]] = row

    rows = [results[k] for k in order if k in results]
    _write_rows(ctx, rows)
    if limit is not None:
        ctx.log(f"stopped at a usage limit ({limit.provider}): {len(rows)} of {len(order)} calls finished and "
                f"saved; rerun with the same cache to resume")
        raise limit

    n_errors = sum(1 for r in rows if r["error"])
    ctx.log(f"{len(rows)} rows written, {n_errors} with errors")
    for model, digest, billing in sorted({(r["model"], r["model_digest"], r["billing"]) for r in rows if r["model"]},
                                         key=str):
        ctx.log(f"model {model}, version/digest {digest}, billing {billing}")

    scores = {m.name: _case_means(rows, n_cases, repeats, lambda r, n=m.name: r["scores"][n]) for m in metrics}
    summary_rows: list[SummaryRow] = []
    for name, per_case in scores.items():
        values = [v for v in per_case if v is not None]
        if not values:
            ctx.log(f"no valid scores for {name}")
            continue
        mean, low, high = bootstrap_ci(values, seed=seed)
        summary_rows.append(SummaryRow(name=name, value=mean, ci_low=low, ci_high=high, n=len(values)))
    summary_rows.append(SummaryRow(name=ERROR_ROW, value=n_errors / len(rows) if rows else 0.0, ci_low=None,
                                   ci_high=None, n=len(rows)))
    with_completion = [r for r in rows if r["latency_s"] is not None]
    if with_completion:
        lat = [v for v in _case_means(rows, n_cases, repeats, lambda r: r["latency_s"]) if v is not None]
        mean, low, high = bootstrap_ci(lat, seed=seed)
        summary_rows.append(SummaryRow(name=LATENCY_ROW, value=mean, ci_low=low, ci_high=high, n=len(lat), unit="s"))
        if all(r["cost_usd"] is not None for r in with_completion):
            cost = [v for v in _case_means(rows, n_cases, repeats, lambda r: r["cost_usd"]) if v is not None]
            mean, low, high = bootstrap_ci(cost, seed=seed)
            summary_rows.append(SummaryRow(name=COST_ROW, value=mean, ci_low=low, ci_high=high, n=len(cost),
                                           unit="USD"))
        else:
            ctx.log("cost row left out: at least one call has no known price")
    path = ctx.write_summary(summary_rows, status=status, title=title)
    return RunResult(path=ctx.path, summary=json.loads(path.read_text(encoding="utf-8")), n_errors=n_errors,
                     scores=scores)


def _write_rows(ctx: RunContext, rows: list[dict]) -> None:
    with (ctx.path / "results.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
