"""The `evalh` command: run a suite, compare two runs, gate a candidate against a baseline, smoke-test a provider."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from evalharness.cases import load_suite
from evalharness.gate import gate
from evalharness.llm.base import ArmUnavailableError, Message, UsageLimitError, get_client
from evalharness.metrics import exact_match, token_f1
from evalharness.report import compare, format_summary
from evalharness.runlog import BusyMachineError
from evalharness.runner import run

METRICS = {"exact_match": exact_match, "token_f1": token_f1}
LIVE_PROVIDERS = {"ollama", "openai_compat", "claude_code", "codex_cli", "gemini_cli"}
PAY_PER_TOKEN = {"openai", "anthropic", "gemini"}


class CliError(Exception):
    """A usage problem: printed as `evalh: <message>` with exit code 2."""


def _load_system(spec: str):
    if ":" not in spec:
        raise CliError(f"--system must look like module:function, got {spec!r}")
    module_name, _, func_name = spec.partition(":")
    sys.path.insert(0, os.getcwd())
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise CliError(f"cannot import system module {module_name!r}: {e}") from e
    try:
        return getattr(module, func_name)
    except AttributeError as e:
        raise CliError(f"module {module_name!r} has no function {func_name!r}") from e


def _cmd_run(a: argparse.Namespace) -> int:
    names = [n.strip() for n in a.metrics.split(",") if n.strip()]
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        raise CliError(f"unknown metric(s) {unknown}; the command line offers {sorted(METRICS)} "
                       "(json_schema_valid and rubric_judge need parameters: use the Python API)")
    try:
        suite = load_suite(a.suite)
    except (FileNotFoundError, ValueError) as e:
        raise CliError(f"suite problem: {e}") from e
    system = _load_system(a.system)
    result = run(suite, system, [METRICS[n]() for n in names], out_dir=Path(a.out), concurrency=a.concurrency,
                 repeats=a.repeats, status=a.status, title=a.title or f"{suite.name} ({a.system})",
                 allow_busy=a.allow_busy)
    print(format_summary(result.summary))
    print(f"run folder: {result.path}")
    return 0


def _cmd_compare(a: argparse.Namespace) -> int:
    print(compare(a.a, a.b))
    return 0


def _cmd_gate(a: argparse.Namespace) -> int:
    result = gate(Path(a.baseline), Path(a.candidate), max_drop=a.max_drop)
    print("\n".join(result.details))
    print("gate: PASSED" if result.passed else "gate: FAILED")
    return 0 if result.passed else 1


def _cmd_smoke(a: argparse.Namespace) -> int:
    if a.provider in PAY_PER_TOKEN:
        raise CliError(f"{a.provider} is a pay-per-token client: it is mock-tested only and never called live "
                       "from this tool")
    if not a.live:
        raise CliError("smoke makes one real call; pass --live to allow it")
    if a.provider not in LIVE_PROVIDERS:
        raise CliError(f"smoke supports {sorted(LIVE_PROVIDERS)}, got {a.provider!r}")
    kw = {"base_url": a.base_url} if a.base_url else {}
    try:
        client = get_client(a.provider, **kw)
    except ArmUnavailableError as e:
        raise CliError(str(e)) from e
    except TypeError as e:
        raise CliError(f"cannot build {a.provider}: {e} (openai_compat needs --base-url)") from e
    model = a.model or ("qwen3:1.7b" if a.provider == "ollama" else "default")
    out = client.complete([Message(role="user", content="Reply with the single word OK")], model=model,
                          max_tokens=1024)
    cost = {"local": "0 (local; electricity not counted)",
            "subscription": f"{out.cost_usd} USD API-equivalent (not paid)" if out.cost_usd is not None
            else "unknown (no price for this model)",
            "api": f"{out.cost_usd} USD"}[out.billing]
    print(f"provider: {a.provider}\nmodel: {out.model}\nversion/digest: {out.model_digest}\nbilling: {out.billing}\n"
          f"reply: {out.text!r}\nlatency: {out.latency_s:.2f} s\ntokens: {out.input_tokens} in, "
          f"{out.output_tokens} out\ncost: {cost}")
    return 0 if "OK" in out.text.upper() else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalh", description="Run, compare and gate LLM evaluation suites.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a suite with a system and metrics")
    r.add_argument("suite")
    r.add_argument("--system", required=True, help="module:function taking a Case and returning an Output")
    r.add_argument("--metrics", required=True, help="comma-separated: " + ", ".join(sorted(METRICS)))
    r.add_argument("--out", default="runs")
    r.add_argument("--title")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--status", choices=["Verified", "Reported"], default="Verified")
    r.add_argument("--allow-busy", action="store_true", help="run on a busy machine (the run is labelled so)")
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("compare", help="compare two run folders")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=_cmd_compare)

    g = sub.add_parser("gate", help="exit 1 when the candidate is worse than the baseline")
    g.add_argument("--baseline", required=True)
    g.add_argument("--candidate", required=True)
    g.add_argument("--max-drop", type=float, required=True)
    g.set_defaults(func=_cmd_gate)

    s = sub.add_parser("smoke", help="one real call to a provider")
    s.add_argument("--provider", required=True)
    s.add_argument("--model")
    s.add_argument("--base-url", help="for openai_compat")
    s.add_argument("--live", action="store_true", help="required: this makes a real call")
    s.set_defaults(func=_cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as e:
        print(f"evalh: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"evalh: {e}", file=sys.stderr)
        return 2
    except BusyMachineError as e:
        print(f"evalh: {e}", file=sys.stderr)
        return 2
    except UsageLimitError as e:
        print(f"evalh: usage limit reached ({e.provider}); finished work is saved, rerun to resume", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
