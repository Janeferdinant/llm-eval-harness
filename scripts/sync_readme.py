"""Render the README Summary block from the runs listed in runs/published.txt.

    python scripts/sync_readme.py [--check] [--root PATH]

Without --check the README is rewritten. With --check nothing is written, and the exit code is 1
when the README differs from what the runs say (or a published run is missing).
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

START = "<!-- summary:start -->"
END = "<!-- summary:end -->"
EMPTY = "_No published runs yet._"


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _label(summary: dict) -> str:
    return f"Verified ({summary['date']})" if summary["status"] == "Verified" else "Reported"


def _render_run(summary: dict) -> str:
    machine = summary.get("machine", {})
    where = ", ".join(p for p in (machine.get("cpu"), machine.get("gpu")) if p)
    lines = [
        f"**{summary['title']}**: {_label(summary)}",
        "",
        f"Run `{summary['run_id']}`, commit `{summary['git_sha']}`{' (dirty)' if summary.get('dirty') else ''}"
        + (f", {where}" if where else "") + ".",
        "",
        "| Metric | Value | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for r in summary["rows"]:
        unit = f" {r['unit']}" if r.get("unit") else ""
        ci = "n/a" if r["ci_low"] is None or r["ci_high"] is None else f"{_fmt(r['ci_low'])} to {_fmt(r['ci_high'])}"
        lines.append(f"| {r['name']} | {_fmt(r['value'])}{unit} | {ci} | {r['n']} |")
    return "\n".join(lines)


def render(root: Path) -> str:
    published = root / "runs" / "published.txt"
    ids = [ln.strip() for ln in published.read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.lstrip().startswith("#")] if published.exists() else []
    blocks = []
    for run_id in ids:
        path = root / "runs" / run_id / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"published run {run_id} has no {path.relative_to(root)}")
        blocks.append(_render_run(json.loads(path.read_text(encoding="utf-8"))))
    return "\n\n".join(blocks) if blocks else EMPTY


def apply(readme: str, block: str) -> str:
    if readme.count(START) != 1 or readme.count(END) != 1 or readme.index(START) > readme.index(END):
        raise ValueError(f"README must contain exactly one {START} ... {END} pair")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{block}\n{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 when out of sync")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)
    root: Path = args.root
    readme_path = root / "README.md"
    try:
        current = readme_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        wanted = apply(current, render(root))
    except (FileNotFoundError, ValueError) as e:
        print(f"sync_readme: {e}", file=sys.stderr)
        return 1
    if wanted == current:
        return 0
    if args.check:
        diff = difflib.unified_diff(current.splitlines(), wanted.splitlines(), "README.md", "from runs", lineterm="")
        print("README Summary is out of sync with runs/:\n" + "\n".join(diff), file=sys.stderr)
        return 1
    readme_path.write_text(wanted, encoding="utf-8", newline="\n")
    print(f"updated {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
