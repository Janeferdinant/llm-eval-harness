"""Prepare the judge calibration set (run from the repo root).

    uv run python calibration/prepare.py fetch          download SQuAD 2.0 dev, sample 100 question IDs
    uv run python calibration/prepare.py answers [--fresh]  qwen3:8b answers the 100 questions (a Verified run;
                                                        --fresh uses an empty cache so timings are real)
    uv run python calibration/prepare.py sheet          write calibration/labels.csv for you to fill in
    uv run python calibration/prepare.py import-labels  check the human labels, write calibration/labels_public.csv

SQuAD 2.0 is CC BY-SA 4.0. The dev file and labels.csv (which holds SQuAD questions) stay in git-ignored places.
Only question_ids.txt, labels_public.csv (id, label) and the answers run are committed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from evalharness import calibration as cal
from evalharness.cases import Case, Suite
from evalharness.llm.base import Message
from evalharness.llm.cache import CachingClient
from evalharness.llm.ollama import OllamaClient
from evalharness.metrics import Output
from evalharness.runner import run

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "calibration"
WORK = CAL / "work"
DEV = WORK / "dev-v2.0.json"
IDS = CAL / "question_ids.txt"
SHEET = CAL / "labels.csv"
PUBLIC = CAL / "labels_public.csv"
ANSWERS_RUN = WORK / "answers_run.txt"
MODEL = "qwen3:8b"
SEED = 0


def read_ids() -> list[str]:
    return [line.strip() for line in IDS.read_text(encoding="utf-8").splitlines() if line.strip()]


def selected_items() -> list[dict]:
    wanted = set(read_ids())
    return [i for i in cal.load_squad(DEV) if i["id"] in wanted]


def fetch() -> None:
    cal.fetch_squad(DEV, expected_sha256=cal.SQUAD_DEV_SHA256)
    print(f"downloaded {DEV.stat().st_size} bytes, sha256 matches the pinned value")
    if not IDS.exists():
        ids = cal.sample_ids(cal.load_squad(DEV), n_answerable=60, n_unanswerable=40, seed=SEED)
        IDS.write_text("\n".join(ids) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(ids)} question IDs to {IDS}")
    else:
        print(f"{IDS} already exists; kept")


def answers(fresh: bool = False) -> None:
    items = selected_items()
    if len(items) != len(read_ids()):
        sys.exit("the dev file does not contain every sampled question ID")
    suite = Suite(name="squad2-calibration", version="1", cases=[
        Case(id=i["id"], input={"question": i["question"], "context": i["context"]},
             reference={"answers": i["answers"]}, tags=["unanswerable" if i["is_impossible"] else "answerable"])
        for i in items])
    llm = OllamaClient()
    digest = llm._digest(MODEL)
    if digest is None:
        sys.exit(f"{MODEL} is not pulled in Ollama")
    cache_dir = ROOT / ".cache" / (f"fresh-{time.strftime('%Y%m%d-%H%M%S')}" if fresh else "answers")
    client = CachingClient(llm, cache_dir, salt=f"{MODEL}@{digest}")
    by_id = {i["id"]: i for i in items}

    def system(case: Case) -> Output:
        system_text, user_text = cal.answer_prompt(by_id[case.id])
        c = client.complete([Message(role="system", content=system_text), Message(role="user", content=user_text)],
                            model=MODEL, temperature=0.0, max_tokens=1024, seed=SEED)
        return Output(text=c.text, completion=c)

    notes = {"model": MODEL, "model digest": digest, "temperature": 0.0, "seed": SEED, "max_tokens": 1024,
             "num_ctx": llm.num_ctx, "think": llm.think, "response cache": "fresh, empty" if fresh else "shared",
             "ollama": "localhost:11434"}
    result = run(suite, system, [cal.SquadF1Metric()], out_dir=ROOT / "runs", concurrency=1, notes=notes,
                 title=f"SQuAD 2.0 answers, {MODEL} (100 questions)")
    WORK.mkdir(exist_ok=True)
    ANSWERS_RUN.write_text(str(result.path), encoding="utf-8")
    empty = sum(1 for line in (result.path / "results.jsonl").read_text(encoding="utf-8").splitlines()
                if not json.loads(line)["output"])
    print(f"run folder: {result.path}\nerror rows: {result.n_errors}, empty answers: {empty}")


def sheet() -> None:
    run_dir = Path(ANSWERS_RUN.read_text(encoding="utf-8").strip())
    given = {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        given[row["case_id"]] = row["output"] or ""
    items = selected_items()
    missing = [i["id"] for i in items if not given.get(i["id"])]
    if missing:
        sys.exit(f"{len(missing)} questions have no answer; rerun `answers` first: {missing[:5]}")
    cal.write_sheet(items, given, SHEET)
    print(f"wrote {SHEET} with {len(items)} rows. Fill in the `label` column (see calibration/LABELLING.md).")


def import_labels() -> None:
    counts = cal.import_labels(SHEET, PUBLIC)
    print(f"wrote {PUBLIC}: {counts}")
    small = [name for name, n in counts.items() if n < 5]
    if small or len(counts) < 3:
        print("note: a label class has fewer than 5 items (or is missing); say so in the README limits")


if __name__ == "__main__":
    commands = {"fetch": fetch, "answers": answers, "sheet": sheet, "import-labels": import_labels}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    if sys.argv[1] == "answers":
        answers(fresh="--fresh" in sys.argv[2:])
    else:
        commands[sys.argv[1]]()
