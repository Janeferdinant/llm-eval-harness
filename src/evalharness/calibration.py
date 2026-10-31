"""Judge calibration: sample SQuAD 2.0 questions, build the labelling sheet, and compare a judge with human labels.

No SQuAD text is committed (it is CC BY-SA 4.0 and this repo is not). Only question IDs and the labels
(id, label) leave the working folder.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evalharness.cases import Case
from evalharness.metrics import Output, token_f1
from evalharness.runlog import SummaryRow
from evalharness.stats import cohen_kappa, spearman

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_DEV_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"  # downloaded 2026-10-30
LABEL_VALUES = {"wrong": 0, "partly": 1, "correct": 2}
LABEL_NAMES = {v: k for k, v in LABEL_VALUES.items()}
SHEET_COLUMNS = ["id", "question", "gold", "model_answer", "label"]
F1_THRESHOLDS = (0.3, 0.8)  # fixed in advance: F1 below 0.3 counts as wrong, below 0.8 as partly correct


# ---- data --------------------------------------------------------------------------------------

def fetch_squad(dest: Path, *, url: str = SQUAD_URL, expected_sha256: str | None = None,
                opener: Callable[[], bytes] | None = None) -> Path:
    """Download the SQuAD 2.0 dev set to `dest` after checking that it is that file. Nothing is written otherwise."""
    if opener is None:
        def opener() -> bytes:
            request = urllib.request.Request(url, headers={"User-Agent": "llm-eval-harness"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
    blob = opener()
    try:
        data = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"the download is not valid JSON ({e})") from e
    version = data.get("version") if isinstance(data, dict) else None
    if version != "v2.0" or not isinstance(data.get("data"), list):
        raise ValueError(f"expected the SQuAD v2.0 dev set, got version {version!r}")
    digest = hashlib.sha256(blob).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"sha256 mismatch: expected {expected_sha256}, got {digest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return dest


def load_squad(path: Path) -> list[dict]:
    """Flatten the SQuAD file into items: id, question, context, answers (texts), is_impossible."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                items.append({"id": qa["id"], "question": qa["question"], "context": paragraph["context"],
                              "answers": [a["text"] for a in qa.get("answers", [])],
                              "is_impossible": bool(qa.get("is_impossible", False))})
    return items


def sample_ids(items: Sequence[dict], *, n_answerable: int = 60, n_unanswerable: int = 40, seed: int = 0) -> list[str]:
    """A seeded sample of question IDs, stratified by whether the question can be answered. Sorted."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for label, n, wanted in (("answerable", n_answerable, False), ("unanswerable", n_unanswerable, True)):
        pool = sorted(i["id"] for i in items if i["is_impossible"] == wanted)
        if len(pool) < n:
            raise ValueError(f"not enough {label} questions: need {n}, have {len(pool)}")
        chosen += rng.sample(pool, n)
    return sorted(chosen)


def answer_prompt(item: dict) -> tuple[str, str]:
    system = ("Answer the question using only the paragraph. Reply with the shortest answer that fits, with no "
              "explanation. If the paragraph does not contain the answer, reply exactly: unanswerable")
    return system, f"Paragraph:\n{item['context']}\n\nQuestion: {item['question']}"


def gold_text(item: dict) -> str:
    unique = list(dict.fromkeys(item["answers"]))  # SQuAD repeats an answer once per annotator
    return " | ".join(unique) if unique else "(unanswerable)"


# ---- the labelling sheet -----------------------------------------------------------------------

def _read_rows(path: Path) -> list[dict]:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # a spreadsheet may have re-saved it in the Windows code page
        text = raw.decode("cp1252")
    return list(csv.DictReader(text.splitlines(), skipinitialspace=False))


def write_sheet(items: Sequence[dict], answers: dict[str, str], path: Path) -> Path:
    """The file you fill in: one row per question with an empty `label` column. Refuses to overwrite labels."""
    path = Path(path)
    if path.exists() and any((r.get("label") or "").strip() for r in _read_rows(path)):
        raise FileExistsError(f"{path} already has labels filled in; move it away before writing a new sheet")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in items:
            writer.writerow({"id": item["id"], "question": item["question"], "gold": gold_text(item),
                             "model_answer": answers[item["id"]], "label": ""})
    return path


def import_labels(sheet: Path, public: Path) -> dict[str, int]:
    """Check the filled-in sheet and write `id,label` (no SQuAD text). Returns the count per label."""
    rows = _read_rows(sheet)
    problems, out = [], []
    for r in rows:
        label = (r.get("label") or "").strip().lower()
        if label not in LABEL_VALUES:
            problems.append(f"{r['id']}: label {label!r} is not one of {sorted(LABEL_VALUES)}")
        else:
            out.append((r["id"], label))
    if problems:
        raise ValueError("labels need fixing:\n" + "\n".join(problems))
    Path(public).parent.mkdir(parents=True, exist_ok=True)
    with Path(public).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "label"])
        writer.writerows(out)
    return {name: sum(1 for _, lab in out if lab == name) for name in ("correct", "partly", "wrong")
            if any(lab == name for _, lab in out)}


def load_labels(public: Path) -> dict[str, int]:
    return {r["id"]: LABEL_VALUES[r["label"]] for r in _read_rows(public)}


# ---- scoring helpers ---------------------------------------------------------------------------

def squad_f1(gold: list[str], answer: str) -> float:
    """Token F1 against the best gold answer. With no gold answer, only abstaining scores (the SQuAD 2.0 rule)."""
    abstained = answer.strip().lower().rstrip(".") in {"", "unanswerable"}
    if not gold:
        return 1.0 if abstained else 0.0
    if abstained:
        return 0.0
    case = Case(id="x", input={}, reference={"answers": gold})
    return token_f1().score(case, Output(text=answer)).value


class SquadF1Metric:
    """squad_f1 as a Metric: the case reference is {"answers": [...]}, empty for an unanswerable question."""

    name = "SQuAD 2.0 F1"

    def score(self, case: Case, output: Output):
        from evalharness.metrics import Score
        ref = case.reference if isinstance(case.reference, dict) else {}
        return Score(value=squad_f1(list(ref.get("answers", [])), output.text))


# ---- analysis ----------------------------------------------------------------------------------

@dataclass
class JudgeReport:
    n: int
    kappa: float
    kappa_low: float
    kappa_high: float
    confusion: list[list[int]]  # rows: human label (0 wrong, 1 partly, 2 correct); columns: the judge's score
    length_vs_judge: float
    length_vs_labels: float
    total_latency_s: float | None
    total_cost_usd: float | None

    @property
    def length_bias_gap(self) -> float:
        return self.length_vs_judge - self.length_vs_labels


@dataclass
class BaselineReport:
    n: int
    spearman: float
    kappa: float


def _bootstrap_kappa(a: list[int], b: list[int], *, seed: int, n_resamples: int, level: float = 0.95):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    values = []
    for _ in range(n_resamples):
        pick = rng.choice(idx, size=len(idx), replace=True)
        k = cohen_kappa(a_arr[pick].tolist(), b_arr[pick].tolist(), weights="linear")
        if not np.isnan(k):
            values.append(k)
    if not values:
        return float("nan"), float("nan")
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return float(low), float(high)


@dataclass
class JudgeInputs:
    scores: dict[str, int]
    latencies: dict[str, float]
    costs: dict[str, float | None]
    failed: list[str]


def judge_inputs_from_run(results_path: Path, metric: str = "rubric judge") -> JudgeInputs:
    """Read a finished judging run back: the judge's 0-2 score, latency and cost for each case it scored."""
    inputs = JudgeInputs(scores={}, latencies={}, costs={}, failed=[])
    for line in Path(results_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        score = row["scores"].get(metric)
        if row.get("error") or score is None:
            inputs.failed.append(row["case_id"])
            continue
        call = row["metric_calls"][metric]
        inputs.scores[row["case_id"]] = int(score)
        inputs.latencies[row["case_id"]] = call["latency_s"]
        inputs.costs[row["case_id"]] = call["cost_usd"]
    return inputs


def analyze_judge(labels: dict[str, int], judge: dict[str, int], *, answers: dict[str, str],
                  latencies: dict[str, float] | None = None, costs: dict[str, float | None] | None = None,
                  seed: int = 0, n_resamples: int = 2000) -> JudgeReport:
    """Compare a judge's 0-2 scores with human labels: weighted kappa (linear) with a bootstrap interval over the
    items, the confusion matrix, and how answer length relates to the judge's scores versus to human labels."""
    if set(labels) != set(judge):
        raise ValueError("the labels and the judge's scores must cover the same items")
    ids = sorted(labels)
    a, b = [labels[i] for i in ids], [judge[i] for i in ids]
    kappa = cohen_kappa(a, b, weights="linear")
    low, high = _bootstrap_kappa(a, b, seed=seed, n_resamples=n_resamples)
    confusion = [[0] * 3 for _ in range(3)]
    for x, y in zip(a, b, strict=True):
        confusion[x][y] += 1
    length = [len(answers[i].split()) for i in ids]
    latency = sum(latencies[i] for i in ids) if latencies is not None else None
    cost = None
    if costs is not None and all(costs.get(i) is not None for i in ids):
        cost = float(sum(costs[i] for i in ids))
    return JudgeReport(n=len(ids), kappa=kappa, kappa_low=low, kappa_high=high, confusion=confusion,
                       length_vs_judge=spearman(length, b), length_vs_labels=spearman(length, a),
                       total_latency_s=latency, total_cost_usd=cost)


def analyze_baseline(items: Sequence[dict], answers: dict[str, str], labels: dict[str, int]) -> BaselineReport:
    """Token F1 as a judge that needs no LLM: its Spearman correlation with human labels, and its weighted kappa
    after mapping F1 to 0-2 with the fixed thresholds."""
    ids = [i["id"] for i in items]
    f1 = [squad_f1(i["answers"], answers[i["id"]]) for i in items]
    mapped = [0 if v < F1_THRESHOLDS[0] else 1 if v < F1_THRESHOLDS[1] else 2 for v in f1]
    given = [labels[i] for i in ids]
    return BaselineReport(n=len(ids), spearman=spearman(f1, given), kappa=cohen_kappa(given, mapped, weights="linear"))


def judge_name_from_title(title: str) -> str:
    """The judge's name from a judging run's title, `Judge: <name>, 100 labelled answers[ (busy machine)]`."""
    t = title.removesuffix(" (busy machine)")
    if not t.startswith("Judge: ") or not t.endswith(", 100 labelled answers"):
        raise ValueError(f"not the title of a judging run: {title!r}")
    return t.removeprefix("Judge: ").removesuffix(", 100 labelled answers")


def baseline_summary_rows(b: BaselineReport) -> list[SummaryRow]:
    rows = []
    if not np.isnan(b.spearman):
        rows.append(SummaryRow(name="token F1 baseline: Spearman with human labels", value=b.spearman, ci_low=None,
                               ci_high=None, n=b.n))
    if not np.isnan(b.kappa):
        rows.append(SummaryRow(name="token F1 baseline: weighted kappa vs human labels", value=b.kappa, ci_low=None,
                               ci_high=None, n=b.n))
    return rows


def summary_rows(judge_name: str, r: JudgeReport) -> list[SummaryRow]:
    rows = []
    if not np.isnan(r.kappa):
        rows.append(SummaryRow(name=f"weighted kappa vs human labels, {judge_name}", value=r.kappa,
                               ci_low=None if np.isnan(r.kappa_low) else r.kappa_low,
                               ci_high=None if np.isnan(r.kappa_high) else r.kappa_high, n=r.n))
    for label, value in ((f"answer length vs judge score (Spearman), {judge_name}", r.length_vs_judge),
                         (f"answer length vs human labels (Spearman), {judge_name}", r.length_vs_labels)):
        if not np.isnan(value):
            rows.append(SummaryRow(name=label, value=value, ci_low=None, ci_high=None, n=r.n))
    if r.total_latency_s is not None:
        rows.append(SummaryRow(name=f"latency per judgement, {judge_name}", value=r.total_latency_s / r.n,
                               ci_low=None, ci_high=None, n=r.n, unit="s"))
    if r.total_cost_usd is not None:
        rows.append(SummaryRow(name=f"cost per 100 judgements, API-equivalent (USD), {judge_name}",
                               value=r.total_cost_usd / r.n * 100, ci_low=None, ci_high=None, n=r.n, unit="USD"))
    return rows
