import csv
import hashlib
import json
import math

import pytest

from evalharness import calibration as cal


def squad_items(n_ans=30, n_imp=30):
    items = []
    for i in range(n_ans):
        items.append({"id": f"a{i:03d}", "question": f"Question {i}?", "context": f"Paragraph {i}.",
                      "answers": [f"answer {i}"], "is_impossible": False})
    for i in range(n_imp):
        items.append({"id": f"u{i:03d}", "question": f"Unanswerable {i}?", "context": f"Paragraph u{i}.",
                      "answers": [], "is_impossible": True})
    return items


def squad_json(items):
    qas_by_context: dict[str, list] = {}
    for it in items:
        qas = qas_by_context.setdefault(it["context"], [])
        qas.append({"id": it["id"], "question": it["question"], "is_impossible": it["is_impossible"],
                    "answers": [{"text": t, "answer_start": 0} for t in it["answers"]]})
    paragraphs = [{"context": c, "qas": q} for c, q in qas_by_context.items()]
    return {"version": "v2.0", "data": [{"title": "t", "paragraphs": paragraphs}]}


# ---- data --------------------------------------------------------------------------------------

def test_load_squad_flattens_paragraphs_and_questions(tmp_path):
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(squad_json(squad_items(3, 2))), encoding="utf-8")
    items = cal.load_squad(p)
    assert len(items) == 5 and {i["id"] for i in items} == {"a000", "a001", "a002", "u000", "u001"}
    assert next(i for i in items if i["id"] == "u000")["is_impossible"] is True
    assert next(i for i in items if i["id"] == "a001")["answers"] == ["answer 1"]


def test_sampling_is_deterministic_and_stratified():
    items = squad_items(80, 80)
    ids = cal.sample_ids(items, n_answerable=60, n_unanswerable=40, seed=0)
    assert ids == cal.sample_ids(list(reversed(items)), n_answerable=60, n_unanswerable=40, seed=0)
    assert len(ids) == 100 and len(set(ids)) == 100
    assert sum(i.startswith("a") for i in ids) == 60 and sum(i.startswith("u") for i in ids) == 40
    assert ids != cal.sample_ids(items, n_answerable=60, n_unanswerable=40, seed=1)


def test_sampling_fails_when_a_stratum_is_too_small():
    with pytest.raises(ValueError, match="answerable"):
        cal.sample_ids(squad_items(10, 50), n_answerable=60, n_unanswerable=40)


def test_fetch_squad_refuses_hash_mismatch_or_missing_file(tmp_path):
    good = json.dumps(squad_json(squad_items(2, 2))).encode("utf-8")
    src = tmp_path / "source.json"
    src.write_bytes(good)
    dest = tmp_path / "work" / "dev.json"
    cal.fetch_squad(dest, opener=lambda: src.read_bytes(), expected_sha256=hashlib.sha256(good).hexdigest())
    assert dest.read_bytes() == good
    bad_dest = tmp_path / "work2" / "dev.json"
    with pytest.raises(ValueError, match="sha256"):
        cal.fetch_squad(bad_dest, opener=lambda: src.read_bytes(), expected_sha256="0" * 64)
    assert not bad_dest.exists()
    with pytest.raises(ValueError, match="v2.0"):
        cal.fetch_squad(tmp_path / "w3" / "d.json", opener=lambda: b'{"version": "v1.1", "data": []}')
    with pytest.raises(ValueError, match="JSON"):
        cal.fetch_squad(tmp_path / "w4" / "d.json", opener=lambda: b"<html>not found</html>")
    with pytest.raises(FileNotFoundError):
        cal.load_squad(tmp_path / "nope.json")


# ---- prompt, sheet, labels ---------------------------------------------------------------------

def test_answer_prompt_has_paragraph_question_and_abstain_instruction():
    system, user = cal.answer_prompt({"question": "Who?", "context": "Ann did it."})
    assert "unanswerable" in system and "Ann did it." in user and "Who?" in user


def test_gold_text_marks_unanswerable_questions():
    assert cal.gold_text({"answers": ["a", "b"]}) == "a | b"
    assert cal.gold_text({"answers": []}) == "(unanswerable)"


def test_gold_text_drops_repeated_answers_keeping_order():
    item = {"answers": ["9th century", "9th century", "the 9th century", "9th century"]}
    assert cal.gold_text(item) == "9th century | the 9th century"


def test_sheet_has_empty_label_column_and_all_ids(tmp_path):
    items = squad_items(3, 2)
    answers = {i["id"]: f"model says {i['id']}" for i in items}
    path = tmp_path / "labels.csv"
    cal.write_sheet(items, answers, path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["id"] for r in rows] == [i["id"] for i in items]
    assert list(rows[0]) == ["id", "question", "gold", "model_answer", "label"]
    assert all(r["label"] == "" and r["model_answer"] for r in rows)
    assert rows[3]["gold"] == "(unanswerable)"


def test_sheet_refuses_to_overwrite_a_filled_in_sheet(tmp_path):
    items = squad_items(2, 0)
    path = tmp_path / "labels.csv"
    cal.write_sheet(items, {i["id"]: "x" for i in items}, path)
    text = path.read_text(encoding="utf-8").replace(",\r\n", ",correct\r\n", 1).replace(",\n", ",correct\n", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(FileExistsError, match="labels"):
        cal.write_sheet(items, {i["id"]: "x" for i in items}, path)


def fill(path, labels):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r, lab in zip(rows, labels, strict=True):
        r["label"] = lab
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_import_labels_validates_values(tmp_path):
    items = squad_items(3, 1)
    sheet = tmp_path / "labels.csv"
    cal.write_sheet(items, {i["id"]: "x" for i in items}, sheet)
    fill(sheet, ["Correct", " partly ", "WRONG", "correct"])
    public = tmp_path / "labels_public.csv"
    counts = cal.import_labels(sheet, public)
    assert counts == {"correct": 2, "partly": 1, "wrong": 1}
    with public.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["id", "label"] and [r["label"] for r in rows] == ["correct", "partly", "wrong", "correct"]
    assert cal.load_labels(public) == {"a000": 2, "a001": 1, "a002": 0, "u000": 2}

    fill(sheet, ["correct", "", "wrong", "mostly"])
    with pytest.raises(ValueError) as e:
        cal.import_labels(sheet, tmp_path / "x.csv")
    assert "a001" in str(e.value) and "u000" in str(e.value) and "mostly" in str(e.value)
    assert not (tmp_path / "x.csv").exists()


# ---- scoring helpers ---------------------------------------------------------------------------

def test_squad_f1_follows_the_squad2_abstention_rule():
    assert cal.squad_f1([], "unanswerable") == 1.0
    assert cal.squad_f1([], "The answer is 5") == 0.0
    assert cal.squad_f1(["5"], "unanswerable") == 0.0
    assert cal.squad_f1(["Eiffel Tower", "the tower"], "tower") == 1.0


# ---- analysis ----------------------------------------------------------------------------------

IDS = [f"i{i:03d}" for i in range(60)]
LABELS = {i: (n % 3) for n, i in enumerate(IDS)}


def test_analyze_weighted_kappa_on_synthetic_labels():
    perfect = cal.analyze_judge(LABELS, dict(LABELS), answers={i: "x" for i in IDS})
    assert perfect.kappa == 1.0 and perfect.kappa_low == 1.0 and perfect.n == 60
    shifted = {i: (v + 1) % 3 if n % 4 == 0 else v for n, (i, v) in enumerate(LABELS.items())}
    r = cal.analyze_judge(LABELS, shifted, answers={i: "x" for i in IDS})
    assert 0 < r.kappa < 1 and r.kappa_low <= r.kappa <= r.kappa_high


def test_analyze_is_reproducible_with_a_seed():
    shifted = {i: (v + 1) % 3 if n % 4 == 0 else v for n, (i, v) in enumerate(LABELS.items())}
    a = cal.analyze_judge(LABELS, shifted, answers={i: "x" for i in IDS}, seed=1)
    b = cal.analyze_judge(LABELS, shifted, answers={i: "x" for i in IDS}, seed=1)
    assert (a.kappa_low, a.kappa_high) == (b.kappa_low, b.kappa_high)


def test_analyze_confusion_matrix_counts():
    labels = {"a": 0, "b": 0, "c": 1, "d": 2, "e": 2}
    judge = {"a": 0, "b": 1, "c": 1, "d": 2, "e": 1}
    r = cal.analyze_judge(labels, judge, answers={k: "x" for k in labels})
    assert r.confusion == [[1, 1, 0], [0, 1, 0], [0, 1, 1]]  # rows: human label, columns: the judge's score


def test_analyze_requires_the_same_items():
    with pytest.raises(ValueError, match="same items"):
        cal.analyze_judge({"a": 0, "b": 1}, {"a": 0}, answers={"a": "x", "b": "y"})


def test_analyze_length_bias_correlation_sign():
    answers = {i: "word " * (n + 1) for n, i in enumerate(IDS)}
    labels = {i: (n * 7) % 3 for n, i in enumerate(IDS)}           # unrelated to length
    judge = {i: min(2, n // 20) for n, i in enumerate(IDS)}        # rises with length
    r = cal.analyze_judge(labels, judge, answers=answers)
    assert r.length_vs_judge > 0.7 and abs(r.length_vs_labels) < 0.3
    assert r.length_bias_gap == pytest.approx(r.length_vs_judge - r.length_vs_labels)


def test_analyze_records_time_and_cost_per_judgement():
    judge = dict(LABELS)
    r = cal.analyze_judge(LABELS, judge, answers={i: "x" for i in IDS}, latencies={i: 0.5 for i in IDS},
                          costs={i: 0.01 for i in IDS})
    assert r.total_latency_s == pytest.approx(30.0) and r.total_cost_usd == pytest.approx(0.6)
    unknown = cal.analyze_judge(LABELS, judge, answers={i: "x" for i in IDS}, costs={**{i: 0.01 for i in IDS[:-1]},
                                                                                   IDS[-1]: None})
    assert unknown.total_cost_usd is None


def test_token_f1_baseline_report_against_labels():
    items = squad_items(30, 30)
    answers = {i["id"]: (i["answers"][0] if n % 2 == 0 else "something else") if not i["is_impossible"]
               else ("unanswerable" if n % 3 else "a guess") for n, i in enumerate(items)}
    labels = {i["id"]: 2 if cal.squad_f1(i["answers"], answers[i["id"]]) == 1.0 else 0 for i in items}
    base = cal.analyze_baseline(items, answers, labels)
    assert base.spearman > 0.9 and base.kappa > 0.9 and base.n == 60


def test_summary_rows_have_n_and_ci():
    r = cal.analyze_judge(LABELS, dict(LABELS), answers={i: "x" for i in IDS}, latencies={i: 1.0 for i in IDS},
                          costs={i: 0.0 for i in IDS})
    rows = cal.summary_rows("qwen3:8b", r)
    names = [x.name for x in rows]
    assert "weighted kappa vs human labels, qwen3:8b" in names
    assert all(x.n == 60 for x in rows if x.name.startswith("weighted kappa"))
    assert any("latency" in n for n in names) and not any(math.isnan(x.value) for x in rows)


def test_squad_f1_metric_scores_case_references():
    from evalharness.cases import Case
    from evalharness.metrics import Output
    metric = cal.SquadF1Metric()
    answerable = Case(id="a", input={}, reference={"answers": ["Eiffel Tower"]})
    unanswerable = Case(id="u", input={}, reference={"answers": []})
    assert metric.name == "SQuAD 2.0 F1"
    assert metric.score(answerable, Output(text="the Eiffel Tower")).value == 1.0
    assert metric.score(unanswerable, Output(text="unanswerable")).value == 1.0
    assert metric.score(unanswerable, Output(text="Paris")).value == 0.0


def test_baseline_summary_rows():
    items = squad_items(30, 30)
    answers = {i["id"]: (i["answers"][0] if not i["is_impossible"] else "unanswerable") for i in items}
    labels = {i["id"]: 2 for i in items}
    labels["a000"] = 0
    answers["a000"] = "wrong"
    base = cal.analyze_baseline(items, answers, labels)
    rows = cal.baseline_summary_rows(base)
    names = [r.name for r in rows]
    assert names == ["token F1 baseline: Spearman with human labels",
                     "token F1 baseline: weighted kappa vs human labels"]
    assert all(r.n == 60 for r in rows)


def test_judge_name_from_a_judging_run_title():
    assert cal.judge_name_from_title("Judge: qwen3:8b, 100 labelled answers") == "qwen3:8b"
    assert cal.judge_name_from_title("Judge: qwen3:8b, 100 labelled answers (busy machine)") == "qwen3:8b"
    codex = "Judge: Codex CLI (gpt-6.1-sol), 100 labelled answers"
    assert cal.judge_name_from_title(codex) == "Codex CLI (gpt-6.1-sol)"
    with pytest.raises(ValueError, match="judging run"):
        cal.judge_name_from_title("Token F1 as a judge, against human labels")


def test_judge_inputs_from_run_reads_scores_latency_and_cost(tmp_path):
    def call(latency, cost):
        return {"rubric judge": {"latency_s": latency, "cost_usd": cost}}

    rows = [
        {"case_id": "a", "scores": {"rubric judge": 2.0}, "error": None, "metric_calls": call(1.5, 0.01)},
        {"case_id": "b", "scores": {"rubric judge": 0.0}, "error": None, "metric_calls": call(2.5, None)},
        {"case_id": "c", "scores": {"rubric judge": None}, "error": "JudgeError: x", "metric_calls": call(9.0, 0.5)},
    ]
    path = tmp_path / "results.jsonl"
    path.write_text(chr(10).join(json.dumps(r) for r in rows) + chr(10), encoding="utf-8")
    got = cal.judge_inputs_from_run(path)
    assert got.scores == {"a": 2, "b": 0} and got.latencies == {"a": 1.5, "b": 2.5}
    assert got.costs == {"a": 0.01, "b": None} and got.failed == ["c"]
