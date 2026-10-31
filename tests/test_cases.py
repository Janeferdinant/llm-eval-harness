import json

import pytest
import yaml

from evalharness.cases import Case, Suite, load_suite


def write_suite(folder, cases, *, meta=None, newline="\n", bom=False, blank_between=False):
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"name": "demo", "version": "1"} if meta is None else meta
    (folder / "suite.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    sep = newline + (newline if blank_between else "")
    body = sep.join(json.dumps(c, ensure_ascii=False) for c in cases) + newline
    data = body.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    (folder / "cases.jsonl").write_bytes(data)
    return folder


CASES = [
    {"id": "a", "input": {"question": "2+2?"}, "reference": "4"},
    {"id": "b", "input": {"question": "capital of France?"}, "reference": {"answer": "Paris"}, "tags": ["geo"]},
    {"id": "c", "input": {"question": "café ☕"}},
]


def test_load_suite_roundtrip(tmp_path):
    suite = load_suite(write_suite(tmp_path / "s", CASES))
    assert isinstance(suite, Suite) and suite.name == "demo" and suite.version == "1"
    assert suite.cases == [Case(**c) for c in CASES]
    assert suite.cases[1].reference == {"answer": "Paris"} and suite.cases[1].tags == ["geo"]
    assert suite.cases[2].reference is None and suite.cases[2].tags == []


def test_duplicate_case_ids_rejected(tmp_path):
    dup = [{"id": "dup-1", "input": {}}, {"id": "dup-1", "input": {}}]
    with pytest.raises(ValueError, match="dup-1"):
        load_suite(write_suite(tmp_path / "s", dup))


def test_suite_without_version_rejected(tmp_path):
    with pytest.raises(ValueError, match="version"):
        load_suite(write_suite(tmp_path / "s", CASES, meta={"name": "demo"}))


def test_load_suite_tolerates_bom_crlf_blank_lines(tmp_path):
    suite = load_suite(write_suite(tmp_path / "s", CASES, newline="\r\n", bom=True, blank_between=True))
    assert [c.id for c in suite.cases] == ["a", "b", "c"]


def test_case_with_empty_id_rejected():
    with pytest.raises(ValueError):
        Case(id="", input={})


def test_missing_files_give_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="suite.yaml"):
        load_suite(tmp_path / "empty")


def test_bad_jsonl_line_reports_line_number(tmp_path):
    folder = write_suite(tmp_path / "s", CASES)
    with (folder / "cases.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
    with pytest.raises(ValueError, match="line 4"):
        load_suite(folder)
