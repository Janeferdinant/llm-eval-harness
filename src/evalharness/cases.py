"""Cases and suites: a suite is a folder with suite.yaml (name, version) and cases.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class Case(BaseModel):
    id: str
    input: dict
    reference: str | dict | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("case id must not be empty")
        return v


class Suite(BaseModel):
    name: str
    version: str
    cases: list[Case]

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
        return self


def load_suite(path: str | Path) -> Suite:
    folder = Path(path)
    meta_path, cases_path = folder / "suite.yaml", folder / "cases.jsonl"
    for p in (meta_path, cases_path):
        if not p.exists():
            raise FileNotFoundError(f"suite file missing: {p}")
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8-sig")) or {}
    for key in ("name", "version"):
        if key not in meta:
            raise ValueError(f"{meta_path} must define '{key}'")
    cases = []
    for lineno, line in enumerate(cases_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(Case(**json.loads(line)))
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise ValueError(f"{cases_path}: line {lineno}: {e}") from e
    return Suite(name=str(meta["name"]), version=str(meta["version"]), cases=cases)
