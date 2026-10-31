"""json_schema_valid: does the output parse into a pydantic model?"""
from __future__ import annotations

from pydantic import BaseModel, ValidationError

from evalharness.cases import Case
from evalharness.llm.base import extract_json
from evalharness.metrics.base import Output, Score


class _JsonSchemaValid:
    name = "json schema valid"

    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def score(self, case: Case, output: Output) -> Score:
        try:
            payload = output.data if output.data is not None else extract_json(output.text)
        except ValueError:
            return Score(value=0.0, detail="no JSON object found in the output")
        try:
            self.schema.model_validate(payload)
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "(root)"
            return Score(value=0.0, detail=f"{where}: {first['msg']}")
        return Score(value=1.0)


def json_schema_valid(schema: type[BaseModel]) -> _JsonSchemaValid:
    return _JsonSchemaValid(schema)
