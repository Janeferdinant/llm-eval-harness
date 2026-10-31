"""Clients for the subscription command line tools (Claude Code, Codex CLI) and their agent route.

Hosted arms are measured through the tools' own non-interactive modes. They are not raw-API numbers: the tool's
own system prompt is part of what is measured, and none of the tools accepts a temperature or a seed.
"""
from __future__ import annotations

import json
import re

LIMIT_PATTERN = re.compile(r"usage limit|hit your limit|limit reached|out of usage|rate.?limit", re.IGNORECASE)


class UnexpectedToolsError(RuntimeError):
    """The CLI started with, or used, a tool, MCP server, skill or plugin the run did not allow."""


class BillingError(RuntimeError):
    """The run would be billed per token or as overage, not against the subscription."""


def strict_schema(schema: dict) -> dict:
    """A JSON Schema in the strict form structured-output endpoints accept: every object closed
    (`additionalProperties: false`), every property required, and no `default` or schema `title` keywords."""
    if isinstance(schema, list):
        return [strict_schema(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for key, value in schema.items():
        if key == "default" or (key == "title" and isinstance(value, str)):
            continue
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: strict_schema(sub) for name, sub in value.items()}  # keys are names, not keywords
        else:
            out[key] = strict_schema(value)
    if out.get("type") == "object" or "properties" in out:
        out["required"] = list(out.get("properties", {}))
        out["additionalProperties"] = False
    return out


def parse_jsonl(stdout: str) -> list[dict]:
    """Every JSON object line of a CLI's output; other lines (banners, warnings) are skipped."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
