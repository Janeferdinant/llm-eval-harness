"""API-equivalent cost from pricing.yaml. Unknown or expired prices give None, never 0."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

PRICING_PATH = Path(__file__).with_name("pricing.yaml")


class PriceEntry(BaseModel):
    provider: str
    model: str
    input_per_m: float
    output_per_m: float
    as_of: date
    valid_until: date | None = None


def load_pricing(path: str | Path = PRICING_PATH) -> list[PriceEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [PriceEntry(**e) for e in data.get("entries", [])]


def cost_usd(model: str, input_tokens: int, output_tokens: int, *, billing: Literal["local", "subscription", "api"],
             pricing: list[PriceEntry] | None = None, today: date | None = None) -> float | None:
    """Local calls cost 0 (electricity is not counted). Subscription calls are priced as if they were API
    calls (API-equivalent, not paid). A model with no current entry gives None."""
    if billing == "local":
        return 0.0
    today = today or date.today()
    for e in pricing if pricing is not None else load_pricing():
        if e.model == model and (e.valid_until is None or today <= e.valid_until):
            return round(input_tokens * e.input_per_m / 1e6 + output_tokens * e.output_per_m / 1e6, 10)
    return None
