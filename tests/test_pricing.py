from datetime import date

import yaml

from evalharness.llm.pricing import PRICING_PATH, cost_usd, load_pricing


def table(tmp_path, entries):
    p = tmp_path / "pricing.yaml"
    p.write_text(yaml.safe_dump({"entries": entries}), encoding="utf-8")
    return load_pricing(p)


def test_cost_from_pricing_table(tmp_path):
    entry = {"provider": "x", "model": "m", "input_per_m": 1.0, "output_per_m": 2.0, "as_of": "2026-10-30"}
    t = table(tmp_path, [entry])
    assert cost_usd("m", 1000, 500, billing="api", pricing=t) == 0.002


def test_unknown_model_cost_is_none_not_zero(tmp_path):
    t = table(tmp_path, [])
    assert cost_usd("mystery", 1000, 500, billing="api", pricing=t) is None
    assert cost_usd("mystery", 1000, 500, billing="subscription", pricing=t) is None


def test_local_billing_is_zero_even_for_unknown_model(tmp_path):
    assert cost_usd("qwen3:8b", 1000, 500, billing="local", pricing=table(tmp_path, [])) == 0.0


def test_expired_price_is_none(tmp_path):
    t = table(tmp_path, [{"provider": "x", "model": "m", "input_per_m": 1.0, "output_per_m": 2.0,
                          "as_of": "2026-10-30", "valid_until": "2026-11-30"}])
    assert cost_usd("m", 1000, 500, billing="api", pricing=t, today=date(2026, 11, 30)) == 0.002
    assert cost_usd("m", 1000, 500, billing="api", pricing=t, today=date(2026, 12, 1)) is None


def test_shipped_pricing_file_entries_are_dated_and_complete():
    entries = load_pricing(PRICING_PATH)
    assert entries, "pricing.yaml must not be empty"
    for e in entries:
        assert e.as_of and e.input_per_m >= 0 and e.output_per_m >= 0 and e.provider and e.model
