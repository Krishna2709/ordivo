"""Public report rendering rejects individual data and keeps evidence links local."""

from pathlib import Path

import pytest

from scripts import build_v2_evidence_report as report


def row() -> dict:
    return {
        "run_id": "v2_pilot", "commit": "a" * 40, "source": "expedia",
        "task": "observed_choice", "comparison": "base vs adapted",
        "metric": "ndcg@5", "value": 0.42, "eligible_groups": 100,
        "population": "held-out search groups", "split_sha256": "b" * 64,
        "receipt": "reports/runs/v2_pilot/aggregate.json",
    }


def payload(record: dict) -> dict:
    return {"schema_version": "ordivo-v2-public-aggregates-v1", "rows": [record]}


def test_render_escapes_public_text_and_links_checked_in_receipt(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "reports/runs/v2_pilot/aggregate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    record = row()
    record["comparison"] = "base < adapted"
    output = report.render_rows(payload(record))
    assert "base &lt; adapted" in output
    assert 'href="runs/v2_pilot/aggregate.json"' in output
    assert "100 · held-out search groups" in output


@pytest.mark.parametrize("mutation", [
    {"raw_rows": [{"query": "private"}]},
    {"receipt": "reports/runs/../private/aggregate.json"},
    {"value": float("nan")},
    {"eligible_groups": 0},
])
def test_render_rejects_unreviewable_or_nonaggregate_fields(monkeypatch, tmp_path: Path, mutation):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    record = row() | mutation
    with pytest.raises(ValueError):
        report.render_rows(payload(record))


def test_log_loss_can_exceed_one(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "reports/runs/v2_pilot/aggregate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    record = row() | {"metric": "nll", "value": 1.52}
    assert "1.5200" in report.render_rows(payload(record))
