"""Insert reviewed aggregate v2 receipts into the self-contained HTML report.

Input is a deliberately narrow public schema, not a run manifest or prediction file:
{
  "schema_version": "ordivo-v2-public-aggregates-v1",
  "rows": [{"run_id": "...", "commit": "40 hex", "source": "expedia",
            "task": "observed_choice", "comparison": "...", "metric": "ndcg@5",
            "value": 0.43, "eligible_groups": 123,
            "population": "judged source candidate groups", "split_sha256": "64 hex",
            "receipt": "reports/runs/.../aggregate.json"}]
}
The operator must review source rights, cohort definitions and receipt contents.
This script validates public shape and escaping; it does not certify the science.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/ordivo-v2-experiment.html"
START = "<!-- V2_EVIDENCE_START -->"
END = "<!-- V2_EVIDENCE_END -->"
SOURCES = {"expedia", "esci", "mslr", "swissmetro", "sushi", "synthetic_mcdm"}
TASKS = {"observed_choice", "stated_choice", "graded_relevance", "explicit_ranking", "synthetic_solver"}
METRICS = {"top1", "ndcg@5", "ndcg@10", "mrr", "pair_agreement", "nll", "brier", "ece", "constraint_violation_rate", "served_violation_rate", "coverage"}
ROW_KEYS = {"run_id", "commit", "source", "task", "comparison", "metric", "value", "eligible_groups", "population", "split_sha256", "receipt"}


def _public_text(value: object, name: str, limit: int = 90) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{name} must be short printable text")
    return value


def render_rows(payload: object) -> str:
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "rows"}:
        raise ValueError("expected only schema_version and rows")
    if payload["schema_version"] != "ordivo-v2-public-aggregates-v1":
        raise ValueError("unsupported schema version")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        raise ValueError("rows must contain 1 to 100 aggregate records")
    rendered = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError(f"row {index}: unexpected or missing field")
        run_id = _public_text(row["run_id"], "run_id", 60)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("run_id must be a public slug")
        commit = _public_text(row["commit"], "commit", 40)
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("commit must be full SHA-1")
        split = _public_text(row["split_sha256"], "split_sha256", 64)
        if not re.fullmatch(r"[0-9a-f]{64}", split):
            raise ValueError("split_sha256 must be SHA-256")
        source, task, metric = (row[key] for key in ("source", "task", "metric"))
        if source not in SOURCES or task not in TASKS or metric not in METRICS:
            raise ValueError("source, task, or metric is not in the public allowlist")
        comparison = _public_text(row["comparison"], "comparison")
        value = row["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("metric value must be finite and nonnegative")
        if metric not in {"nll", "brier"} and value > 1:
            raise ValueError("this metric must be on a unit scale")
        groups = row["eligible_groups"]
        if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
            raise ValueError("eligible_groups must be positive")
        population = _public_text(row["population"], "population", 140)
        receipt = _public_text(row["receipt"], "receipt", 200)
        # Public links are confined to checked-in aggregate receipts. No URL schemes,
        # traversal, fragments or per-row prediction artifacts are accepted.
        if not re.fullmatch(r"reports/runs/[A-Za-z0-9_/-]+/(aggregate|summary)\.json", receipt) or ".." in receipt:
            raise ValueError("receipt must be an aggregate/summary JSON under reports/runs")
        if not (ROOT / receipt).is_file():
            raise ValueError(f"receipt does not exist: {receipt}")
        cells = [f"{source} · {task}", comparison, metric, f"{value:.4f}", f"{groups} · {population}"]
        safe_cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
        safe_href = html.escape(receipt.removeprefix("reports/"), quote=True)
        rendered.append(f'<tr>{safe_cells}<td><a href="{safe_href}">{html.escape(run_id)}</a><small> commit {commit[:12]}; split {split[:12]}</small></td></tr>')
    return ('<div class="table-wrap"><table><caption class="note">V2 measured results on explicitly eligible groups</caption>'
            '<thead><tr><th scope="col">Task and source</th><th scope="col">Comparison</th><th scope="col">Metric</th>'
            '<th scope="col">Result</th><th scope="col">Eligible groups</th><th scope="col">Receipt</th></tr></thead><tbody>'
            + "\n".join(rendered) + "</tbody></table></div>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_json", type=Path, help="reviewed, sanitized aggregate JSON")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    payload = json.loads(args.aggregate_json.read_text(encoding="utf-8"))
    replacement = render_rows(payload)
    source = REPORT.read_text(encoding="utf-8")
    if source.count(START) != 1 or source.count(END) != 1 or source.index(START) >= source.index(END):
        raise ValueError("report evidence markers are missing or ambiguous")
    before, rest = source.split(START, 1)
    _, after = rest.split(END, 1)
    result = before + START + "\n" + replacement + "\n" + END + after
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
