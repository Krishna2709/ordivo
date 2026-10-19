"""Remote exact expected NDCG under uniform permutations of tied score blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from liquid_mcdm.data.behavioral import validate_episode, validate_scores
from liquid_mcdm.evaluation.ranking import ndcg_at_k, ranked_ids, tie_expected_ndcg_at_k
from liquid_mcdm.evaluation.target_policy import targets
from liquid_mcdm.runtime import require_remote_gpu


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(data: Path, baseline_dir: Path, output: Path) -> dict:
    require_remote_gpu()
    commit = os.environ.get("ORDIVO_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("controller-verified source commit required")
    output = Path(output).resolve()
    if Path("/workspace") not in output.parents or output.exists():
        raise ValueError("new remote /workspace output required")
    aggregate_path = baseline_dir / "aggregate.json"
    aggregate_sha256 = _hash(aggregate_path)
    aggregate = json.loads(aggregate_path.read_text())
    manifest_path, test_path = data / "manifest.json", data / "test.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if (_hash(manifest_path) != aggregate["manifest_sha256"] or
            _hash(test_path) != manifest["split_sha256"]["test"] or
            aggregate["source"] != "amazon_esci"):
        raise ValueError("baseline source or frozen test manifest mismatch")
    episodes = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    expected = {e["episode_id"] for e in episodes}
    if len(expected) != len(episodes):
        raise ValueError("duplicate test episode")
    by_model = {}
    output.mkdir()
    for model in ("lambdarank", "lexical_overlap", "random_seed42"):
        scores_path = baseline_dir / f"{model}_scores.json"
        scores = json.loads(scores_path.read_text())
        if set(scores) != expected:
            raise ValueError("incomplete tie-analysis score coverage")
        rows = []
        for episode in episodes:
            validate_episode(episode)
            prediction = scores[episode["episode_id"]]
            validate_scores(episode, prediction)
            target = targets(episode)
            gain = target["gains_by_id"]
            judged = {i: prediction[i] for i in gain}
            canonical = ndcg_at_k(ranked_ids(judged), gain, 10)
            expected_ndcg, sensitive = tie_expected_ndcg_at_k(judged, gain, 10)
            rows.append({"episode_id": episode["episode_id"], "canonical_ndcg@10": canonical,
                         "tie_expected_ndcg@10": expected_ndcg, "tie_metric_sensitive": sensitive})
        eligible = [r for r in rows if r["canonical_ndcg@10"] is not None]
        if not eligible:
            raise ValueError("no positive-ideal-gain test queries for tie analysis")
        canonical_macro = sum(r["canonical_ndcg@10"] for r in eligible) / len(eligible)
        expected_macro = sum(r["tie_expected_ndcg@10"] for r in eligible) / len(eligible)
        prior = aggregate["metrics"][model]["by_source_task_policy"]
        prior_macro = next(iter(prior.values()))["macro"]["ndcg@10"]
        if prior_macro["eligible"] != len(eligible) or abs(prior_macro["value"] - canonical_macro) > 1e-10:
            raise ValueError("tie analysis does not reproduce baseline canonical NDCG")
        (output / f"{model}_per_query.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        by_model[model] = {"expected_queries": len(episodes), "eligible_ndcg_queries": len(eligible),
                           "tie_metric_sensitive_queries": sum(r["tie_metric_sensitive"] for r in rows),
                           "canonical_ndcg@10": canonical_macro,
                           "uniform_tie_expected_ndcg@10": expected_macro,
                           "expected_minus_canonical": expected_macro - canonical_macro,
                           "scores_sha256": _hash(scores_path)}
    report = {"schema_version": "ordivo-tie-analysis-v1", "code_revision": commit,
              "baseline_code_revision": aggregate["code_revision"],
              "baseline_aggregate_sha256": aggregate_sha256,
              "manifest_sha256": aggregate["manifest_sha256"],
              "method": "exact_uniform_permutation_within_equal_score_blocks",
              "by_model": by_model}
    (output / "aggregate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.data, args.baseline_dir, args.output)
    print(json.dumps({"output": str(args.output / "aggregate.json"),
                      "manifest_sha256": report["manifest_sha256"]}))


if __name__ == "__main__":
    main()
