"""Remote-only complete-group specialist benchmark.

Usage: python scripts/benchmark_behavioral.py --config CONFIG --data DATA --output OUTPUT
DATA contains train.jsonl, validation.jsonl, calibration.jsonl, test.jsonl and
manifest.json. All per-query material stays under remote /workspace storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liquid_mcdm.evaluation.ranking import evaluate_ranking
from liquid_mcdm.models.lambdarank import fit_ranker, predict_ranker
from liquid_mcdm.models.ranking_baselines import lexical_scores, numerical_heuristic, random_scores
from liquid_mcdm.models.ranking_baselines import fit_conditional_logit, predict_conditional_logit
from liquid_mcdm.runtime import require_remote_gpu


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(report: dict) -> dict:
    return {key: value for key, value in report.items() if key != "per_query"}


def _revision() -> str:
    revision = os.environ.get("ORDIVO_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("ORDIVO_SOURCE_COMMIT must be a controller-verified 40-hex archive commit")
    return revision


def validate_candidate_pool_scope(source: str, configured: str, declared: str) -> None:
    expected = {"expedia_personalized_sort_2013": "source_exposed_candidates",
                "amazon_esci": "source_task1_reduced_candidate_list",
                "swissmetro": "three_survey_modes"}.get(source)
    if expected is None or configured != expected or declared != expected:
        raise ValueError("candidate pool scope differs from source/config policy")


def objective_name(source: str) -> str:
    return {"expedia_personalized_sort_2013": "declared_click_booking_gain_0_1_5_lambdarank",
            "amazon_esci": "source_graded_relevance_gain_lambdarank",
            "swissmetro": "stated_choice_conditional_logit_nll"}[source]


def validate_respondent_splits(split_episodes: dict[str, list[dict]]) -> None:
    """Panel rows from one respondent may never cross a split boundary."""
    respondents = {}
    for split, episodes in split_episodes.items():
        for episode in episodes:
            respondent = episode["supervision"]["respondent_id"]
            prior = respondents.setdefault(respondent, split)
            if prior != split:
                raise ValueError(f"Swissmetro respondent overlaps {prior} and {split}")


def benchmark(config_path: Path, data_dir: Path, output_dir: Path) -> dict:
    runtime = require_remote_gpu()
    code_revision = _revision()
    output_dir = output_dir.resolve()
    if output_dir != Path("/workspace") and Path("/workspace") not in output_dir.parents:
        raise ValueError("benchmark output must be under remote /workspace storage")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config = yaml.safe_load(config_path.read_text())
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("target_policy_id") != config["target_policy"]["policy_id"]:
        raise ValueError("manifest target policy differs from config")
    validate_candidate_pool_scope(config["source"], config.get("candidate_pool_scope"),
                                  manifest.get("candidate_pool_scope"))
    for split in ("train", "validation", "calibration", "test"):
        expected_hash = manifest.get("split_sha256", {}).get(split)
        if not expected_hash or _digest(data_dir / f"{split}.jsonl") != expected_hash:
            raise ValueError(f"split hash mismatch or absent: {split}")
    # Read all split identities before fitting. Calibration is never used here.
    train = _read_jsonl(data_dir / "train.jsonl")
    validation = _read_jsonl(data_dir / "validation.jsonl")
    calibration = _read_jsonl(data_dir / "calibration.jsonl")
    test = _read_jsonl(data_dir / "test.jsonl")
    split_ids = {name: {e["episode_id"] for e in episodes} for name, episodes in
                 (("train", train), ("validation", validation),
                  ("calibration", calibration), ("test", test))}
    if any(len(split_ids[name]) != len(episodes) for name, episodes in
           (("train", train), ("validation", validation),
            ("calibration", calibration), ("test", test))):
        raise ValueError("duplicate episode in split")
    if any(split_ids[a] & split_ids[b] for a in split_ids for b in split_ids if a < b):
        raise ValueError("episode overlaps splits")
    if config["source"] == "swissmetro":
        validate_respondent_splits({"train": train, "validation": validation,
                                    "calibration": calibration, "test": test})
    if any(e["source"] != config["source"] for group in (train, validation, calibration, test)
           for e in group):
        raise ValueError("source differs from config")
    if manifest.get("episode_count") != sum(map(len, (train, validation, calibration, test))):
        raise ValueError("manifest episode count mismatch")
    if not train or not validation or not test:
        raise ValueError("train, validation, and test splits must be nonempty")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    candidates = {}
    candidates["random_seed42"] = {e["episode_id"]: random_scores(e, config["seed"])
                                   for e in test}
    if config["source"] == "amazon_esci":
        candidates["lexical_overlap"] = {e["episode_id"]: lexical_scores(e, config["feature_policy"])
                                         for e in test}
    elif config["source"] == "expedia_personalized_sort_2013":
        candidates["price_ascending"] = {e["episode_id"]: numerical_heuristic(
            e, config["feature_policy"], {"price_usd": -1.0}) for e in test}
        candidates["rating_descending"] = {e["episode_id"]: numerical_heuristic(
            e, config["feature_policy"], {"prop_review_score": 1.0}) for e in test}
    elif config["source"] == "swissmetro":
        candidates["travel_time_ascending"] = {e["episode_id"]: numerical_heuristic(
            e, config["feature_policy"], {"travel_time_minutes": -1.0}) for e in test}
        candidates["cost_ascending"] = {e["episode_id"]: numerical_heuristic(
            e, config["feature_policy"], {"cost_chf": -1.0}) for e in test}
    if config["source"] == "swissmetro":
        specialist = fit_conditional_logit(train, config["feature_policy"], seed=config["seed"],
                                           **config.get("conditional_logit", {}))
        model_dir = output_dir / "model"
        model_dir.mkdir()
        _write_json(model_dir / "conditional_logit.json", specialist)
        reloaded = json.loads((model_dir / "conditional_logit.json").read_text())
        original = predict_conditional_logit(specialist, validation)
        restored = predict_conditional_logit(reloaded, validation)
        if original != restored:
            raise ValueError("conditional logit scores changed on bundle reload")
        validation_report = evaluate_ranking(validation, restored, config["target_policy"])
        _write_json(model_dir / "validation_per_query.json", validation_report["per_query"])
        _write_json(model_dir / "validation_aggregate.json", _aggregate(validation_report))
        candidates["conditional_logit"] = predict_conditional_logit(reloaded, test)
        model_metadata = {"model_kind": "conditional_logit", "train_queries": len(train),
                          "validation_queries": len(validation), "reload_score_parity": True,
                          "validation_metrics": _aggregate(validation_report)}
    else:
        receipt = fit_ranker(train, validation, config, output_dir / "model")
        candidates["lambdarank"] = predict_ranker(output_dir / "model", test)
        model_metadata = {"model_kind": "lambdarank",
                          "model_best_iteration": receipt["bundle"]["best_iteration"],
                          "lightgbm_version": receipt["bundle"]["lightgbm_version"]}
    aggregate = {}
    for name, predictions in candidates.items():
        report = evaluate_ranking(test, predictions, config["target_policy"])
        _write_json(output_dir / f"{name}_per_query.json", report["per_query"])
        _write_json(output_dir / f"{name}_scores.json", predictions)
        aggregate[name] = _aggregate(report)
    metadata = {"experiment_id": config["experiment_id"], "source": config["source"],
                "code_revision": code_revision,
                "code_revision_basis": "controller_verified_archive_env",
                "runtime_provider": runtime["provider"], "config_sha256": _digest(config_path),
                "manifest_sha256": _digest(manifest_path),
                "source_sha256": manifest.get("source_sha256", {}),
                "source_terms": manifest.get("source_terms"),
                "split_policy_id": manifest.get("split_policy_id"),
                "sampling": manifest.get("sampling"),
                "candidate_pool_scope": manifest.get("candidate_pool_scope"),
                "prediction_event": config["feature_policy"].get("prediction_event"),
                "ranking_objective": objective_name(config["source"]),
                "split_counts": {"train": len(train), "validation": len(validation),
                                 "calibration_unused": len(calibration), "test": len(test)},
                "model": model_metadata,
                "elapsed_seconds": time.monotonic() - started,
                "metrics": aggregate}
    _write_json(output_dir / "aggregate.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.config, args.data, args.output)
    print(json.dumps({"experiment_id": result["experiment_id"],
                      "split_counts": result["split_counts"],
                      "aggregate_path": str(args.output / "aggregate.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
