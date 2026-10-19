"""Invented complete groups prove paired held-out selection and denominators."""

import copy
import hashlib
import json
import math

import pytest

from liquid_mcdm.evaluation.neural_heldout import (
    eligibility, evaluate_neural_pair, select_whole_test_units,
)
from scripts.evaluate_v2_neural_heldout import _source


def swiss(respondent, row, selected="train"):
    return {"schema_version": "ordivo-behavioral-v1",
            "episode_id": f"swissmetro:{respondent}:{row}", "source": "swissmetro",
            "group_id": f"{respondent}:{row}", "task_type": "stated_choice",
            "context": {}, "criteria": [], "constraints": [], "preferences": {},
            "alternatives": [{"id": "train", "available": True, "features": {}},
                             {"id": "car", "available": True, "features": {}}],
            "supervision": {"kind": "stated_choice", "policy_id": "swissmetro-choice-v1",
                            "selected_id": selected, "respondent_id": respondent},
            "exposure": {}, "provenance": {}}


def esci(group, grades):
    return {"schema_version": "ordivo-behavioral-v1", "episode_id": f"amazon_esci:{group}",
            "source": "amazon_esci", "group_id": group, "task_type": "graded_relevance",
            "context": {}, "criteria": [], "constraints": [], "preferences": {},
            "alternatives": [{"id": "a", "available": True, "features": {}},
                             {"id": "b", "available": True, "features": {}}],
            "supervision": {"kind": "graded_relevance", "policy_id": "esci-task1-official-gain-v1",
                            "grades_by_id": grades, "unjudged_ids": []},
            "exposure": {}, "provenance": {}}


def test_selects_whole_respondents_independent_of_row_order_and_labels():
    rows = [swiss("r1", 1), swiss("r1", 2), swiss("r2", 1), swiss("r3", 1)]
    selected, units = select_whole_test_units(rows, source="swissmetro", seed=42, limit=2)
    shuffled = copy.deepcopy(list(reversed(rows)))
    for row in shuffled:
        row["supervision"]["selected_id"] = "car"
    selected2, units2 = select_whole_test_units(shuffled, source="swissmetro", seed=42, limit=2)
    assert units == units2
    assert {e["episode_id"] for e in selected} == {e["episode_id"] for e in selected2}
    assert {e["group_id"].split(":")[0] for e in selected} == {u.split(":")[1] for u in units}
    with pytest.raises(ValueError):
        select_whole_test_units(rows, source="swissmetro", seed=42, limit=51)


def test_paired_choice_metrics_and_missing_prediction_failure():
    row = swiss("r1", 1)
    base = {row["episode_id"]: {"train": .25, "car": .75}}
    adapted = {row["episode_id"]: {"train": .75, "car": .25}}
    report = evaluate_neural_pair([row], base, adapted, {})
    accuracy = report["metrics"]["choice_accuracy@1"]
    assert accuracy == {"eligible": 1, "excluded_zero_ideal_gain": 0,
                        "base": 0, "adapted": 1, "adapted_minus_base": 1}
    assert report["metrics"]["choice_nll"]["base"] == pytest.approx(-math.log(.25))
    assert report["metrics"]["choice_brier"]["adapted"] == pytest.approx(.125)
    with pytest.raises(ValueError, match="incomplete paired"):
        evaluate_neural_pair([row], base, {}, {})
    with pytest.raises(ValueError, match="zero neural probability"):
        evaluate_neural_pair([row], {row["episode_id"]: {"train": 0., "car": 1.}}, adapted, {})


def test_conditional_booking_exclusion_and_overflow_are_explicit():
    row = swiss("r1", 1)
    row["source"] = "expedia_personalized_sort_2013"
    row["episode_id"] = "expedia_personalized_sort_2013:q1"
    row["task_type"] = "observed_choice"
    row["supervision"] = {"kind": "observed_choice", "policy_id": "expedia-click-booking-v1",
                          "outcomes_by_id": {"train": {"booking_bool": 0}, "car": {"booking_bool": 0}}}
    policy = {"prediction_event": "conditional_booking_choice"}
    assert eligibility(row, policy) == (None, "not_exactly_one_booking")
    report = evaluate_neural_pair([row], {}, {}, policy)
    assert report["selected_episodes"] == 1 and report["admitted_queries"] == 0
    assert report["excluded"]["not_exactly_one_booking"] == 1
    row["supervision"]["outcomes_by_id"]["train"]["booking_bool"] = 1
    report = evaluate_neural_pair([row], {}, {}, policy, {row["episode_id"]})
    assert report["excluded"]["prompt_overflow"] == 1


def test_esci_zero_gain_excluded_per_metric_but_group_scored():
    positive = esci("q1", {"a": "E", "b": "I"})
    zero = esci("q2", {"a": "I", "b": "I"})
    base = {e["episode_id"]: {"a": .25, "b": .75} for e in (positive, zero)}
    adapted = {e["episode_id"]: {"a": .75, "b": .25} for e in (positive, zero)}
    report = evaluate_neural_pair([positive, zero], base, adapted, {})
    metric = report["metrics"]["ndcg@10"]
    assert report["scored_base"] == 2 and report["scored_adapted"] == 2
    assert metric["eligible"] == 1 and metric["excluded_zero_ideal_gain"] == 1
    assert metric["adapted"] == 1


def test_frozen_manifest_hashes_and_respondent_units_guard_heldout(tmp_path):
    rows = {"train": [swiss("r1", 1)], "validation": [swiss("r2", 1)],
            "calibration": [swiss("r3", 1)], "test": [swiss("r4", 1), swiss("r4", 2)]}
    for split, episodes in rows.items():
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    manifest = {"split_sha256": {split: hashlib.sha256(
        (tmp_path / f"{split}.jsonl").read_bytes()).hexdigest() for split in rows}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    selected, info = _source({"directory": str(tmp_path)}, digest, seed=42,
                             limit=1, max_episodes=5)
    assert len(selected) == 2 and info["selected_units"] == 1
    rows["test"][0]["supervision"]["respondent_id"] = "r1"
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(e) + "\n" for e in rows["test"]))
    with pytest.raises(ValueError, match="test file hash"):
        _source({"directory": str(tmp_path)}, digest, seed=42, limit=1, max_episodes=5)
    manifest["split_sha256"]["test"] = hashlib.sha256((tmp_path / "test.jsonl").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    updated_digest = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="split unit overlap"):
        _source({"directory": str(tmp_path)}, updated_digest, seed=42, limit=1, max_episodes=5)
