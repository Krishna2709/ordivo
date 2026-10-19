"""Invented groups prove order changes preserve candidate identity and paired coverage."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.diagnose_v2_candidate_order import order_stability_report, permute_candidates
from scripts.evaluate_v2_neural_heldout import _source


def swiss(respondent, row):
    return {"schema_version": "ordivo-behavioral-v1",
            "episode_id": f"swissmetro:{respondent}:{row}", "source": "swissmetro",
            "group_id": f"{respondent}:{row}", "task_type": "stated_choice", "context": {},
            "criteria": [], "constraints": [], "preferences": {},
            "alternatives": [{"id": "train", "available": True, "features": {}},
                             {"id": "car", "available": True, "features": {}}],
            "supervision": {"kind": "stated_choice", "policy_id": "swissmetro-choice-v1",
                            "selected_id": "train", "respondent_id": respondent},
            "exposure": {}, "provenance": {}}


def expedia(name, bookings):
    return {"episode_id": f"expedia:{name}", "source": "expedia_personalized_sort_2013",
            "group_id": name, "task_type": "observed_choice",
            "alternatives": [{"id": "a", "available": True, "features": {}},
                             {"id": "b", "available": True, "features": {}}],
            "supervision": {"outcomes_by_id": {
                "a": {"booking_bool": bookings[0]},
                "b": {"booking_bool": bookings[1]}}}}


def test_permutation_changes_only_candidate_order_and_ignores_labels():
    episode = swiss("r1", 1)
    episode["alternatives"].append({"id": "bike", "available": False, "features": {"time": 12}})
    original = copy.deepcopy(episode)
    changed = permute_candidates(episode, 42)
    assert [a["id"] for a in changed["alternatives"]] != [a["id"] for a in episode["alternatives"]]
    assert sorted(changed["alternatives"], key=lambda a: a["id"]) == sorted(
        episode["alternatives"], key=lambda a: a["id"])
    changed["alternatives"] = original["alternatives"]
    assert changed == original and episode == original
    relabeled = copy.deepcopy(episode)
    relabeled["supervision"]["selected_id"] = "car"
    assert [a["id"] for a in permute_candidates(relabeled, 42)["alternatives"]] == [
        a["id"] for a in permute_candidates(episode, 42)["alternatives"]]


def test_top1_flip_and_l1_use_candidate_ids_and_require_full_pairs():
    episode = swiss("r1", 1)
    episode_id = episode["episode_id"]
    before = {episode_id: {"train": .8, "car": .2}}
    after = {episode_id: {"car": .7, "train": .3}}
    report = order_stability_report([episode], before, after, set(), set())
    assert report["admitted_pairs"] == 1
    assert report["top1_flip_rate"] == 1
    assert report["mean_l1_probability_change"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="incomplete paired"):
        order_stability_report([episode], before, {}, set(), set())


def test_overflow_has_explicit_denominator_and_no_partial_score():
    rows = [swiss("r1", 1), swiss("r2", 1)]
    first = rows[0]["episode_id"]
    second = rows[1]["episode_id"]
    probs = {first: {"train": .6, "car": .4}}
    report = order_stability_report(rows, probs, probs, set(), {second})
    assert report["selected_episodes"] == 2
    assert report["admitted_pairs"] == 1
    assert report["excluded"]["overflow_permuted"] == 1
    assert report["failed_queries"] == 0


def test_conditional_booking_excludes_zero_and_multiple_before_pair_coverage():
    rows = [expedia("zero", (False, False)), expedia("one", (True, False)),
            expedia("multiple", (True, True))]
    admitted = rows[1]["episode_id"]
    prediction = {admitted: {"a": .7, "b": .3}}
    policy = {"prediction_event": "conditional_booking_choice"}
    report = order_stability_report(rows, prediction, prediction, set(), set(), policy)
    assert report["selected_episodes"] == 3
    assert report["admitted_pairs"] == 1
    assert report["excluded"]["not_exactly_one_booking"] == 2
    assert [row["status"] for row in report["per_query"]] == [
        "not_exactly_one_booking", "scored", "not_exactly_one_booking"]
    with pytest.raises(ValueError, match="incomplete paired"):
        order_stability_report(rows, {}, prediction, set(), set(), policy)
    with pytest.raises(ValueError, match="ineligible.*overflow"):
        order_stability_report(rows, prediction, prediction,
                               {rows[0]["episode_id"]}, set(), policy)


def test_validation_source_selection_is_separate_from_test(tmp_path):
    import hashlib
    import json

    rows = {"train": [swiss("train-r", 1)], "validation": [swiss("val-r", 1), swiss("val-r", 2)],
            "calibration": [swiss("cal-r", 1)], "test": [swiss("test-r", 1)]}
    for split, episodes in rows.items():
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    manifest = {"split_sha256": {split: hashlib.sha256(
        (tmp_path / f"{split}.jsonl").read_bytes()).hexdigest() for split in rows}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    selected, meta = _source({"directory": str(tmp_path)}, digest, seed=42, limit=1,
                             max_episodes=5, partition="validation", max_units=20)
    assert {e["episode_id"] for e in selected} == {e["episode_id"] for e in rows["validation"]}
    assert meta["selected_partition"] == "validation" and meta["selected_units"] == 1
    with pytest.raises(ValueError, match="partition"):
        _source({"directory": str(tmp_path)}, digest, seed=42, limit=1,
                max_episodes=5, partition="calibration")


def test_direct_script_launch_resolves_shared_evaluator_import():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, "scripts/diagnose_v2_candidate_order.py", "--help"],
        cwd=root, env=env, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--pilot-output" in completed.stdout
