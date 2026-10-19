"""Hand-calculated grouped metrics; fixtures are invented, never source rows."""

import math

import pytest

from liquid_mcdm.evaluation.ranking import evaluate_ranking, ndcg_at_k, ranked_ids
from liquid_mcdm.evaluation.target_policy import targets


def episode(kind="observed_choice", source="expedia_personalized_sort_2013", suffix="one"):
    ids = ["a", "b"]
    supervision = {
        "observed_choice": {"kind": kind, "policy_id": "expedia-click-booking-v1",
                            "outcomes_by_id": {"a": {"click_bool": 1, "booking_bool": 1},
                                               "b": {"click_bool": 0, "booking_bool": 0}}},
        "graded_relevance": {"kind": kind, "policy_id": "esci-task1-official-gain-v1",
                             "grades_by_id": {"a": "E", "b": "I"}},
        "stated_choice": {"kind": kind, "policy_id": "swissmetro-choice-v1",
                          "selected_id": "a", "respondent_id": "r1"},
    }[kind]
    return {"schema_version": "ordivo-behavioral-v1", "episode_id": f"{source}:{suffix}",
            "source": source, "task_type": kind, "group_id": suffix, "context": {},
            "criteria": [], "constraints": [], "preferences": {},
            "alternatives": [{"id": i, "features": {"price_usd": j + 1}} for j, i in enumerate(ids)],
            "supervision": supervision, "exposure": {}, "provenance": {}}


def test_hand_calculated_perfect_and_reverse():
    e = episode()
    perfect = evaluate_ranking([e], {e["episode_id"]: {"a": 2, "b": 1}})
    row = perfect["per_query"][0]
    assert row["metrics"]["ndcg@5"] == 1
    assert row["metrics"]["booking_mrr"] == 1
    reverse = evaluate_ranking([e], {e["episode_id"]: {"a": 1, "b": 2}})
    assert reverse["per_query"][0]["metrics"]["booking_mrr"] == .5
    assert ndcg_at_k(["b", "a"], {"a": 1, "b": 0}, 1) == 0
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 0}, 10) == 1


def test_expedia_gains_and_separate_zero_event_denominators():
    booked = episode()
    click_only = episode(suffix="click")
    click_only["supervision"]["outcomes_by_id"]["a"]["booking_bool"] = 0
    zero = episode(suffix="zero")
    zero["supervision"]["outcomes_by_id"]["a"] = {"click_bool": 0, "booking_bool": 0}
    assert targets(booked)["gains_by_id"] == {"a": 5, "b": 0}
    predictions = {e["episode_id"]: {"a": 2, "b": 1} for e in (booked, click_only, zero)}
    report = evaluate_ranking([booked, click_only, zero], predictions)
    macro = next(iter(report["by_source_task_policy"].values()))["macro"]
    assert macro["ndcg@5"]["eligible"] == 2
    assert macro["click_ndcg@5"]["eligible"] == 2
    assert macro["booking_ndcg@5"]["eligible"] == 1
    assert macro["booking_mrr"]["eligible"] == 1
    assert macro["booking_ndcg@5"]["excluded_zero_ideal_gain"] == 2


def test_canonical_ties_and_complete_coverage():
    e = episode()
    assert ranked_ids({"b": 0.5, "a": 0.5}) == ["a", "b"]
    with pytest.raises(ValueError, match="incomplete prediction coverage"):
        evaluate_ranking([e], {})
    with pytest.raises(ValueError):
        evaluate_ranking([e], {e["episode_id"]: {"a": 1}})
    with pytest.raises(ValueError):
        evaluate_ranking([e, e], {e["episode_id"]: {"a": 1, "b": 0}})
    with pytest.raises(ValueError):
        evaluate_ranking([e], {e["episode_id"]: {"a": math.inf, "b": 0}})


def test_esci_categories_are_relevance_not_events():
    e = episode("graded_relevance", "amazon_esci", "q1")
    e["alternatives"] += [{"id": "c", "features": {}}, {"id": "d", "features": {}}]
    e["supervision"]["grades_by_id"].update({"c": "C", "d": "S"})
    target = targets(e)
    assert target["gains_by_id"] == {"a": 1, "b": 0, "c": .1, "d": .01}
    assert target["events_by_name"] == {}
    result = evaluate_ranking([e], {e["episode_id"]: {"a": 4, "c": 3, "d": 2, "b": 1}})
    assert result["per_query"][0]["metrics"]["ndcg@5"] == 1


def test_stated_choice_uses_available_candidates_only():
    e = episode("stated_choice", "swissmetro", "r1-1")
    e["alternatives"][1]["available"] = False
    report = evaluate_ranking([e], {e["episode_id"]: {"a": 1}})
    assert report["per_query"][0]["metrics"]["choice_accuracy@1"] == 1
