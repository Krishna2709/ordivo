import copy
import json

import pytest

from liquid_mcdm.data.behavioral import inference_input, validate_episode, validate_scores


def fixture(kind="observed_choice"):
    ids = ["a", "b"]
    supervision = {
        "observed_choice": {"outcomes_by_id": {id_: {"click_bool": int(id_ == "a"), "booking_bool": 0} for id_ in ids}},
        "graded_relevance": {"grades_by_id": {"a": "E", "b": "I"}, "unjudged_ids": []},
        "stated_choice": {"selected_id": "a", "respondent_id": "person-1"},
        "explicit_ranking": {"ordered_groups": [["a"], ["b"]], "unranked_ids": []},
    }[kind]
    return {"schema_version": "ordivo-behavioral-v1", "episode_id": f"test_source:g1", "source": "test_source",
            "task_type": kind, "group_id": "g1", "context": {"query": "shoes", "position": 99},
            "criteria": [], "constraints": [], "preferences": {},
            "alternatives": [{"id": id_, "features": {"price": 2, "booking_bool": 1}, "available": True} for id_ in ids],
            "supervision": {"kind": kind, "policy_id": "fixture-v1", **supervision},
            "exposure": {"position_by_id": {"a": 1, "b": 2}}, "provenance": {"raw_file_sha256": "fixture"}}


@pytest.mark.parametrize("kind", ["observed_choice", "graded_relevance", "stated_choice", "explicit_ranking"])
def test_projection_has_no_labels_or_provenance(kind):
    episode = fixture(kind)
    validate_episode(episode)
    policy = {"context_features": ["query"], "alternative_features": ["price"]}
    if kind == "observed_choice":
        policy["prediction_event"] = "combined_click_booking_relevance"
    projected = inference_input(episode, policy)
    assert set(projected) == ({"task_type", "context", "criteria", "constraints", "preferences", "alternatives"} | ({"prediction_event"} if kind == "observed_choice" else set()))
    assert set(projected["alternatives"][0]) == {"id", "features", "available"}
    assert projected["context"] == {"query": "shoes"}
    assert projected["alternatives"][0]["features"] == {"price": 2}
    assert "fixture" not in json.dumps(projected)
    validate_scores(episode, {"a": 0.5, "b": -1.0})


def test_rejects_forbidden_feature_request_and_nested_preference_leak():
    episode = fixture()
    policy = {"context_features": ["query"], "alternative_features": ["price"], "prediction_event": "combined_click_booking_relevance"}
    with pytest.raises(ValueError):
        inference_input(episode, {**policy, "context_features": ["position"]})
    episode["preferences"] = {"source": "hidden"}
    with pytest.raises(ValueError):
        inference_input(episode, policy)


def test_observed_event_is_explicit_and_not_derived_from_label():
    episode = fixture()
    policy = {"context_features": ["query"], "alternative_features": ["price"]}
    with pytest.raises(ValueError, match="prediction_event"):
        inference_input(episode, policy)
    ranking = inference_input(episode, {**policy, "prediction_event": "combined_click_booking_relevance"})
    with pytest.raises(ValueError, match="exactly one booking"):
        inference_input(episode, {**policy, "prediction_event": "conditional_booking_choice"})
    episode["supervision"]["outcomes_by_id"]["a"]["booking_bool"] = 1
    choice = inference_input(episode, {**policy, "prediction_event": "conditional_booking_choice"})
    assert ranking["prediction_event"] == "combined_click_booking_relevance"
    assert choice["prediction_event"] == "conditional_booking_choice"


@pytest.mark.parametrize("change", [
    lambda e: e["alternatives"].append(copy.deepcopy(e["alternatives"][0])),
    lambda e: e["supervision"]["outcomes_by_id"].pop("b"),
    lambda e: e.update(episode_id="another:g1"),
    lambda e: e["supervision"]["outcomes_by_id"]["a"].update(click_bool=2),
])
def test_invalid_episode(change):
    episode = fixture()
    change(episode)
    with pytest.raises(ValueError):
        validate_episode(episode)


def test_unavailable_positive_and_scores():
    episode = fixture()
    episode["alternatives"][0]["available"] = False
    with pytest.raises(ValueError):
        validate_episode(episode)
    episode["supervision"]["outcomes_by_id"]["a"]["click_bool"] = 0
    validate_episode(episode)
    validate_scores(episode, {"b": 1.0})
    for scores in ({"a": 0.0, "b": 1.0}, {}, {"b": float("nan")}, {"b": float("inf")}):
        with pytest.raises(ValueError):
            validate_scores(episode, scores)


def test_partial_relevance_requires_explicit_unjudged_mask():
    episode = fixture("graded_relevance")
    episode["supervision"]["grades_by_id"].pop("b")
    with pytest.raises(ValueError):
        validate_episode(episode)
    episode["supervision"]["unjudged_ids"] = ["b"]
    validate_episode(episode)


def test_sources_can_reuse_local_group_id_without_episode_collision():
    first, second = fixture(), fixture("graded_relevance")
    second["source"] = "different_source"
    second["episode_id"] = "different_source:g1"
    validate_episode(first)
    validate_episode(second)
    assert first["episode_id"] != second["episode_id"]
