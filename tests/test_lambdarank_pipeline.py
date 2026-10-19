"""Small invented groups test feature boundaries and saved ranker parity."""

import json
import numpy as np
import pytest

from liquid_mcdm.evaluation.ranking import evaluate_ranking
from liquid_mcdm.models.lambdarank import fit_ranker, predict_ranker
from liquid_mcdm.models.ranking_baselines import (
    FeatureState, fit_conditional_logit, lexical_scores, numerical_heuristic,
    predict_conditional_logit, random_scores,
)
from scripts.benchmark_behavioral import (
    _revision, objective_name, validate_candidate_pool_scope, validate_respondent_splits,
)


POLICY = {"prediction_event": "combined_click_booking_relevance", "context_features": [],
          "alternative_features": ["price_usd", "hotel_class"],
          "numeric_features": ["alternative.price_usd"],
          "categorical_features": ["alternative.hotel_class"]}


def example(number, *, price_a=1., category="known"):
    return {"schema_version": "ordivo-behavioral-v1", "episode_id": f"expedia_personalized_sort_2013:{number}",
            "source": "expedia_personalized_sort_2013", "task_type": "observed_choice",
            "group_id": str(number), "context": {}, "criteria": [], "constraints": [],
            "preferences": {}, "alternatives": [
                {"id": "a", "features": {"price_usd": price_a, "hotel_class": category}},
                {"id": "b", "features": {"price_usd": 3., "hotel_class": "known"}}],
            "supervision": {"kind": "observed_choice", "policy_id": "expedia-click-booking-v1",
                            "outcomes_by_id": {"a": {"click_bool": 1, "booking_bool": 1},
                                               "b": {"click_bool": 0, "booking_bool": 0}}},
            "exposure": {}, "provenance": {}}


def test_feature_fit_is_train_only_and_unknown_category_is_sentinel():
    state = FeatureState(POLICY).fit([example(1, price_a=1)])
    x, group, ids = state.transform([example(2, price_a=None, category="never_seen")])
    assert group == [2] and ids == [["a", "b"]]
    assert x[0, 0] == 2.0  # train median of 1 and 3
    assert x[0, 1] == 1.0  # missingness channel
    assert x[0, 2] == -1.0
    assert "never_seen" not in state.categories["alternative.hotel_class"]


def test_random_control_ignores_candidate_order():
    e = example(1)
    before = random_scores(e)
    e["alternatives"].reverse()
    assert random_scores(e) == before


def test_missing_numerical_heuristic_field_ranks_below_known_value():
    e = example(3, price_a=None)
    scores = numerical_heuristic(e, POLICY, {"price_usd": -1})
    assert scores["a"] < scores["b"]


def test_saved_ranker_reload_and_metric_parity(tmp_path):
    pytest.importorskip("lightgbm")
    train = [example(i) for i in range(10)]
    validation = [example(i) for i in range(10, 13)]
    config = {"seed": 42, "feature_policy": POLICY,
              "target_policy": {"policy_id": "expedia-click-booking-v1"},
              "max_trees": 8, "patience": 3, "lightgbm": {"min_data_in_leaf": 1}}
    receipt = fit_ranker(train, validation, config, tmp_path / "model")
    reloaded = predict_ranker(tmp_path / "model", validation)
    reported = evaluate_ranking(validation, reloaded, config["target_policy"])
    expected = receipt["validation"]["per_query"]
    actual = reported["per_query"]
    assert [r["metrics"] for r in actual] == [r["metrics"] for r in expected]
    assert receipt["bundle"]["best_iteration"] > 0
    assert receipt["bundle"]["reload_max_abs_difference"] <= 1e-12
    assert sum(receipt["bundle"]["train_group_sizes"]) == 20
    assert all(set(scores) == {"a", "b"} and np.isfinite(list(scores.values())).all()
               for scores in reloaded.values())


def test_training_groups_cannot_reenter_validation(tmp_path):
    pytest.importorskip("lightgbm")
    e = example(1)
    config = {"feature_policy": POLICY,
              "target_policy": {"policy_id": "expedia-click-booking-v1"}}
    with pytest.raises(ValueError, match="overlap"):
        fit_ranker([e], [e], config, tmp_path / "model")


def test_lexical_baseline_keeps_source_judgments_out_of_projection():
    e = {"schema_version": "ordivo-behavioral-v1", "episode_id": "amazon_esci:q1",
         "source": "amazon_esci", "task_type": "graded_relevance", "group_id": "q1",
         "context": {"query": "blue shoe"}, "criteria": [], "constraints": [], "preferences": {},
         "alternatives": [{"id": "a", "features": {"product_title": "blue shoe"}},
                          {"id": "b", "features": {"product_title": "red hat"}}],
         "supervision": {"kind": "graded_relevance", "policy_id": "esci-task1-official-gain-v1",
                         "grades_by_id": {"a": "E", "b": "I"}},
         "exposure": {}, "provenance": {}}
    scores = lexical_scores(e, {"context_features": ["query"],
                                "alternative_features": ["product_title"]})
    assert scores["a"] > scores["b"]


def test_runner_accepts_only_source_specific_candidate_pool_scopes():
    validate_candidate_pool_scope("amazon_esci", "source_task1_reduced_candidate_list",
                                  "source_task1_reduced_candidate_list")
    validate_candidate_pool_scope("expedia_personalized_sort_2013", "source_exposed_candidates",
                                  "source_exposed_candidates")
    validate_candidate_pool_scope("swissmetro", "three_survey_modes", "three_survey_modes")
    with pytest.raises(ValueError, match="candidate pool scope"):
        validate_candidate_pool_scope("amazon_esci", "source_exposed_candidates",
                                      "source_task1_reduced_candidate_list")


def test_runner_archive_revision_requires_controller_env(monkeypatch):
    monkeypatch.delenv("ORDIVO_SOURCE_COMMIT", raising=False)
    with pytest.raises(ValueError, match="ORDIVO_SOURCE_COMMIT"):
        _revision()
    commit = "a" * 40
    monkeypatch.setenv("ORDIVO_SOURCE_COMMIT", commit)
    assert _revision() == commit


def swissmetro(number, respondent, selected="train"):
    return {"schema_version": "ordivo-behavioral-v1",
            "episode_id": f"swissmetro:{respondent}:{number}", "source": "swissmetro",
            "task_type": "stated_choice", "group_id": f"{respondent}:{number}",
            "context": {}, "criteria": [], "constraints": [], "preferences": {},
            "alternatives": [
                {"id": "train", "features": {"travel_time_minutes": 10, "cost_chf": 20}},
                {"id": "car", "features": {"travel_time_minutes": 25, "cost_chf": 40}},
                {"id": "swissmetro", "available": False,
                 "features": {"travel_time_minutes": 5, "cost_chf": 10}}],
            "supervision": {"kind": "stated_choice", "policy_id": "swissmetro-choice-v1",
                            "selected_id": selected, "respondent_id": respondent},
            "exposure": {}, "provenance": {}}


def test_swissmetro_respondent_disjointness_beyond_episode_ids():
    splits = {"train": [swissmetro(1, "person-a")],
              "validation": [swissmetro(2, "person-a")], "calibration": [], "test": []}
    with pytest.raises(ValueError, match="respondent overlaps"):
        validate_respondent_splits(splits)


def test_swissmetro_objective_is_stated_choice_not_relevance():
    assert objective_name("swissmetro") == "stated_choice_conditional_logit_nll"


def test_conditional_logit_only_scores_available_modes():
    policy = {"context_features": [], "alternative_features": ["travel_time_minutes", "cost_chf"],
              "numeric_features": ["alternative.travel_time_minutes", "alternative.cost_chf"]}
    train = [swissmetro(1, "person-a"), swissmetro(2, "person-b", selected="car")]
    bundle = fit_conditional_logit(train, policy, iterations=5)
    scores = predict_conditional_logit(bundle, [swissmetro(3, "person-c")])
    reloaded = json.loads(json.dumps(bundle))
    assert predict_conditional_logit(reloaded, [swissmetro(3, "person-c")]) == scores
    assert set(scores["swissmetro:person-c:3"]) == {"train", "car"}
    assert np.isfinite(list(scores["swissmetro:person-c:3"].values())).all()
