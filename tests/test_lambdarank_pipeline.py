"""Small invented groups test feature boundaries and saved ranker parity."""

import numpy as np
import pytest

from liquid_mcdm.evaluation.ranking import evaluate_ranking
from liquid_mcdm.models.lambdarank import fit_ranker, predict_ranker
from liquid_mcdm.models.ranking_baselines import FeatureState, lexical_scores, random_scores


POLICY = {"context_features": [], "alternative_features": ["price_usd", "hotel_class"],
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
    assert sum(receipt["bundle"]["train_group_sizes"]) == 20
    assert all(set(scores) == {"a", "b"} and np.isfinite(list(scores.values())).all()
               for scores in reloaded.values())


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
