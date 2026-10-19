"""Per-domain LightGBM LambdaRank with validation-only model selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from liquid_mcdm.evaluation.ranking import evaluate_ranking
from liquid_mcdm.evaluation.target_policy import ESCI_POLICY, EXPEDIA_POLICY, targets
from liquid_mcdm.models.ranking_baselines import FeatureState


def _labels(episodes: list[Mapping], policy: Mapping) -> list[int]:
    output = []
    for episode in episodes:
        target = targets(episode, policy)
        if target["policy_id"] == EXPEDIA_POLICY:
            output.extend(target["raw_labels_by_id"][i]["grade"] for i in target["available_ids"])
        elif target["policy_id"] == ESCI_POLICY:
            if target["unjudged_count"]:
                raise ValueError("LambdaRank cannot train on unjudged ESCI candidates")
            output.extend({"I": 0, "S": 1, "C": 2, "E": 3}[target["raw_labels_by_id"][i]]
                          for i in target["available_ids"])
        else:
            raise ValueError("LambdaRank supports Expedia and fully judged ESCI only")
    return output


def _predictions(episodes, ids, groups, values):
    if sum(groups) != len(values):
        raise ValueError("group sizes do not cover prediction rows")
    predictions, start = {}, 0
    for episode, candidate_ids, count in zip(episodes, ids, groups, strict=True):
        predictions[episode["episode_id"]] = dict(zip(candidate_ids,
            (float(v) for v in values[start:start + count]), strict=True))
        start += count
    return predictions


def fit_ranker(train: list[Mapping], validation: list[Mapping], config: Mapping,
               output: Path) -> dict:
    """Fit, select with our NDCG@5, persist bundle, and return validation receipt."""
    import lightgbm as lgb

    if not train or not validation:
        raise ValueError("train and validation must both contain complete groups")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    policy = config["target_policy"]
    policy_id = policy["policy_id"]
    if any(e["supervision"]["policy_id"] != policy_id for e in train + validation):
        raise ValueError("target policy mismatch")
    feature_state = FeatureState(config["feature_policy"]).fit(train)
    x_train, train_groups, train_ids = feature_state.transform(train)
    x_validation, validation_groups, validation_ids = feature_state.transform(validation)
    labels_train, labels_validation = _labels(train, policy), _labels(validation, policy)
    if sum(train_groups) != len(labels_train) or sum(validation_groups) != len(labels_validation):
        raise ValueError("contiguous group sizes do not cover labels")
    gains = [0, 1, 5] if policy_id == EXPEDIA_POLICY else [0, 0.01, 0.1, 1]
    seed = int(config.get("seed", 42))
    params = {"objective": "lambdarank", "metric": "None", "label_gain": gains,
              "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
              "num_threads": 4, "deterministic": True, "force_col_wise": True,
              "seed": seed, "verbosity": -1}
    params.update(config.get("lightgbm", {}))
    # Frozen gains cannot be overridden by an experiment config.
    params["label_gain"] = gains
    params["metric"] = "None"
    training = lgb.Dataset(x_train, label=labels_train, group=train_groups, free_raw_data=False)
    development = lgb.Dataset(x_validation, label=labels_validation,
                              group=validation_groups, reference=training, free_raw_data=False)

    def validation_ndcg5(values, _dataset):
        predictions = _predictions(validation, validation_ids, validation_groups, values)
        report = evaluate_ranking(validation, predictions, policy)
        segment = next(iter(report["by_source_task_policy"].values()))
        metric = segment["macro"]["ndcg@5"]
        # No validation signal is an error, not a score of zero or one.
        if metric["eligible"] == 0:
            raise ValueError("validation has no positive-ideal-gain query")
        return "independent_ndcg_at_5", metric["value"], True

    booster = lgb.train(params, training, num_boost_round=int(config.get("max_trees", 500)),
                        valid_sets=[development], valid_names=["validation"],
                        feval=validation_ndcg5,
                        callbacks=[lgb.early_stopping(int(config.get("patience", 50)),
                                                      first_metric_only=True, verbose=False)])
    best_iteration = booster.best_iteration
    validation_values = booster.predict(x_validation, num_iteration=best_iteration)
    validation_predictions = _predictions(validation, validation_ids, validation_groups, validation_values)
    validation_report = evaluate_ranking(validation, validation_predictions, policy)
    output.mkdir(parents=True)
    booster.save_model(str(output / "model.txt"), num_iteration=best_iteration)
    bundle = {"schema_version": "ordivo-ranker-bundle-v1", "target_policy": policy,
              "feature_state": feature_state.to_dict(), "seed": seed,
              "best_iteration": best_iteration, "lightgbm_version": lgb.__version__,
              "train_queries": len(train), "validation_queries": len(validation),
              "train_group_sizes": train_groups, "validation_group_sizes": validation_groups}
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True))
    receipt = {"bundle": bundle, "validation": validation_report,
               "model_dir": str(output)}
    (output / "validation_metrics.json").write_text(json.dumps(validation_report, indent=2, sort_keys=True))
    return receipt


def predict_ranker(model_dir: Path, episodes) -> dict[str, dict[str, float]]:
    import lightgbm as lgb

    model_dir = Path(model_dir)
    bundle = json.loads((model_dir / "bundle.json").read_text())
    episodes = list(episodes)
    if any(e["supervision"]["policy_id"] != bundle["target_policy"]["policy_id"]
           for e in episodes):
        raise ValueError("saved target policy does not match episodes")
    state = FeatureState.from_dict(bundle["feature_state"])
    matrix, groups, ids = state.transform(episodes)
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    values = booster.predict(matrix, num_iteration=bundle["best_iteration"])
    return _predictions(episodes, ids, groups, values)
