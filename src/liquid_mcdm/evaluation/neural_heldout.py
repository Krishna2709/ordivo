"""Pure paired held-out selection and metrics for a bounded neural probe."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from liquid_mcdm.evaluation.ranking import ndcg_at_k, ranked_ids
from liquid_mcdm.evaluation.target_policy import targets


def split_unit(episode: Mapping) -> str:
    return f"{episode['source']}:{episode['supervision'].get('respondent_id', episode['group_id'])}"


def select_whole_test_units(episodes: list[Mapping], *, source: str, seed: int,
                            limit: int) -> tuple[list[Mapping], list[str]]:
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("held-out selection limit must be 1..50 whole split units")
    if any(e["source"] != source for e in episodes):
        raise ValueError("selection input contains another source")
    units = {split_unit(e) for e in episodes}
    chosen = sorted(units, key=lambda unit: (
        hashlib.sha256(f"{seed}|{source}|test|{unit}".encode()).digest(), unit))[:limit]
    selected = [e for e in episodes if split_unit(e) in chosen]
    return selected, chosen


def eligibility(episode: Mapping, feature_policy: Mapping) -> tuple[str | None, str | None]:
    """Return task metric population and exclusion reason, without using labels for selection."""
    available = [a["id"] for a in episode["alternatives"] if a.get("available", True)]
    if len(available) < 2:
        return None, "fewer_than_two_available"
    task = episode["task_type"]
    if task == "stated_choice":
        return "choice", None
    if task == "graded_relevance":
        return "relevance", None
    if task == "observed_choice" and feature_policy.get("prediction_event") == "conditional_booking_choice":
        events = episode["supervision"]["outcomes_by_id"]
        if sum(bool(events[i]["booking_bool"]) for i in available) != 1:
            return None, "not_exactly_one_booking"
        return "choice", None
    raise ValueError("unsupported held-out task or prediction event")


def _choice_metrics(episode: Mapping, probabilities: Mapping[str, float]) -> dict:
    eligible = [a["id"] for a in episode["alternatives"] if a.get("available", True)]
    selected = (episode["supervision"]["selected_id"] if episode["task_type"] == "stated_choice"
                else next(i for i in eligible if episode["supervision"]["outcomes_by_id"][i]["booking_bool"]))
    if probabilities[selected] <= 0:
        raise ValueError("chosen option has zero neural probability")
    winning = ranked_ids(probabilities)[0]
    return {"choice_accuracy@1": float(winning == selected),
            "choice_nll": -math.log(probabilities[selected]),
            "choice_brier": sum((probabilities[i] - float(i == selected)) ** 2 for i in eligible)}


def _validate_probabilities(episode: Mapping, values: Mapping[str, float]) -> None:
    expected = {a["id"] for a in episode["alternatives"] if a.get("available", True)}
    if set(values) != expected or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                     or v < 0 for v in values.values()):
        raise ValueError("missing, extra, negative, or nonfinite neural probability")
    if abs(sum(values.values()) - 1.0) > 1e-5:
        raise ValueError("neural probabilities do not sum to one")


def evaluate_neural_pair(episodes: list[Mapping], base: Mapping, adapted: Mapping,
                         feature_policy: Mapping, overflow_ids: set[str] | None = None) -> dict:
    """Score the exact same admitted groups; reject any missing model response."""
    overflow_ids = set(overflow_ids or ())
    ids = [e["episode_id"] for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selected episode")
    if not overflow_ids <= set(ids):
        raise ValueError("overflow ID outside selected groups")
    admitted, excluded = [], {"fewer_than_two_available": 0,
                              "not_exactly_one_booking": 0, "prompt_overflow": 0}
    for episode in episodes:
        population, reason = eligibility(episode, feature_policy)
        if reason:
            excluded[reason] += 1
        elif episode["episode_id"] in overflow_ids:
            excluded["prompt_overflow"] += 1
        else:
            admitted.append((episode, population))
    expected = {e["episode_id"] for e, _ in admitted}
    if set(base) != expected or set(adapted) != expected:
        raise ValueError("incomplete paired neural prediction coverage")
    rows = []
    for episode, population in admitted:
        episode_id = episode["episode_id"]
        models = {}
        for name, predictions in (("base", base), ("adapted", adapted)):
            probabilities = predictions[episode_id]
            _validate_probabilities(episode, probabilities)
            if population == "choice":
                models[name] = _choice_metrics(episode, probabilities)
            else:
                target = targets(episode)
                judged_order = [i for i in ranked_ids(probabilities) if i in target["gains_by_id"]]
                models[name] = {"ndcg@10": ndcg_at_k(judged_order, target["gains_by_id"], 10)}
        rows.append({"episode_id": episode_id, "source": episode["source"],
                     "population": population, "candidate_count": len(base[episode_id]),
                     "base": models["base"], "adapted": models["adapted"]})
    metrics = sorted({metric for row in rows for metric in row["base"]})
    aggregate = {}
    for metric in metrics:
        paired = [(row["base"][metric], row["adapted"][metric]) for row in rows
                  if metric in row["base"] and row["base"][metric] is not None
                  and row["adapted"][metric] is not None]
        aggregate[metric] = {"eligible": len(paired),
                             "excluded_zero_ideal_gain": sum(metric in row["base"] and
                                 row["base"][metric] is None for row in rows),
                             "base": sum(a for a, _ in paired) / len(paired) if paired else None,
                             "adapted": sum(b for _, b in paired) / len(paired) if paired else None,
                             "adapted_minus_base": sum(b - a for a, b in paired) / len(paired)
                             if paired else None}
    return {"schema_version": "ordivo-neural-heldout-v1", "selected_episodes": len(episodes),
            "selected_units": len({split_unit(e) for e in episodes}),
            "admitted_queries": len(admitted), "scored_base": len(base),
            "scored_adapted": len(adapted), "failed_queries": 0,
            "excluded": excluded, "metrics": aggregate, "per_query": rows}


def selected_units_sha256(units: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(units), separators=(",", ":")).encode()).hexdigest()
