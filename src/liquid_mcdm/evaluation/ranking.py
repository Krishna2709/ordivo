"""Independent group-level ranking metrics with a complete prediction ledger."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isfinite, log2

from liquid_mcdm.evaluation.target_policy import targets


def ranked_ids(scores: Mapping[str, float]) -> list[str]:
    """Sort descending scores; canonical IDs break exact score ties."""
    return sorted(scores, key=lambda candidate_id: (-scores[candidate_id], candidate_id))


def ndcg_at_k(order: list[str], gains: Mapping[str, float], k: int) -> float | None:
    if k < 1:
        raise ValueError("k must be positive")
    ideal = sorted(gains.values(), reverse=True)[:k]
    denominator = sum(g / log2(rank + 2) for rank, g in enumerate(ideal))
    if denominator == 0:
        return None
    return sum(gains.get(i, 0.0) / log2(rank + 2)
               for rank, i in enumerate(order[:k])) / denominator


def _macro(rows: list[dict], field: str) -> dict:
    values = [row["metrics"].get(field) for row in rows]
    eligible = [value for value in values if value is not None]
    return {"value": sum(eligible) / len(eligible) if eligible else None,
            "eligible": len(eligible), "excluded_zero_ideal_gain": len(values) - len(eligible)}


def _estimand(rows: list[dict]) -> str:
    names = {("observed_choice", "expedia_personalized_sort_2013"):
             "ranking_within_exposed_hotel_search_candidates",
             ("graded_relevance", "amazon_esci"):
             "ranking_within_judged_task1_reduced_candidates",
             ("stated_choice", "swissmetro"):
             "choice_among_available_survey_modes"}
    keys = {(row["task_type"], row["source"]) for row in rows}
    return names.get(next(iter(keys)), "task_specific_candidate_pool_metrics") if len(keys) == 1 else "task_specific_candidate_pool_metrics"


def evaluate_ranking(episodes: Iterable[Mapping], predictions: Mapping[str, Mapping[str, float]],
                     policy: Mapping | None = None) -> dict:
    """Evaluate all expected episodes; fail the whole call on missing/extra scores.

    Macro values never pool incompatible target types. Every episode receives a
    per-query row, including groups with no positive event.
    """
    from liquid_mcdm.data.behavioral import validate_scores

    episodes = list(episodes)
    expected_ids = [e["episode_id"] for e in episodes]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate namespaced episode_id")
    missing, extra = sorted(set(expected_ids) - set(predictions)), sorted(set(predictions) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"incomplete prediction coverage: missing={missing}, extra={extra}")
    rows = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        scores = predictions[episode_id]
        validate_scores(episode, scores)
        if any(not isfinite(float(score)) for score in scores.values()):
            raise ValueError(f"non-finite prediction: {episode_id}")
        target = targets(episode, policy)
        order = ranked_ids(scores)
        gain = target["gains_by_id"]
        # Unjudged relevance items are not negatives and do not consume metric ranks.
        judged_order = [i for i in order if i in gain]
        metrics = {f"ndcg@{k}": ndcg_at_k(judged_order, gain, k) for k in (5, 10)}
        if target["task_type"] == "observed_choice":
            for name in ("click", "booking"):
                event = target["events_by_name"][name]
                for k in (5, 10):
                    metrics[f"{name}_ndcg@{k}"] = ndcg_at_k(order, event, k)
            booking = target["events_by_name"]["booking"]
            first_booking = next((rank for rank, i in enumerate(order, 1) if booking[i]), None)
            metrics["booking_mrr"] = 1.0 / first_booking if first_booking else None
            metrics["booking_hit@1"] = float(bool(booking[order[0]])) if first_booking else None
        elif target["task_type"] == "stated_choice":
            selected = episode["supervision"]["selected_id"]
            rank = order.index(selected) + 1
            metrics["choice_accuracy@1"] = float(rank == 1)
            metrics["choice_mrr"] = 1.0 / rank
        elif target["task_type"] == "explicit_ranking":
            groups = episode["supervision"]["ordered_groups"]
            rank_by_id = {i: r for r, group in enumerate(groups) for i in group}
            comparable = [(a, b) for a in rank_by_id for b in rank_by_id
                          if rank_by_id[a] < rank_by_id[b]]
            positions = {i: r for r, i in enumerate(order)}
            metrics["pair_agreement"] = (sum(positions[a] < positions[b] for a, b in comparable)
                                          / len(comparable) if comparable else None)
        rows.append({"episode_id": episode_id, "source": episode["source"],
                     "task_type": target["task_type"], "policy_id": target["policy_id"],
                     "candidate_count": len(order), "judged_count": len(target["judged_ids"]),
                     "unjudged_count": target["unjudged_count"], "ranking": order,
                     "tied_score_count": len(scores) - len(set(scores.values())),
                     "metrics": metrics})
    slices = {}
    for source, task, target_policy in sorted({(r["source"], r["task_type"], r["policy_id"]) for r in rows}):
        group = [r for r in rows if (r["source"], r["task_type"], r["policy_id"]) ==
                 (source, task, target_policy)]
        fields = sorted({field for row in group for field in row["metrics"]})
        slices[f"{source}|{task}|{target_policy}"] = {
            "query_count": len(group), "tied_score_queries": sum(r["tied_score_count"] > 0 for r in group),
            "unjudged_candidates": sum(r["unjudged_count"] for r in group),
            "macro": {field: _macro(group, field) for field in fields}}
    return {"schema_version": "ordivo-ranking-eval-v1", "expected_queries": len(episodes),
            "scored_queries": len(rows), "failed_queries": 0, "missing_predictions": [],
            "estimand": _estimand(rows),
            "prediction_tie_break": "canonical_candidate_id_ascending",
            "per_query": rows, "by_source_task_policy": slices}
