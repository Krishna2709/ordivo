"""Source-independent, JSON-serializable choice and relevance boundary."""

from __future__ import annotations

import math
from typing import Any, TypedDict

SCHEMA_VERSION = "ordivo-behavioral-v1"
TASK_TYPES = {"observed_choice", "stated_choice", "graded_relevance", "explicit_ranking"}
FORBIDDEN_FEATURES = {
    "source", "group_id", "episode_id", "supervision", "exposure", "provenance",
    "position", "position_by_id", "random_bool", "click_bool", "booking_bool",
    "esci_label", "grade", "choice", "selected_id", "respondent_id",
    "srch_id", "prop_id", "query_id", "product_id", "example_id", "id",
}


class DecisionEpisode(TypedDict):
    schema_version: str
    episode_id: str
    source: str
    task_type: str
    group_id: str
    context: dict[str, Any]
    criteria: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    preferences: dict[str, Any]
    alternatives: list[dict[str, Any]]
    supervision: dict[str, Any]
    exposure: dict[str, Any]
    provenance: dict[str, Any]


class DecisionInput(TypedDict):
    task_type: str
    context: dict[str, Any]
    criteria: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    preferences: dict[str, Any]
    alternatives: list[dict[str, Any]]


CandidateScores = dict[str, float]


def _ids(episode: DecisionEpisode) -> tuple[set[str], set[str]]:
    alternatives = episode["alternatives"]
    if not alternatives:
        raise ValueError("episode has no alternatives")
    ids = [candidate["id"] for candidate in alternatives]
    if any(not isinstance(id_, str) or not id_ for id_ in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique nonempty strings")
    if any(type(candidate.get("available", True)) is not bool for candidate in alternatives):
        raise ValueError("available must be boolean")
    return set(ids), {candidate["id"] for candidate in alternatives if candidate.get("available", True)}


def validate_episode(episode: DecisionEpisode) -> None:
    """Reject malformed labels, ID collisions and unavailable positive choices."""
    if episode.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported behavioral schema")
    source, group, episode_id = (episode.get(key) for key in ("source", "group_id", "episode_id"))
    if not all(isinstance(value, str) and value for value in (source, group, episode_id)):
        raise ValueError("source, group_id and episode_id are required")
    if episode_id != f"{source}:{group}":
        raise ValueError("episode_id must namespace source and group_id")
    kind = episode.get("task_type")
    supervision = episode.get("supervision")
    if kind not in TASK_TYPES or not isinstance(supervision, dict) or supervision.get("kind") != kind:
        raise ValueError("task_type and supervision kind must match")
    if not supervision.get("policy_id"):
        raise ValueError("versioned target policy_id required")
    for key, expected in (("context", dict), ("criteria", list), ("constraints", list),
                          ("preferences", dict), ("exposure", dict), ("provenance", dict)):
        if not isinstance(episode.get(key), expected):
            raise ValueError(f"{key} must be {expected.__name__}")
    ids, available = _ids(episode)
    for candidate in episode["alternatives"]:
        if not isinstance(candidate.get("features"), dict):
            raise ValueError("candidate features must be an object")
    if kind == "observed_choice":
        outcomes = supervision.get("outcomes_by_id")
        if not isinstance(outcomes, dict) or set(outcomes) != ids:
            raise ValueError("observed outcomes must cover every candidate")
        for id_, events in outcomes.items():
            if not isinstance(events, dict) or any(type(events.get(k)) not in (bool, int) or events[k] not in (0, 1) for k in ("click_bool", "booking_bool")):
                raise ValueError("click and booking must be binary")
            if id_ not in available and (events["click_bool"] or events["booking_bool"]):
                raise ValueError("unavailable candidate has positive outcome")
    elif kind == "stated_choice":
        if supervision.get("selected_id") not in available:
            raise ValueError("selected candidate must exist and be available")
        if "respondent_id" in supervision and not str(supervision["respondent_id"]):
            raise ValueError("respondent_id cannot be empty")
    elif kind == "graded_relevance":
        grades = supervision.get("grades_by_id")
        unjudged = supervision.get("unjudged_ids", [])
        if not isinstance(grades, dict) or not isinstance(unjudged, list) or len(unjudged) != len(set(unjudged)):
            raise ValueError("invalid judgment mask")
        if set(grades) & set(unjudged) or set(grades) | set(unjudged) != ids:
            raise ValueError("judged and explicit unjudged IDs must partition candidates")
        if any(v is None for v in grades.values()):
            raise ValueError("null is not a judgment")
    else:
        ordered = supervision.get("ordered_groups")
        unranked = supervision.get("unranked_ids", [])
        if not isinstance(ordered, list) or not all(isinstance(group, list) and group for group in ordered):
            raise ValueError("ordered_groups must contain nonempty tie groups")
        ranked = [id_ for group in ordered for id_ in group]
        if len(ranked) != len(set(ranked)) or set(ranked) & set(unranked) or set(ranked) | set(unranked) != ids:
            raise ValueError("ranking and unranked IDs must partition candidates")


def _approved(values: dict[str, Any], names: list[str], where: str) -> dict[str, Any]:
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{where} allowlist required")
    if set(names) & FORBIDDEN_FEATURES:
        raise ValueError(f"{where} allowlist contains forbidden field")
    missing = set(names) - set(values)
    if missing:
        raise ValueError(f"{where} allowlist absent fields: {sorted(missing)}")
    return {name: _safe_declared(values[name]) for name in names}


def _safe_declared(value: Any) -> Any:
    """Preserve actual criteria/preferences while rejecting embedded leakage."""
    if isinstance(value, dict):
        if set(value) & FORBIDDEN_FEATURES:
            raise ValueError("declared decision input contains forbidden field")
        return {key: _safe_declared(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_declared(item) for item in value]
    return value


def inference_input(episode: DecisionEpisode, feature_policy: dict[str, Any]) -> DecisionInput:
    """Project only declared, user-observable input fields; identities remain keys."""
    validate_episode(episode)
    if not isinstance(feature_policy, dict):
        raise ValueError("feature policy required")
    context_names = feature_policy.get("context_features")
    feature_names = feature_policy.get("alternative_features")
    if not isinstance(feature_names, list):
        raise ValueError("alternative_features allowlist required")
    alternatives = []
    for candidate in episode["alternatives"]:
        alternatives.append({
            "id": candidate["id"],
            "features": _approved(candidate["features"], feature_names, "alternative"),
            "available": candidate.get("available", True),
        })
    return {
        "task_type": episode["task_type"],
        "context": _approved(episode["context"], context_names, "context"),
        "criteria": _safe_declared(episode["criteria"]),
        "constraints": _safe_declared(episode["constraints"]),
        "preferences": _safe_declared(episode["preferences"]),
        "alternatives": alternatives,
    }


def validate_scores(episode: DecisionEpisode, scores: CandidateScores) -> None:
    validate_episode(episode)
    _, available = _ids(episode)
    if not isinstance(scores, dict) or set(scores) != available:
        raise ValueError("scores must cover exactly available candidate IDs")
    if any(type(score) not in (float, int) or not math.isfinite(score) for score in scores.values()):
        raise ValueError("scores must be finite numbers")
