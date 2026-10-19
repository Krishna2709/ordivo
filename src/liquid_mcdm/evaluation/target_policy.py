"""Versioned, task-specific target projections for complete decision episodes."""

from __future__ import annotations

from collections.abc import Mapping


EXPEDIA_POLICY = "expedia-click-booking-v1"
ESCI_POLICY = "esci-task1-official-gain-v1"
SWISSMETRO_POLICY = "swissmetro-choice-v1"
EXPLICIT_POLICY = "explicit-ranking-v1"

# ESCI Task 1's published trec_eval conversion: E=4, C=3, S=2, I=1.
ESCI_GAINS = {"E": 1.0, "C": 0.1, "S": 0.01, "I": 0.0}


def targets(episode: Mapping, policy: Mapping | None = None) -> dict:
    """Return raw labels and gains by eligible ID, with explicit judgment masks.

    A policy may name ``policy_id`` but may not silently change frozen source gains.
    Unjudged candidates are excluded from graded metrics and counted separately.
    """
    from liquid_mcdm.data.behavioral import validate_episode

    validate_episode(episode)
    ids = [a["id"] for a in episode["alternatives"] if a.get("available", True)]
    supervision = episode["supervision"]
    kind = supervision["kind"]
    expected = {"observed_choice": EXPEDIA_POLICY,
                "graded_relevance": ESCI_POLICY,
                "stated_choice": SWISSMETRO_POLICY,
                "explicit_ranking": EXPLICIT_POLICY}[kind]
    declared = supervision.get("policy_id")
    if declared != expected or (policy and policy.get("policy_id", expected) != expected):
        raise ValueError(f"unsupported {kind} target policy: {declared}")
    raw, gains, events = {}, {}, {}
    judged = set(ids)
    if kind == "observed_choice":
        for candidate_id in ids:
            event = supervision["outcomes_by_id"][candidate_id]
            click, booking = bool(event["click_bool"]), bool(event["booking_bool"])
            grade = 2 if booking else 1 if click else 0
            raw[candidate_id] = {"click_bool": click, "booking_bool": booking, "grade": grade}
            gains[candidate_id] = (0.0, 1.0, 5.0)[grade]
        events = {"click": {i: float(raw[i]["click_bool"]) for i in ids},
                  "booking": {i: float(raw[i]["booking_bool"]) for i in ids}}
    elif kind == "graded_relevance":
        grades = supervision["grades_by_id"]
        judged = set(grades) & set(ids)
        if not judged and ids:
            raise ValueError("no judged available candidates")
        for candidate_id in ids:
            if candidate_id in judged:
                grade = grades[candidate_id]
                if grade not in ESCI_GAINS:
                    raise ValueError(f"unknown ESCI grade: {grade}")
                raw[candidate_id], gains[candidate_id] = grade, ESCI_GAINS[grade]
    elif kind == "stated_choice":
        selected = supervision["selected_id"]
        if selected not in ids:
            raise ValueError("selected candidate is unavailable")
        raw = {i: i == selected for i in ids}
        gains = {i: float(i == selected) for i in ids}
        events = {"choice": gains.copy()}
    else:
        groups = supervision["ordered_groups"]
        judged = {i for group in groups for i in group}
        if len(judged) != sum(map(len, groups)):
            raise ValueError("duplicate ranked candidate")
        if not judged <= set(ids):
            raise ValueError("ranked candidate unavailable")
        for rank, group in enumerate(groups):
            for candidate_id in group:
                raw[candidate_id] = rank
                gains[candidate_id] = float(len(groups) - rank)
    return {"task_type": kind, "policy_id": expected, "raw_labels_by_id": raw,
            "gains_by_id": gains, "events_by_name": events,
            "judged_ids": sorted(judged), "available_ids": ids,
            "unjudged_count": len(set(ids) - judged)}
