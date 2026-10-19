"""Expedia Personalized Sort labeled search adapter."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .behavioral import DecisionEpisode, SCHEMA_VERSION, validate_episode
from .grouped import file_sha256, grouped_rows

SOURCE = "expedia_personalized_sort_2013"
CONTEXT_FIELDS = ("srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count", "srch_room_count", "srch_saturday_night_bool")
FEATURE_FIELDS = ("price_usd", "prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2")
NUMERIC_FIELDS = set(CONTEXT_FIELDS) | set(FEATURE_FIELDS)
CRITERIA = [{"name": "price_usd", "direction": "min", "weight": None},
            {"name": "prop_starrating", "direction": "max", "weight": None},
            {"name": "prop_review_score", "direction": "max", "weight": None}]


def _number(value: str | None) -> int | float | None:
    if value is None or value == "" or value.strip().upper() == "NULL":
        return None
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError("non-finite Expedia numeric field")
    return int(number) if number.is_integer() else number


def iter_episodes(path: Path, policy: dict) -> Iterator[DecisionEpisode]:
    path = Path(path)
    source_hash = file_sha256(path)
    profile = policy["source_profile"] = {"raw_rows": 0, "selected_rows": 0, "omitted_by_reason": {}}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"srch_id", "prop_id", "click_bool", "booking_bool", "random_bool", "position"} | set(CONTEXT_FIELDS) | set(FEATURE_FIELDS)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"missing Expedia fields: {sorted(required - set(reader.fieldnames or []))}")
        def counted_rows():
            for row in reader:
                profile["raw_rows"] += 1
                profile["selected_rows"] += 1
                if any(row[name] is None or not row[name].strip() or row[name].strip().upper() == "NULL" for name in ("srch_id", "prop_id")):
                    raise ValueError("Expedia search and property IDs cannot be missing")
                # Keep only the approved dictionary/exposure/label columns in
                # the disk spool; the competition CSV has many large excluded
                # competitor/history fields.
                yield {name: row[name] for name in required}
        with grouped_rows(counted_rows(), group_key="srch_id", row_key="prop_id", temp_dir=policy.get("temp_dir")) as groups:
            completed_groups = 0
            for group_id, rows in groups:
                completed_groups += 1
                first = rows[0]
                context = {name: _number(first[name]) for name in CONTEXT_FIELDS}
                random_bool = _binary(first["random_bool"])
                for row in rows[1:]:
                    if {name: _number(row[name]) for name in CONTEXT_FIELDS} != context or _binary(row["random_bool"]) != random_bool:
                        raise ValueError(f"conflicting search context or randomization in {group_id}")
                alternatives = []
                outcomes = {}
                positions = {}
                for row in rows:
                    candidate_id = str(row["prop_id"])
                    alternatives.append({"id": candidate_id, "features": {name: _number(row[name]) for name in FEATURE_FIELDS}, "available": True})
                    outcomes[candidate_id] = {"click_bool": _binary(row["click_bool"]), "booking_bool": _binary(row["booking_bool"])}
                    positions[candidate_id] = _number(row["position"])
                episode: DecisionEpisode = {
                    "schema_version": SCHEMA_VERSION, "episode_id": f"{SOURCE}:{group_id}", "source": SOURCE,
                    "task_type": "observed_choice", "group_id": group_id, "context": context,
                    "criteria": CRITERIA, "constraints": [], "preferences": {}, "alternatives": alternatives,
                    "supervision": {"kind": "observed_choice", "policy_id": "expedia-click-booking-v1", "outcomes_by_id": outcomes},
                    "exposure": {"random_bool": random_bool, "position_by_id": positions},
                    "provenance": {"raw_file_sha256": source_hash, "adapter_version": "expedia-v1"},
                }
                validate_episode(episode)
                yield episode
            profile["raw_groups"] = completed_groups
            profile["selected_groups"] = completed_groups
            profile["omitted_groups_by_reason"] = {}


def _binary(value: str | None) -> int:
    if value not in ("0", "1"):
        raise ValueError(f"expected binary Expedia field, got {value!r}")
    return int(value)
