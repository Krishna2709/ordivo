"""Swissmetro stated-choice panel adapter."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from .behavioral import DecisionEpisode, SCHEMA_VERSION, validate_episode
from .grouped import file_sha256

SOURCE = "swissmetro"
MODES = ((1, "train", "TRAIN_AV", "TRAIN_TT", "TRAIN_CO", "TRAIN_HE"),
         (2, "swissmetro", "SM_AV", "SM_TT", "SM_CO", "SM_HE"),
         (3, "car", "CAR_AV", "CAR_TT", "CAR_CO", None))


def iter_episodes(path: Path, policy: dict) -> Iterator[DecisionEpisode]:
    path = Path(path)
    raw_hash = file_sha256(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"ID", "CHOICE", "PURPOSE", "GA", "WHO", "LUGGAGE"}
        for _, _, availability, time, cost, headway in MODES:
            required.update((availability, time, cost))
            if headway:
                required.add(headway)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing Swissmetro fields: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=1):
            respondent = row["ID"]
            if not respondent:
                raise ValueError(f"empty Swissmetro respondent at row {row_index}")
            choice = _integer(row["CHOICE"], "CHOICE")
            if choice == 0:
                # Biogeme's official example excludes no-choice rows; record
                # skipped count in the acquisition profile, not a fake target.
                continue
            if choice not in (1, 2, 3):
                raise ValueError(f"invalid Swissmetro choice {choice}")
            alternatives = []
            for code, name, availability, time, cost, headway in MODES:
                available = _integer(row[availability], availability)
                if available not in (0, 1):
                    raise ValueError(f"invalid availability {availability}={available}")
                alternatives.append({
                    "id": name,
                    "available": bool(available),
                    "features": {
                        "travel_time_minutes": _number(row[time]),
                        "cost_chf": _number(row[cost]),
                        "headway_minutes": _number(row[headway]) if headway else None,
                    },
                })
            episode: DecisionEpisode = {
                "schema_version": SCHEMA_VERSION,
                "episode_id": f"{SOURCE}:{respondent}:{row_index}",
                "source": SOURCE, "task_type": "stated_choice", "group_id": f"{respondent}:{row_index}",
                "context": {name.lower(): _number(row[name]) for name in ("PURPOSE", "GA", "WHO", "LUGGAGE")},
                "criteria": [{"name": "travel_time_minutes", "direction": "min", "weight": None},
                             {"name": "cost_chf", "direction": "min", "weight": None}],
                "constraints": [], "preferences": {}, "alternatives": alternatives,
                "supervision": {"kind": "stated_choice", "policy_id": "swissmetro-choice-v1",
                                "selected_id": MODES[choice - 1][1], "respondent_id": respondent},
                "exposure": {},
                "provenance": {"raw_file_sha256": raw_hash, "source_row": row_index, "adapter_version": "swissmetro-v1"},
            }
            validate_episode(episode)
            yield episode


def _integer(value: str, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer {name}={value!r}") from exc
    return number


def _number(value: str | None) -> int | float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError("non-finite Swissmetro field")
    return int(number) if number.is_integer() else number
