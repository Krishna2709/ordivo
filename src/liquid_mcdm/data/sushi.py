"""Official SUSHI3 2016 questionnaire rankings, for research evaluation only."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Iterator

from .behavioral import DecisionEpisode, SCHEMA_VERSION, validate_episode
from .grouped import file_sha256

SOURCE = "sushi3_2016"
SERIES = (("a", 10), ("b", 100))
ITEM_FIELDS = ("item_name", "style", "major_group", "minor_group", "oiliness",
               "eating_frequency", "normalized_price", "shop_availability")
OFFICIAL_SHA256 = {
    "order_a": "c56da325c0f58f9a38bafc650ed699b48dd2fb9f6ad64ba61ffc288d1ae4cc24",
    "order_b": "a241c87331e4eabea5841aae0c8a982453cccd01b0a0f3b6a4e9cfabcdab563b",
    "items": "568d73580f82a9ffba7ee1a3750506a0517c9ce904eec41559fe4ab573911fde",
    "respondents": "592d345613b6fbc1d7d126775eb5fb5628619e855c292166250f09651716eca2",
}
OFFICIAL_RESPONDENTS = 5000


def _tab_rows(path: Path, width: int) -> list[list[str]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != width:
                raise ValueError(f"{path.name} row {line_number} has {len(fields)} fields, expected {width}")
            rows.append(fields)
    return rows


def _order_rows(path: Path, universe: int) -> Iterator[list[str]]:
    with Path(path).open(encoding="utf-8") as handle:
        if handle.readline().split() != [str(universe), "1"]:
            raise ValueError(f"unexpected SUSHI order header in {path.name}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.split()
            if len(fields) != 12 or fields[:2] != ["0", "10"]:
                raise ValueError(f"invalid SUSHI ranking row {line_number}")
            ranked = fields[2:]
            if len(set(ranked)) != 10 or any(not item.isdecimal() or int(item) >= universe for item in ranked):
                raise ValueError(f"invalid SUSHI candidate set at row {line_number}")
            yield ranked


def iter_episodes(order_a: Path, order_b: Path, items: Path, respondents: Path,
                  policy: dict, *, verify_official: bool = True) -> Iterator[DecisionEpisode]:
    """Keep each complete original ten-item set and align respondents by row."""
    order_a, order_b, items, respondents = map(Path, (order_a, order_b, items, respondents))
    item_rows = _tab_rows(items, 9)
    item_features = {}
    for row in item_rows:
        item_id = row[0]
        if not item_id.isdecimal() or item_id in item_features:
            raise ValueError("duplicate or invalid SUSHI item ID")
        item_features[item_id] = dict(zip(ITEM_FIELDS, row[1:]))
    if set(item_features) != {str(i) for i in range(100)}:
        raise ValueError("SUSHI item metadata must cover IDs 0..99")
    respondent_rows = _tab_rows(respondents, 11)
    respondent_ids = [row[0] for row in respondent_rows]
    if any(not id_ for id_ in respondent_ids) or len(set(respondent_ids)) != len(respondent_ids):
        raise ValueError("SUSHI respondent IDs must be unique and nonempty")
    source_hashes = {"order_a": file_sha256(order_a), "order_b": file_sha256(order_b),
                     "items": file_sha256(items), "respondents": file_sha256(respondents)}
    if verify_official and source_hashes != OFFICIAL_SHA256:
        raise ValueError("official SUSHI source hash mismatch")
    if verify_official and len(respondent_rows) != OFFICIAL_RESPONDENTS:
        raise ValueError("official SUSHI source must have 5,000 respondent rows")
    profile = policy["source_profile"] = {"raw_rows": 0, "selected_rows": 0,
                                          "omitted_by_reason": {}}
    orders = ( _order_rows(order_a, 10), _order_rows(order_b, 100) )
    for index, (respondent, a_ranked, b_ranked) in enumerate(
            zip_longest(respondent_ids, *orders), start=1):
        if respondent is None or a_ranked is None or b_ranked is None:
            raise ValueError("SUSHI respondent/order row count mismatch")
        for (series, _), ranked in zip(SERIES, (a_ranked, b_ranked)):
            profile["raw_rows"] += 1
            profile["selected_rows"] += 1
            group_id = f"{series}:{respondent}"
            alternatives = [{"id": item_id, "features": item_features[item_id].copy(),
                             "available": True} for item_id in sorted(ranked, key=int)]
            episode: DecisionEpisode = {
                "schema_version": SCHEMA_VERSION, "episode_id": f"{SOURCE}:{group_id}",
                "source": SOURCE, "task_type": "explicit_ranking", "group_id": group_id,
                "context": {}, "criteria": [], "constraints": [], "preferences": {},
                "alternatives": alternatives,
                "supervision": {"kind": "explicit_ranking", "policy_id": "sushi3-2016-order-v1",
                                "respondent_id": respondent,
                                "ordered_groups": [[item_id] for item_id in ranked], "unranked_ids": []},
                "exposure": {},
                "provenance": {"raw_file_sha256": source_hashes, "source_row": index,
                               "series": series, "adapter_version": "sushi3-2016-v1"},
            }
            validate_episode(episode)
            yield episode
    profile["raw_groups"] = profile["raw_rows"]
    profile["selected_groups"] = profile["selected_rows"]
    profile["omitted_groups_by_reason"] = {}
    if verify_official and profile["raw_rows"] != 2 * OFFICIAL_RESPONDENTS:
        raise ValueError("official SUSHI source must have 10,000 ranking episodes")
