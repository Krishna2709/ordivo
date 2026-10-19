"""Invented complete searches prove a label-independent bounded source sample."""

import hashlib
import json

import pytest

from scripts.sample_behavioral_groups import (
    FROZEN_PARENT_SHA256, SOURCE, _require_parent_binding, _select_split,
)


def episode(group, booked=False):
    return {"episode_id": f"{SOURCE}:{group}", "group_id": group, "source": SOURCE,
            "alternatives": [
                {"id": "a", "features": {"price_usd": 12.5}, "available": True},
                {"id": "b", "features": {"price_usd": 20.0}, "available": True}],
            "supervision": {"kind": "observed_choice", "outcomes_by_id": {
                "a": {"click_bool": int(booked), "booking_bool": int(booked)},
                "b": {"click_bool": 0, "booking_bool": 0}}}}


def write_rows(path, rows):
    path.write_bytes(b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(path, digest, seen=None, split="train"):
    return _select_split(path, source=SOURCE, split=split, seed=42, cap=2,
                         expected_sha256=digest, seen_groups={} if seen is None else seen)


def test_sample_selection_ignores_source_order_and_outcomes(tmp_path):
    rows = [episode(f"q{i}", i % 2 == 0) for i in range(4)]
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first_hash = write_rows(first, rows)
    second_hash = write_rows(second, list(reversed(rows)))
    chosen_a, counts_a = select(first, first_hash)
    chosen_b, counts_b = select(second, second_hash)
    assert [item[1] for item in chosen_a] == [item[1] for item in chosen_b]
    assert counts_a == counts_b == {"full_groups": 4, "full_candidates": 8,
                                    "selected_groups": 2, "selected_candidates": 4,
                                    "excluded_groups": 2, "excluded_candidates": 4}
    original_by_id = {row["episode_id"]: row for row in rows}
    assert all(json.loads(raw) == original_by_id[episode_id]
               for _, episode_id, raw in chosen_a)
    relabeled = [episode(f"q{i}", i % 2 != 0) for i in range(4)]
    third = tmp_path / "relabeled.jsonl"
    third_hash = write_rows(third, relabeled)
    chosen_c, _ = select(third, third_hash)
    assert [item[1] for item in chosen_a] == [item[1] for item in chosen_c]


def test_sample_rejects_hash_mismatch_and_split_group_overlap(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train_hash = write_rows(train, [episode("q1")])
    val_hash = write_rows(validation, [episode("q1")])
    with pytest.raises(ValueError, match="hash differs"):
        select(train, "0" * 64)
    seen = {}
    select(train, train_hash, seen=seen)
    with pytest.raises(ValueError, match="split overlap"):
        select(validation, val_hash, seen=seen, split="validation")


def test_sample_rejects_duplicate_group_inside_split(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    digest = write_rows(path, [episode("q1"), episode("q1")])
    with pytest.raises(ValueError, match="duplicate group"):
        select(path, digest)


def test_predeclared_parent_hash_rejects_internally_consistent_other_manifest():
    config = {"expected_parent_manifest_sha256": FROZEN_PARENT_SHA256}
    # This could describe a self-consistent replacement four-way source. The
    # predeclared acquisition audit, not merely self-consistency, must bind it.
    replacement = json.dumps({"episode_count": 4, "split_sha256": {"train": "a" * 64}}).encode()
    with pytest.raises(ValueError, match="audited parent hash"):
        _require_parent_binding(config, replacement)
    with pytest.raises(ValueError, match="audited parent hash"):
        _require_parent_binding({"expected_parent_manifest_sha256": "0" * 64}, replacement)
