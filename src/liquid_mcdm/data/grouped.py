"""Deterministic source splits and disk-backed group assembly."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .behavioral import DecisionEpisode, validate_episode


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_split(source: str, group_id: str, policy: dict, official_split: str | None = None) -> str:
    policy_id = policy.get("split_policy_id")
    if source == "expedia_personalized_sort_2013":
        if policy_id != "expedia-2013-split-v2-calibration" or official_split is not None:
            raise ValueError("Expedia requires namespaced search hash split")
        digest = hashlib.sha256(f"expedia-2013|split-v1|42|{group_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        return "train" if bucket < 80 else "validation" if bucket < 85 else "calibration" if bucket < 90 else "test"
    if source == "amazon_esci":
        if policy_id != "esci-official-train-test-v2-calibration" or official_split not in ("train", "test"):
            raise ValueError("ESCI requires official train/test partition")
        if official_split == "test":
            return "test"
        digest = hashlib.sha256(f"amazon-esci|validation-v1|42|{group_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        return "validation" if bucket < 5 else "calibration" if bucket < 10 else "train"
    if source == "swissmetro":
        if policy_id != "swissmetro-respondent-v2-calibration" or official_split is not None:
            raise ValueError("Swissmetro requires respondent hash split")
        # group_id is the respondent ID; observations inherit that assignment.
        digest = hashlib.sha256(f"swissmetro|respondent-v1|42|{group_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        return "train" if bucket < 80 else "validation" if bucket < 85 else "calibration" if bucket < 90 else "test"
    raise ValueError(f"no registered split policy for {source}")


@contextmanager
def grouped_rows(rows: Iterable[dict[str, Any]], *, group_key: str, row_key: str,
                 temp_dir: Path | None = None) -> Iterator[Iterator[tuple[str, list[dict[str, Any]]]]]:
    """Spool arbitrary row order to SQLite, then yield one complete group at a time.

    Peak Python memory is one candidate group. SQLite's PRIMARY KEY detects
    repeated source identities rather than silently deduplicating them.
    """
    with tempfile.TemporaryDirectory(dir=temp_dir) as directory:
        connection = sqlite3.connect(str(Path(directory) / "groups.sqlite3"))
        try:
            connection.execute("CREATE TABLE rows (group_id TEXT, row_id TEXT, payload TEXT, PRIMARY KEY(group_id,row_id))")
            for row in rows:
                group_id, row_id = str(row[group_key]), str(row[row_key])
                if not group_id or not row_id:
                    raise ValueError("empty source group or row key")
                try:
                    connection.execute("INSERT INTO rows VALUES (?,?,?)", (group_id, row_id, json.dumps(row, sort_keys=True, allow_nan=False)))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate source key ({group_id}, {row_id})") from exc
            connection.commit()

            def iterate() -> Iterator[tuple[str, list[dict[str, Any]]]]:
                cursor = connection.execute("SELECT group_id,payload FROM rows ORDER BY group_id,row_id")
                current: str | None = None
                group: list[dict[str, Any]] = []
                for group_id, payload in cursor:
                    if current is not None and group_id != current:
                        yield current, group
                        group = []
                    current = group_id
                    group.append(json.loads(payload))
                if current is not None:
                    yield current, group

            yield iterate()
        finally:
            connection.close()


def write_dataset(episodes: Iterable[DecisionEpisode], output: Path, policy: dict) -> dict[str, Any]:
    """Finalize complete immutable split files and manifest in one rename."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(policy.get("source_files"), dict) or not policy.get("split_policy_id"):
        raise ValueError("source_files and split_policy_id required for dataset manifest")
    source_hashes = {name: file_sha256(Path(path)) for name, path in sorted(policy["source_files"].items())}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    seen: set[str] = set()
    retained_rows = 0
    candidate_rows = 0
    handles = {split: (stage / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "validation", "calibration", "test")}
    try:
        try:
            for episode in episodes:
                validate_episode(episode)
                if episode["episode_id"] in seen:
                    raise ValueError(f"duplicate episode {episode['episode_id']}")
                seen.add(episode["episode_id"])
                retained_rows += 1 if episode["source"] == "swissmetro" else len(episode["alternatives"])
                candidate_rows += len(episode["alternatives"])
                if len(episode["alternatives"]) == 1:
                    diagnostics["singleton_groups"] += 1
                for candidate in episode["alternatives"]:
                    for name, value in candidate["features"].items():
                        if value is None:
                            missing[name] += 1
                if episode["task_type"] == "observed_choice":
                    for event in ("click_bool", "booking_bool"):
                        positives = sum(int(v[event]) for v in episode["supervision"]["outcomes_by_id"].values())
                        if positives == 0:
                            diagnostics[f"zero_{event}_groups"] += 1
                        elif positives > 1:
                            diagnostics[f"multiple_{event}_groups"] += 1
                split_group = episode["supervision"].get("respondent_id", episode["group_id"]) if episode["source"] == "swissmetro" else episode["group_id"]
                official = episode["provenance"].get("official_split")
                split = assign_split(episode["source"], str(split_group), policy, official)
                handles[split].write(json.dumps(episode, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
                counts[split] += 1
                task_counts[episode["task_type"]] += 1
        finally:
            for handle in handles.values():
                handle.close()
        profile = policy.get("source_profile")
        if profile is None:
            raise ValueError("source adapter did not provide raw-row accounting")
        omitted = profile["omitted_by_reason"]
        if profile["raw_rows"] != profile["selected_rows"] + sum(omitted.values()):
            raise ValueError("source raw rows do not reconcile to selected plus omitted")
        if profile["selected_rows"] != retained_rows:
            raise ValueError("selected source rows do not reconcile to retained candidate rows")
        profile = {**profile, "retained_rows": retained_rows, "retained_groups": len(seen), "candidate_rows": candidate_rows}
        manifest = {
            "manifest_version": "ordivo-grouped-v1",
            "split_policy_id": policy["split_policy_id"],
            "source_sha256": source_hashes,
            "source_terms": policy.get("source_terms"),
            "target_policy_id": policy.get("target_policy_id"),
            "candidate_pool_scope": policy.get("candidate_pool_scope", "source_exposed_candidates"),
            "judgment_policy": policy.get("judgment_policy"),
            "source_profile": profile,
            "episode_counts": {split: counts[split] for split in handles},
            "task_counts": dict(sorted(task_counts.items())),
            "episode_count": len(seen),
            "diagnostics": dict(sorted(diagnostics.items())),
            "missing_candidate_feature_counts": dict(sorted(missing.items())),
            "split_sha256": {split: file_sha256(stage / f"{split}.jsonl") for split in handles},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
