"""Remote-only streamed, label-independent complete-group sample of a source dataset."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
from pathlib import Path

import yaml

from liquid_mcdm.runtime import require_remote_gpu


SPLITS = ("train", "validation", "calibration", "test")
POLICY = "sha256_smallest_source_split_episode_id_v1"
SOURCE = "expedia_personalized_sort_2013"
FROZEN_CAPS = {"train": 20000, "validation": 2000, "calibration": 2000, "test": 5000}
FROZEN_PARENT_SHA256 = "3b14e3f30466a2da868ac636c8888bc42366f865737715a57a67f32866a02d92"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(seed: int, source: str, split: str, episode_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{source}|{split}|{episode_id}".encode()).digest()


def _require_parent_binding(config: dict, parent_bytes: bytes) -> str:
    expected = config.get("expected_parent_manifest_sha256")
    actual = hashlib.sha256(parent_bytes).hexdigest()
    if expected != FROZEN_PARENT_SHA256 or actual != expected:
        raise ValueError("full source manifest differs from predeclared audited parent hash")
    return actual


def _select_split(path: Path, *, source: str, split: str, seed: int, cap: int,
                  expected_sha256: str, seen_groups: dict[str, str]) -> tuple[list[tuple[bytes, str, bytes]], dict]:
    """Keep K smallest identity hashes, preserving every selected JSONL record byte-for-byte."""
    digest = hashlib.sha256()
    heap: list[tuple[int, str, bytes, bytes]] = []
    count = candidate_count = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row["source"] != source:
                raise ValueError(f"source differs in {split}")
            episode_id, group_id = row["episode_id"], row["group_id"]
            if episode_id != f"{source}:{group_id}":
                raise ValueError("episode and group IDs are not source-bound")
            if group_id in seen_groups:
                raise ValueError(f"duplicate group or split overlap: {group_id}")
            seen_groups[group_id] = split
            count += 1
            candidate_count += len(row["alternatives"])
            key = _key(seed, source, split, episode_id)
            entry = (-int.from_bytes(key, "big"), episode_id, key, raw_line)
            if len(heap) < cap:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"full {split} file hash differs from parent manifest")
    selected = sorted(((entry[2], entry[1], entry[3]) for entry in heap),
                      key=lambda entry: (entry[0], entry[1]))
    selected_candidates = sum(len(json.loads(line)["alternatives"]) for _, _, line in selected)
    return selected, {"full_groups": count, "full_candidates": candidate_count,
                      "selected_groups": len(selected), "selected_candidates": selected_candidates,
                      "excluded_groups": count - len(selected),
                      "excluded_candidates": candidate_count - selected_candidates}


def sample(config_path: Path, data_dir: Path, output: Path) -> dict:
    require_remote_gpu()
    commit = os.environ.get("ORDIVO_SOURCE_COMMIT", "")
    archive_sha256 = os.environ.get("ORDIVO_SOURCE_SHA256", "")
    if (not re.fullmatch(r"[0-9a-f]{40}", commit) or
            not re.fullmatch(r"[0-9a-f]{64}", archive_sha256)):
        raise ValueError("controller-verified source commit and archive SHA-256 required")
    output = Path(output).resolve()
    if Path("/workspace") not in output.parents or output.exists():
        raise ValueError("new /workspace output directory required")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    caps = config.get("group_caps")
    if (config.get("source") != SOURCE or config.get("selection_policy") != POLICY or
            config.get("seed") != 42 or type(config.get("seed")) is not int or
            caps != FROZEN_CAPS):
        raise ValueError("invalid predeclared source, policy, seed or four-way caps")
    parent_path = data_dir / "manifest.json"
    parent_bytes = parent_path.read_bytes()
    parent_sha256 = _require_parent_binding(config, parent_bytes)
    parent = json.loads(parent_bytes)
    if (parent.get("source") != SOURCE or
            parent.get("candidate_pool_scope") != "source_exposed_candidates" or
            parent.get("target_policy_id") != "expedia-click-booking-v1" or
            parent.get("judgment_policy") != "complete_observed_click_booking_per_exposed_candidate" or
            set(parent.get("split_sha256", {})) != set(SPLITS)):
        raise ValueError("full Expedia source manifest/policy mismatch")
    seen_groups: dict[str, str] = {}
    selected_by_split, counts = {}, {}
    for split in SPLITS:
        selected, info = _select_split(data_dir / f"{split}.jsonl", source=SOURCE,
                                       split=split, seed=config["seed"], cap=caps[split],
                                       expected_sha256=parent["split_sha256"][split],
                                       seen_groups=seen_groups)
        if info["full_groups"] != parent["episode_counts"][split]:
            raise ValueError(f"full {split} episode count differs from parent manifest")
        selected_by_split[split], counts[split] = selected, info
    if (sum(info["full_groups"] for info in counts.values()) != parent["episode_count"] or
            len(seen_groups) != parent["episode_count"] or
            sum(info["full_candidates"] for info in counts.values()) !=
            parent["source_profile"]["candidate_rows"]):
        raise ValueError("full source group accounting differs from parent manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    split_hashes = {}
    selection_hashes = {}
    for split in SPLITS:
        target = output / f"{split}.jsonl"
        with target.open("wb") as handle:
            for _, _, raw_line in selected_by_split[split]:
                handle.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
        split_hashes[split] = _sha256(target)
        selection_hashes[split] = hashlib.sha256(json.dumps(
            [episode_id for _, episode_id, _ in selected_by_split[split]],
            separators=(",", ":")).encode()).hexdigest()
    sampling = {"policy": POLICY, "seed": config["seed"],
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "parent_manifest_sha256": parent_sha256,
                "parent_split_sha256": parent["split_sha256"],
                "group_caps": caps, "by_split": counts,
                "selected_episode_ids_sha256": selection_hashes,
                "complete_groups_preserved": True,
                "label_independent_selection": True}
    manifest = {"manifest_version": "ordivo-grouped-bounded-v1", "source": SOURCE,
                "task_counts": {"observed_choice": sum(info["selected_groups"] for info in counts.values())},
                "episode_count": sum(info["selected_groups"] for info in counts.values()),
                "episode_counts": {split: counts[split]["selected_groups"] for split in SPLITS},
                "split_sha256": split_hashes,
                "split_policy_id": parent["split_policy_id"] + ":bounded-hash-v1",
                "target_policy_id": parent["target_policy_id"],
                "candidate_pool_scope": parent["candidate_pool_scope"],
                "judgment_policy": parent.get("judgment_policy"),
                "source_sha256": parent["source_sha256"],
                "source_terms": parent.get("source_terms"), "sampling": sampling}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {"status": "complete", "source_code_revision": commit,
               "source_archive_sha256": archive_sha256,
               "sample_manifest_sha256": _sha256(output / "manifest.json"),
               "sampling": sampling,
               "selected_split_sha256": split_hashes,
               "source": SOURCE}
    (output / "sampling_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sample(args.config, args.data, args.output)
    print(json.dumps({"status": report["status"], "source": report["source"],
                      "sample_manifest_sha256": report["sample_manifest_sha256"],
                      "counts": report["sampling"]["by_split"]}, sort_keys=True))


if __name__ == "__main__":
    main()
