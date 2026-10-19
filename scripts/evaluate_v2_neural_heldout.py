"""Remote-only base d1 versus frozen pilot adapter on identical held-out groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import yaml

from liquid_mcdm.data.behavioral import validate_episode
from liquid_mcdm.evaluation.neural_heldout import (
    eligibility, evaluate_neural_pair, select_whole_test_units, selected_units_sha256, split_unit,
)
from liquid_mcdm.runtime import require_remote_gpu
from liquid_mcdm.training.behavioral_rendering import render_behavioral
from liquid_mcdm.training.behavioral_rendering import PROMPT_VERSION
from liquid_mcdm.training.option_policy import option_logits


MODEL = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
SPLITS = ("train", "validation", "calibration", "test")


def _controller_identity() -> dict:
    commit = os.environ.get("ORDIVO_SOURCE_COMMIT", "")
    digest = os.environ.get("ORDIVO_SOURCE_SHA256", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("controller-verified source commit and archive SHA-256 are required")
    return {"commit": commit, "archive_sha256": digest}


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _require_eligible_metrics(report: dict) -> None:
    if not any(metric["eligible"] > 0 for metric in report["metrics"].values()):
        raise ValueError("no eligible paired metric in source")


def _source(source_spec: dict, expected_manifest_sha: str, *, seed: int, limit: int,
            max_episodes: int, deadline: float | None = None,
            partition: str = "test", max_units: int = 50) -> tuple[list[dict], dict]:
    if partition not in SPLITS or partition in ("train", "calibration"):
        raise ValueError("paired evaluation partition must be validation or test")
    if type(limit) is not int or not 1 <= limit <= max_units:
        raise ValueError("held-out selection limit must be 1..50 whole split units")
    directory = Path(source_spec["directory"])
    manifest_bytes = (directory / "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != expected_manifest_sha:
        raise ValueError("source manifest changed since pilot selection")
    manifest = json.loads(manifest_bytes)
    if set(manifest.get("split_sha256", {})) != set(SPLITS):
        raise ValueError("four-way hashed split manifest required")
    seen_episodes, units_by_split, counts = set(), {}, {}
    source_name = None
    for split in SPLITS:
        units = set()
        file_digest = hashlib.sha256()
        count = 0
        with (directory / f"{split}.jsonl").open("rb") as handle:
            for raw_line in handle:
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("held-out source integrity scan time cap reached")
                file_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                episode = json.loads(raw_line)
                if source_name is None:
                    source_name = episode["source"]
                elif source_name != episode["source"]:
                    raise ValueError("mixed source in one manifest")
                episode_id = episode["episode_id"]
                if episode_id in seen_episodes:
                    raise ValueError("duplicate episode across frozen splits")
                seen_episodes.add(episode_id)
                units.add(split_unit(episode))
                count += 1
        if file_digest.hexdigest() != manifest["split_sha256"][split]:
            raise ValueError(f"{split} file hash differs from manifest")
        units_by_split[split], counts[split] = units, count
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if units_by_split[left] & units_by_split[right]:
                raise ValueError(f"split unit overlap: {left}/{right}")
    selected_units = sorted(units_by_split[partition], key=lambda unit: (
        hashlib.sha256(f"{seed}|{source_name}|{partition}|{unit}".encode()).digest(), unit))[:limit]
    chosen = set(selected_units)
    rows = []
    with (directory / f"{partition}.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("held-out selected-group load time cap reached")
            if line.strip():
                episode = json.loads(line)
                if split_unit(episode) in chosen:
                    validate_episode(episode)
                    rows.append(episode)
                    if len(rows) > max_episodes:
                        raise ValueError("selected whole units exceed episode bound")
    checked_ids = {e["episode_id"] for e in rows}
    if (partition == "test"):
        checked, checked_units = select_whole_test_units(rows, source=source_name, seed=seed,
                                                         limit=limit)
        materialized = ({e["episode_id"] for e in checked} == checked_ids and
                        set(checked_units) == chosen)
    else:
        materialized = {split_unit(e) for e in rows} == chosen
    if not materialized or len(checked_ids) != len(rows):
        raise ValueError("selected unit materialization changed")
    return rows, {"source": source_name, "manifest_sha256": digest,
                  "selected_partition": partition,
                  "source_sha256": manifest.get("source_sha256", {}),
                  "split_sha256": manifest["split_sha256"],
                  "complete_episode_counts": counts,
                  "complete_test_units": len(units_by_split["test"]),
                  "complete_partition_units": len(units_by_split[partition]),
                  "selected_units": len(selected_units),
                  "selected_units_sha256": selected_units_sha256(selected_units)}


def _score_model(model, tokenizer, rendered: list[tuple[dict, dict]], deadline: float,
                 progress: dict[str, int], name: str) -> dict:
    import torch

    model.eval()
    predictions = {}
    with torch.no_grad():
        for episode, row in rendered:
            if time.monotonic() > deadline:
                raise TimeoutError("held-out inference time cap reached")
            input_ids = tokenizer(row["prompt"], add_special_tokens=False,
                                  return_tensors="pt").to("cuda")
            answer = model(**input_ids, logits_to_keep=1).logits[0, -1]
            mask = torch.tensor(row["eligible"], dtype=torch.bool, device=answer.device)
            scores = option_logits(answer, row["readout_groups"], mask).float()
            probabilities = torch.softmax(scores, dim=-1).detach().cpu().tolist()
            predictions[episode["episode_id"]] = {
                candidate_id: float(probabilities[i]) for i, candidate_id in enumerate(row["candidate_ids"])
                if row["eligible"][i]}
            progress[name] = len(predictions)
    return predictions


def run(pilot_config_path: Path, pilot_output: Path, eval_config_path: Path, output: Path) -> dict:
    require_remote_gpu()
    identity = _controller_identity()
    output = Path(output).resolve()
    if Path("/workspace") not in output.parents or output.exists():
        raise ValueError("new output directory under /workspace required")
    pilot_config_bytes = Path(pilot_config_path).read_bytes()
    pilot_config = yaml.safe_load(pilot_config_bytes)
    eval_config_bytes = Path(eval_config_path).read_bytes()
    eval_config = yaml.safe_load(eval_config_bytes)
    receipt = json.loads((Path(pilot_output) / "pilot_receipt.json").read_text())
    if (receipt.get("status") != "pilot_completed" or receipt.get("best_step", 0) < 1 or
            pilot_config.get("model_id") != MODEL or pilot_config.get("revision") != REVISION or
            receipt.get("model_id") != MODEL or receipt.get("revision") != REVISION):
        raise ValueError("frozen completed pilot and pinned base revision required")
    if receipt.get("config_sha256") != hashlib.sha256(pilot_config_bytes).hexdigest():
        raise ValueError("pilot config changed after training")
    if (receipt.get("prompt_version") != PROMPT_VERSION or
            receipt.get("effective_hyperparameters", {}).get("max_length") != pilot_config.get("max_length")):
        raise ValueError("pilot prompt version or maximum length changed")
    if not (Path(pilot_output) / "best_adapter" / "adapter_config.json").is_file():
        raise ValueError("best pilot adapter checkpoint missing")
    if (type(eval_config.get("seed")) is not int or
            not 1 <= eval_config.get("max_runtime_seconds", 0) <= 7200 or
            not 1 <= eval_config.get("max_selected_episodes_per_source", 0) <= 500):
        raise ValueError("invalid bounded held-out evaluation config")
    if eval_config["seed"] != pilot_config["seed"]:
        raise ValueError("held-out selection seed differs from frozen pilot seed")
    specs = pilot_config["sources"]
    if set(eval_config["max_test_units_per_source"]) != {s["name"] for s in specs}:
        raise ValueError("held-out source limits must match pilot domains")
    started = time.monotonic()
    deadline = started + eval_config["max_runtime_seconds"]
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    import torch
    from peft import PeftModel

    all_rendered, source_meta, excluded = {}, {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        if time.monotonic() > deadline:
            raise TimeoutError("held-out tokenizer load time cap reached")
        for spec in specs:
            name = spec["name"]
            provenance = receipt["source_provenance"][name]
            if provenance["feature_policy"] != spec["feature_policy"]:
                raise ValueError("feature policy differs from frozen pilot")
            limit = eval_config["max_test_units_per_source"][name]
            rows, meta = _source(spec, provenance["manifest_sha256"], seed=eval_config["seed"],
                                 limit=limit,
                                 max_episodes=eval_config["max_selected_episodes_per_source"],
                                 deadline=deadline)
            source_meta[name] = meta
            rendered, overflow = [], set()
            for episode in rows:
                if time.monotonic() > deadline:
                    raise TimeoutError("held-out rendering time cap reached")
                population, reason = eligibility(episode, spec["feature_policy"])
                if reason:
                    continue
                try:
                    row = render_behavioral(episode, tokenizer, spec["feature_policy"],
                                            max_length=pilot_config["max_length"])
                except ValueError as exc:
                    if "exceeds max_length" not in str(exc):
                        raise
                    overflow.add(episode["episode_id"])
                    continue
                if row["candidate_ids"] != [a["id"] for a in episode["alternatives"]]:
                    raise ValueError("rendered candidate coverage differs from source")
                rendered.append((episode, row))
            all_rendered[name] = (rows, rendered)
            excluded[name] = overflow
        if any(not items[1] for items in all_rendered.values()):
            raise ValueError("no eligible rendered queries in at least one pilot domain")
    except Exception as exc:
        _write(output / "failure.json", {
            "status": "incomplete", "phase": "preparation", "failure_type": type(exc).__name__,
            "selected_episodes_by_source": {name: len(items[0]) for name, items in all_rendered.items()},
            "admitted_rendered_by_source": {name: len(items[1]) for name, items in all_rendered.items()},
            "aggregate_published": False})
        raise
    _write(output / "selection_manifest.json", {"sources": source_meta,
                                                  "evaluation_config_sha256": hashlib.sha256(eval_config_bytes).hexdigest(),
                                                  "pilot_config_sha256": receipt["config_sha256"],
                                                  "pilot_best_step": receipt["best_step"],
                                                  "pilot_source_identity": receipt["source_identity"],
                                                  "evaluation_source_identity": identity})
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_progress, adapted_progress = {}, {}
    try:
        base = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION,
                                                          dtype=dtype, device_map="auto")
        if time.monotonic() > deadline:
            raise TimeoutError("held-out base model load time cap reached")
        base_predictions = {name: _score_model(base, tokenizer, items[1], deadline,
                                               base_progress, name)
                            for name, items in all_rendered.items()}
        del base
        torch.cuda.empty_cache()
        fresh = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION,
                                                           dtype=dtype, device_map="auto")
        adapted = PeftModel.from_pretrained(fresh, Path(pilot_output) / "best_adapter")
        if time.monotonic() > deadline:
            raise TimeoutError("held-out adapter load time cap reached")
        adapted_predictions = {name: _score_model(adapted, tokenizer, items[1], deadline,
                                                  adapted_progress, name)
                               for name, items in all_rendered.items()}
        del adapted, fresh
        torch.cuda.empty_cache()
        aggregates = {}
        for name, (rows, _) in all_rendered.items():
            _write(output / f"{name}_base_probabilities.json", base_predictions[name])
            _write(output / f"{name}_adapted_probabilities.json", adapted_predictions[name])
            report = evaluate_neural_pair(rows, base_predictions[name], adapted_predictions[name],
                                          next(s["feature_policy"] for s in specs if s["name"] == name),
                                          excluded[name])
            _require_eligible_metrics(report)
            _write(output / f"{name}_per_query.json", report.pop("per_query"))
            aggregates[name] = {**report, "source": source_meta[name]["source"],
                                "manifest_sha256": source_meta[name]["manifest_sha256"],
                                "selected_units_sha256": source_meta[name]["selected_units_sha256"]}
    except Exception as exc:
        _write(output / "failure.json", {
            "status": "incomplete", "failure_type": type(exc).__name__,
            "selected_episodes_by_source": {name: len(items[0]) for name, items in all_rendered.items()},
            "admitted_rendered_by_source": {name: len(items[1]) for name, items in all_rendered.items()},
            "base_scored_by_source": base_progress,
            "adapted_scored_by_source": adapted_progress,
            "aggregate_published": False})
        raise
    aggregate = {"schema_version": "ordivo-neural-heldout-run-v1", "status": "complete",
                 "claim_scope": "exploratory_bounded_probe_not_protected_release_test",
                 "source_identity": identity, "pilot_best_step": receipt["best_step"],
                 "evaluation_config_sha256": hashlib.sha256(eval_config_bytes).hexdigest(),
                 "pilot_config_sha256": receipt["config_sha256"],
                 "base_model": MODEL, "revision": REVISION, "seed": eval_config["seed"],
                 "elapsed_seconds": time.monotonic() - started,
                 "by_source": aggregates, "paired_interpretation": "small_bounded_heldout_probe_no_calibration"}
    _write(output / "aggregate.json", aggregate)
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.pilot_config, args.pilot_output, args.config, args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output / "aggregate.json"),
                      "source_identity": report["source_identity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
