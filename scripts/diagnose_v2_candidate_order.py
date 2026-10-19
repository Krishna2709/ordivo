"""Remote validation-only candidate-order stability diagnostic for a frozen pilot adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import yaml

from liquid_mcdm.evaluation.neural_heldout import _validate_probabilities
from liquid_mcdm.evaluation.ranking import ranked_ids
from liquid_mcdm.runtime import require_remote_gpu
from liquid_mcdm.training.behavioral_rendering import PROMPT_VERSION, render_behavioral
try:
    from scripts.evaluate_v2_neural_heldout import (
        MODEL, REVISION, _controller_identity, _score_model, _source, _write,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from evaluate_v2_neural_heldout import (  # type: ignore[no-redef]
        MODEL, REVISION, _controller_identity, _score_model, _source, _write,
    )


def permute_candidates(episode: dict, seed: int) -> dict:
    """Change only alternative order, independently of labels and feature values."""
    if type(seed) is not int:
        raise ValueError("integer candidate-order seed required")
    result = copy.deepcopy(episode)
    alternatives = result["alternatives"]
    if len(alternatives) < 2:
        raise ValueError("candidate-order diagnostic requires at least two alternatives")
    original_ids = [a["id"] for a in alternatives]
    alternatives.sort(key=lambda alternative: (
        hashlib.sha256(f"{seed}|candidate-order|{episode['episode_id']}|{alternative['id']}".encode()).digest(),
        alternative["id"]))
    if [a["id"] for a in alternatives] == original_ids:
        alternatives.reverse()
    if sorted(a["id"] for a in alternatives) != sorted(original_ids):
        raise ValueError("permutation changed candidate membership")
    return result


def order_stability_report(episodes: list[dict], original: dict, permuted: dict,
                           original_overflow: set[str], permuted_overflow: set[str]) -> dict:
    """Compare probabilities by canonical candidate ID on the paired admitted cohort."""
    selected = {e["episode_id"]: e for e in episodes}
    if len(selected) != len(episodes):
        raise ValueError("duplicate selected episode")
    if not original_overflow <= set(selected) or not permuted_overflow <= set(selected):
        raise ValueError("overflow ID outside selected cohort")
    fewer_than_two = {e["episode_id"] for e in episodes
                      if sum(a.get("available", True) for a in e["alternatives"]) < 2}
    admitted = set(selected) - fewer_than_two - original_overflow - permuted_overflow
    if set(original) != admitted or set(permuted) != admitted:
        raise ValueError("incomplete paired candidate-order prediction coverage")
    if not admitted:
        raise ValueError("no paired candidate-order predictions")
    rows = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        status = ("fewer_than_two_available" if episode_id in fewer_than_two else
                  "overflow_both" if episode_id in original_overflow and episode_id in permuted_overflow else
                  "overflow_original" if episode_id in original_overflow else
                  "overflow_permuted" if episode_id in permuted_overflow else "scored")
        row = {"episode_id": episode_id, "status": status}
        if status == "scored":
            before, after = original[episode_id], permuted[episode_id]
            _validate_probabilities(episode, before)
            _validate_probabilities(episode, after)
            row.update({"original_top1_id": ranked_ids(before)[0],
                        "permuted_top1_id": ranked_ids(after)[0],
                        "top1_flipped": ranked_ids(before)[0] != ranked_ids(after)[0],
                        "l1_probability_change": sum(abs(before[i] - after[i]) for i in before),
                        "original_probabilities_by_id": before,
                        "permuted_probabilities_by_id": after})
        rows.append(row)
    scored = [r for r in rows if r["status"] == "scored"]
    return {"schema_version": "ordivo-candidate-order-v1",
            "selected_episodes": len(episodes), "admitted_pairs": len(scored),
            "failed_queries": 0,
            "excluded": {reason: sum(r["status"] == reason for r in rows)
                         for reason in ("fewer_than_two_available", "overflow_original",
                                        "overflow_permuted", "overflow_both")},
            "top1_flip_rate": sum(r["top1_flipped"] for r in scored) / len(scored),
            "mean_l1_probability_change": sum(r["l1_probability_change"] for r in scored) / len(scored),
            "per_query": rows}


def run(pilot_config_path: Path, pilot_output: Path, diagnostic_config_path: Path,
        output: Path) -> dict:
    require_remote_gpu()
    identity = _controller_identity()
    output = Path(output).resolve()
    if Path("/workspace") not in output.parents or output.exists():
        raise ValueError("new output directory under /workspace required")
    pilot_config_bytes = Path(pilot_config_path).read_bytes()
    pilot_config = yaml.safe_load(pilot_config_bytes)
    config_bytes = Path(diagnostic_config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    receipt = json.loads((Path(pilot_output) / "pilot_receipt.json").read_bytes())
    if (receipt.get("status") != "pilot_completed" or receipt.get("best_step", 0) < 1 or
            pilot_config.get("model_id") != MODEL or pilot_config.get("revision") != REVISION or
            receipt.get("model_id") != MODEL or receipt.get("revision") != REVISION or
            receipt.get("config_sha256") != hashlib.sha256(pilot_config_bytes).hexdigest() or
            receipt.get("prompt_version") != PROMPT_VERSION or
            receipt.get("effective_hyperparameters", {}).get("max_length") != pilot_config.get("max_length")):
        raise ValueError("frozen completed pilot identity/config/prompt mismatch")
    if not (Path(pilot_output) / "best_adapter" / "adapter_config.json").is_file():
        raise ValueError("best pilot adapter checkpoint missing")
    if (type(config.get("seed")) is not int or config["seed"] != pilot_config["seed"] or
            type(config.get("max_runtime_seconds")) is not int or
            not 1 <= config["max_runtime_seconds"] <= 3600 or
            type(config.get("max_selected_episodes_per_source")) is not int or
            not 1 <= config["max_selected_episodes_per_source"] <= 250):
        raise ValueError("invalid bounded validation diagnostic config")
    specs = pilot_config["sources"]
    limits = config.get("max_validation_units_per_source")
    if (not isinstance(limits, dict) or set(limits) != {s["name"] for s in specs} or
            any(type(value) is not int or not 1 <= value <= 20 for value in limits.values())):
        raise ValueError("validation source limits must match pilot domains and be at most 20")
    started = time.monotonic()
    deadline = started + config["max_runtime_seconds"]
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from peft import PeftModel
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    prepared, source_meta = {}, {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        if time.monotonic() > deadline:
            raise TimeoutError("validation diagnostic tokenizer load time cap reached")
        for spec in specs:
            name = spec["name"]
            if spec["feature_policy"] != receipt["source_provenance"][name]["feature_policy"]:
                raise ValueError("feature policy differs from frozen pilot")
            rows, meta = _source(spec, receipt["source_provenance"][name]["manifest_sha256"],
                                 seed=config["seed"], limit=limits[name],
                                 max_episodes=config["max_selected_episodes_per_source"],
                                 deadline=deadline, partition="validation", max_units=20)
            source_meta[name] = meta
            original_rendered, permuted_rendered = [], []
            original_overflow, permuted_overflow = set(), set()
            for episode in rows:
                if time.monotonic() > deadline:
                    raise TimeoutError("validation diagnostic rendering time cap reached")
                episode_id = episode["episode_id"]
                if sum(a.get("available", True) for a in episode["alternatives"]) < 2:
                    continue
                changed = permute_candidates(episode, config["seed"])
                pair = []
                for variant, overflows in ((episode, original_overflow), (changed, permuted_overflow)):
                    try:
                        rendered = render_behavioral(variant, tokenizer, spec["feature_policy"],
                                                     max_length=pilot_config["max_length"])
                    except ValueError as exc:
                        if "exceeds max_length" not in str(exc):
                            raise
                        overflows.add(episode_id)
                        pair.append(None)
                        continue
                    if rendered["candidate_ids"] != [a["id"] for a in variant["alternatives"]]:
                        raise ValueError("rendered candidate coverage differs from source")
                    pair.append(rendered)
                if all(item is not None for item in pair):
                    original_rendered.append((episode, pair[0]))
                    permuted_rendered.append((episode, pair[1]))
            prepared[name] = (rows, original_rendered, permuted_rendered,
                              original_overflow, permuted_overflow)
        if any(not value[1] for value in prepared.values()):
            raise ValueError("no paired rendered validation queries in at least one pilot domain")
    except Exception as exc:
        _write(output / "failure.json", {"status": "incomplete", "phase": "preparation",
                                          "failure_type": type(exc).__name__,
                                          "selected_episodes_by_source": {k: len(v[0]) for k, v in prepared.items()},
                                          "aggregate_published": False})
        raise
    _write(output / "selection_manifest.json", {"sources": source_meta,
                                                  "partition": "validation",
                                                  "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                                                  "pilot_config_sha256": receipt["config_sha256"],
                                                  "pilot_best_step": receipt["best_step"],
                                                  "pilot_source_identity": receipt["source_identity"],
                                                  "diagnostic_source_identity": identity})
    progress_original, progress_permuted = {}, {}
    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION,
                                                          dtype=dtype, device_map="auto")
        model = PeftModel.from_pretrained(base, Path(pilot_output) / "best_adapter")
        if time.monotonic() > deadline:
            raise TimeoutError("validation diagnostic model load time cap reached")
        reports = {}
        for name, (rows, original_rows, permuted_rows, original_overflow,
                   permuted_overflow) in prepared.items():
            original = _score_model(model, tokenizer, original_rows, deadline, progress_original, name)
            permuted = _score_model(model, tokenizer, permuted_rows, deadline, progress_permuted, name)
            report = order_stability_report(rows, original, permuted,
                                            original_overflow, permuted_overflow)
            _write(output / f"{name}_per_query.json", report.pop("per_query"))
            reports[name] = {**report, "source": source_meta[name]["source"],
                             "manifest_sha256": source_meta[name]["manifest_sha256"],
                             "selected_units": source_meta[name]["selected_units"],
                             "selected_units_sha256": source_meta[name]["selected_units_sha256"]}
        del model, base
        torch.cuda.empty_cache()
    except Exception as exc:
        _write(output / "failure.json", {"status": "incomplete", "phase": "inference",
                                          "failure_type": type(exc).__name__,
                                          "original_scored_by_source": progress_original,
                                          "permuted_scored_by_source": progress_permuted,
                                          "aggregate_published": False})
        raise
    aggregate = {"schema_version": "ordivo-candidate-order-run-v1", "status": "complete",
                 "claim_scope": "validation_only_diagnostic_no_model_selection",
                 "partition": "validation", "source_identity": identity,
                 "pilot_source_identity": receipt["source_identity"],
                 "pilot_best_step": receipt["best_step"],
                 "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                 "pilot_config_sha256": receipt["config_sha256"],
                 "prompt_version": PROMPT_VERSION,
                 "elapsed_seconds": time.monotonic() - started, "by_source": reports}
    _write(output / "aggregate.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pilot_config, args.pilot_output, args.config, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output / "aggregate.json"),
                      "source_identity": result["source_identity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
