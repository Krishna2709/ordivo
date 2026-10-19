"""Remote-only official-source acquisition with streamed files and checksums.

Run on an approved remote workspace; raw files must stay out of Git and public
artifacts. Expedia requires the account holder's Kaggle access configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ESCI_REPO = "https://github.com/amazon-science/esci-data.git"
BIOGEME_REPO = "https://github.com/michelbierlaire/biogeme.git"
ESCI_FILES = ("shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
              "shopping_queries_dataset/shopping_queries_dataset_products.parquet")
SWISSMETRO_FILE = "src/biogeme/data/data/swissmetro.dat"
SUSHI_URL = "https://www.kamishima.net/asset/sushi3-2016.zip"
SUSHI_MEMBERS = ("sushi3a.5000.10.order", "sushi3b.5000.10.order", "sushi3.idata", "sushi3.udata")


def _run(*args: str, cwd: Path | None = None, env: dict | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _head(repo: str) -> str:
    lines = _run("git", "ls-remote", repo, "HEAD").splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"could not resolve unique HEAD for {repo}")
    return lines[0].split()[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(output: Path, source: str, revision: str, files: list[Path], url: str, rights: str) -> dict:
    manifest = {"source": source, "source_url": url, "revision": revision, "rights_reference": rights,
                "files": {str(path.relative_to(output)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                          for path in files}}
    (output / "acquisition.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def acquire_esci(output: Path) -> dict:
    revision = _head(ESCI_REPO)
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    _run("git", "clone", "--filter=blob:none", "--depth=1", "--no-checkout", ESCI_REPO, str(output / "repo"), env=env)
    repo = output / "repo"
    _run("git", "checkout", revision, "--", *ESCI_FILES, cwd=repo, env=env)
    _run("git", "lfs", "pull", "--include=" + ",".join(ESCI_FILES), "--exclude=", cwd=repo)
    files = [repo / name for name in ESCI_FILES]
    if any(not path.exists() or path.stat().st_size < 1024 for path in files):
        raise RuntimeError("ESCI data missing or LFS pointer not resolved")
    return _record(output, "amazon_esci", revision, files, ESCI_REPO,
                   "https://github.com/amazon-science/esci-data/blob/main/LICENSE (Apache-2.0); retain NOTICE")


def acquire_swissmetro(output: Path) -> dict:
    revision = _head(BIOGEME_REPO)
    output.mkdir(parents=True, exist_ok=False)
    file = output / "swissmetro.dat"
    url = f"https://raw.githubusercontent.com/michelbierlaire/biogeme/{revision}/{SWISSMETRO_FILE}"
    with urllib.request.urlopen(url, timeout=90) as response, file.open("wb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(block)
    if file.stat().st_size < 1024:
        raise RuntimeError("Swissmetro download too small")
    return _record(output, "swissmetro", revision, [file], url,
                   "https://biogeme.epfl.ch/ (research/education data); verify dataset-specific redistribution terms")


def acquire_sushi(output: Path) -> dict:
    """Fetch only the author's corrected archive; keep it private on remote storage."""
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "sushi3-2016.zip"
    with urllib.request.urlopen(SUSHI_URL, timeout=90) as response, archive.open("wb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(block)
    files = []
    with zipfile.ZipFile(archive) as zipped:
        for basename in SUSHI_MEMBERS:
            member = f"sushi3-2016/{basename}"
            if member not in zipped.namelist():
                raise RuntimeError(f"official SUSHI archive lacks {member}")
            destination = output / basename
            with zipped.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            files.append(destination)
    receipt = _record(output, "sushi3_2016", "2016-corrected-archive", [archive, *files],
                      SUSHI_URL, "https://www.kamishima.net/sushi/ (research use; redistribution prohibited)")
    receipt["acquisition_method"] = "official_author_site"
    (output / "acquisition.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def acquire_expedia(output: Path, existing_archive: Path | None = None, expected_archive_sha256: str | None = None) -> dict:
    """Stream the labeled member of the official Deflate64 competition archive."""
    if existing_archive is None:
        if not os.environ.get("KAGGLE_API_TOKEN"):
            raise RuntimeError("KAGGLE_API_TOKEN is required for official Expedia retrieval")
        if shutil.which("kaggle") is None:
            raise RuntimeError("official Kaggle CLI is unavailable")
        output.mkdir(parents=True, exist_ok=False)
    else:
        existing_archive = Path(existing_archive)
        if not existing_archive.is_file() or existing_archive.parent.resolve() != output.resolve():
            raise ValueError("existing archive must be data.zip inside output directory")
        if (output / "acquisition.json").exists():
            raise FileExistsError("Expedia acquisition already completed")
    # The competition lists data.zip containing train.csv and unlabeled test.csv.
    # The CLI enforces existing account access/rules; this command never joins
    # the competition or accepts new terms.
    if existing_archive is None:
        _run("kaggle", "competitions", "download", "expedia-personalized-sort", "-f", "data.zip", "-p", str(output))
    archive = existing_archive or output / "data.zip"
    if not archive.exists():
        raise RuntimeError("official Kaggle download did not produce data.zip")
    archive_sha256 = _sha256(archive)
    if expected_archive_sha256 and archive_sha256 != expected_archive_sha256:
        raise ValueError("existing Expedia archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as zipped:
        matches = [name for name in zipped.namelist() if Path(name).name == "train.csv"]
        if len(matches) != 1:
            raise RuntimeError("official Expedia archive lacks a unique train.csv")
        expected_bytes = zipped.getinfo(matches[0]).file_size
        if zipped.getinfo(matches[0]).compress_type != 9:
            raise RuntimeError("unexpected Expedia compression method; inspect archive before extraction")
        if shutil.which("7z") is None:
            raise RuntimeError("7z with Deflate64 support is required on the remote runtime")
        train = output / "train.csv"
        with train.open("wb") as destination:
            result = subprocess.run(("7z", "x", "-so", str(archive), matches[0]), stdout=destination,
                                    stderr=subprocess.PIPE, check=False)
        if result.returncode:
            train.unlink(missing_ok=True)
            raise RuntimeError(f"7z failed to extract Expedia train.csv (exit {result.returncode})")
        if train.stat().st_size != expected_bytes:
            train.unlink(missing_ok=True)
            raise RuntimeError("extracted Expedia train.csv size mismatch")
    receipt = _record(output, "expedia_personalized_sort_2013", "kaggle-competition-data.zip", [train],
                      "https://www.kaggle.com/c/expedia-personalized-sort/data",
                      "https://www.kaggle.com/c/expedia-personalized-sort/rules (competition terms; no redistribution assumed)")
    receipt["download_archive_sha256"] = archive_sha256
    receipt["download_archive_bytes"] = archive.stat().st_size
    tool_probe = subprocess.run(("7z",), text=True, capture_output=True, check=False)
    receipt["extraction_tool_version"] = next((line.strip() for line in tool_probe.stdout.splitlines() if "7-Zip" in line), "7z version unavailable")
    receipt["acquisition_method"] = "official_kaggle_cli" if existing_archive is None else "verified_preexisting_archive"
    (output / "acquisition.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=("esci", "swissmetro", "expedia", "sushi", "build-esci", "build-swissmetro", "build-expedia", "build-sushi"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="remote raw source file, or ESCI acquisition repo directory")
    parser.add_argument("--existing-archive", type=Path, help="already downloaded data.zip in --output; no download")
    parser.add_argument("--expected-archive-sha256", help="required hash comparison for a preexisting archive")
    args = parser.parse_args()
    if args.source.startswith("build-"):
        if args.input is None:
            parser.error("--input required for build")
        from liquid_mcdm.data.grouped import write_dataset
        from liquid_mcdm.data import esci, expedia, swissmetro, sushi
        kind = args.source.removeprefix("build-")
        if kind == "esci":
            root = args.input
            examples, products = (root / file for file in ESCI_FILES)
            policy = {"source": "amazon_esci", "split_policy_id": "esci-official-train-test-v2-calibration", "target_policy_id": "esci-task1-official-gain-v1",
                      "candidate_pool_scope": "source_task1_reduced_candidate_list", "judgment_policy": "explicit_judged_and_unjudged_partition",
                      "source_terms": "https://github.com/amazon-science/esci-data/blob/main/LICENSE",
                      "source_files": {"examples": examples, "products": products}}
            episodes = esci.iter_episodes(examples, products, policy)
        elif kind == "swissmetro":
            policy = {"source": "swissmetro", "split_policy_id": "swissmetro-respondent-v2-calibration", "target_policy_id": "swissmetro-choice-v1",
                      "candidate_pool_scope": "three_survey_modes", "judgment_policy": "one_stated_available_choice_per_survey_scenario",
                      "source_terms": "https://biogeme.epfl.ch/", "source_files": {"survey": args.input}}
            episodes = swissmetro.iter_episodes(args.input, policy)
        elif kind == "sushi":
            root = args.input
            files = [root / name for name in SUSHI_MEMBERS]
            policy = {"source": "sushi3_2016", "split_policy_id": "sushi3-2016-respondent-v1-calibration", "target_policy_id": "sushi3-2016-order-v1",
                      "candidate_pool_scope": "original_ten_item_questionnaire_sets", "judgment_policy": "complete_strict_order_of_exposed_ten",
                      "source_terms": "https://www.kamishima.net/sushi/ (research use; redistribution prohibited)",
                      "source_files": dict(zip(("order_a", "order_b", "items", "respondents"), files))}
            episodes = sushi.iter_episodes(*files, policy)
        else:
            policy = {"source": "expedia_personalized_sort_2013", "split_policy_id": "expedia-2013-split-v2-calibration", "target_policy_id": "expedia-click-booking-v1",
                      "candidate_pool_scope": "source_exposed_candidates", "judgment_policy": "complete_observed_click_booking_per_exposed_candidate",
                      "source_terms": "https://www.kaggle.com/c/expedia-personalized-sort/rules", "source_files": {"training": args.input}}
            episodes = expedia.iter_episodes(args.input, policy)
        print(json.dumps(write_dataset(episodes, args.output, policy), sort_keys=True))
        return
    if args.source == "expedia" and args.existing_archive and not args.expected_archive_sha256:
        parser.error("--expected-archive-sha256 required with --existing-archive")
    manifest = (acquire_esci(args.output) if args.source == "esci" else
                acquire_swissmetro(args.output) if args.source == "swissmetro" else
                acquire_sushi(args.output) if args.source == "sushi" else
                acquire_expedia(args.output, args.existing_archive, args.expected_archive_sha256))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
