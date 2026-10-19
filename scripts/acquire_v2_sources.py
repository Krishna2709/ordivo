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


def acquire_expedia(output: Path) -> dict:
    """Use the official Kaggle CLI with a scoped bearer token; never accept terms."""
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise RuntimeError("KAGGLE_API_TOKEN is required for official Expedia retrieval")
    if shutil.which("kaggle") is None:
        raise RuntimeError("official Kaggle CLI is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    # The competition lists data.zip containing train.csv and unlabeled test.csv.
    # The CLI enforces existing account access/rules; this command never joins
    # the competition or accepts new terms.
    _run("kaggle", "competitions", "download", "expedia-personalized-sort", "-f", "data.zip", "-p", str(output))
    archive = output / "data.zip"
    if not archive.exists():
        raise RuntimeError("official Kaggle download did not produce data.zip")
    archive_sha256 = _sha256(archive)
    with zipfile.ZipFile(archive) as zipped:
        matches = [name for name in zipped.namelist() if Path(name).name == "train.csv"]
        if len(matches) != 1:
            raise RuntimeError("official Expedia archive lacks a unique train.csv")
        train = output / "train.csv"
        with zipped.open(matches[0]) as source, train.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    archive.unlink()
    receipt = _record(output, "expedia_personalized_sort_2013", "kaggle-competition-data.zip", [train],
                      "https://www.kaggle.com/c/expedia-personalized-sort/data",
                      "https://www.kaggle.com/c/expedia-personalized-sort/rules (competition terms; no redistribution assumed)")
    receipt["download_archive_sha256"] = archive_sha256
    (output / "acquisition.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=("esci", "swissmetro", "expedia", "build-esci", "build-swissmetro", "build-expedia"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="remote raw source file, or ESCI acquisition repo directory")
    args = parser.parse_args()
    if args.source.startswith("build-"):
        if args.input is None:
            parser.error("--input required for build")
        from liquid_mcdm.data.grouped import write_dataset
        from liquid_mcdm.data import esci, expedia, swissmetro
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
        else:
            policy = {"source": "expedia_personalized_sort_2013", "split_policy_id": "expedia-2013-split-v2-calibration", "target_policy_id": "expedia-click-booking-v1",
                      "candidate_pool_scope": "source_exposed_candidates", "judgment_policy": "complete_observed_click_booking_per_exposed_candidate",
                      "source_terms": "https://www.kaggle.com/c/expedia-personalized-sort/rules", "source_files": {"training": args.input}}
            episodes = expedia.iter_episodes(args.input, policy)
        print(json.dumps(write_dataset(episodes, args.output, policy), sort_keys=True))
        return
    manifest = acquire_esci(args.output) if args.source == "esci" else acquire_swissmetro(args.output) if args.source == "swissmetro" else acquire_expedia(args.output)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
