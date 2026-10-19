"""Amazon ESCI reduced ranking adapter with locale-aware disk joins."""

from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterator

from .behavioral import DecisionEpisode, SCHEMA_VERSION, validate_episode
from .grouped import file_sha256, grouped_rows

SOURCE = "amazon_esci"
PRODUCT_FIELDS = ("product_title", "product_description", "product_bullet_point", "product_brand", "product_color")


def _rows(path: Path, columns: list[str], batch_size: int) -> Iterator[dict]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("ESCI Parquet ingestion requires pyarrow on the remote runtime") from exc
        parquet = pq.ParquetFile(path)
        missing = set(columns) - set(parquet.schema.names)
        if missing:
            raise ValueError(f"missing ESCI columns: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield row
    else:
        raise ValueError("ESCI source must be CSV or Parquet")


def iter_episodes(examples: Path, products: Path, policy: dict) -> Iterator[DecisionEpisode]:
    examples, products = Path(examples), Path(products)
    batch_size = int(policy.get("batch_size", 4096))
    source_hashes = {"examples_sha256": file_sha256(examples), "products_sha256": file_sha256(products)}
    profile = policy["source_profile"] = {"raw_rows": 0, "selected_rows": 0, "omitted_by_reason": {"outside_task1_reduced": 0}}
    raw_groups: set[str] = set()
    selected_groups: set[str] = set()
    with tempfile.TemporaryDirectory(dir=policy.get("temp_dir")) as directory:
        connection = sqlite3.connect(str(Path(directory) / "products.sqlite3"))
        try:
            connection.execute("CREATE TABLE products(locale TEXT, product_id TEXT, payload TEXT, PRIMARY KEY(locale,product_id))")
            for row in _rows(products, ["product_id", "product_locale", *PRODUCT_FIELDS], batch_size):
                key = (str(row["product_locale"]), str(row["product_id"]))
                try:
                    connection.execute("INSERT INTO products VALUES (?,?,?)", (*key, json.dumps({name: row.get(name) for name in PRODUCT_FIELDS}, allow_nan=False)))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"duplicate ESCI product key {key}") from exc
            connection.commit()
            columns = ["example_id", "query", "query_id", "product_id", "product_locale", "esci_label", "small_version", "split"]
            def selected_rows() -> Iterator[dict]:
                for row in _rows(examples, columns, batch_size):
                    profile["raw_rows"] += 1
                    group_id = f"{row['product_locale']}:{row['query_id']}"
                    raw_groups.add(group_id)
                    if str(row["small_version"]) not in ("1", "True", "true"):
                        profile["omitted_by_reason"]["outside_task1_reduced"] += 1
                        continue
                    profile["selected_rows"] += 1
                    selected_groups.add(group_id)
                    row = dict(row)
                    row["_group"] = group_id
                    yield row
            with grouped_rows(selected_rows(), group_key="_group", row_key="example_id", temp_dir=policy.get("temp_dir")) as groups:
                for group_id, rows in groups:
                    first = rows[0]
                    query, official = first["query"], first["split"]
                    if official not in ("train", "test"):
                        raise ValueError(f"invalid official ESCI split {official!r}")
                    alternatives = []
                    grades = {}
                    for row in rows:
                        if row["query"] != query or row["split"] != official:
                            raise ValueError(f"conflicting ESCI query or official split in {group_id}")
                        key = (str(row["product_locale"]), str(row["product_id"]))
                        result = connection.execute("SELECT payload FROM products WHERE locale=? AND product_id=?", key).fetchone()
                        if result is None:
                            raise ValueError(f"missing ESCI product {key}")
                        candidate_id = str(row["product_id"])
                        if candidate_id in grades:
                            raise ValueError(f"duplicate ESCI candidate in {group_id}: {candidate_id}")
                        label = row["esci_label"]
                        if label not in ("E", "S", "C", "I"):
                            raise ValueError(f"invalid ESCI category {label!r}")
                        alternatives.append({"id": candidate_id, "features": json.loads(result[0]), "available": True})
                        grades[candidate_id] = label
                    episode: DecisionEpisode = {
                        "schema_version": SCHEMA_VERSION, "episode_id": f"{SOURCE}:{group_id}", "source": SOURCE,
                        "task_type": "graded_relevance", "group_id": group_id,
                        "context": {"query": query, "locale": str(first["product_locale"])},
                        "criteria": [], "constraints": [], "preferences": {}, "alternatives": alternatives,
                        "supervision": {"kind": "graded_relevance", "policy_id": "esci-task1-official-gain-v1", "grades_by_id": grades, "unjudged_ids": []},
                        "exposure": {}, "provenance": {**source_hashes, "official_split": official, "adapter_version": "esci-v1"},
                    }
                    validate_episode(episode)
                    yield episode
                profile["raw_groups"] = len(raw_groups)
                profile["selected_groups"] = len(selected_groups)
                profile["omitted_groups_by_reason"] = {"outside_task1_reduced": len(raw_groups - selected_groups)}
        finally:
            connection.close()
