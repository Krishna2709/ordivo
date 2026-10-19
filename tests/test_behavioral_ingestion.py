import csv
import hashlib
import json

import pytest

from liquid_mcdm.data import esci, expedia
from liquid_mcdm.data.grouped import assign_split, write_dataset


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def expedia_row(search, hotel, click=0, booking=0):
    return {"srch_id": search, "prop_id": hotel, "click_bool": str(click), "booking_bool": str(booking),
            "random_bool": "1", "position": "2", "srch_length_of_stay": "3", "srch_booking_window": "2",
            "srch_adults_count": "2", "srch_children_count": "0", "srch_room_count": "1",
            "srch_saturday_night_bool": "0", "price_usd": "100", "prop_starrating": "4",
            "prop_review_score": "0", "prop_location_score1": "2.5", "prop_location_score2": ""}


def test_expedia_complete_groups_shuffle_and_split(tmp_path):
    rows = [expedia_row("2", "b", 1), expedia_row("1", "a"), expedia_row("2", "a", booking=1), expedia_row("1", "b")]
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    write_csv(first, rows)
    write_csv(second, list(reversed(rows)))
    policy = {"temp_dir": tmp_path, "split_policy_id": "expedia-2013-split-v2-calibration", "source_files": {"train": first}, "target_policy_id": "expedia-click-booking-v1"}
    episodes1 = list(expedia.iter_episodes(first, policy))
    episodes2 = list(expedia.iter_episodes(second, policy))
    for episode in episodes1 + episodes2:
        episode["provenance"].pop("raw_file_sha256")
    assert episodes1 == episodes2
    assert [len(e["alternatives"]) for e in episodes1] == [2, 2]
    output = tmp_path / "dataset"
    manifest = write_dataset(expedia.iter_episodes(first, policy), output, policy)
    assert manifest["episode_count"] == 2
    assert manifest["source_sha256"]["train"] == hashlib.sha256(first.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write_dataset([], output, policy)


def test_expedia_rejects_duplicate_and_conflicting_context(tmp_path):
    path = tmp_path / "expedia.csv"
    rows = [expedia_row("1", "a"), expedia_row("1", "a")]
    write_csv(path, rows)
    with pytest.raises(ValueError, match="duplicate"):
        list(expedia.iter_episodes(path, {"temp_dir": tmp_path}))
    rows[1]["prop_id"] = "b"
    rows[1]["srch_length_of_stay"] = "4"
    write_csv(path, rows)
    with pytest.raises(ValueError, match="conflicting"):
        list(expedia.iter_episodes(path, {"temp_dir": tmp_path}))


def test_esci_locale_join_and_official_test(tmp_path):
    products = tmp_path / "products.csv"
    examples = tmp_path / "examples.csv"
    write_csv(products, [{"product_id": "same", "product_locale": locale, "product_title": title,
                          "product_description": "", "product_bullet_point": "", "product_brand": "", "product_color": ""}
                         for locale, title in (("us", "shoe"), ("es", "zapato"))])
    rows = [{"example_id": i, "query": q, "query_id": "q1", "product_id": "same", "product_locale": locale,
             "esci_label": label, "small_version": "1", "split": "test"}
            for i, q, locale, label in (("1", "shoe", "us", "E"), ("2", "zapato", "es", "C"))]
    write_csv(examples, rows)
    policy = {"temp_dir": tmp_path, "split_policy_id": "esci-official-train-test-v2-calibration", "source_files": {"examples": examples, "products": products}, "target_policy_id": "esci-task1-official-gain-v1"}
    episodes = list(esci.iter_episodes(examples, products, policy))
    assert len(episodes) == 2
    assert {e["alternatives"][0]["features"]["product_title"] for e in episodes} == {"shoe", "zapato"}
    assert all(assign_split(e["source"], e["group_id"], policy, e["provenance"]["official_split"]) == "test" for e in episodes)
    rows[0]["product_id"] = "missing"
    write_csv(examples, rows)
    with pytest.raises(ValueError, match="missing ESCI product"):
        list(esci.iter_episodes(examples, products, policy))


def test_esci_train_only_validation_and_calibration():
    policy = {"split_policy_id": "esci-official-train-test-v2-calibration"}
    assert all(assign_split("amazon_esci", f"us:q{i}", policy, "test") == "test" for i in range(100))
    derived = {assign_split("amazon_esci", f"us:q{i}", policy, "train") for i in range(2000)}
    assert derived == {"train", "validation", "calibration"}
