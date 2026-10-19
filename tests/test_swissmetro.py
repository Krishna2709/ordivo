import csv

import pytest

from liquid_mcdm.data.grouped import assign_split, write_dataset
from liquid_mcdm.data.swissmetro import iter_episodes


def row(person, choice=1, availability="1"):
    return {"ID": person, "CHOICE": str(choice), "PURPOSE": "1", "GA": "0", "WHO": "1", "LUGGAGE": "0",
            "TRAIN_AV": availability, "TRAIN_TT": "120", "TRAIN_CO": "40", "TRAIN_HE": "30",
            "SM_AV": "1", "SM_TT": "80", "SM_CO": "60", "SM_HE": "20",
            "CAR_AV": "1", "CAR_TT": "140", "CAR_CO": "50"}


def test_respondent_disjoint_and_availability(tmp_path):
    path = tmp_path / "swissmetro.dat"
    rows = [row("person-1"), row("person-1", choice=2), row("person-2", choice=3), row("person-3", choice=0)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    episodes = list(iter_episodes(path, {}))
    assert len(episodes) == 3
    policy = {"split_policy_id": "swissmetro-respondent-v2-calibration"}
    splits = [assign_split(e["source"], e["supervision"]["respondent_id"], policy) for e in episodes]
    assert splits[0] == splits[1]
    dataset_policy = {"source": "swissmetro", "split_policy_id": "swissmetro-respondent-v2-calibration", "source_files": {"survey": path}}
    manifest = write_dataset(iter_episodes(path, dataset_policy), tmp_path / "processed", dataset_policy)
    assert manifest["source_profile"] == {"raw_rows": 4, "selected_rows": 3, "omitted_by_reason": {"no_choice": 1}, "raw_groups": 4, "selected_groups": 3, "omitted_groups_by_reason": {"no_choice": 1}, "retained_rows": 3, "retained_groups": 3, "candidate_rows": 9}
    rows[0]["TRAIN_AV"] = "0"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="available"):
        list(iter_episodes(path, {}))
