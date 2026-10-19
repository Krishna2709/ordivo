"""Official SUSHI order format and respondent-disjoint control."""

import json

from liquid_mcdm.data.grouped import write_dataset
from liquid_mcdm.data import sushi


def _fixture(tmp_path):
    a, b = (tmp_path / name for name in ("sushi3a.5000.10.order", "sushi3b.5000.10.order"))
    items, people = tmp_path / "sushi3.idata", tmp_path / "sushi3.udata"
    a.write_text("10 1\n0 10 9 8 7 6 5 4 3 2 1 0\n0 10 0 1 2 3 4 5 6 7 8 9\n")
    b.write_text("100 1\n0 10 99 40 30 20 10 9 8 7 6 5\n0 10 5 6 7 8 9 10 20 30 40 99\n")
    items.write_text("".join("\t".join([str(i), f"item{i}", "0", "1", "2", "0.5", "0.2", "0.3", "0.4"]) + "\n" for i in range(100)))
    people.write_text("101\t" + "\t".join(["0"] * 10) + "\n999\t" + "\t".join(["1"] * 10) + "\n")
    return a, b, items, people


def _policy(paths):
    return {"source": sushi.SOURCE, "split_policy_id": "sushi3-2016-respondent-v1-calibration",
            "target_policy_id": "sushi3-2016-order-v1", "source_files": dict(zip(("order_a", "order_b", "items", "respondents"), paths)),
            "candidate_pool_scope": "original_ten_item_questionnaire_sets", "judgment_policy": "complete_strict_order_of_exposed_ten"}


def test_sushi_preserves_original_sets_without_rank_order_leakage(tmp_path):
    paths = _fixture(tmp_path)
    policy = _policy(paths)
    episodes = list(sushi.iter_episodes(*paths, policy))
    assert len(episodes) == 4
    assert {episode["supervision"]["respondent_id"] for episode in episodes} == {"101", "999"}
    assert episodes[0]["supervision"]["ordered_groups"][0] == ["9"]
    assert [candidate["id"] for candidate in episodes[0]["alternatives"]] == [str(i) for i in range(10)]
    assert episodes[1]["supervision"]["ordered_groups"][0] == ["99"]
    assert {candidate["id"] for candidate in episodes[1]["alternatives"]} == {"99", "40", "30", "20", "10", "9", "8", "7", "6", "5"}
    assert episodes[0]["alternatives"][0]["features"]["item_name"] == "item0"
    output = tmp_path / "dataset"
    manifest = write_dataset(sushi.iter_episodes(*paths, policy), output, policy)
    assert manifest["source_profile"]["raw_rows"] == 4
    assert manifest["source_profile"]["retained_rows"] == 4
    assert manifest["source_profile"]["candidate_rows"] == 40
    assert sum(manifest["episode_counts"].values()) == 4
    respondents_by_split = {}
    for split in ("train", "validation", "calibration", "test"):
        lines = (output / f"{split}.jsonl").read_text().splitlines()
        assert len(lines) in (0, 2, 4)
        for line in lines:
            respondent = json.loads(line)["supervision"]["respondent_id"]
            assert respondent not in respondents_by_split or respondents_by_split[respondent] == split
            respondents_by_split[respondent] = split
    assert set(respondents_by_split) == {"101", "999"}


def test_sushi_rejects_malformed_original_candidate_set(tmp_path):
    paths = _fixture(tmp_path)
    a = paths[0]
    a.write_text(a.read_text().replace("9 8 7 6 5 4 3 2 1 0", "9 9 7 6 5 4 3 2 1 0"))
    policy = _policy(paths)
    try:
        list(sushi.iter_episodes(*paths, policy))
    except ValueError as exc:
        assert "candidate set" in str(exc)
    else:
        raise AssertionError("duplicate ranked item was accepted")
