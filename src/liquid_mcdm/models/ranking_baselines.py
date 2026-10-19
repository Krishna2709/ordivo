"""Transparent task specialists and train-fitted feature projection."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping

import numpy as np

from liquid_mcdm.data.behavioral import inference_input


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"\w+", str(value).casefold()))


def lexical_scores(episode: Mapping, feature_policy: Mapping) -> dict[str, float]:
    """ESCI lexical overlap, with all text taken from allowlisted inference fields."""
    projected = inference_input(episode, feature_policy)
    query = _tokens(projected["context"].get("query", ""))
    scores = {}
    for alternative in projected["alternatives"]:
        if not alternative.get("available", True):
            continue
        features = alternative["features"]
        title = _tokens(features.get("product_title", ""))
        description = _tokens(features.get("product_description", ""))
        scores[alternative["id"]] = float(2 * len(query & title) + len(query & description))
    return scores


def numerical_heuristic(episode: Mapping, feature_policy: Mapping,
                        weights: Mapping[str, float]) -> dict[str, float]:
    projected = inference_input(episode, feature_policy)
    result = {}
    missing = []
    for alternative in projected["alternatives"]:
        if not alternative.get("available", True):
            continue
        values = alternative["features"]
        selected = [values.get(field) for field in weights]
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
               for value in selected):
            missing.append(alternative["id"])
            continue
        result[alternative["id"]] = sum(float(coefficient) * float(values[field])
                                         for field, coefficient in weights.items())
    # A missing price/time/rating cannot win by masquerading as a favorable zero.
    floor = min(result.values(), default=0.0) - 1.0
    result.update({candidate_id: floor for candidate_id in missing})
    return result


def random_scores(episode: Mapping, seed: int = 42) -> dict[str, float]:
    """Seeded hash control invariant to source row order and candidate order."""
    return {a["id"]: int.from_bytes(hashlib.sha256(
        f'{seed}|{episode["episode_id"]}|{a["id"]}'.encode()).digest()[:8], "big") / 2**64
        for a in episode["alternatives"] if a.get("available", True)}


class FeatureState:
    """Train-only medians and categorical vocabularies over named source fields."""

    def __init__(self, policy: Mapping):
        self.policy = dict(policy)
        self.numeric = list(policy.get("numeric_features", []))
        self.categorical = list(policy.get("categorical_features", []))
        self.lexical = bool(policy.get("lexical_overlap", False))
        for name in self.numeric + self.categorical:
            if name.startswith("context."):
                approved = policy.get("context_features", [])
                field = name.removeprefix("context.")
            elif name.startswith("alternative."):
                approved = policy.get("alternative_features", [])
                field = name.removeprefix("alternative.")
            else:
                raise ValueError(f"feature must be explicitly scoped: {name}")
            if field not in approved:
                raise ValueError(f"feature absent from inference allowlist: {name}")
        self.medians: dict[str, float] = {}
        self.categories: dict[str, dict[str, int]] = {}

    def _projection(self, episode: Mapping) -> dict:
        return inference_input(episode, self.policy)

    def _rows(self, episodes: Iterable[Mapping]):
        for episode in episodes:
            projected = self._projection(episode)
            for alternative in projected["alternatives"]:
                if alternative.get("available", True):
                    yield projected, alternative

    def fit(self, train: Iterable[Mapping]) -> "FeatureState":
        numeric = {name: [] for name in self.numeric}
        categories = {name: set() for name in self.categorical}
        for projected, alternative in self._rows(train):
            for name in self.numeric:
                value = _field(projected, alternative, name)
                if isinstance(value, (float, int)) and math.isfinite(float(value)):
                    numeric[name].append(float(value))
            for name in self.categorical:
                value = _field(projected, alternative, name)
                if value is not None:
                    categories[name].add(str(value))
        self.medians = {name: float(np.median(values)) if values else 0.0
                        for name, values in numeric.items()}
        self.categories = {name: {value: i for i, value in enumerate(sorted(values))}
                           for name, values in categories.items()}
        return self

    def transform(self, episodes: Iterable[Mapping]) -> tuple[np.ndarray, list[int], list[list[str]]]:
        matrix, groups, ids = [], [], []
        for episode in episodes:
            projected = self._projection(episode)
            eligible = [a for a in projected["alternatives"] if a.get("available", True)]
            if not eligible:
                raise ValueError("empty eligible group")
            groups.append(len(eligible))
            ids.append([a["id"] for a in eligible])
            for alternative in eligible:
                row = []
                for name in self.numeric:
                    value = _field(projected, alternative, name)
                    valid = isinstance(value, (int, float)) and math.isfinite(float(value))
                    row.extend((float(value) if valid else self.medians[name], float(not valid)))
                for name in self.categorical:
                    value = _field(projected, alternative, name)
                    row.append(float(self.categories[name].get(str(value), -1)
                                     if value is not None else -1))
                if self.lexical:
                    query = _tokens(projected["context"].get("query", ""))
                    title = _tokens(alternative["features"].get("product_title", ""))
                    description = _tokens(alternative["features"].get("product_description", ""))
                    row.extend((float(len(query & title)), float(len(query & description))))
                matrix.append(row)
        if not matrix or not matrix[0]:
            raise ValueError("feature projection has no rows or no features")
        return np.asarray(matrix, dtype=np.float32), groups, ids

    def to_dict(self) -> dict:
        return {"policy": self.policy, "numeric": self.numeric, "categorical": self.categorical,
                "lexical": self.lexical, "medians": self.medians, "categories": self.categories}

    @classmethod
    def from_dict(cls, payload: Mapping) -> "FeatureState":
        state = cls(payload["policy"])
        state.medians = dict(payload["medians"])
        state.categories = {k: dict(v) for k, v in payload["categories"].items()}
        return state


def _field(projected: Mapping, alternative: Mapping, name: str):
    if name.startswith("context."):
        return projected["context"].get(name.removeprefix("context."))
    if name.startswith("alternative."):
        return alternative["features"].get(name.removeprefix("alternative."))
    raise ValueError(f"feature must be explicitly scoped: {name}")


def fit_conditional_logit(train: list[Mapping], feature_policy: Mapping,
                          *, seed: int = 42, iterations: int = 500,
                          learning_rate: float = 0.05, l2: float = 0.01) -> dict:
    """Swissmetro availability-aware linear conditional choice control."""
    if any(e["task_type"] != "stated_choice" for e in train):
        raise ValueError("conditional logit requires stated choices")
    state = FeatureState(feature_policy).fit(train)
    matrix, groups, ids = state.transform(train)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0] = 1
    x = (matrix - means) / scales
    weights = np.zeros(x.shape[1], dtype=np.float64)
    starts = np.cumsum([0] + groups)
    for _ in range(iterations):
        gradient = l2 * weights
        for j, episode in enumerate(train):
            block = x[starts[j]:starts[j + 1]]
            logits = block @ weights
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
            target = ids[j].index(episode["supervision"]["selected_id"])
            gradient += (probability @ block - block[target]) / len(train)
        weights -= learning_rate * gradient
    return {"state": state.to_dict(), "means": means.tolist(), "scales": scales.tolist(),
            "weights": weights.tolist(), "seed": seed, "iterations": iterations,
            "learning_rate": learning_rate, "l2": l2, "policy_id": "swissmetro-choice-v1"}


def predict_conditional_logit(bundle: Mapping, episodes: Iterable[Mapping]) -> dict[str, dict[str, float]]:
    episodes = list(episodes)
    state = FeatureState.from_dict(bundle["state"])
    matrix, groups, ids = state.transform(episodes)
    x = (matrix - np.asarray(bundle["means"])) / np.asarray(bundle["scales"])
    logits = x @ np.asarray(bundle["weights"])
    predictions, offset = {}, 0
    for episode, size, candidate_ids in zip(episodes, groups, ids, strict=True):
        predictions[episode["episode_id"]] = dict(zip(candidate_ids, logits[offset:offset + size].tolist(), strict=True))
        offset += size
    return predictions
