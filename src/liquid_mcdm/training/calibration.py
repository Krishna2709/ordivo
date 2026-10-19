"""Manifest-bound temperature fitting on a disjoint calibration partition."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


EVENT_POPULATIONS = {
    'conditional_booking_choice': 'exactly_one_booking_exposed_group',
    'stated_choice': 'eligible_stated_choice_group',
}
SPLITS = ('train', 'validation', 'calibration', 'test')


def _hash_ids(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class CalibrationEvidence:
    manifest_sha256: str
    split_sha256: dict[str, str]
    group_ids: dict[str, frozenset[str]]
    population_group_ids: frozenset[str]
    event: str
    candidate_population: str


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    event: str
    candidate_population: str
    manifest_sha256: str
    calibration_groups_sha256: str
    calibration_groups_count: int
    fitted_split: str = 'calibration'
    objective: str = 'nll'

    def apply(self, option_scores: torch.Tensor, *, event: str,
              candidate_population: str, manifest_sha256: str) -> torch.Tensor:
        if (event, candidate_population, manifest_sha256) != (self.event, self.candidate_population,
                                                               self.manifest_sha256):
            raise ValueError('calibration event, population, or manifest lineage mismatch')
        if option_scores.ndim < 2 or not option_scores.is_floating_point():
            raise ValueError('option_scores must be a batched floating tensor')
        _check_scores(option_scores)
        return torch.softmax(option_scores.float() / self.temperature, dim=-1)


def load_calibration_evidence(manifest_path: str | Path, *, event: str,
                              candidate_population: str) -> CalibrationEvidence:
    """Verify four frozen split files against the manifest and their group disjointness."""
    if EVENT_POPULATIONS.get(event) != candidate_population:
        raise ValueError('event and candidate population protocol disagree')
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (event == 'conditional_booking_choice' and
            manifest.get('candidate_pool_scope') != 'source_exposed_candidates'):
        raise ValueError('conditional booking requires exposed candidate pool manifest')
    hashes = manifest.get('split_sha256')
    if not isinstance(hashes, dict) or set(hashes) != set(SPLITS):
        raise ValueError('manifest must hash four split files')
    groups = {}
    split_units = {}
    population_ids = set()
    for split in SPLITS:
        content = (manifest_path.parent / f'{split}.jsonl').read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes[split]:
            raise ValueError(f'{split} file differs from frozen manifest')
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        ids = [row['episode_id'] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate episode in {split}')
        groups[split] = frozenset(ids)
        if split == 'calibration':
            for row in rows:
                if event == 'stated_choice' and row.get('task_type') == 'stated_choice':
                    population_ids.add(row['episode_id'])
                elif event == 'conditional_booking_choice' and row.get('task_type') == 'observed_choice':
                    outcomes = row['supervision']['outcomes_by_id']
                    count = sum(bool(outcomes[candidate['id']]['booking_bool'])
                                for candidate in row['alternatives'] if candidate.get('available', True))
                    if count == 1:
                        population_ids.add(row['episode_id'])
        split_units[split] = frozenset(
            f"{row['source']}:{row['supervision'].get('respondent_id', row['group_id'])}"
            for row in rows)
    if not groups['calibration']:
        raise ValueError('empty calibration split')
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if groups[left] & groups[right] or split_units[left] & split_units[right]:
                raise ValueError(f'group overlap between {left} and {right}')
    return CalibrationEvidence(hashlib.sha256(manifest_bytes).hexdigest(), dict(hashes),
                               groups, frozenset(population_ids), event, candidate_population)


def _check_scores(scores: torch.Tensor) -> None:
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError('invalid option scores')
    if not torch.isfinite(scores).any(dim=-1).all():
        raise ValueError('each group needs an eligible option')


def fit_temperature(option_scores: torch.Tensor, target: torch.Tensor, *,
                    evidence: CalibrationEvidence, group_ids: list[str]) -> TemperatureCalibration:
    """Fit one scalar on specified calibration groups of a verified frozen manifest."""
    if not isinstance(evidence, CalibrationEvidence) or evidence.event not in EVENT_POPULATIONS:
        raise ValueError('verified calibration evidence required')
    if EVENT_POPULATIONS[evidence.event] != evidence.candidate_population:
        raise ValueError('event and candidate population protocol disagree')
    if not isinstance(group_ids, list) or len(group_ids) != len(set(group_ids)):
        raise ValueError('group IDs must be unique')
    if not set(group_ids) <= evidence.group_ids['calibration']:
        raise ValueError('fitting groups must belong to calibration split')
    if not set(group_ids) <= evidence.population_group_ids:
        raise ValueError('fitting groups violate candidate population protocol')
    if set(group_ids) != evidence.population_group_ids:
        raise ValueError('fit requires full eligible calibration population')
    if not isinstance(option_scores, torch.Tensor) or option_scores.ndim != 2:
        raise ValueError('option_scores must be [groups, options]')
    if not option_scores.is_floating_point():
        raise ValueError('option scores must be floats')
    _check_scores(option_scores)
    if option_scores.shape[0] < 2 or option_scores.shape[1] < 2 or len(group_ids) != option_scores.shape[0]:
        raise ValueError('at least two aligned groups with two options are required')
    if not isinstance(target, torch.Tensor) or target.dtype != torch.long or target.shape != option_scores.shape[:1]:
        raise ValueError('target must have one long index per group')
    if ((target < 0) | (target >= option_scores.shape[1])).any():
        raise ValueError('target index outside options')
    if not torch.isfinite(option_scores.gather(1, target[:, None])).all():
        raise ValueError('target is unavailable')
    scores = option_scores.detach().float()
    labels = target.detach()
    log_t = torch.linspace(-4.0, 4.0, 801, device=scores.device)
    losses = []
    for chunk in log_t.split(80):
        scaled = scores.unsqueeze(0) / chunk.exp()[:, None, None]
        losses.append(F.cross_entropy(scaled.reshape(-1, scores.shape[-1]),
                                      labels.repeat(chunk.numel()), reduction='none')
                      .reshape(chunk.numel(), -1).mean(dim=-1))
    best = torch.cat(losses).argmin()
    return TemperatureCalibration(float(log_t[best].exp().cpu()), evidence.event,
                                  evidence.candidate_population, evidence.manifest_sha256,
                                  _hash_ids(group_ids), len(group_ids))
