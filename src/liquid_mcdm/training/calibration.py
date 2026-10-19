"""Post-hoc temperature fitting on a disjoint calibration partition."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    event: str
    fitted_split: str
    objective: str = 'nll'

    def apply(self, option_scores: torch.Tensor) -> torch.Tensor:
        if option_scores.ndim < 2 or not option_scores.is_floating_point():
            raise ValueError('option_scores must be a batched floating tensor')
        _check_scores(option_scores)
        return torch.softmax(option_scores.float() / self.temperature, dim=-1)


def _check_scores(scores: torch.Tensor) -> None:
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError('invalid option scores')
    if not torch.isfinite(scores).any(dim=-1).all():
        raise ValueError('each group needs an eligible option')


def fit_temperature(option_scores: torch.Tensor, target: torch.Tensor, *,
                    split: str, event: str) -> TemperatureCalibration:
    """Fit one scalar on held-out calibration groups, separate from selection."""
    if split != 'calibration':
        raise ValueError('temperature fitting requires the disjoint calibration split')
    if event not in {'conditional_booking_choice', 'stated_choice'}:
        raise ValueError('event must be conditional_booking_choice or stated_choice')
    if not isinstance(option_scores, torch.Tensor) or option_scores.ndim != 2:
        raise ValueError('option_scores must be [groups, options]')
    if not option_scores.is_floating_point():
        raise ValueError('option scores must be floats')
    _check_scores(option_scores)
    if option_scores.shape[0] < 2 or option_scores.shape[1] < 2:
        raise ValueError('at least two groups with two options are required')
    if not isinstance(target, torch.Tensor) or target.dtype != torch.long or target.shape != option_scores.shape[:1]:
        raise ValueError('target must have one long index per group')
    if ((target < 0) | (target >= option_scores.shape[1])).any():
        raise ValueError('target index outside options')
    if not torch.isfinite(option_scores.gather(1, target[:, None])).all():
        raise ValueError('target is unavailable')
    # Bounded one-dimensional search is deterministic, stable and never updates
    # the trained model. Fine grid is adequate for this post-hoc control.
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
    return TemperatureCalibration(float(log_t[best].exp().cpu()), event.strip(), split)
