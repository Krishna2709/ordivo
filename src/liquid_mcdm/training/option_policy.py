"""Differentiable d1 answer-slot readout and conditional choice scores.

Groups must come from the pinned ``d1_prompt.readout_ids`` for the exact
rendered question. They contain an alias token and, when tokenizable, its
leading-space form. This module never treats option probabilities as sequence
log probabilities.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def option_logits(answer_logits: torch.Tensor, readout_groups: list[list[int]],
                  eligible: torch.Tensor | None = None) -> torch.Tensor:
    """MAX each token group and mask unavailable options; preserve autograd."""
    if not isinstance(answer_logits, torch.Tensor) or not answer_logits.is_floating_point():
        raise TypeError('answer_logits must be a floating tensor')
    if answer_logits.ndim < 1 or answer_logits.shape[-1] < 1:
        raise ValueError('answer_logits must end in a nonempty vocabulary axis')
    if not readout_groups or any(not group for group in readout_groups):
        raise ValueError('every option needs a nonempty token group')
    flat = [index for group in readout_groups for index in group]
    if any(type(index) is not int or index < 0 or index >= answer_logits.shape[-1] for index in flat):
        raise ValueError('readout token ID outside vocabulary')
    if len(flat) != len(set(flat)):
        raise ValueError('readout token IDs must be unique across groups')
    selected = answer_logits[..., flat]
    if not torch.isfinite(selected).all():
        raise ValueError('nonfinite readout token logits')
    scores = torch.stack([answer_logits[..., group].max(dim=-1).values
                          for group in readout_groups], dim=-1)
    if eligible is not None:
        if not isinstance(eligible, torch.Tensor) or eligible.dtype != torch.bool:
            raise TypeError('eligible must be a bool tensor')
        if eligible.shape != scores.shape:
            raise ValueError('eligible mask must match option scores exactly')
        if not eligible.any(dim=-1).all():
            raise ValueError('every episode needs an eligible option')
        scores = scores.masked_fill(~eligible, -torch.inf)
    return scores


def option_log_probabilities(answer_logits: torch.Tensor,
                             readout_groups: list[list[int]],
                             eligible: torch.Tensor | None = None) -> torch.Tensor:
    return F.log_softmax(option_logits(answer_logits, readout_groups, eligible).float(), dim=-1)


def option_probabilities(answer_logits: torch.Tensor, readout_groups: list[list[int]],
                         eligible: torch.Tensor | None = None) -> torch.Tensor:
    return option_log_probabilities(answer_logits, readout_groups, eligible).exp()


def choice_loss(answer_logits: torch.Tensor, readout_groups: list[list[int]],
                target: torch.Tensor, *, eligible: torch.Tensor | None = None,
                objective: str = 'nll') -> torch.Tensor:
    """Mean conditional NLL or multiclass Brier over complete choice groups."""
    scores = option_logits(answer_logits, readout_groups, eligible).float()
    if not isinstance(target, torch.Tensor) or target.dtype != torch.long:
        raise TypeError('target must be a long tensor of option indices')
    if target.shape != scores.shape[:-1]:
        raise ValueError('one target index is required per complete group')
    if ((target < 0) | (target >= scores.shape[-1])).any():
        raise ValueError('target index outside options')
    if eligible is not None and not eligible.gather(-1, target.unsqueeze(-1)).all():
        raise ValueError('target is unavailable')
    logp = F.log_softmax(scores, dim=-1)
    if objective == 'nll':
        loss = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    elif objective == 'brier':
        one_hot = F.one_hot(target, scores.shape[-1]).to(logp.dtype)
        loss = (logp.exp() - one_hot).square().sum(dim=-1)
    else:
        raise ValueError('objective must be nll or brier')
    return loss.mean()
