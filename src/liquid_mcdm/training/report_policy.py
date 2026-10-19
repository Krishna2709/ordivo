"""Experimental RLCD-inspired stochastic probability-report objective.

This is a continuous report policy over one complete categorical choice group.
It is neither TRL GRPO nor a reproduction of TypeSafe/Laya training.
"""
from __future__ import annotations

import math

import torch
from torch.distributions import Dirichlet, kl_divergence

from .calibration import EVENT_POPULATIONS


def _probabilities(scores: torch.Tensor, eligible: torch.Tensor, floor: float):
    selected = scores[eligible].float()
    if not torch.isfinite(selected).all():
        raise ValueError('nonfinite eligible scores')
    probabilities = torch.softmax(selected, dim=-1)
    # Positive concentration is required by Dirichlet. The floor is a declared
    # report-policy transform and is recorded separately from raw d1 readout.
    probabilities = probabilities.clamp_min(floor)
    return probabilities / probabilities.sum()


def reinforce_objective(log_prob: torch.Tensor, rewards: torch.Tensor,
                        kl: torch.Tensor, *, kl_weight: float):
    """Score-function loss with independent detached leave-one-out baseline."""
    if log_prob.ndim != 1 or rewards.shape != log_prob.shape or log_prob.numel() < 4:
        raise ValueError('at least four aligned report log probabilities/rewards required')
    if not torch.isfinite(log_prob).all() or not torch.isfinite(rewards).all():
        raise ValueError('nonfinite report log probability or reward')
    if kl.ndim != 0 or not torch.isfinite(kl):
        raise ValueError('finite scalar KL required')
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError('KL weight must be finite and nonnegative')
    baseline = (rewards.sum() - rewards) / (rewards.numel() - 1)
    advantage = (rewards - baseline).detach()
    loss = -(log_prob * advantage).mean() + kl_weight * kl
    return loss, advantage


def report_policy_loss(option_scores: torch.Tensor, reference_scores: torch.Tensor,
                       eligible: torch.Tensor, target_index: int, *, event: str,
                       candidate_population: str, concentration: float = 20.0,
                       report_count: int = 8, kl_weight: float = 0.0,
                       probability_floor: float = 1e-6):
    """Sample reports, score them by negative Brier, and form REINFORCE loss.

    `option_scores` are the differentiable MAX-pooled d1 option logits. The
    reference is frozen. Reports and rewards are sampled without pathwise
    gradients, so the only reward gradient uses true Dirichlet log density.
    """
    if EVENT_POPULATIONS.get(event) != candidate_population:
        raise ValueError('unsupported categorical event/population protocol')
    if not isinstance(option_scores, torch.Tensor) or option_scores.ndim != 1 or not option_scores.is_floating_point():
        raise ValueError('option_scores must be a floating option vector')
    if not isinstance(reference_scores, torch.Tensor) or reference_scores.shape != option_scores.shape:
        raise ValueError('reference scores must align with options')
    if not isinstance(eligible, torch.Tensor) or eligible.dtype != torch.bool or eligible.shape != option_scores.shape:
        raise ValueError('eligible must be an aligned bool mask')
    if int(eligible.sum()) < 2:
        raise ValueError('at least two eligible options required')
    if not isinstance(target_index, int) or not 0 <= target_index < eligible.numel() or not eligible[target_index]:
        raise ValueError('target must name an eligible option')
    if not math.isfinite(concentration) or concentration <= 0:
        raise ValueError('concentration must be finite and positive')
    if not math.isfinite(probability_floor) or not 0 < probability_floor < 1 / int(eligible.sum()):
        raise ValueError('probability floor is invalid for option count')
    if not isinstance(report_count, int) or report_count < 4:
        raise ValueError('at least four reports required')
    if option_scores.device != reference_scores.device or option_scores.device != eligible.device:
        raise ValueError('scores and mask must share a device')
    probabilities = _probabilities(option_scores, eligible, probability_floor)
    reference = _probabilities(reference_scores.detach(), eligible, probability_floor)
    policy = Dirichlet(concentration * probabilities)
    frozen = Dirichlet(concentration * reference)
    reports = policy.sample((report_count,))
    log_prob = policy.log_prob(reports)
    position = int(eligible[:target_index].sum())
    one_hot = torch.zeros_like(probabilities)
    one_hot[position] = 1
    rewards = -(reports - one_hot).square().sum(dim=-1).detach()
    kl = kl_divergence(policy, frozen)
    loss, advantage = reinforce_objective(log_prob, rewards, kl, kl_weight=kl_weight)
    full_reports = torch.zeros((report_count, eligible.numel()), dtype=reports.dtype,
                               device=reports.device)
    full_reports[:, eligible] = reports
    return {'loss': loss, 'reports': full_reports.detach(), 'rewards': rewards,
            'advantages': advantage, 'log_prob': log_prob,
            'kl': kl, 'policy_probabilities': probabilities,
            'mean_reward': rewards.mean(),
            'reward_variance': rewards.var(unbiased=False),
            'advantage_variance': advantage.var(unbiased=False),
            'zero_variance_group': bool(torch.all(rewards == rewards[0]))}


def analytic_expected_brier(option_scores: torch.Tensor, eligible: torch.Tensor,
                            target_index: int, *, concentration: float,
                            probability_floor: float = 1e-6) -> torch.Tensor:
    """Exact expected Brier under fixed-concentration Dirichlet for test/control."""
    probabilities = _probabilities(option_scores, eligible, probability_floor)
    position = int(eligible[:target_index].sum())
    one_hot = torch.zeros_like(probabilities)
    one_hot[position] = 1
    return (probabilities - one_hot).square().sum() + (
        1 - probabilities.square().sum()) / (concentration + 1)
