import pytest
import torch

from liquid_mcdm.training.report_policy import (analytic_expected_brier,
                                                 reinforce_objective,
                                                 report_policy_loss)


def test_dirichlet_report_gradient_matches_analytic_expected_brier():
    torch.manual_seed(42)
    scores = torch.tensor([0.3, -0.2], requires_grad=True)
    mask = torch.tensor([True, True])
    result = report_policy_loss(
        scores, scores.detach(), mask, 0, event='stated_choice',
        candidate_population='eligible_stated_choice_group', concentration=10,
        report_count=20000, kl_weight=0)
    result['loss'].backward()
    sampled_gradient = scores.grad.detach().clone()
    exact_scores = scores.detach().clone().requires_grad_(True)
    analytic_expected_brier(exact_scores, mask, 0, concentration=10).backward()
    assert torch.allclose(sampled_gradient, exact_scores.grad, atol=0.04, rtol=0.08)
    assert result['reward_variance'] > 0
    assert not result['zero_variance_group']


def test_constant_reward_has_zero_advantage_and_gradient():
    log_prob = torch.tensor([-0.3, -0.6, -1.0, -0.8], requires_grad=True)
    reward = torch.full((4,), -0.5)
    loss, advantage = reinforce_objective(log_prob, reward, torch.tensor(0.),
                                          kl_weight=0)
    loss.backward()
    assert torch.count_nonzero(advantage) == 0
    assert torch.count_nonzero(log_prob.grad) == 0


def test_event_mask_kl_and_invalid_inputs():
    scores = torch.tensor([0.2, 9., -0.1], requires_grad=True)
    reference = torch.tensor([-0.1, 9., 0.2])
    eligible = torch.tensor([True, False, True])
    result = report_policy_loss(
        scores, reference, eligible, 2, event='conditional_booking_choice',
        candidate_population='exactly_one_booking_exposed_group', report_count=8,
        kl_weight=0.1)
    assert torch.all(result['reports'][:, 1] == 0)
    assert torch.allclose(result['reports'].sum(-1), torch.ones(8), atol=1e-6)
    assert result['kl'] > 0 and torch.isfinite(result['loss'])
    result['loss'].backward()
    assert torch.isfinite(scores.grad).all() and scores.grad[1] == 0
    with pytest.raises(ValueError, match='protocol'):
        report_policy_loss(scores.detach(), reference, eligible, 2, event='stated_choice',
                           candidate_population='exactly_one_booking_exposed_group')
    with pytest.raises(ValueError, match='eligible'):
        report_policy_loss(scores.detach(), reference, eligible, 1, event='stated_choice',
                           candidate_population='eligible_stated_choice_group')
