import math

import pytest
import torch

from liquid_mcdm.training.option_policy import (choice_loss, option_logits,
                                                 option_probabilities)


def test_pinned_max_then_softmax_and_gradient():
    logits = torch.tensor([0., 2., 3., 1., 4., -1.], requires_grad=True)
    groups = [[1, 2], [3, 4]]
    probabilities = option_probabilities(logits, groups)
    expected = torch.softmax(torch.tensor([3., 4.]), 0)
    assert torch.allclose(probabilities, expected)
    loss = choice_loss(logits, groups, torch.tensor(0), objective='nll')
    loss.backward()
    assert math.isfinite(loss.item())
    assert logits.grad[2] < 0 and logits.grad[4] > 0
    assert logits.grad[[0, 1, 3, 5]].abs().sum() == 0


def test_mask_coverage_brier_and_stability():
    logits = torch.tensor([[10000., -10000., 8.], [2., 3., 4.]], requires_grad=True)
    eligible = torch.tensor([[True, False], [True, True]])
    groups = [[0], [1, 2]]
    probabilities = option_probabilities(logits, groups, eligible)
    assert torch.equal(probabilities[0], torch.tensor([1., 0.]))
    loss = choice_loss(logits, groups, torch.tensor([0, 1]), eligible=eligible,
                       objective='brier')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() == 0


@pytest.mark.parametrize('groups', [[], [[]], [[-1]], [[99]], [[0], [0]]])
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        option_logits(torch.zeros(3), groups)


def test_invalid_masks_targets_and_logits():
    logits = torch.tensor([1., 2.])
    with pytest.raises(ValueError, match='eligible option'):
        option_logits(logits, [[0], [1]], torch.tensor([False, False]))
    with pytest.raises(ValueError, match='unavailable'):
        choice_loss(logits, [[0], [1]], torch.tensor(1), eligible=torch.tensor([True, False]))
    with pytest.raises(ValueError, match='outside'):
        choice_loss(logits, [[0], [1]], torch.tensor(2))
    with pytest.raises(ValueError, match='nonfinite'):
        option_logits(torch.tensor([float('nan'), 0.]), [[0], [1]])
