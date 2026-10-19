import pytest
import torch

from liquid_mcdm.training.calibration import fit_temperature


def test_temperature_fit_reduces_validation_nll():
    scores = torch.tensor([[8., 0.], [8., 0.], [0., 8.], [0., 8.]])
    target = torch.tensor([0, 1, 1, 0])
    fitted = fit_temperature(scores, target, split='calibration',
                             event='conditional_booking_choice')
    assert fitted.event == 'conditional_booking_choice'
    assert fitted.temperature > 1
    before = torch.nn.functional.cross_entropy(scores, target)
    after = torch.nn.functional.nll_loss(fitted.apply(scores).log(), target)
    assert after < before


def test_variable_group_padding_and_masked_target():
    scores = torch.tensor([[3., 1., float('-inf')], [0., 2., 1.]])
    target = torch.tensor([0, 1])
    fitted = fit_temperature(scores, target, split='calibration', event='stated_choice')
    assert fitted.apply(scores)[0, 2] == 0
    with pytest.raises(ValueError, match='unavailable'):
        fit_temperature(scores, torch.tensor([2, 1]), split='calibration',
                        event='stated_choice')


@pytest.mark.parametrize('split', ['test', 'train', 'validation', 'final_test', ''])
def test_no_test_or_train_fitting_api(split):
    with pytest.raises(ValueError, match='disjoint calibration'):
        fit_temperature(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long),
                        split=split, event='booking')
