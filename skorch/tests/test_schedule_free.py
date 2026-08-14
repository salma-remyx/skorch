"""Tests for using schedule-free optimizers with skorch nets."""

import pytest
import torch
from torch import nn

from skorch import NeuralNetRegressor
from skorch.dataset import ValidSplit
from skorch.schedule_free import set_optimizer_mode


class RecordingScheduleFreeOptimizer(torch.optim.SGD):
    """Stand-in for a schedule-free optimizer (e.g. from the
    facebookresearch/schedule_free package), which exposes train/eval
    methods that must be called when switching between training and
    evaluation.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_calls = []

    def train(self):
        self.mode_calls.append('train')

    def eval(self):
        self.mode_calls.append('eval')


@pytest.fixture
def mode_recorder():
    """Return a fitted net whose optimizer records its train/eval calls;
    the recorded modes are accessible via net.optimizer_.mode_calls.

    """
    module = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))

    net = NeuralNetRegressor(
        module,
        optimizer=RecordingScheduleFreeOptimizer,
        lr=0.1,
        max_epochs=2,
        train_split=ValidSplit(2),
        verbose=0,
    )
    X = torch.randn(20, 2).float()
    y = torch.randn(20, 1).float()
    net.fit(X, y)
    return net


class TestOptimizerModeToggling:
    def test_optimizer_mode_toggled_between_train_and_eval(self, mode_recorder):
        # fitting runs the validation loop after each training epoch;
        # therefore, both train and eval modes must have been set
        modes = mode_recorder.optimizer_.mode_calls
        assert 'train' in modes
        assert 'eval' in modes

        # training happens in batches, hence many train calls, whereas
        # validation is entered once per epoch
        n_epochs = mode_recorder.max_epochs
        assert modes.count('eval') >= n_epochs
        assert modes.count('train') >= modes.count('eval')

    def test_prediction_sets_eval_mode(self, mode_recorder):
        mode_recorder.optimizer_.mode_calls.clear()
        X = torch.randn(4, 2).float()
        mode_recorder.predict(X)
        # the optimizer should be in eval mode for predictions, without
        # switching back to train mode
        assert mode_recorder.optimizer_.mode_calls == ['eval']

    def test_evaluation_step_sets_optimizer_eval_mode(self, mode_recorder):
        net = mode_recorder
        optimizer = net.optimizer_
        # emulate the state right after training
        optimizer.mode_calls.clear()
        batch = (torch.randn(3, 2).float(), torch.randn(3, 1).float())
        net.evaluation_step(batch)
        assert optimizer.mode_calls == ['eval']

    def test_regular_optimizer_unaffected(self):
        # regular optimizers don't have train/eval methods; setting the
        # mode should be a no-op, not raise
        optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
        set_optimizer_mode(optimizer, True)
        set_optimizer_mode(optimizer, False)
        assert isinstance(optimizer, torch.optim.SGD)
