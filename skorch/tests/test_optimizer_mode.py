"""Tests for optimizer train/eval mode switching (skorch/optimizer_mode.py)"""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from skorch import NeuralNetClassifier
from skorch.optimizer_mode import optimizer_mode
from skorch.optimizer_mode import set_optimizer_mode


class TrainModeTrackingSGD(torch.optim.SGD):
    """Plain torch optimizer that tracks mode the schedule-free way.

    torch.optim.Optimizer defines neither ``train`` nor ``eval``, so this
    mimics the contract of the schedule-free optimizers: per-param-group
    ``train_mode`` and no-argument ``train()``/``eval()`` methods.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            group['train_mode'] = True

    def eval(self):
        for group in self.param_groups:
            group['train_mode'] = False


class ModuleTrackingMode(nn.Module):
    """Module whose output depends on the mode of the optimizer."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(20, 2)

    # pylint: disable=arguments-differ
    def forward(self, X):
        return self.dense(X)


class TestSetOptimizerMode:
    def test_no_op_for_plain_optimizer(self):
        # plain optimizers have no train/eval, mode stays undetermined
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.1)
        assert optimizer_mode(optimizer) is None
        set_optimizer_mode(optimizer, True)  # does not raise
        assert optimizer_mode(optimizer) is None

    def test_sets_mode_via_no_argument_convention(self):
        param = nn.Parameter(torch.zeros(1))
        optimizer = TrainModeTrackingSGD([param], lr=0.1)
        assert optimizer_mode(optimizer) is False

        set_optimizer_mode(optimizer, True)
        assert optimizer_mode(optimizer) is True
        set_optimizer_mode(optimizer, False)
        assert optimizer_mode(optimizer) is False

    def test_sets_mode_via_boolean_convention(self):
        # optimizers that mirror torch.nn.Module take a boolean
        param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.1)

        def train(mode=True):
            optimizer.training = mode

        def do_eval():
            optimizer.training = False

        optimizer.train = Mock(wraps=train)
        optimizer.eval = Mock(wraps=do_eval)
        optimizer.training = False

        set_optimizer_mode(optimizer, True)
        optimizer.train.assert_called_once_with(True)
        assert optimizer.training is True

        set_optimizer_mode(optimizer, False)
        optimizer.eval.assert_called_once_with()
        assert optimizer.training is False

    def test_ignores_non_optimizers(self):
        module = nn.Linear(1, 1)
        module.eval()
        # a module has train/eval but is not an optimizer
        assert optimizer_mode(module) is None
        set_optimizer_mode(module, True)  # does not raise / no effect
        assert module.training is False  # untouched by the optimizer path


class TestOptimizerModeDuringFit:
    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        return classifier_data

    def test_optimizer_toggled_between_train_and_eval(self, data):
        # _set_training is called for every train/validation batch; the
        # optimizer mode must follow, otherwise evaluation (e.g. the valid
        # loss) is computed at the training weights.
        calls = []

        class TrackingOptimizer(TrainModeTrackingSGD):
            def train(self):
                calls.append('train')
                super().train()

            def eval(self):
                calls.append('eval')
                super().eval()

        X, y = data
        net = NeuralNetClassifier(
            ModuleTrackingMode,
            optimizer=TrackingOptimizer,
            criterion=nn.CrossEntropyLoss,
            max_epochs=1,
            lr=0.1,
            batch_size=64,
        )
        net.fit(X, y)

        # both modes occurred during fit
        assert 'train' in calls
        assert 'eval' in calls
        # and the end state is evaluation mode, since predict/eval steps
        # run last
        assert optimizer_mode(net.optimizer_) is False

    def test_predict_leaves_optimizer_in_eval_mode(self, data):
        X, y = data
        net = NeuralNetClassifier(
            ModuleTrackingMode,
            optimizer=TrainModeTrackingSGD,
            criterion=nn.CrossEntropyLoss,
            max_epochs=1,
            lr=0.1,
            batch_size=64,
        )
        net.fit(X, y)
        assert optimizer_mode(net.optimizer_) is False
        net.predict(X[:10])
        assert optimizer_mode(net.optimizer_) is False
