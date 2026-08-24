"""Tests for optimizer train/eval switching (schedule-free optimizers)."""

import numpy as np
import pytest
import torch
from torch import nn

from skorch import NeuralNetClassifier
from skorch.schedule_free import optimizer_switches_modes


class SwitchingOptimizer(torch.optim.SGD):
    """Optimizer that follows the module-like train/eval contract.

    Real schedule-free optimizers (Defazio et al., 2024) expose ``train()`` and
    ``eval()`` and expect to be switched in lockstep with the module; this
    subclass stands in for them without requiring the ``schedulefree`` package.

    """

    def train(self, mode=True):
        self.mode = 'train' if mode else 'eval'
        return self


class TestOptimizerSwitchesModes:
    def test_true_for_optimizer_overriding_train(self):
        optimizer = SwitchingOptimizer(nn.Linear(2, 1).parameters(), lr=0.1)
        assert optimizer_switches_modes(optimizer) is True

    @pytest.mark.parametrize('make_optimizer', [
        lambda params: torch.optim.SGD(params, lr=0.1),
        lambda params: torch.optim.Adam(params, lr=0.1),
    ])
    def test_false_for_ordinary_optimizer(self, make_optimizer):
        optimizer = make_optimizer(nn.Linear(2, 1).parameters())
        assert optimizer_switches_modes(optimizer) is False

    def test_false_for_non_optimizer(self):
        assert optimizer_switches_modes(nn.Linear(2, 1)) is False


class TestSetTrainingSwitchesOptimizer:
    @pytest.fixture(scope='module')
    def module_cls(self, classifier_module):
        return classifier_module

    @pytest.fixture(scope='module')
    def data(self, classifier_data):
        return classifier_data

    @pytest.fixture
    def optimizer_cls(self):
        return SwitchingOptimizer

    def test_optimizer_switched_through_fit_and_predict(
            self, module_cls, data, optimizer_cls):
        # the switching optimizer records the mode each phase left it in
        modes = []

        class RecordingOptimizer(optimizer_cls):
            """Switching optimizer that records its mode on every switch."""
            def train(self, mode=True):
                super().train(mode)
                modes.append(self.mode)
                return self

        net = NeuralNetClassifier(
            module_cls,
            optimizer=RecordingOptimizer,
            max_epochs=2,
        )
        X, y = data[0][:100], data[1][:100]
        net.fit(X, y)

        # training starts in train mode, alternating with eval for validation
        assert modes[0] == 'train'
        assert 'eval' in modes

        net.predict(X)
        # prediction leaves the optimizer in eval mode, so that gradients are
        # taken at the primal iterate and predictions at the averaged one
        assert net.optimizer_.mode == 'eval'
        assert modes[-1] == 'eval'

    def test_train_called_once_per_switch(self, module_cls):
        calls = []

        class SpyingOptimizer(SwitchingOptimizer):
            """Record the argument of every train() call."""
            def train(self, mode=True):
                calls.append(mode)
                return super().train(mode)

        net = NeuralNetClassifier(
            module_cls, optimizer=torch.optim.SGD, max_epochs=1,
        ).initialize()
        # swap in the spying optimizer after initialization, the same way a
        # user-supplied optimizer would be switched
        net.optimizer_ = SpyingOptimizer(
            nn.Linear(20, 2).parameters(), lr=0.1)
        net._set_training(True)
        net._set_training(False)
        assert calls == [True, False]

    def test_module_and_optimizer_switched_together(
            self, module_cls, optimizer_cls):
        net = NeuralNetClassifier(
            module_cls, optimizer=optimizer_cls, max_epochs=1,
        ).initialize()
        net._set_training(True)
        assert net.module_.training is True
        assert net.optimizer_.mode == 'train'
        net._set_training(False)
        assert net.module_.training is False
        assert net.optimizer_.mode == 'eval'

    def test_net_still_learns_with_switching_optimizer(
            self, module_cls, data, optimizer_cls):
        # the mode switches must not perturb the ordinary training loop
        net = NeuralNetClassifier(
            module_cls,
            optimizer=optimizer_cls,
            max_epochs=2,
            lr=0.05,
        )
        X, y = data[0][:200], data[1][:200]
        net.fit(X, y)
        y_pred = net.predict(X)
        assert np.mean(y_pred == y) > 0.5

    def test_plain_optimizer_untouched(self, module_cls, data):
        # an ordinary optimizer is never switched, which is safe precisely
        # because torch.optim.Optimizer has no train method at all
        assert 'train' not in vars(torch.optim.Optimizer)

        net = NeuralNetClassifier(module_cls, max_epochs=1)
        X, y = data[0][:100], data[1][:100]
        net.fit(X, y)
        net.predict(X)

        assert not hasattr(net.optimizer_, 'train')
        assert not hasattr(net.optimizer_, 'eval')

    def test_missing_optimizer_tolerated(self, module_cls):
        # after trim_for_prediction, the optimizer is gone but _set_training
        # is still called
        net = NeuralNetClassifier(
            module_cls,
            optimizer=SwitchingOptimizer,
            max_epochs=1,
        ).initialize()
        net.optimizer_ = None
        net._set_training(True)
        assert net.module_.training is True
