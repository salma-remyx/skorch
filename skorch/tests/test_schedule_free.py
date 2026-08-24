"""Tests for skorch.schedule_free and its wiring into NeuralNet.

The optimizer mode switching is tested with a stand-in optimizer instead of
the ``schedulefree`` package, which is an optional dependency that is not
installed for these tests. The stand-in reproduces the interface that matters
here: a ``train_mode`` entry in each param group, and argument-less
``train()``/``eval()`` methods that record when the optimizer switches
between the point it takes gradients at and the point it evaluates at.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from skorch import NeuralNetRegressor
from skorch.schedule_free import set_optimizer_training
from skorch.schedule_free import supports_training_mode


class ScheduleFreeSGD(torch.optim.SGD):
    """Stand-in for a schedule-free optimizer.

    Mimics ``schedulefree.SGDScheduleFree``: the mode lives in the param
    groups as ``train_mode``, and ``train()``/``eval()`` are argument-less.
    The parameter values at the time of a switch are recorded, so that a test
    can tell apart the gradient point from the evaluation point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for group in self.param_groups:
            group['train_mode'] = False
        # every switch ever made, never cleared; ``(mode, first_parameter)``
        self.switch_history = []

    def _switch(self, mode):
        for group in self.param_groups:
            group['train_mode'] = mode
        # the parameter as seen by the module at this point of the schedule
        self.switch_history.append(
            (mode, self.param_groups[0]['params'][0].detach().clone())
        )

    def train(self):
        self._switch(True)
        return self

    def eval(self):
        self._switch(False)
        return self


class RecordingModule(nn.Module):
    """Module that records its training mode and parameters per call."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1, 1)
        self.call_log = []

    def forward(self, x):
        self.call_log.append(
            (self.training, self.dense.weight.detach().clone())
        )
        return self.dense(x)


class TestSupportsTrainingMode:
    def test_plain_optimizer_is_not_train_eval_aware(self):
        param = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.SGD([param], lr=0.1)
        assert not supports_training_mode(optimizer)

    def test_schedule_free_optimizer_is_detected(self):
        param = torch.zeros(1, requires_grad=True)
        optimizer = ScheduleFreeSGD([param], lr=0.1)
        assert supports_training_mode(optimizer)

    def test_optimizer_without_eval_method_is_not_detected(self):
        # train_mode and train() alone are not enough, eval() is needed too
        class TrainOnlySGD(ScheduleFreeSGD):
            """Has a training mode but no way to leave it"""

            eval = None

        param = torch.zeros(1, requires_grad=True)
        optimizer = TrainOnlySGD([param], lr=0.1)
        assert not supports_training_mode(optimizer)

    def test_mock_optimizer_is_not_detected(self):
        # a Mock stands in for an optimizer in some tests; it must not look
        # like it supports a training mode, since any attribute access on it
        # succeeds
        assert not supports_training_mode(Mock())


class TestSetOptimizerTraining:
    def test_is_noop_for_plain_optimizer(self):
        param = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.SGD([param], lr=0.1)
        set_optimizer_training(optimizer, False)
        assert not any('train_mode' in g for g in optimizer.param_groups)

    def test_accepts_none(self):
        # a net trimmed for prediction has None in place of its optimizers
        set_optimizer_training(None, False)  # does not raise

    def test_switches_mode(self):
        param = torch.zeros(1, requires_grad=True)
        optimizer = ScheduleFreeSGD([param], lr=0.1)

        set_optimizer_training(optimizer, True)
        assert all(g['train_mode'] for g in optimizer.param_groups)

        set_optimizer_training(optimizer, False)
        assert not any(g['train_mode'] for g in optimizer.param_groups)

    def test_redundant_switch_is_skipped(self):
        # switching moves parameters, so an already active mode must not be
        # set again
        param = torch.zeros(1, requires_grad=True)
        optimizer = ScheduleFreeSGD([param], lr=0.1)

        set_optimizer_training(optimizer, True)
        n_switches = len(optimizer.switch_history)
        set_optimizer_training(optimizer, True)

        assert len(optimizer.switch_history) == n_switches


class TestSetTrainingPropagatesToOptimizer:
    """The net should switch the optimizer's mode along with the module's."""

    @pytest.fixture
    def data(self):
        # enough samples for the default validation split
        X = np.arange(20, dtype=np.float32).reshape(-1, 1)
        y = (2 * np.arange(20, dtype=np.float32)).reshape(-1, 1)
        return X, y

    @pytest.fixture
    def net(self, data):
        return NeuralNetRegressor(
            RecordingModule,
            criterion=nn.MSELoss,
            optimizer=ScheduleFreeSGD,
            lr=0.05,
            max_epochs=1,
            batch_size=4,
        ).initialize()

    def test_set_training_switches_optimizer_mode(self, net):
        # pylint: disable=protected-access
        net._set_training(True)
        assert all(g['train_mode'] for g in net.optimizer_.param_groups)

        net._set_training(False)
        assert not any(g['train_mode'] for g in net.optimizer_.param_groups)

    def test_fit_switches_optimizer_back_and_forth(self, net, data):
        X, y = data
        net.fit(X, y)

        modes = [mode for mode, _ in net.optimizer_.switch_history]
        assert True in modes, "optimizer was never switched to train mode"
        assert False in modes, "optimizer was never switched to eval mode"

        # after fit, the optimizer rests in eval mode, which is when
        # schedule-free optimizers expose their averaged weights
        assert not any(g['train_mode'] for g in net.optimizer_.param_groups)
        assert net.module_.training is False

    def test_predict_reads_the_eval_point_of_the_optimizer(self, net, data):
        """The mode switch is what makes schedule-free optimizers work.

        In eval mode the optimizer writes its averaged weights into the
        parameters. If the net never switched the optimizer, the module would
        keep reading the iterate used for gradients, and that iterate is what
        predictions would come from. Here every weight the module reads during
        prediction must be one the optimizer switched to in eval mode, and
        none of them may be a train-mode weight.
        """
        X, y = data
        net.fit(X, y)

        net.module_.call_log.clear()
        n_switches = len(net.optimizer_.switch_history)
        net.predict(X)

        # fit ended in eval mode, so no further switch is needed here and
        # none must happen -- switching would move the weights away from the
        # averaged ones that were just established
        assert len(net.optimizer_.switch_history) == n_switches

        train_weights = [w for mode, w in net.optimizer_.switch_history
                         if mode]
        eval_weights = [w for mode, w in net.optimizer_.switch_history
                        if not mode]

        for training_mode, module_weight in net.module_.call_log:
            assert training_mode is False
            assert any(
                torch.equal(module_weight, w) for w in eval_weights
            ), "module read a weight that is not an eval point"
            assert not any(
                torch.equal(module_weight, w) for w in train_weights
            ), "module read the train point instead of the eval point"

    def test_validation_uses_eval_mode(self, net, data):
        # the default train split runs validation during fit; the optimizer
        # must not stay in train mode for it
        X, y = data
        net.fit(X, y)
        assert not any(g['train_mode'] for g in net.optimizer_.param_groups)


class TestTrimmedNet:
    def test_trimmed_net_does_not_raise(self):
        # after trim_for_prediction, optimizers are None; _set_training is
        # called on the trimmed net and must not raise
        X = np.arange(20, dtype=np.float32).reshape(-1, 1)
        y = (2 * np.arange(20, dtype=np.float32)).reshape(-1, 1)
        net = NeuralNetRegressor(
            RecordingModule,
            criterion=nn.MSELoss,
            optimizer=ScheduleFreeSGD,
            lr=0.05,
            max_epochs=1,
            batch_size=4,
        ).fit(X, y)

        net.trim_for_prediction()
        # pylint: disable=protected-access
        net._set_training(True)
        net._set_training(False)
        net.predict(X)
