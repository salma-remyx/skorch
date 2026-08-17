"""Tests for schedule-free optimizer support (schedule_free.py + net.py)"""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
from skorch.callbacks import ScheduleFreeMode


#################
# fake schedule-free optimizer #
#################


class FakeScheduleFreeSGD(torch.optim.SGD):
    """Mimics the train/eval contract of schedule-free optimizers.

    The real ones (arXiv:2405.15682) interpolate parameters between the
    training iterate and an averaged iterate; for these tests it is enough
    to record the mode transitions.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_calls = []
        self.train_mode = True

    def train(self):
        self.mode_calls.append('train')
        self.train_mode = True

    def eval(self):
        self.mode_calls.append('eval')
        self.train_mode = False


@pytest.fixture
def module_cls():
    """Small regressor module."""

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(1, 1)

        def forward(self, X):
            return self.dense(X)

    return MyModule


@pytest.fixture
def data():
    X = np.array([0, 2, 3, 0, 1, 5]).astype(np.float32).reshape(-1, 1)
    y = np.array([-1, 0, 5, 4, 2, 3]).astype(np.float32).reshape(-1, 1)
    return X, y


@pytest.fixture
def net(module_cls, data):
    """A fitted net whose optimizer records train/eval calls."""
    X, y = data
    net = NeuralNetRegressor(
        module_cls,
        max_epochs=2,
        lr=0.01,
        optimizer=FakeScheduleFreeSGD,
        batch_size=3,
        verbose=0,
    )
    net.fit(X, y)
    return net


class TestSetTrainingPropagatesToOptimizer:
    """The call site: net._set_training must switch optimizer mode too."""

    def test_optimizer_mode_follows_module_mode(self, net):
        net._set_training(True)
        assert net.optimizer_.train_mode is True
        net._set_training(False)
        assert net.optimizer_.train_mode is False

    def test_optimizer_eval_called_during_validation(self, net):
        # training alternates between module train mode (train batches) and
        # eval mode (validation batches); the optimizer must follow along
        calls = net.optimizer_.mode_calls
        assert 'train' in calls
        assert 'eval' in calls

    def test_plain_optimizer_unaffected(self, module_cls, data):
        # a regular optimizer has no train/eval and must not break
        X, y = data
        net = NeuralNetRegressor(
            module_cls, max_epochs=1, lr=0.01, verbose=0)
        net.fit(X, y)
        assert net.module_.training is False
        net.predict(X)

    def test_trim_for_prediction_does_not_raise(self, net, data):
        # optimizer is set to None when trimmed; _set_training must skip it
        net.trim_for_prediction()
        assert net.optimizer_ is None
        net._set_training(False)


class TestScheduleFreeModeCallback:
    """The callback guards the schedule-free contract."""

    def test_detects_schedule_free_optimizer(self, net):
        callback = ScheduleFreeMode()
        callback.initialize()
        callback.on_train_begin(net)
        assert callback.found_schedule_free_ == ['optimizer']

    def test_warns_when_optimizer_not_schedule_free(
            self, module_cls, data):
        X, y = data
        net = NeuralNetRegressor(
            module_cls,
            max_epochs=1,
            lr=0.01,
            verbose=0,
            callbacks=[ScheduleFreeMode()],
        )
        with pytest.warns(UserWarning, match='schedule-free'):
            net.fit(X, y)

    def test_no_warning_when_disabled(self, module_cls, data):
        X, y = data
        net = NeuralNetRegressor(
            module_cls,
            max_epochs=1,
            lr=0.01,
            verbose=0,
            callbacks=[ScheduleFreeMode(warn_if_unsupported=False)],
        )
        with patch('skorch.callbacks.schedule_free.warnings.warn') as warn:
            net.fit(X, y)
        assert not warn.called

    def test_raises_when_scheduler_callback_present(self, net):
        # sneak an LRScheduler into the fitted net's callbacks, then
        # re-trigger on_train_begin as fit would
        net.callbacks_.append(
            ('lr_scheduler', LRScheduler(policy='ExponentialLR', gamma=0.1)))
        callback = ScheduleFreeMode()
        callback.initialize()
        with pytest.raises(ValueError, match='scheduler'):
            callback.on_train_begin(net)

    def test_scheduler_conflict_warns_when_not_enforced(self, net):
        net.callbacks_.append(
            ('lr_scheduler', LRScheduler(policy='ExponentialLR', gamma=0.1)))
        callback = ScheduleFreeMode(enforce_no_scheduler=False)
        callback.initialize()
        with pytest.warns(UserWarning, match='scheduler'):
            callback.on_train_begin(net)

    def test_scheduler_alone_is_no_error(self, module_cls, data):
        # an LRScheduler on a plain optimizer is a normal, valid setup
        X, y = data
        net = NeuralNetRegressor(
            module_cls,
            max_epochs=1,
            lr=0.01,
            verbose=0,
            callbacks=[
                ('lr_scheduler', LRScheduler(
                    policy='ExponentialLR', gamma=0.1)),
                ScheduleFreeMode(warn_if_unsupported=False),
            ],
        )
        net.fit(X, y)
        assert len(net.history) == 1

    def test_callback_is_cloneable(self):
        from sklearn.base import clone
        clone(ScheduleFreeMode())

    def test_callback_params_roundtrip(self):
        callback = ScheduleFreeMode(enforce_no_scheduler=False)
        params = callback.get_params()
        assert params['enforce_no_scheduler'] is False
        assert params['warn_if_unsupported'] is True

    def test_initialize_resets_state(self, net):
        callback = ScheduleFreeMode(warn_if_unsupported=False)
        callback.on_train_begin(net)
        assert callback.found_schedule_free_
        callback.initialize()
        assert callback.found_schedule_free_ == []
