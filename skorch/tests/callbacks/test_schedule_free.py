"""Tests for schedule_free.py"""
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import SGD

from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint
from skorch.callbacks import ScheduleFree


class ScheduleFreeSGD(SGD):
    """Minimal stand-in for a schedule-free optimizer.

    Implements the train/eval protocol the paper's optimizers expose:
    the parameters hold the training point ("z") while in train mode and
    the averaged evaluation point ("x") while in eval mode, mirroring how
    e.g. ``schedulefree.AdamWScheduleFree`` swaps the two in and out of
    the parameter tensors. Switching is idempotent in both directions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = True
        self.mode_calls = []
        self._z = None

    def _params(self):
        for group in self.param_groups:
            yield from group['params']

    def train(self):
        if not self.train_mode and self._z is not None:
            for param, z in zip(self._params(), self._z):
                param.data.copy_(z)
        self.train_mode = True
        self.mode_calls.append('train')
        return self

    def eval(self):
        if self.train_mode:
            # cache the training point, write the averaged point into the
            # parameters, as the averaging step would
            self._z = [param.data.clone() for param in self._params()]
            for param in self._params():
                param.data.mul_(1.01)
        self.train_mode = False
        self.mode_calls.append('eval')
        return self


@pytest.fixture
def data():
    X, y = [], []
    for class_id in range(2):
        center = 1.0 if class_id else -1.0
        pts = 0.5 * torch.randn(50, 4) + center
        X.append(pts)
        y.append(torch.full((50,), class_id, dtype=torch.long))
    return torch.cat(X), torch.cat(y)


@pytest.fixture
def module_cls():
    class MyModule(nn.Module):
        """Outputs probabilities, as NeuralNetClassifier's NLLLoss expects."""

        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(4, 2)

        # pylint: disable=arguments-differ
        def forward(self, X):
            return torch.softmax(self.dense(X), dim=-1)

    return MyModule


class ObservingScheduleFree(ScheduleFree):
    """ScheduleFree that also records the optimizer mode at epoch end."""

    def initialize(self):
        super().initialize()
        self.epoch_end_modes = []
        return self

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self, net, **kwargs):
        super().on_epoch_end(net, **kwargs)
        self.epoch_end_modes.append(net.optimizer_.train_mode)


class TestSetOptimizerTraining:
    """NeuralNet._set_training toggles optimizers with train/eval."""

    def test_optimizer_follows_module_into_eval_mode(self, module_cls, data):
        # a validation_step or predict leaves a schedule-free optimizer in
        # eval mode, i.e. on the averaged trajectory
        X, y = data
        net = NeuralNetClassifier(
            module_cls, optimizer=ScheduleFreeSGD, max_epochs=1)
        net.fit(X, y)
        net.predict(X)
        assert net.optimizer_.train_mode is False

    def test_optimizer_follows_module_into_train_mode(self, module_cls, data):
        X, y = data
        net = NeuralNetClassifier(
            module_cls, optimizer=ScheduleFreeSGD, max_epochs=1)
        net.fit(X, y)
        # forward(training=True) is what e.g. active learning loops use
        net.forward(X, training=True)
        assert net.optimizer_.train_mode is True

    def test_plain_optimizer_is_untouched(self, module_cls, data):
        # an optimizer without train/eval must not be switched, and must
        # not break anything either
        X, y = data
        net = NeuralNetClassifier(
            module_cls, optimizer=SGD, lr=0.05, max_epochs=1)
        net.fit(X, y)
        net.predict(X)
        assert isinstance(net.optimizer_, SGD)
        assert not isinstance(net.optimizer_, ScheduleFreeSGD)

    def test_trimmed_net_does_not_raise(self, module_cls, data):
        # trim_for_prediction drops the optimizers; _set_training is
        # called after that and must tolerate optimizers being None
        X, y = data
        net = NeuralNetClassifier(
            module_cls, optimizer=ScheduleFreeSGD, max_epochs=1)
        net.fit(X, y)
        net.trim_for_prediction()
        net.predict(X)


class TestScheduleFreeCallback:
    """The callback keeps the net on the evaluation trajectory."""

    @pytest.fixture
    def net_factory(self, module_cls):
        def _make(callbacks, **kwargs):
            return NeuralNetClassifier(
                module_cls,
                optimizer=ScheduleFreeSGD,
                lr=0.05,
                max_epochs=2,
                callbacks=callbacks,
                train_split=False,
                **kwargs
            )
        return _make

    def test_optimizer_in_eval_mode_between_epochs(self, net_factory, data):
        X, y = data
        observer = ObservingScheduleFree()
        net = net_factory([('schedule_free', observer)])
        net.fit(X, y)
        # at the moment epoch-end callbacks (EarlyStopping, Checkpoint)
        # run, the optimizer is already on the averaged trajectory
        assert observer.epoch_end_modes == [False, False]

    def test_optimizer_in_eval_mode_after_fit(self, net_factory, data):
        X, y = data
        net = net_factory([ScheduleFree()])
        net.fit(X, y)
        assert net.optimizer_.train_mode is False

    def test_parameters_change_across_mode_switch(self, net_factory, data):
        # the train/eval switch is not a no-op: the training point and
        # the evaluation point are genuinely different parameters, and
        # switching between the two is lossless in both directions
        X, y = data
        net = net_factory([ScheduleFree()])
        net.fit(X, y)
        optimizer = net.optimizer_
        assert optimizer.train_mode is False

        eval_point = net.module_.dense.weight.data.clone()
        optimizer.train()
        train_point = net.module_.dense.weight.data.clone()
        assert not torch.allclose(eval_point, train_point)

        optimizer.eval()
        assert torch.allclose(eval_point, net.module_.dense.weight.data)

    def test_missing_optimizer_raises(self, net_factory, data):
        X, y = data
        callback = ScheduleFree(optimizer_name='does_not_exist')
        net = net_factory([callback])
        with pytest.raises(ValueError) as exc:
            net.fit(X, y)
        assert "does_not_exist" in str(exc.value)

    def test_syncs_checkpoint_to_eval_trajectory(
            self, net_factory, data, tmp_path):
        # a checkpoint written while the optimizer is in train mode must
        # end up holding the averaged parameters
        X, y = data
        dirname = str(tmp_path)
        checkpoint = Checkpoint(
            monitor=None, dirname=dirname, f_params='params.pt',
            f_optimizer=None, f_history=None, f_pickle=None)
        net = net_factory([('schedule_free', ScheduleFree()), checkpoint])
        net.fit(X, y)

        saved = torch.load(
            '{}/params.pt'.format(dirname), weights_only=True)
        assert torch.allclose(
            saved['dense.weight'], net.module_.dense.weight.data)

    def test_no_checkpoint_resync_without_checkpoint_callback(
            self, net_factory, data):
        # without a Checkpoint there is nothing to re-save; the mode
        # switch itself must still happen
        X, y = data
        callback = ScheduleFree()
        net = net_factory([('schedule_free', callback)])
        net.fit(X, y)
        assert callback.eval_active_ is True

    def test_state_reset_on_reinitialize(self, net_factory, data):
        X, y = data
        callback = ScheduleFree()
        net = net_factory([callback])
        net.fit(X, y)
        net.initialize()
        assert callback.eval_active_ is None


class TestScheduleFreeProtocol:
    """The helper works on any optimizer exposing the protocol."""

    def test_call_mode_returns_false_without_protocol(self):
        from skorch.callbacks.schedule_free import _call_mode
        optimizer = Mock(spec=['step', 'zero_grad'])
        assert _call_mode(optimizer, True) is False

    def test_call_mode_switches_when_protocol_present(self):
        from skorch.callbacks.schedule_free import _call_mode
        optimizer = Mock(spec=['train', 'eval', 'step'])
        assert _call_mode(optimizer, True) is True
        optimizer.train.assert_called_once_with()

    def test_training_loss_decreases(self, module_cls, data):
        # the integration actually optimizes: the schedule-free
        # trajectory reduces the training loss
        X, y = data
        net = NeuralNetClassifier(
            module_cls,
            optimizer=ScheduleFreeSGD,
            lr=0.05,
            max_epochs=5,
            callbacks=[ScheduleFree()],
            train_split=False,
        )
        net.fit(X, y)
        losses = net.history[:, 'train_loss']
        assert losses[-1] < losses[0]
        assert np.isfinite(losses).all()
