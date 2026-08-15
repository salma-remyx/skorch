"""Tests for optimizers that distinguish training from evaluation mode.

Schedule-free optimizers (Defazio et al., "The Road Less Scheduled",
https://arxiv.org/abs/2405.15682) evaluate gradients at an iterate ``y`` that
differs from the averaged iterate ``x`` used for validation and prediction.
They therefore expose ``train()``/``eval()`` methods that rewrite the
parameters, and those calls must happen in lockstep with the module's mode
switches. The tests below use a small optimizer that reproduces exactly that
behavior.

"""
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from skorch import NeuralNetRegressor
from skorch.optimizer_mode import set_optimizer_mode
from skorch.optimizer_mode import supports_train_eval_mode
from skorch.toy import make_regressor


class ScheduleFreeSGD(Optimizer):
    """Minimal schedule-free optimizer, modeled on the reference implementation.

    Keeps an averaged iterate ``z`` in its state and swaps the parameters
    between the gradient point ``y`` (train mode) and the averaged iterate
    ``x`` (eval mode), which is all that is needed to test the mode wiring.

    """

    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.train_mode = False

    @torch.no_grad()
    def _lerp_params(self, weight):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'z' not in state:
                    continue
                p.lerp_(end=state['z'].to(p.device), weight=weight)

    @property
    def momentum(self):
        return self.param_groups[0]['momentum']

    @torch.no_grad()
    def train(self):
        """Set the parameters to the gradient point y."""
        if not self.train_mode:
            # Set p to y
            self._lerp_params(1 - self.momentum)
            self.train_mode = True

    @torch.no_grad()
    def eval(self):
        """Set the parameters to the averaged iterate x."""
        if self.train_mode:
            # Set p to x
            self._lerp_params(1 - 1 / self.momentum)
            self.train_mode = False

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise RuntimeError(
                "Optimizer was not in train mode when step is called. Please "
                "insert .train() and .eval() calls on the optimizer."
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        lr = self.param_groups[0]['lr']
        momentum = self.momentum
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['k'] = 0
                # Interpolation weight, decaying with the step count. In the
                # reference implementation this is weight/weight_sum, which
                # shrinks as more steps are averaged in; the decay is what
                # makes y and z two genuinely different iterates.
                ckp1 = 1.0 / (state['k'] + 2)
                # Apply step to z: plain SGD on the averaged iterate
                state['z'].sub_(p.grad, alpha=lr)
                # Apply step to y in place
                p.lerp_(end=state['z'], weight=ckp1)
                p.add_(p.grad, alpha=lr * (momentum * (1 - ckp1) - 1))
                state['k'] += 1
        return loss


class TestSupportsTrainEvalMode:
    @pytest.mark.parametrize('optimizer', [
        torch.optim.SGD,
        torch.optim.Adam,
        torch.optim.AdamW,
        torch.optim.RMSprop,
    ])
    def test_plain_optimizers_do_not_support_it(self, optimizer):
        module = nn.Linear(2, 1)
        opt = optimizer(module.parameters(), lr=0.1)
        assert not supports_train_eval_mode(opt)

    def test_schedule_free_optimizer_supports_it(self):
        module = nn.Linear(2, 1)
        opt = ScheduleFreeSGD(module.parameters())
        assert supports_train_eval_mode(opt)

    def test_object_without_the_methods(self):
        assert not supports_train_eval_mode(object())
        assert not supports_train_eval_mode(None)

    def test_non_callable_attributes_do_not_count(self):
        optimizer = Mock(spec=[])
        optimizer.train = 'not callable'
        optimizer.eval = 'not callable'
        assert not supports_train_eval_mode(optimizer)


class TestSetOptimizerMode:
    def test_noop_for_plain_optimizer(self):
        module = nn.Linear(2, 1)
        opt = torch.optim.SGD(module.parameters(), lr=0.1)
        # A plain optimizer has no 'train' attribute at all; if the mode
        # switch tried to call it, this would raise an AttributeError.
        set_optimizer_mode(opt, True)
        assert not hasattr(opt, 'train')

    def test_calls_train_or_eval(self):
        module = nn.Linear(2, 1)
        opt = ScheduleFreeSGD(module.parameters())
        set_optimizer_mode(opt, True)
        assert opt.train_mode is True
        set_optimizer_mode(opt, False)
        assert opt.train_mode is False


class TestSetTrainingPropagatesToOptimizer:
    @pytest.fixture(scope='module')
    def module_cls(self):
        return make_regressor(input_units=1, output_units=1)

    @pytest.fixture
    def data(self):
        X = np.array([0, 2, 3, 0, 1, 4]).astype(np.float32).reshape(-1, 1)
        y = np.array([-1, 0, 5, 4, 2, 3]).astype(np.float32).reshape(-1, 1)
        return X, y

    @pytest.fixture
    def net(self, module_cls, data):
        net = NeuralNetRegressor(
            module_cls,
            max_epochs=1,
            lr=0.1,
            optimizer=ScheduleFreeSGD,
            batch_size=4,
            verbose=0,
        )
        return net.fit(*data)

    def test_optimizer_mode_follows_module_mode(self, net):
        # pylint: disable=protected-access
        net._set_training(True)
        assert net.module_.training is True
        assert net.optimizer_.train_mode is True

        net._set_training(False)
        assert net.module_.training is False
        assert net.optimizer_.train_mode is False

    def test_fitting_leaves_optimizer_in_eval_mode(self, net):
        # After fit, the net is set to eval mode for later prediction; the
        # optimizer must have followed, so that the parameters hold the
        # averaged iterate rather than the gradient point.
        assert net.module_.training is False
        assert net.optimizer_.train_mode is False

    def test_steps_all_ran_in_train_mode(self, net):
        # ScheduleFreeSGD.step raises if it is not in train mode, so the fact
        # that fit completed means every step happened in train mode.
        assert len(net.history) == 1
        assert net.history[0, 'train_batch_count'] > 0

    def test_eval_mode_holds_the_averaged_iterate(self, net, data):
        # pylint: disable=protected-access
        net._set_training(True)
        gradient_point = {
            k: v.clone() for k, v in net.module_.state_dict().items()
        }
        net._set_training(False)
        averaged = net.module_.state_dict()

        # Eval mode must leave the parameters at the averaged iterate x, not
        # at the gradient point y where the gradients were computed.
        assert any(
            not torch.allclose(averaged[k], v) for k, v in gradient_point.items()
        )

        # And switching back must restore the gradient point exactly, so that
        # the next train step continues from where it left off.
        net._set_training(True)
        restored = net.module_.state_dict()
        assert all(
            torch.allclose(restored[k], v) for k, v in gradient_point.items()
        )

    def test_predict_works_after_training(self, net, data):
        # predict() switches to eval mode internally; the optimizer must not
        # interfere with that.
        y_pred = net.predict(data[0])
        assert y_pred.shape == data[1].shape

    def test_plain_optimizer_is_not_touched(self, module_cls, data):
        net = NeuralNetRegressor(
            module_cls,
            max_epochs=1,
            lr=0.1,
            optimizer=torch.optim.SGD,
            batch_size=4,
            verbose=0,
        )
        net.fit(*data)
        # pylint: disable=protected-access
        assert not hasattr(net.optimizer_, 'train_mode')
        assert not hasattr(net.optimizer_, 'train')
        # Setting the mode must not raise even though the optimizer has no
        # train/eval methods at all.
        net._set_training(True)
        assert net.module_.training is True
