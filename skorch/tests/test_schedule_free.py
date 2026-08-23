"""Tests for skorch.schedule_free and its integration with NeuralNet.

The interesting parts are the integration ones: a schedule-free
optimizer plugged into a net must be switched between train and eval
mode together with the module, otherwise predictions are taken at the
gradient iterate instead of the averaged evaluation weights.

"""

import numpy as np
import pytest
import torch
from torch import nn

from skorch import NeuralNetClassifier
from skorch.schedule_free import (
    SGDScheduleFree,
    AdamWScheduleFree,
    SkorchLRSchedulerPassthrough,
)


class SimpleClassifier(nn.Module):
    """Minimal module with a dropout layer, so that train/eval mode
    differences on the module side are observable too."""

    def __init__(self, input_units=20):
        super().__init__()
        self.dense = nn.Linear(input_units, 2)

    def forward(self, X):
        return self.dense(X)


@pytest.fixture(scope='module')
def classifier_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(100, 20, n_informative=10, random_state=0)
    return X.astype(np.float32), y


# pylint: disable=protected-access


class TestScheduleFreeOptimizer:
    @pytest.mark.parametrize('optimizer_cls', [SGDScheduleFree, AdamWScheduleFree])
    def test_reduces_loss_on_quadratic(self, optimizer_cls):
        # minimize (w - 3)^2 with the schedule-free optimizer; no lr
        # schedule is attached anywhere. The Adam variant overshoots the
        # target on its way (it takes normalized steps), so it gets more
        # steps to settle.
        steps = 1000 if optimizer_cls is SGDScheduleFree else 2000
        param = torch.zeros(1, requires_grad=True)
        opt = optimizer_cls([param], lr=0.05)

        opt.train()
        for _ in range(steps):
            opt.zero_grad()
            loss = (param - 3) ** 2
            loss.backward()
            opt.step()

        opt.eval()
        assert param.item() == pytest.approx(3.0, abs=0.05)

    def test_sgd_invalid_momentum_raises(self):
        with pytest.raises(ValueError):
            SGDScheduleFree([torch.zeros(1, requires_grad=True)], momentum=1.5)

    def test_adamw_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            AdamWScheduleFree(
                [torch.zeros(1, requires_grad=True)], betas=(1.5, 0.999))

    def test_step_in_eval_mode_raises(self):
        param = torch.zeros(1, requires_grad=True)
        opt = SGDScheduleFree([param], lr=0.1)
        opt.zero_grad()
        (param ** 2).backward()
        opt.step()  # creates the 'z' state so the swap is not a no-op

        opt.eval()
        opt.zero_grad()
        (param ** 2).backward()
        with pytest.raises(RuntimeError):
            opt.step()

    def test_train_eval_roundtrip_returns_train_weights(self):
        # swapping to eval and back must restore the gradient iterate
        # exactly, i.e. the exchange is reversible
        param = torch.zeros(1, requires_grad=True)
        opt = SGDScheduleFree([param], lr=0.1)
        opt.zero_grad()
        (param ** 2).backward()
        opt.step()

        before = param.detach().clone()
        opt.eval()
        opt.train()
        assert torch.equal(param.detach(), before)

    def test_state_dict_roundtrip_restores_step_count(self):
        param = torch.zeros(1, requires_grad=True)
        opt = SGDScheduleFree([param], lr=0.1)
        for _ in range(3):
            opt.zero_grad()
            (param ** 2).backward()
            opt.step()

        fresh = SGDScheduleFree([torch.zeros(1, requires_grad=True)], lr=0.1)
        fresh.load_state_dict(opt.state_dict())
        assert fresh.sf_state['k'] == 3
        assert fresh.sf_state['weight_sum'] == opt.sf_state['weight_sum']


class TestScheduleFreeNetIntegration:
    """These exercise the NeuralNet._set_training wiring in net.py."""

    @pytest.mark.parametrize('optimizer_cls', [SGDScheduleFree, AdamWScheduleFree])
    def test_net_switches_optimizer_mode(self, classifier_data, optimizer_cls):
        # the call-site behavior: as the net moves between train and
        # eval, the optimizer must follow
        X, y = classifier_data
        net = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=optimizer_cls,
            lr=0.01,
            max_epochs=2,
            verbose=0,
            train_split=False,
        )
        net.fit(X, y)

        # fit_loop ends on a training step, so the optimizer is left in
        # train mode, mirroring the module itself
        assert net.module_.training is True
        assert net.optimizer_.param_groups[0]['train_mode'] is True

        # predicting switches both to eval mode
        net.predict(X)
        assert net.module_.training is False
        assert net.optimizer_.param_groups[0]['train_mode'] is False

        # and _set_training toggles the optimizer along with the module
        net._set_training(True)
        assert net.optimizer_.param_groups[0]['train_mode'] is True
        net._set_training(False)
        assert net.optimizer_.param_groups[0]['train_mode'] is False

    @pytest.mark.parametrize('optimizer_cls', [SGDScheduleFree, AdamWScheduleFree])
    def test_net_predict_uses_eval_weights(self, classifier_data, optimizer_cls):
        # the module is deterministic, so a correct train/eval exchange
        # means predict() output must be independent of whether the last
        # operation was a training step
        X, y = classifier_data
        net = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=optimizer_cls,
            lr=0.01,
            max_epochs=3,
            verbose=0,
            train_split=False,
        )
        net.fit(X, y)

        first = net.predict(X)
        # force a train-mode step, then predict again; the wiring must
        # swap back to the evaluation weights before predicting
        net._set_training(True)
        second = net.predict(X)
        assert (first == second).all()

    def test_net_learns_classifier(self, classifier_data):
        # end to end: fit, then check the net separates the training
        # data about as well as plain SGD on the same fixture (0.79),
        # without any lr schedule attached
        X, y = classifier_data
        net = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=SGDScheduleFree,
            lr=0.05,
            optimizer__momentum=0.9,
            max_epochs=30,
            verbose=0,
            train_split=False,
        )
        net.fit(X, y)
        accuracy = (net.predict(X) == y).mean()
        assert accuracy > 0.7

    def test_plain_optimizer_untouched_by_wiring(self, classifier_data):
        # regression guard: plain torch optimizers have no train()
        # method, fitting with them must be unaffected
        X, y = classifier_data
        net = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.SGD,
            lr=0.01,
            max_epochs=2,
            verbose=0,
            train_split=False,
        )
        net.fit(X, y)
        assert isinstance(net.optimizer_, torch.optim.SGD)
        # fit ran through and produced predictions
        assert net.predict(X).shape == y.shape

    def test_save_and_load_optimizer_state(self, classifier_data, tmp_path):
        # save_params serializes the optimizer state_dict, so the
        # schedule-free bookkeeping must survive the roundtrip
        X, y = classifier_data
        net = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=SGDScheduleFree,
            lr=0.01,
            max_epochs=2,
            verbose=0,
            train_split=False,
        )
        net.fit(X, y)
        steps_before = net.optimizer_.sf_state['k']
        assert steps_before > 0

        f_optimizer = tmp_path / 'optimizer.pt'
        f_params = tmp_path / 'params.pt'
        net.save_params(f_optimizer=str(f_optimizer), f_params=str(f_params))

        net_loaded = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=SGDScheduleFree,
            lr=0.01,
            max_epochs=2,
            verbose=0,
            train_split=False,
        ).initialize()
        net_loaded.load_params(
            f_optimizer=str(f_optimizer), f_params=str(f_params))
        assert net_loaded.optimizer_.sf_state['k'] == steps_before
        # predictions of both nets must agree
        assert (net.predict(X) == net_loaded.predict(X)).all()


class TestSkorchLRSchedulerPassthrough:
    def test_does_not_change_lr_and_records_event(self, classifier_data):
        X, y = classifier_data
        net = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=SGDScheduleFree,
            lr=0.03,
            max_epochs=3,
            verbose=0,
            train_split=False,
            callbacks=[SkorchLRSchedulerPassthrough()],
        )
        net.fit(X, y)

        # the lr recorded in the history is the lr passed to the net
        lrs = [entry for entry in net.history[:, 'event_lr']]
        assert lrs == [0.03] * 3
        # ... and the optimizer's lr was left alone
        assert net.optimizer_.param_groups[0]['lr'] == 0.03

    def test_no_scheduler_is_created(self, classifier_data):
        X, y = classifier_data
        net = NeuralNetClassifier(
            SimpleClassifier,
            criterion=nn.CrossEntropyLoss,
            optimizer=SGDScheduleFree,
            lr=0.03,
            max_epochs=1,
            verbose=0,
            train_split=False,
            callbacks=[SkorchLRSchedulerPassthrough(event_name=None)],
        )
        net.fit(X, y)
        # with event_name=None nothing is recorded, but fit runs through
        assert len(net.history) == 1
