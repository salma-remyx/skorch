"""Tests for skorch.schedule_free."""

import numpy as np
import pytest
import torch

from skorch import NeuralNetClassifier
from skorch.callbacks import Callback, ScheduleFreeMode
from skorch.dataset import ValidSplit
from skorch.schedule_free import ScheduleFreeAdamW


def _fit_net(classifier_module, X, y, **kwargs):
    params = dict(
        max_epochs=2,
        lr=0.02,
        verbose=0,
        train_split=ValidSplit(0.2),
        callbacks=[ScheduleFreeMode()],
        optimizer=ScheduleFreeAdamW,
    )
    params.update(kwargs)
    net = NeuralNetClassifier(classifier_module, **params)
    net.fit(X, y)
    return net


@pytest.fixture
def clf_data(classifier_data):
    X, y = classifier_data
    # 200 rows split 80/20 with the default batch size of 128 gives
    # a 40-row validation set, i.e. a single full batch.
    return X[:200], y[:200]


class TestScheduleFreeAdamW:
    @pytest.fixture
    def param_and_optimizer(self):
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = ScheduleFreeAdamW([param], lr=0.05)
        return param, optimizer

    def _descend(self, param, optimizer, n_steps=20):
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()

    def test_zero_state_initialized_like_param(self, param_and_optimizer):
        param, optimizer = param_and_optimizer
        state = optimizer.state[param]
        # z starts out as a copy of the parameter, so that the first
        # average is over identical iterates.
        assert torch.allclose(state['z'], param.data)

    def test_train_mode_by_default(self, param_and_optimizer):
        _, optimizer = param_and_optimizer
        assert optimizer.param_groups[0]['train_mode'] is True

    def test_step_requires_train_mode(self, param_and_optimizer):
        param, optimizer = param_and_optimizer
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.eval()
        with pytest.raises(RuntimeError) as exc:
            optimizer.step()
        expected = "step() was called while the optimizer was in eval mode"
        assert expected in str(exc.value)

    def test_step_moves_towards_minimum(self, param_and_optimizer):
        param, optimizer = param_and_optimizer
        start = param.data.norm().item()
        self._descend(param, optimizer, n_steps=400)
        # Schedule-free steps are small early on: the average only picks
        # up weight as the run progresses, so compare against the start
        # rather than demanding a specific tolerance.
        assert param.data.norm().item() < 0.1 * start

    def test_mode_switch_is_reversible(self, param_and_optimizer):
        param, optimizer = param_and_optimizer
        self._descend(param, optimizer, n_steps=5)

        y = param.data.clone()
        optimizer.eval()
        x = param.data.clone()
        # y and x are distinct iterates that only agree at step 0.
        assert not torch.allclose(y, x)

        optimizer.train()
        assert torch.allclose(param.data, y)

    def test_eval_does_not_change_z(self, param_and_optimizer):
        param, optimizer = param_and_optimizer
        self._descend(param, optimizer, n_steps=5)
        z_before = optimizer.state[param]['z'].clone()

        optimizer.eval()
        optimizer.train()
        assert torch.allclose(optimizer.state[param]['z'], z_before)

    @pytest.mark.parametrize('kwargs', [
        {'lr': -1},
        {'betas': (-0.1, 0.999)},
        {'betas': (0.9, 1.0)},
        {'eps': -1e-8},
        {'weight_decay': -1},
        {'warmup_steps': -3},
    ])
    def test_invalid_arguments_raise(self, kwargs):
        param = torch.nn.Parameter(torch.ones(1))
        with pytest.raises(ValueError):
            ScheduleFreeAdamW([param], **kwargs)

    def test_warmup_limits_first_step(self):
        torch.manual_seed(0)
        no_warmup = torch.nn.Parameter(torch.tensor([1.0]))
        warmup = torch.nn.Parameter(torch.tensor([1.0]))

        opt_plain = ScheduleFreeAdamW([no_warmup], lr=0.1)
        opt_warm = ScheduleFreeAdamW([warmup], lr=0.1, warmup_steps=10)

        for optimizer, param in ((opt_plain, no_warmup), (opt_warm, warmup)):
            optimizer.zero_grad()
            (param ** 2).sum().backward()
            optimizer.step()

        # With warmup, only a fraction of the learning rate applies.
        assert opt_warm.param_groups[0]['lr_max'] < opt_plain.param_groups[0]['lr_max']
        assert warmup.data.norm() > no_warmup.data.norm()

    def test_weight_decay_pulls_params_to_zero(self):
        torch.manual_seed(0)
        no_decay = torch.nn.Parameter(torch.tensor([10.0]))
        decay = torch.nn.Parameter(torch.tensor([10.0]))

        opt_plain = ScheduleFreeAdamW([no_decay], lr=0.01)
        opt_decay = ScheduleFreeAdamW([decay], lr=0.01, weight_decay=0.1)

        for _ in range(50):
            for optimizer, param in ((opt_plain, no_decay), (opt_decay, decay)):
                optimizer.zero_grad()
                (param ** 2).sum().backward()
                optimizer.step()

        assert decay.data.norm() < no_decay.data.norm()

    def test_works_with_multiple_param_groups(self):
        torch.manual_seed(0)
        p1 = torch.nn.Parameter(torch.tensor([1.0]))
        p2 = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = ScheduleFreeAdamW(
            [{'params': [p1], 'lr': 0.1}, {'params': [p2], 'lr': 0.01}])
        for _ in range(10):
            optimizer.zero_grad()
            ((p1 ** 2).sum() + (p2 ** 2).sum()).backward()
            optimizer.step()
        # The higher-lr group converges faster.
        assert p1.data.norm() < p2.data.norm()

    def test_state_survives_pickle_roundtrip(self, classifier_module, clf_data):
        import pickle

        net = _fit_net(classifier_module, *clf_data)
        X, _ = clf_data
        X = X[:10]
        y_pred_before = torch.as_tensor(net.predict(X))

        net_restored = pickle.loads(pickle.dumps(net))
        y_pred_after = torch.as_tensor(net_restored.predict(X))
        assert torch.allclose(y_pred_before, y_pred_after)
        assert net_restored.optimizer_.param_groups[0]['train_mode'] is False


class TestScheduleFreeMode:
    @pytest.fixture
    def net(self, classifier_module, clf_data):
        return _fit_net(classifier_module, *clf_data)

    def test_optimizer_in_eval_mode_after_fit(self, net):
        # The averaged iterate is what should be used after training.
        assert net.optimizer_.param_groups[0]['train_mode'] is False

    def test_params_leave_train_iterate_after_fit(self, net):
        # After fit the parameters must no longer sit at y, the point
        # gradients are computed at, but at x, the averaged iterate.
        param = next(net.module_.parameters())
        z = net.optimizer_.state[param]['z']
        beta1 = net.optimizer_.param_groups[0]['betas'][0]

        net.optimizer_.train()
        y = param.data.clone()
        net.optimizer_.eval()

        assert not torch.allclose(param.data, y)
        # x is recovered from the (y, z) pair, so it must reproduce
        # exactly when re-entering eval mode.
        assert torch.allclose(param.data, y.lerp(z, 1 - 1 / beta1))

    def test_mode_switches_between_train_and_valid_batches(
            self, classifier_module, clf_data):
        modes = []

        class Recorder(Callback):
            def on_batch_begin(self, net, batch=None, training=None, **kwargs):
                modes.append(
                    (training, net.optimizer_.param_groups[0]['train_mode']))

        net = _fit_net(
            classifier_module, *clf_data, callbacks=[ScheduleFreeMode(), Recorder()])
        X, y = clf_data
        net.partial_fit(X, y)

        # Every training batch sees train mode, every validation batch
        # sees eval mode; the two must agree element-wise.
        assert modes
        assert all(train_mode == training for training, train_mode in modes)
        assert {training for training, _ in modes} == {True, False}

    def test_callback_is_noop_for_regular_optimizer(
            self, classifier_module, clf_data):
        X, y = clf_data
        net = NeuralNetClassifier(
            classifier_module,
            max_epochs=1,
            lr=0.02,
            verbose=0,
            train_split=None,
            callbacks=[ScheduleFreeMode()],
        )
        # Plain SGD has no train()/eval(); fitting must still work.
        net.fit(X, y)
        assert len(net.history) == 1

    def test_loss_decreases(self, classifier_module, clf_data):
        X, y = clf_data
        net = _fit_net(classifier_module, X, y, max_epochs=5)
        train_losses = [h['train_loss'] for h in net.history]
        assert train_losses[-1] < train_losses[0]

    def test_accuracy_beats_chance(self, classifier_module, classifier_data):
        X, y = classifier_data
        net = _fit_net(
            classifier_module, X, y, max_epochs=20, train_split=None)
        assert np.mean(net.predict(X) == y) > 0.8

    def test_params_dont_change_during_evaluation(self, net, clf_data):
        X, _ = clf_data
        params_before = [p.data.clone() for p in net.module_.parameters()]
        net.predict(X[:10])
        params_after = [p.data for p in net.module_.parameters()]
        for before, after in zip(params_before, params_after):
            assert torch.allclose(before, after)

    def test_warm_start_resumes_training(self, net, clf_data):
        n_epochs = len(net.history)
        X, y = clf_data
        net.partial_fit(X, y)
        assert len(net.history) == n_epochs + 2
        # Still at the averaged iterate once training ends.
        assert net.optimizer_.param_groups[0]['train_mode'] is False
