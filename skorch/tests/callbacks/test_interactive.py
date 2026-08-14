"""Tests for interactive.py"""

import numpy as np
import pytest
from torch import nn

from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint
from skorch.callbacks import InteractiveTraining


@pytest.fixture
def module_cls():
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(4, 2)

        # pylint: disable=unused-argument
        def forward(self, X, **kwargs):
            return self.dense(X)

    return Module


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 4).astype(np.float32)
    y = rng.randint(0, 2, size=40).astype(np.int64)
    return X, y


@pytest.fixture
def net(module_cls):
    return NeuralNetClassifier(
        module_cls,
        max_epochs=2,
        batch_size=8,
        iterator_train__shuffle=False,
        verbose=0,
    )


class TestInteractiveTraining:
    def test_set_lr_knob_from_controller(self, net, data):
        X, y = data
        seen = []

        def controller(metrics, net):
            seen.append(metrics)
            if metrics['batch'] == 2 and metrics['epoch'] == 1:
                return [{'action': 'set_knob', 'name': 'lr', 'value': 0.5}]
            return []

        net.set_params(callbacks=[InteractiveTraining(controller)])
        net.fit(X, y)

        assert net.optimizer_.param_groups[0]['lr'] == 0.5
        # the controller was consulted and received training metrics
        assert seen
        assert 'train_loss' in seen[0]

    def test_stop_action_ends_training_early(self, net, data):
        X, y = data
        net.set_params(max_epochs=20)

        def controller(metrics, net):
            if metrics['batch'] == 3:
                return [{'action': 'stop'}]
            return []

        net.set_params(callbacks=[InteractiveTraining(controller)])
        net.fit(X, y)

        assert len(net.history) == 1
        assert len(net.history[-1]['batches']) == 3

    def test_checkpoint_action_saves_model(self, net, data, tmp_path):
        X, y = data
        checkpoint = Checkpoint(dirname=str(tmp_path), monitor=None)

        def controller(metrics, net):
            if metrics['batch'] == 1:
                return [{'action': 'checkpoint'}]
            return []

        net.set_params(callbacks=[checkpoint, InteractiveTraining(controller)])
        net.fit(X, y)

        assert (tmp_path / 'params.pt').exists()

    def test_checkpoint_action_without_checkpoint_warns(self, net, data):
        X, y = data

        def controller(metrics, net):
            if metrics['batch'] == 1:
                return [{'action': 'checkpoint'}]
            return []

        net.set_params(callbacks=[InteractiveTraining(controller)])
        with pytest.warns(UserWarning, match='checkpoint'):
            net.fit(X, y)

    def test_unknown_knob_raises(self, net, data):
        X, y = data

        def controller(metrics, net):
            return [{'action': 'set_knob', 'name': 'momentum', 'value': 0.9}]

        net.set_params(callbacks=[InteractiveTraining(controller)])
        with pytest.raises(KeyError, match='momentum'):
            net.fit(X, y)

    def test_every_skips_control_points(self, net, data):
        X, y = data
        net.set_params(max_epochs=1)
        calls = []

        net.set_params(callbacks=[
            InteractiveTraining(lambda m, n: calls.append(m) or [], every=2),
        ])
        net.fit(X, y)

        # 40 samples / batch_size 8 = 5 batches; every=2 consults the
        # controller at control points 1, 3, 5
        assert len(calls) == 3
        assert [m['batch'] for m in calls] == [1, 3, 5]

    def test_epoch_control_point(self, net, data):
        X, y = data
        calls = []

        net.set_params(callbacks=[
            InteractiveTraining(
                lambda m, n: calls.append(m) or [], on='on_epoch_end'),
        ])
        net.fit(X, y)

        assert [m['epoch'] for m in calls] == [1, 2]
        assert 'valid_loss' in calls[-1]

    def test_custom_knob(self, net, data):
        X, y = data
        state = {'momentum': 0.0}
        knobs = {
            'momentum': (
                lambda net: state['momentum'],
                lambda net, value: state.update(momentum=value),
            ),
        }

        def controller(metrics, net):
            return [{'action': 'set_knob', 'name': 'momentum', 'value': 0.9}]

        net.set_params(callbacks=[InteractiveTraining(controller, knobs=knobs)])
        net.fit(X, y)

        assert state['momentum'] == 0.9
