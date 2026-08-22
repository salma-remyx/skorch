"""Tests for callbacks in indicator_correlation.py"""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from skorch import NeuralNetClassifier
from skorch.callbacks import IndicatorCorrelationStopping
from skorch.helper import predefined_split


class _History(list):
    """Records list supporting ``history[-n, column]`` indexing."""

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index, column = item
            return list.__getitem__(self, index)[column]
        return list.__getitem__(self, item)


class _Net:
    """Minimal net stand-in, exposing only what the callback reads."""

    verbose = False

    def __init__(self, records):
        self.history = _History(records)


def _loss_records(train_loss, valid_loss):
    return [
        {'epoch': i + 1, 'train_loss': float(t), 'valid_loss': float(v)}
        for i, (t, v) in enumerate(zip(train_loss, valid_loss))
    ]


def _overfitting_records(n_epochs=30, best_epoch=10):
    """Losses that overfit: valid improves until `best_epoch`, then rises."""
    epochs = np.arange(1, n_epochs + 1)
    train_loss = 1.0 / epochs
    valid_loss = np.where(
        epochs <= best_epoch, 1.2 / epochs, 0.12 + 0.02 * (epochs - best_epoch))
    return _loss_records(train_loss, valid_loss)


def _converging_records(n_epochs=30):
    """Losses that keep improving; no overfitting to detect."""
    epochs = np.arange(1, n_epochs + 1)
    return _loss_records(1.0 / epochs, 1.2 / epochs)


class TestIndicatorCorrelationStopping:

    @pytest.fixture
    def indicator_stopping_cls(self):
        return IndicatorCorrelationStopping

    def test_stops_when_indicators_correlate(
            self, indicator_stopping_cls):
        # In the overfitting regime, gl, pq and up all track the rising
        # validation loss, so their correlation crosses the threshold and
        # training stops.
        records = _overfitting_records()
        callback = indicator_stopping_cls(
            threshold=0.8, sink=lambda *_: None)
        callback.on_train_begin(_Net(records[:1]))

        with pytest.raises(KeyboardInterrupt):
            for n_epochs in range(1, len(records) + 1):
                callback.on_epoch_end(_Net(records[:n_epochs]))

        assert len(_Net(records).history) == len(records)
        assert callback.correlation_ > 0.8

    def test_no_stop_while_validation_improves(
            self, indicator_stopping_cls):
        # While the validation loss keeps setting new bests, the
        # indicators carry no signal and training must not be stopped.
        records = _converging_records()
        callback = indicator_stopping_cls(
            threshold=0.5, sink=lambda *_: None)
        callback.on_train_begin(_Net(records[:1]))

        for n_epochs in range(1, len(records) + 1):
            callback.on_epoch_end(_Net(records[:n_epochs]))

        assert callback.misses_ == 0
        assert callback.correlation_ == 0.0

    def test_stops_later_with_higher_patience(
            self, indicator_stopping_cls):
        # Requiring the condition to hold for more epochs in a row
        # delays the stop, it does not remove it.
        records = _overfitting_records()
        stop_epochs = []
        for patience in (1, 3, 5):
            callback = indicator_stopping_cls(
                threshold=0.8, patience=patience, sink=lambda *_: None)
            callback.on_train_begin(_Net(records[:1]))
            for n_epochs in range(1, len(records) + 1):
                try:
                    callback.on_epoch_end(_Net(records[:n_epochs]))
                except KeyboardInterrupt:
                    stop_epochs.append(n_epochs)
                    break
            else:
                stop_epochs.append(None)

        assert stop_epochs[0] is not None
        assert stop_epochs == sorted(
            e for e in stop_epochs if e is not None)

    def test_correlation_is_floored_at_zero(
            self, indicator_stopping_cls):
        # Anti-correlated indicators disagree and hence point at no
        # common cause; the reported correlation cannot be negative.
        callback = indicator_stopping_cls()
        callback.on_train_begin(_Net([{'train_loss': 0.0, 'valid_loss': 0.0}]))

        correlation = callback._indicator_correlation(
            _Net(_overfitting_records()))

        assert 0.0 <= correlation <= 1.0

    @pytest.mark.parametrize('kwargs, match', [
        ({'strip_length': 1}, 'strip_length'),
        ({'threshold': 0.0}, 'threshold'),
        ({'threshold': 1.5}, 'threshold'),
    ])
    def test_invalid_arguments_raise(
            self, indicator_stopping_cls, kwargs, match):
        callback = indicator_stopping_cls(**kwargs)
        with pytest.raises(ValueError, match=match):
            callback.on_train_begin(_Net([]))

    def test_typical_use_case(
            self, indicator_stopping_cls, classifier_module,
            classifier_data):
        # Integration: the callback reads the losses the net itself
        # records during fit and stops a non-improving net early.
        from sklearn.model_selection import train_test_split

        X_tr, X_val, y_tr, y_val = train_test_split(
            *classifier_data, random_state=0)
        train_dataset = TensorDataset(
            torch.as_tensor(X_tr).float(), torch.as_tensor(y_tr))
        valid_dataset = TensorDataset(
            torch.as_tensor(X_val).float(), torch.as_tensor(y_val))

        max_epochs = 20
        callback = indicator_stopping_cls(sink=lambda *_: None)
        net = NeuralNetClassifier(
            classifier_module,
            max_epochs=max_epochs,
            train_split=predefined_split(valid_dataset),
            callbacks=[callback],
        )
        net.fit(train_dataset)

        assert 0 < len(net.history) <= max_epochs
        assert 0.0 <= callback.correlation_ <= 1.0

    def test_sink_receives_message_on_stop(
            self, indicator_stopping_cls):
        records = _overfitting_records()
        messages = []

        class _VerboseNet(_Net):
            verbose = True

        callback = indicator_stopping_cls(sink=messages.append)
        callback.on_train_begin(_Net(records[:1]))

        with pytest.raises(KeyboardInterrupt):
            for n_epochs in range(1, len(records) + 1):
                callback.on_epoch_end(_VerboseNet(records[:n_epochs]))

        assert messages
        assert 'correlated' in messages[0]
