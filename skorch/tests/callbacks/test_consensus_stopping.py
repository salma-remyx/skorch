"""Tests for the ConsensusStopping callback.

The tests stub the ``net`` argument, as is common for callback tests in
this suite, so the stopping rule itself is exercised without running a
full training loop.

"""

import pytest


class TestConsensusStopping:

    @pytest.fixture
    def consensus_cls(self):
        from skorch.callbacks import ConsensusStopping
        return ConsensusStopping

    @pytest.fixture
    def callback(self, consensus_cls):
        return consensus_cls(sink=lambda *args: None)

    @pytest.fixture
    def stub_net_cls(self):
        """Build a stand-in for a net with the given loss history.

        The stub's ``history`` supports the two access patterns the
        callback relies on: whole-epoch rows and ``history[:, key]``
        column slices.

        """

        class History(list):
            """History stub supporting row and ``[:, key]`` access."""

            def __getitem__(self, i):
                if isinstance(i, tuple) and isinstance(i[1], str):
                    rows = list.__getitem__(self, slice(None))
                    return [row[i[1]] for row in rows]
                return list.__getitem__(self, i)

        class StubNet:
            """Minimal net with a loss history, verbose off by default."""

            def __init__(self, train_loss, valid_loss):
                self.verbose = False
                self.history = History()
                for epoch, (train, valid) in enumerate(
                        zip(train_loss, valid_loss), start=1):
                    self.history.append({
                        'epoch': epoch,
                        'train_loss': train,
                        'valid_loss': valid,
                    })

        return StubNet

    @pytest.fixture
    def train_loss(self):
        # epoch: 1     2     3     4     5     6     7     8     9
        return [0.90, 0.70, 0.55, 0.45, 0.40, 0.36, 0.33, 0.31, 0.30]

    @pytest.fixture
    def overfitting_net(self, stub_net_cls, train_loss):
        """A net whose valid loss turns upward and keeps rising."""
        return stub_net_cls(
            train_loss,
            valid_loss=[0.95, 0.80, 0.70, 0.65, 0.66, 0.68, 0.71, 0.75, 0.80],
        )

    @pytest.fixture
    def healthy_net(self, stub_net_cls, train_loss):
        """A net whose valid loss keeps decreasing."""
        return stub_net_cls(
            train_loss,
            valid_loss=[0.95, 0.80, 0.70, 0.65, 0.63, 0.62, 0.61, 0.60, 0.60],
        )

    def test_exposed_in_callback_namespace(self, consensus_cls):
        import skorch.callbacks
        assert skorch.callbacks.ConsensusStopping is consensus_cls
        assert 'ConsensusStopping' in skorch.callbacks.__all__

    def test_stops_when_indicators_agree(self, callback, overfitting_net):
        callback.on_train_begin(overfitting_net)
        with pytest.raises(KeyboardInterrupt):
            callback.on_epoch_end(overfitting_net)
        # Several online indicators fired and their firing correlated
        assert sum(callback.latest_indicators_.values()) >= 2
        assert callback.latest_correlation_ > 0.5

    def test_healthy_training_does_not_stop(self, callback, healthy_net):
        callback.on_train_begin(healthy_net)
        callback.on_epoch_end(healthy_net)  # does not raise
        assert callback.latest_correlation_ == 0.0

    def test_waits_until_first_strip_is_complete(
            self, callback, overfitting_net):
        # With the default strip_length=5 there is not enough history to
        # evaluate one strip before epoch 6, so the callback stays idle.
        del overfitting_net.history[5:]
        callback.on_train_begin(overfitting_net)
        callback.on_epoch_end(overfitting_net)  # does not raise
        assert callback.latest_indicators_ == {}

    def test_min_agree_above_firing_indicators_blocks_stop(
            self, callback, overfitting_net):
        # Requiring more agreeing indicators than exist prevents the
        # stop even though the firing patterns are correlated.
        callback.set_params(min_agree=5)
        callback.on_train_begin(overfitting_net)
        callback.on_epoch_end(overfitting_net)  # does not raise
        assert callback.latest_correlation_ > 0.5

    def test_correlation_threshold_above_one_blocks_stop(
            self, callback, overfitting_net):
        callback.set_params(correlation_threshold=1.1)
        callback.on_train_begin(overfitting_net)
        callback.on_epoch_end(overfitting_net)  # does not raise

    def test_message_names_the_agreeing_indicators(
            self, consensus_cls, overfitting_net):
        messages = []
        callback = consensus_cls(sink=messages.append)
        callback.on_train_begin(overfitting_net)
        overfitting_net.verbose = True
        with pytest.raises(KeyboardInterrupt):
            callback.on_epoch_end(overfitting_net)
        assert len(messages) == 1
        assert "online indicators correlated above 0.5" in messages[0]
        assert "generalization_loss" in messages[0]

    def test_monitors_can_be_renamed(
            self, callback, stub_net_cls, train_loss):
        net = stub_net_cls(train_loss, [0.9] * len(train_loss))
        for row in net.history:
            row['loss_tr'] = row.pop('train_loss')
            row['loss_va'] = row.pop('valid_loss')
        callback.set_params(monitor_train='loss_tr', monitor_valid='loss_va')
        callback.on_train_begin(net)
        callback.on_epoch_end(net)
        assert set(callback.latest_indicators_) == {
            'generalization_loss', 'progress', 'overfitting_gap',
            'valid_increase'}
