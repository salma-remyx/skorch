"""Callback that forwards train/eval mode to optimizers which need it.

Some optimizers are stateful across the train/eval distinction, not just
across steps: schedule-free optimizers evaluate at an averaged iterate
but take gradient steps at another one, and signal which they want
through their own ``train()``/``eval()`` methods. skorch toggles the
module, never the optimizer, so this callback forwards the net's phase.

"""

from skorch.callbacks import Callback


__all__ = ['ScheduleFreeMode']


class ScheduleFreeMode(Callback):
    """Propagate train/eval mode to a schedule-free optimizer.

    Schedule-free optimizers compute gradients at one iterate and
    evaluate at another, and signal which one they want through their
    own ``train()``/``eval()`` methods. skorch only ever toggles the
    module, not the optimizer, so this callback forwards the net's
    training/evaluation phases to it.

    The switch happens on the batch rather than the epoch boundary:
    skorch runs the validation pass inside the epoch, after the last
    training batch, so waiting for ``on_epoch_end`` would leave the
    optimizer in train mode while the validation loss is computed.

    It is a no-op for optimizers that don't implement those methods, so
    it is safe to leave attached when swapping optimizers:

    >>> from skorch.schedule_free import ScheduleFreeAdamW
    >>> net = NeuralNetClassifier(
    ...     module, optimizer=ScheduleFreeAdamW,
    ...     callbacks=[ScheduleFreeMode()],
    ... )

    """

    def initialize(self):
        self.train_mode_ = None
        return self

    def _set_mode(self, net, training):
        for name in net._optimizers:
            optimizer = getattr(net, name + '_')
            method = 'train' if training else 'eval'
            if hasattr(optimizer, method):
                getattr(optimizer, method)()
        self.train_mode_ = training

    def on_train_begin(self, net, **kwargs):
        self._set_mode(net, training=True)

    def on_batch_begin(self, net, batch=None, training=None, **kwargs):
        # training is None outside of fit, e.g. during predict, in which
        # case the optimizer should be in eval mode already.
        if training is None:
            return
        self._set_mode(net, training=training)

    def on_train_end(self, net, **kwargs):
        # Leave the net at the averaged iterate, so that predict() and
        # any checkpoint taken after fit() see the right weights.
        self._set_mode(net, training=False)
