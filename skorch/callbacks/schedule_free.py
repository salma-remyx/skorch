"""Callback for schedule-free optimization.

Schedule-free optimizers (Defazio et al., "The Road Less Scheduled",
arXiv:2405.15682) remove the learning rate schedule -- and with it the
need to know the stopping step in advance -- by interleaving momentum
and averaging. The price is that they train at one point of the
trajectory (the "z" sequence) while they should be evaluated,
checkpointed, and inferred with at another (the averaged "x"
sequence), and that the switch between the two is signalled through
``optimizer.train()``/``optimizer.eval()``.

:class:`~skorch.callbacks.ScheduleFree` takes care of the bookkeeping
that skorch would otherwise get wrong around such an optimizer:

* during the batches, the optimizer is switched to train/eval mode
  together with the module (``NeuralNet._set_training`` does this on
  its own, so a bare optimizer already gets the basic switching);
* the parameters are put back on the evaluation trajectory at the end
  of each epoch, so that epoch-end callbacks -- ``EarlyStopping``,
  ``Checkpoint``, ``EpochScoring`` -- and the epochs in between see
  the averaged parameters rather than the training ones;
* saved checkpoints are normalized back to the evaluation point, so
  that a checkpoint written while the optimizer happened to be in
  train mode still loads into an inference-ready model.

Use it together with a schedule-free optimizer such as those provided
by the ``schedulefree`` package::

    import schedulefree

    net = NeuralNetClassifier(
        module,
        optimizer=schedulefree.AdamWScheduleFree,
        lr=0.0025,
        callbacks=[ScheduleFree()],
    )

Note that no learning rate scheduler callback is needed or useful with
a schedule-free optimizer; a warmup can still be layered on top if the
optimizer doesn't provide it itself.
"""

from skorch.callbacks import Callback


__all__ = ['ScheduleFree']


class ScheduleFree(Callback):
    """Keep a schedule-free optimizer on its evaluation trajectory.

    The optimizer is switched to eval mode at the end of each epoch and
    back to train mode at the start of the next one, so that the
    parameters the rest of skorch observes between epochs are the
    averaged ones the optimizer is meant to be evaluated at. During the
    batches themselves the switching is left to
    ``NeuralNet._set_training``, which keeps train and validation steps
    on their respective trajectories. Checkpoints saved in train mode
    are re-written at the evaluation point when training ends.

    Use this callback with an optimizer that implements the
    schedule-free ``train()``/``eval()`` protocol, e.g.
    ``schedulefree.AdamWScheduleFree``. It is not needed for optimizers
    that don't have the train/eval distinction.

    Parameters
    ----------
    optimizer_name : str (default='optimizer')
      Name of the optimizer to control, in case the net uses more than
      one (e.g. ``'optimizer2'``).

    sync_checkpoints : bool (default=True)
      Whether to re-write checkpoints saved by a ``Checkpoint``
      callback at the evaluation point once training ends. Disable this
      if you want the files to keep whatever point they were saved at.

    Attributes
    ----------
    eval_active_ : bool or None
      Whether the optimizer is currently in eval mode. ``None`` until
      the callback has seen an epoch.

    """

    def __init__(
            self,
            optimizer_name='optimizer',
            sync_checkpoints=True,
    ):
        self.optimizer_name = optimizer_name
        self.sync_checkpoints = sync_checkpoints

    def initialize(self):
        self.eval_active_ = None
        return self

    # pylint: disable=unused-argument
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # the epoch's training batches run on the training trajectory
        self.eval_active_ = False
        self._train(net)

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # epoch-end callbacks (EarlyStopping, Checkpoint, EpochScoring)
        # run after this, on the averaged parameters
        self.eval_active_ = True
        self._eval(net)

    def on_train_end(self, net, **kwargs):
        # leave the net inference-ready once fitting is over; this also
        # covers a training that was interrupted mid-epoch
        if self.eval_active_ is not True:
            self.eval_active_ = True
            self._eval(net)
        if self.sync_checkpoints:
            _sync_saved_checkpoints(net, self._eval)

    def _optimizer(self, net):
        optimizer = getattr(net, self.optimizer_name + '_', None)
        if optimizer is None:
            raise ValueError(
                "ScheduleFree: optimizer '{}' is not initialized on this "
                "net.".format(self.optimizer_name))
        return optimizer

    def _train(self, net):
        _call_mode(self._optimizer(net), True)

    def _eval(self, net):
        _call_mode(self._optimizer(net), False)


def _call_mode(optimizer, training):
    """Switch an optimizer that supports the train/eval protocol."""
    method = getattr(optimizer, 'train' if training else 'eval', None)
    if not callable(method):
        # an optimizer without the distinction is switched by
        # NeuralNet._set_optimizer_training, and is a no-op here
        return False
    method()
    return True


def _sync_saved_checkpoints(net, set_eval):
    """Make checkpoints written in train mode land on the eval point.

    ``Checkpoint`` saves through ``net.save_params``, so the parameter
    files of any callback that saved while the optimizer was in train
    mode now hold training-point weights. Re-saving them with the
    optimizer in eval mode rewrites them at the averaged point, which
    is where an inference-ready checkpoint belongs.
    """
    for _, callback in net.callbacks_:
        get_formatted_files = getattr(callback, 'get_formatted_files', None)
        if not callable(get_formatted_files):
            continue
        for f_name, f in get_formatted_files(net).items():
            if (f_name.startswith('f_') and f is not None
                    and f_name != 'f_history'):
                set_eval(net)
                net.save_params(**{f_name: f})
