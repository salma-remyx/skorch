"""Support for optimizers with distinct training/evaluation modes.

Some third-party optimizers carry state that depends on whether they are
currently used for training or for evaluation. The most prominent family
are the "schedule-free" optimizers (e.g. ``AdamWScheduleFree`` and
``SGDScheduleFree`` from https://github.com/facebookresearch/schedule_free,
see "The Road Less Scheduled", Defazio et al., 2024). Instead of moving
along the iterate itself, they interpolate between two sets of weights,
``x`` for evaluation and ``y`` for training, and switch between them in
their ``train()``/``eval()`` methods. The weights are only at the true
iterate during evaluation, so skipping the mode switch makes a fitted net
report wrong losses and predictions.

skorch already toggles ``module_.train()``/``module_.eval()`` on every
mode transition through :meth:`.NeuralNet._set_training`. The helpers in
this module apply the same toggling to optimizers, so that mode-switching
optimizers can be passed as ``optimizer=`` to any skorch net without
further changes.

Detection is duck-typed and conservative: the base
``torch.optim.Optimizer`` defines neither ``train`` nor ``eval``, so only
optimizers that opted into mode switching are affected. Two calling
conventions are supported, since third-party optimizers disagree:
schedule-free optimizers take no argument, while optimizers that mirror
``torch.nn.Module`` expect a boolean.

"""

from torch.optim import Optimizer

__all__ = ['optimizer_mode', 'set_optimizer_mode']


def optimizer_mode(optimizer):
    """Return whether ``optimizer`` is in training mode.

    Returns True for training mode, False for evaluation mode, and None if
    the optimizer does not track a mode at all (which is the case for plain
    torch optimizers).

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
      The optimizer whose mode should be determined.

    """
    if not isinstance(optimizer, Optimizer):
        return None
    # Schedule-free optimizers store the mode per param group, whereas
    # optimizers that mirror torch.nn.Module expose a `training` attribute.
    train_mode = optimizer.param_groups[0].get('train_mode')
    if train_mode is not None:
        return train_mode
    return getattr(optimizer, 'training', None)


def set_optimizer_mode(optimizer, training):
    """Set the training/evaluation mode on ``optimizer``, if it has one.

    This is a no-op for plain torch optimizers, since
    ``torch.optim.Optimizer`` defines neither ``train`` nor ``eval``. It
    only acts on optimizers that opted into mode switching, notably
    schedule-free optimizers such as ``AdamWScheduleFree``, for which
    forgetting the switch makes evaluation silently use the training
    weights ``y`` instead of the averaged weights ``x``.

    Call this from :meth:`.NeuralNet._set_training`, alongside toggling
    the modules and criteria of the net.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
      The optimizer whose mode should be set.

    training : bool
      Whether to set the optimizer to training mode (True) or evaluation
      mode (False).

    """
    if not isinstance(optimizer, Optimizer):
        return
    if not hasattr(optimizer, 'train') or not hasattr(optimizer, 'eval'):
        return
    if optimizer_mode(optimizer) == training:
        # Mode-switching optimizers interpolate weights on each call;
        # skip the call when the mode doesn't change, since _set_training
        # is invoked for every batch.
        return

    method = optimizer.train if training else optimizer.eval
    try:
        method(training)
    except TypeError:
        # The optimizer follows the schedule-free convention of taking no
        # argument. If this call raises too, mode switching is actually
        # broken and the error should propagate.
        method()
