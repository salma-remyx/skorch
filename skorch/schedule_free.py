"""Helpers for optimizers that manage their own training state.

Schedule-free optimizers, as proposed in "The Road Less Scheduled"
(Defazio et al., 2024, https://arxiv.org/abs/2405.15682), do not rely on
a learning rate schedule. Instead, they interpolate between the raw and
the averaged weights internally, which requires them to know whether
they are currently training or evaluating/predicting. For that reason,
they expose ``train()`` and ``eval()`` methods, just like torch modules.

The helpers in this module allow skorch to detect and toggle that
optimizer mode. For optimizers that have no mode support (i.e. most
torch optimizers), they are simply no-ops, so any optimizer can be used
interchangeably.

"""

def optimizer_supports_mode(optimizer):
    """Check whether the optimizer supports a training/evaluation mode.

    An optimizer is considered to support mode switching if it defines
    both a ``train`` and an ``eval`` method. This is the case for
    schedule-free optimizers, whereas regular torch optimizers like
    ``torch.optim.SGD`` don't define these methods.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
      The optimizer to check.

    Returns
    -------
    supports_mode : bool
      Whether the optimizer supports switching between training and
      evaluation mode.

    """
    return callable(getattr(optimizer, 'train', None)) and callable(
        getattr(optimizer, 'eval', None))


def set_optimizer_mode(optimizer, training):
    """Set the optimizer to training or evaluation mode.

    If the optimizer supports mode switching (see
    :func:`.optimizer_supports_mode`), call its ``train()`` or ``eval()``
    method. Otherwise, do nothing. Schedule-free optimizers should be in
    eval mode for validation and prediction, otherwise they evaluate at
    a point that doesn't correspond to the averaged weights.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
      The optimizer whose mode should be set.

    training : bool
      Whether to set the optimizer to training mode (True) or
      evaluation mode (False).

    """
    if not optimizer_supports_mode(optimizer):
        return
    if training:
        optimizer.train()
    else:
        optimizer.eval()
