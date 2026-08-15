"""Support for optimizers that distinguish a training from an evaluation mode.

Most PyTorch optimizers are stateless with respect to training vs evaluation:
``torch.optim.SGD`` or ``torch.optim.Adam`` can be stepped at any time and
never need to be told whether the model is training. A growing family of
optimizers does not have that property. The most prominent example are
schedule-free optimizers (Defazio et al., "The Road Less Scheduled",
https://arxiv.org/abs/2405.15682), which replace the learning rate schedule by
weighted iterate averaging and therefore keep two distinct iterates:

- the gradient point ``y``, at which gradients must be evaluated, and
- the averaged iterate ``x``, which is the one that should be used for
  validation, inference and checkpointing.

Since the parameters themselves hold either ``y`` or ``x`` depending on the
mode, such optimizers expose ``train()`` and ``eval()`` methods that must be
called in lockstep with the corresponding calls on the module. skorch already
funnels every one of those transitions through
:meth:`skorch.net.NeuralNet._set_training`; this module provides the
capability check and the mode switch used there.

Optimizers without the two methods are left completely untouched, so nothing
changes for the vast majority of users.

"""


def supports_train_eval_mode(optimizer):
    """Whether an optimizer distinguishes training from evaluation mode.

    An optimizer is considered to support train/eval mode if it has both a
    ``train`` and an ``eval`` attribute that are callable. Plain PyTorch
    optimizers such as ``torch.optim.SGD`` or ``torch.optim.Adam`` do not, and
    are therefore not affected.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer or any object
      The optimizer to check.

    Returns
    -------
    supports_mode : bool
      True if the optimizer can be switched between training and evaluation
      mode, False otherwise.

    """
    # torch.optim.Optimizer itself has no train/eval methods, so an optimizer
    # exposing them is one that implements the protocol itself (e.g. a
    # schedule-free optimizer, or a wrapper around a plain one).
    return callable(getattr(optimizer, 'train', None)) and callable(
        getattr(optimizer, 'eval', None)
    )


def set_optimizer_mode(optimizer, training):
    """Set training/evaluation mode on an optimizer, if it supports it.

    This is a no-op for optimizers that do not distinguish the two modes, see
    :func:`.supports_train_eval_mode`. Note that a mode switch can be an
    expensive operation for optimizers that implement it, since it may rewrite
    the parameters (e.g. converting them from the gradient point to the
    averaged iterate), but it only does actual work when the mode actually
    changes.

    Parameters
    ----------
    optimizer
      The optimizer whose mode should be set.

    training : bool
      Whether to set the optimizer to training mode (True) or evaluation mode
      (False).

    """
    if not supports_train_eval_mode(optimizer):
        return
    if training:
        optimizer.train()
    else:
        optimizer.eval()
