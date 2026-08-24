"""Support for optimizers that are themselves train/eval aware.

Some modern optimizers keep their own training/evaluation mode, most
prominently the "Schedule-Free" optimizers of Defazio et al., "The Road Less
Scheduled" (https://arxiv.org/abs/2405.15682). Their update rule interpolates
between the iterate used for gradients and an averaged sequence, and switches
the parameters between the two. That is why the official ``schedulefree``
package asks users to call ``optimizer.train()`` when training starts and
``optimizer.eval()`` before any evaluation, inference or checkpointing --
in eval mode the averaged weights are what ends up in the module.

skorch used to set training/eval mode only on modules and criteria, silently
skipping optimizers. For schedule-free optimizers that means the optimizer is
never told when a fit ends, so predictions keep coming from the iterate used
for gradients instead of the averaged one, losing the accuracy the method is
known for.

The helpers here let a net switch such optimizers along with its modules, so
that a schedule-free optimizer works as a drop-in ``optimizer=`` argument:

>>> import schedulefree  # doctest: +SKIP
>>> import skorch  # doctest: +SKIP
>>> net = skorch.NeuralNetClassifier(  # doctest: +SKIP
...     module, optimizer=schedulefree.AdamWScheduleFree, lr=0.0025,
... )  # doctest: +SKIP

The detection is based on the interface the ``schedulefree`` package uses: a
``train_mode`` entry in each param group, and argument-less ``train()`` and
``eval()`` methods. Optimizers without that interface, such as
``torch.optim.SGD``, are left untouched.

Note that the ``schedulefree`` package also offers ``ScheduleFreeWrapper``,
which wraps the module and flips its mode along with the optimizer's. skorch
already switches the module itself, so the wrapper is not needed -- and it
could not be used through the ``optimizer__`` prefix anyway, since it needs
the module at construction time.
"""

__all__ = ['supports_training_mode', 'set_optimizer_training']


def supports_training_mode(optimizer):
    """Return True if the optimizer tracks a training mode of its own.

    An optimizer is considered train/eval aware if it has a ``train_mode``
    entry in its param groups and argument-less ``train()`` and ``eval()``
    methods to switch between them. This is the interface of the
    schedule-free optimizers of the ``schedulefree`` package.

    """
    param_groups = getattr(optimizer, 'param_groups', None)
    if not isinstance(param_groups, (list, tuple)) or not param_groups:
        return False
    return callable(getattr(optimizer, 'train', None)) and callable(
        getattr(optimizer, 'eval', None)
    ) and all(
        'train_mode' in group for group in param_groups
    )


def set_optimizer_training(optimizer, training=True):
    """Set training/evaluation mode on an optimizer, if it has one.

    Calls ``optimizer.train()`` or ``optimizer.eval()`` for optimizers that
    are train/eval aware (see :func:`.supports_training_mode`), and does
    nothing for plain optimizers such as ``torch.optim.SGD``. ``None`` is
    also accepted and ignored, so that callers don't need to check for it --
    this happens e.g. for the optimizers of a net that was trimmed for
    prediction.

    The optimizer methods are argument-less, so switching to an already
    active mode would still call them. The ``train_mode`` entry of the param
    groups is consulted first to avoid that redundant call, which matters
    because switching is not free: it moves the parameters between the
    iterate used for gradients and the averaged one.

    Parameters
    ----------
    optimizer : torch optimizer or None
      The optimizer whose mode should be set.

    training : bool (default=True)
      Whether to set the optimizer to training mode (True) or evaluation
      mode (False).

    """
    if optimizer is None:
        return
    if not supports_training_mode(optimizer):
        return

    needs_switch = any(
        group['train_mode'] != training for group in optimizer.param_groups
    )
    if needs_switch:
        if training:
            optimizer.train()
        else:
            optimizer.eval()
