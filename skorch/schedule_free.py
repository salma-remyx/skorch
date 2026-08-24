"""Support for optimizers that switch behavior between training and evaluation.

Some optimizers are not just parameter updaters but track state that depends on
whether the net is training or evaluating. The most prominent family is
"schedule-free" optimizers (Defazio et al., 2024, "The Road Less Scheduled"),
which interpolate between a rapidly-moving primal iterate used for gradients
and a slower averaged iterate used for evaluation. Because of that, these
optimizers follow the ``nn.Module`` convention of exposing ``train()`` and
``eval()`` methods, and expect to be switched in lockstep with the module:
gradients should be taken at the primal iterate, predictions at the averaged
one.

Since skorch users pass such optimizers through the ordinary
``optimizer=...`` slot, the net takes care of switching them at the same time
as it does the modules and criteria. This module contains the helper used to
detect that contract.

"""

from torch.optim import Optimizer


__all__ = ['optimizer_switches_modes']


def optimizer_switches_modes(optimizer):
    """Whether an optimizer follows the module-like train/eval contract.

    ``torch.optim.Optimizer`` itself has no ``train`` method, so an optimizer
    is considered to switch modes if it defines one itself (or inherits one
    from a mixin). The schedule-free optimizers of Defazio et al. (2024) are the
    reference implementation of this contract, but any optimizer honoring it is
    supported.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
      The initialized optimizer to check.

    Returns
    -------
    switches : bool
      True if the optimizer should be switched between training and
      evaluation mode by the net.

    """
    if not isinstance(optimizer, Optimizer):
        return False
    return getattr(type(optimizer), 'train', None) is not None
