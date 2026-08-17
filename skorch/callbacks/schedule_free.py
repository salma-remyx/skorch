"""Callback for schedule-free optimizers.

Implements the integration contract required by schedule-free optimizers
(Defazio et al., "The Road Less Scheduled", 2024, arXiv:2405.15682), such as
the ones provided by the ``schedulefree`` package: the optimizer must be
notified of train/eval transitions, and learning rate schedulers must not be
used on top of it.
"""

import warnings

from skorch.callbacks import Callback


__all__ = ['ScheduleFreeMode']


class ScheduleFreeMode(Callback):
    """Guard rail callback for schedule-free optimizers.

    Schedule-free optimizers (arXiv:2405.15682) replace the learning rate
    schedule with an interpolation between the current iterate and an
    averaged iterate. They need two things from the surrounding training
    loop:

    1. ``optimizer.train()`` before training steps and
       ``optimizer.eval()`` before evaluation, mirroring the module's own
       train/eval mode. :meth:`.NeuralNet._set_training` takes care of this
       automatically for any optimizer that defines those methods.
    2. No learning rate scheduler stepping on top of them, since there is
       no schedule to decay -- adding one fights the interpolation.

    This callback makes requirement 1 explicit and enforces requirement 2
    by raising as soon as a scheduler callback is detected alongside a
    schedule-free optimizer.

    >>> from skorch.callbacks import ScheduleFreeMode
    >>> net = NeuralNetClassifier(  # doctest: +SKIP
    ...     ...,
    ...     callbacks=[ScheduleFreeMode()],
    ... )

    Parameters
    ----------
    enforce_no_scheduler : bool (default=True)
      Whether to raise a ``ValueError`` if an ``LRScheduler`` callback is
      found alongside a schedule-free optimizer. Set to False to only get
      a warning.

    warn_if_unsupported : bool (default=True)
      Whether to warn if none of the net's optimizers define the
      ``train``/``eval`` methods, which usually means the optimizer is not
      schedule-free and this callback has nothing to do.

    Attributes
    ----------
    found_schedule_free_ : list of str
      Names of the optimizers that were identified as schedule-free.

    """
    def __init__(self, enforce_no_scheduler=True, warn_if_unsupported=True):
        self.enforce_no_scheduler = enforce_no_scheduler
        self.warn_if_unsupported = warn_if_unsupported

    def initialize(self):
        self.found_schedule_free_ = []
        return self

    def _is_schedule_free(self, optimizer):
        """A schedule-free optimizer exposes train/eval mode switches."""
        if optimizer is None:
            return False
        return (
            callable(getattr(optimizer, 'train', None))
            and callable(getattr(optimizer, 'eval', None))
        )

    def _scheduler_callbacks(self, net):
        """Names of LR scheduler callbacks currently attached to the net."""
        # imported here to avoid a circular import at module load time
        from skorch.callbacks import LRScheduler

        return [
            name for name, callback in net.callbacks_
            if isinstance(callback, LRScheduler)
        ]

    # pylint: disable=unused-argument,protected-access
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        self.found_schedule_free_ = [
            name for name in net._optimizers
            if self._is_schedule_free(getattr(net, name + '_', None))
        ]

        if not self.found_schedule_free_ and self.warn_if_unsupported:
            warnings.warn(
                "ScheduleFreeMode: none of the net's optimizers defines "
                "train/eval methods. This callback only has an effect with "
                "schedule-free optimizers (arXiv:2405.15682), e.g. from the "
                "'schedulefree' package. The net will train normally but no "
                "schedule-free mode is being applied.",
                UserWarning,
            )

        scheduler_names = self._scheduler_callbacks(net)
        if scheduler_names and self.found_schedule_free_:
            msg = (
                "Schedule-free optimizers must not be combined with a "
                "learning rate scheduler, but found scheduler callback(s): "
                f"{scheduler_names}. Schedule-free optimizers replace the "
                "schedule with an interpolation between iterates, so a "
                "scheduler on top fights that mechanism. Remove the "
                "scheduler callback or set enforce_no_scheduler=False."
            )
            if self.enforce_no_scheduler:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning)
