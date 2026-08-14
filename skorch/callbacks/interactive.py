"""Feedback-driven, interactive control of the training loop.

Implements the core control point of "Interactive Training:
Feedback-Driven Neural Network Optimization" (arXiv:2510.02297) in a
skorch-native way: at regular control points during training, the latest
metrics are handed to a user-supplied controller, which answers with a
list of interventions (set a knob, stop training, write a checkpoint)
that are applied to the running net. The controller may be a human in
the loop, a simple rule, or an automated agent.

"""

__all__ = ['InteractiveTraining']

import warnings

from skorch.callbacks.base import Callback
from skorch.callbacks.training import Checkpoint


def _get_lr(net):
    return net.optimizer_.param_groups[0]['lr']


def _set_lr(net, value):
    for group in net.optimizer_.param_groups:
        group['lr'] = value


class InteractiveTraining(Callback):
    """Consult a controller during training and apply its interventions.

    At each control point (by default, after every training batch), the
    latest metrics from ``net.history`` are passed to the controller as
    ``controller(metrics, net)``. The controller returns an iterable of
    action dicts, which are applied in order:

    - ``{'action': 'set_knob', 'name': name, 'value': value}``: set a
      registered knob (``'lr'`` is always registered) to ``value``.
    - ``{'action': 'stop'}``: gracefully end training, as if the user
      had interrupted it.
    - ``{'action': 'checkpoint'}``: trigger ``save_model`` on every
      :class:`~skorch.callbacks.Checkpoint` callback registered on the
      net.

    Parameters
    ----------
    controller : callable
        Function called as ``controller(metrics, net)`` at each
        consulted control point. Must return an iterable of action
        dicts (possibly empty). The metrics dict contains the latest
        batch record (for ``on='on_batch_end'``) merged with the current
        epoch row of ``net.history``, plus ``'epoch'`` and, per batch,
        ``'batch'`` counters.
    every : int (default=1)
        Consult the controller every ``every`` control points. Values
        above 1 reduce the overhead of expensive controllers.
    on : str (default='on_batch_end')
        Control point; one of ``'on_batch_end'`` (after each training
        batch) or ``'on_epoch_end'``.
    knobs : dict or None (default=None)
        Additional knobs, mapping a name to a ``(getter, setter)`` pair
        of callables with signatures ``getter(net)`` and
        ``setter(net, value)``. The ``'lr'`` knob, which reads and
        writes the learning rate of all optimizer param groups, is
        always available.
    sink : callable (default=print)
        Where applied interventions are logged. Set to
        :func:`~skorch.utils.noop` to silence.

    Examples
    --------
    >>> def lower_lr_when_loss_high(metrics, net):
    ...     if metrics.get('train_loss', 0.0) > 1.0:
    ...         return [{'action': 'set_knob', 'name': 'lr', 'value': 1e-4}]
    ...     return []
    >>> net = NeuralNetClassifier(module, callbacks=[
    ...     ('interactive', InteractiveTraining(lower_lr_when_loss_high)),
    ... ])

    """
    def __init__(
            self,
            controller,
            every=1,
            on='on_batch_end',
            knobs=None,
            sink=print,
    ):
        self.controller = controller
        self.every = every
        self.on = on
        self.knobs = knobs
        self.sink = sink

    def initialize(self):
        self._check_params()
        self.n_control_points_ = 0
        return self

    def on_batch_end(self, net, batch=None, training=None, **kwargs):
        """Serve a control point after each training batch."""
        if training is False:
            return
        if self.on == 'on_batch_end':
            self._control_point(net)

    def on_epoch_end(self, net, **kwargs):
        """Serve a control point after each epoch, if so configured."""
        if self.on == 'on_epoch_end':
            self._control_point(net)

    def _check_params(self):
        if self.on not in ('on_batch_end', 'on_epoch_end'):
            raise ValueError(
                "InteractiveTraining: 'on' must be 'on_batch_end' or "
                "'on_epoch_end', got {!r}.".format(self.on))
        if self.every < 1:
            raise ValueError(
                "InteractiveTraining: 'every' must be at least 1, got {!r}."
                .format(self.every))

    def _control_point(self, net):
        self.n_control_points_ += 1
        if (self.n_control_points_ - 1) % self.every:
            return
        metrics = self._current_metrics(net)
        for action in self.controller(metrics, net) or []:
            self._apply(net, action)

    def _current_metrics(self, net):
        metrics = {key: value for key, value in net.history[-1].items()
                   if key != 'batches'}
        metrics['epoch'] = len(net.history)
        if self.on == 'on_batch_end':
            batches = net.history[-1]['batches']
            metrics['batch'] = len(batches)
            if batches:
                metrics.update(batches[-1])
        return metrics

    def _knobs(self):
        knobs = {'lr': (_get_lr, _set_lr)}
        knobs.update(self.knobs or {})
        return knobs

    def _apply(self, net, action):
        kind = action.get('action')
        if kind == 'set_knob':
            self._set_knob(net, action['name'], action['value'])
        elif kind == 'stop':
            self._sink(
                "InteractiveTraining: stopping training.", net.verbose)
            raise KeyboardInterrupt
        elif kind == 'checkpoint':
            self._save_checkpoints(net)
        else:
            warnings.warn(
                "InteractiveTraining: ignoring unknown action {!r}."
                .format(action))

    def _set_knob(self, net, name, value):
        knobs = self._knobs()
        if name not in knobs:
            raise KeyError(
                "InteractiveTraining: unknown knob {!r}; available knobs: "
                "{}.".format(name, sorted(knobs)))
        getter, setter = knobs[name]
        old_value = getter(net)
        setter(net, value)
        self._sink(
            "InteractiveTraining: set {!r} from {!r} to {!r}.".format(
                name, old_value, value), net.verbose)

    def _save_checkpoints(self, net):
        checkpoints = [cb for _, cb in net.callbacks_
                       if isinstance(cb, Checkpoint)]
        if not checkpoints:
            warnings.warn(
                "InteractiveTraining: 'checkpoint' action requested but no "
                "Checkpoint callback is registered on the net; ignoring.")
            return
        for checkpoint in checkpoints:
            checkpoint.save_model(net)
        self._sink(
            "InteractiveTraining: wrote {} checkpoint(s).".format(
                len(checkpoints)), net.verbose)

    def _sink(self, text, verbose):
        # We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)
