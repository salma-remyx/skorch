""" Callback for early stopping by correlating online indicators.

Implements the "correlation of online indicators" (COI) stopping rule of
Prechelt-style training indicators: instead of trusting a single
overfitting indicator, the rule fires when *several* independent
indicators start agreeing within a short strip of epochs, the
correlation between them being taken as evidence of a common cause
(Reichenbach's principle), which the paper identifies with the onset of
overfitting.

Adapted from: "Early stopping by correlating online indicators in neural
networks" (https://arxiv.org/abs/2402.02513). The indicator definitions
are Prechelt's classical GL/PQ/UP (equations 6-7 of the paper define the
correlation step); no reference implementation was consulted.
"""

import itertools

import numpy as np

from skorch.callbacks import Callback


__all__ = ['IndicatorCorrelationStopping']


def _pearson(x, y):
    """Pearson correlation of two 1d sequences, floored at 0.

    Returns 0.0 for degenerate input (empty strips, zero variance, or
    mismatched lengths), so that a non-informative strip can never be
    read as agreement.
    """
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    if x.size < 2 or x.shape != y.shape:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    norm = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    if norm == 0.0:
        return 0.0
    return float(np.clip((x_centered * y_centered).sum() / norm, 0.0, 1.0))


def _concordant(x, y):
    """Whether two indicator trajectories are both trending upwards.

    Equation (6) of the paper restricts the correlation to indicator
    pairs that actually fired within the strip, allowing for up to k-1
    epochs of asynchrony between them. Since the indicators used here
    are all "larger means more overfitting", firing shows up as an
    upward trend, and an upward trend is exactly what makes a positive
    correlation meaningful rather than incidental.
    """
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    if x.size < 2 or y.size < 2:
        return False
    return bool(_is_firing(x) and _is_firing(y))


def _is_firing(x):
    """Whether an indicator trajectory points at overfitting.

    An indicator that stays flat over the whole strip has not fired;
    correlating it with anything would only measure the other indicator
    against a constant, which says nothing about agreement. An indicator
    whose values merely decay towards 0 as training converges has not
    fired either: the decay of a *relative* excess is the signature of a
    still-improving loss, not of a degrading one.
    """
    x = np.asarray(x, dtype='float64')
    if x.size < 2:
        return False
    return bool((x > x.min()).any() and np.diff(x).max() > 0.0)


def _carry_forward(values):
    """Replace non-finite entries by the last finite one, 0 if none.

    Used for the productivity quotient, which is undefined on strips
    where the training loss made no progress at all.
    """
    values = np.array(values, dtype='float64', copy=True)
    last_finite = 0.0
    for i, value in enumerate(values):
        if np.isfinite(value):
            last_finite = value
        else:
            values[i] = last_finite
    return values


class IndicatorCorrelationStopping(Callback):
    """Early stopping when independent overfitting indicators agree.

    Unlike :class:`skorch.callbacks.EarlyStopping`, which counts how long
    a single monitored score has failed to improve, this callback derives
    several *independent* online indicators from the train/valid loss
    strips already recorded in ``net.history`` and stops when the
    indicators become strongly correlated over a window of epochs. Under
    the hypothesis of the paper, individually noisy indicators that
    nonetheless move together indicate a common cause -- the validation
    loss degrading while the training loss keeps improving, i.e.
    overfitting -- which is a more trustworthy stop signal than any one
    of them.

    The indicators used are Prechelt's classical ones, evaluated per
    epoch over the strip:

      - ``gl``  : generalization loss, the relative increase of the
        validation loss over its best past value;
      - ``pq``  : productivity quotient, generalization loss divided by
        the training progress made over the strip;
      - ``up``  : the relative increase of the validation loss from one
        epoch of the strip to the next.

    Training is stopped when, for at least ``patience`` consecutive
    epochs, the maximum pairwise Pearson correlation between the
    indicators' trajectories over the strip exceeds ``threshold``, with
    only pairs of indicators that are both rising -- i.e. both firing --
    taken into account.

    Examples
    --------
    >>> from skorch.callbacks import IndicatorCorrelationStopping
    >>> net = NeuralNetClassifier(
    ...     my_module, max_epochs=100,
    ...     callbacks=[IndicatorCorrelationStopping()],
    ... )  # doctest: +SKIP

    Parameters
    ----------
    train_loss : str (default='train_loss')
      History column holding the training loss.

    valid_loss : str (default='valid_loss')
      History column holding the validation loss. Note that this implies
      the use of a validation split.

    strip_length : int (default=5)
      Number of past epochs (the "strip", k in the paper) over which the
      indicators and their correlation are computed. The paper uses
      k = 5 for all indicators.

    threshold : float (default=0.8)
      Minimum pairwise Pearson correlation between two indicators for
      their agreement to count as a stop signal (alpha_N in the paper).

    patience : int (default=1)
      Number of consecutive epochs the correlation condition has to hold
      before training is stopped. The paper stops as soon as it holds,
      which corresponds to ``patience=1``.

    min_delta : float (default=0.0)
      Relative amount by which the validation loss must exceed its best
      past value before the indicators are evaluated at all. Setting it
      to a small positive value (e.g. 0.05) guards against stopping on
      epoch-to-epoch noise around a still-improving minimum.

    sink : callable (default=print)
      The target that the information about early stopping is sent to.
      By default, the output is printed to stdout, but the sink could
      also be a logger or :func:`~skorch.utils.noop`.
    """

    def __init__(
            self,
            train_loss='train_loss',
            valid_loss='valid_loss',
            strip_length=5,
            threshold=0.8,
            patience=1,
            min_delta=0.0,
            sink=print,
    ):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.strip_length = strip_length
        self.threshold = threshold
        self.patience = patience
        self.min_delta = min_delta
        self.sink = sink

    # pylint: disable=arguments-differ
    def on_train_begin(self, net, **kwargs):
        if self.strip_length < 2:
            raise ValueError(
                "strip_length must be >= 2, got {}".format(self.strip_length))
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(
                "threshold must lie in (0, 1], got {}".format(self.threshold))
        self.misses_ = 0
        self.correlation_ = 0.0

    def on_epoch_end(self, net, **kwargs):
        n_epochs = len(net.history)
        if n_epochs < self.strip_length:
            return
        valid = self._valid_losses(net)
        # Overfitting indicators are only meaningful once the validation
        # loss has left its optimum behind: while a new best is still
        # being set, generalization loss is 0 by construction and every
        # indicator trivially agrees with it, which would make the
        # correlation fire on perfectly healthy training. Stray noise
        # around a still-improving minimum is likewise no evidence of
        # overfitting, hence the `min_delta` slack.
        if not (valid > valid.min() * (1.0 + self.min_delta)).any():
            self.misses_ = 0
            self.correlation_ = 0.0
            return
        self.correlation_ = self._indicator_correlation(net)
        if self.correlation_ > self.threshold:
            self.misses_ += 1
        else:
            self.misses_ = 0
        if self.misses_ >= self.patience:
            if net.verbose:
                self._sink(
                    "Stopping since online training indicators have been "
                    "correlated (rho={:.3f} > {}) for {} epochs.".format(
                        self.correlation_, self.threshold, self.misses_),
                    verbose=net.verbose)
            raise KeyboardInterrupt

    def _strip(self, net, column):
        """Return the last `strip_length` values of a history column.

        History is indexed by record position, hence the negative indices
        counting back from the current epoch.
        """
        return np.asarray(
            [net.history[-i, column] for i in range(
                self.strip_length, 0, -1)],
            dtype='float64',
        )

    def _valid_losses(self, net):
        """Return all validation losses observed so far."""
        return np.asarray(
            [net.history[-i, self.valid_loss]
             for i in range(len(net.history), 0, -1)],
            dtype='float64',
        )

    def _indicators(self, net):
        """Return the per-epoch indicator values over the current strip.

        The indicators are as defined by Prechelt and used by the paper;
        each is large when overfitting is more likely. ``valid_min`` is
        the best (minimum) validation loss observed so far, not just
        within the strip, since generalization loss is defined against
        the whole training past.
        """
        train = self._strip(net, self.train_loss)
        valid = self._strip(net, self.valid_loss)
        valid_min = self._valid_losses(net).min()

        # gl: generalization loss, the relative excess of the current
        # validation loss over its best past value.
        generalization_loss = 100.0 * (
            valid / (valid_min * (1.0 + self.min_delta)) - 1.0)

        # up: the relative increase of the validation loss from one epoch
        # to the next; positive while validation degrades.
        with np.errstate(divide='ignore', invalid='ignore'):
            uninterrupted = 100.0 * (
                valid / np.roll(valid, 1) - 1.0)[1:]

        # pq: generalization loss penalized by the training progress made
        # over the strip, so that fast progress does not stop training.
        # A strip with no training progress leaves the quotient
        # undefined; carrying it forward from the previous epoch keeps
        # the trajectory comparable across the strip instead of letting
        # an extreme or undefined value dominate the correlation.
        with np.errstate(divide='ignore', invalid='ignore'):
            progress = 100.0 * (train / train[np.argmin(train)] - 1.0)
            productivity = generalization_loss / progress
        productivity = _carry_forward(productivity)

        return {
            'gl': generalization_loss,
            'pq': productivity,
            'up': uninterrupted,
        }

    def _indicator_correlation(self, net):
        """Maximum pairwise Pearson correlation over the indicator strip.

        This is c_coi of equation (6) of the paper: the best correlation
        between any two indicators' value sequences over the strip,
        floored at 0 so that anti-correlated indicators (which disagree,
        and hence do not point at a common cause) never fire.

        Only pairs whose trajectories are *concordant* -- both rising over
        the strip, i.e. both indicators firing -- can reach a high value;
        a rising/decreasing pair is anti-correlated and contributes 0.
        """
        indicators = self._indicators(net)
        best = 0.0
        for (_, first), (_, second) in itertools.combinations(
                sorted(indicators.items()), 2):
            if _concordant(first, second):
                best = max(best, _pearson(first, second))
        return best

    def _sink(self, text, verbose):
        # We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)
