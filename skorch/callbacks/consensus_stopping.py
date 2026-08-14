""" Callbacks that stop training when independent stopping conditions
agree, instead of trusting a single one. """

import numpy as np

from skorch.callbacks import Callback


__all__ = ['ConsensusStopping']


class ConsensusStopping(Callback):
    """Stop training when independent overfitting indicators agree.

    skorch's :class:`.EarlyStopping` trusts a single condition: the
    monitored score must improve by a threshold within ``patience``
    epochs. A single criterion is easily fooled by local irregularities
    in the error curve, so this callback instead evaluates a small
    family of online indicators on the train/valid loss curves that
    skorch already records in the history, and only stops once their
    diagnoses correlate over time.

    Each indicator is a Boolean function evaluated per epoch. The
    following indicators are used, all taken from the stopping-rule
    literature (Prechelt 1997; Lodwick et al. 2009; Vanneschi et al.
    2010):

    * generalization loss: the valid loss rose above ``gl_threshold``
      times the best valid loss seen so far
    * progress: within the last ``strip_length`` epochs, the mean train
      loss exceeds ``p_threshold`` times the minimum train loss in the
      same strip, i.e. training has stalled
    * overfitting gap: the absolute train/valid loss gap grew by more
      than ``og_threshold`` over its minimum so far
    * valid loss increase: the valid loss is higher than it was
      ``strip_length`` epochs ago

    The stopping rule is the consensus of these indicators: training
    stops at epoch ``e`` if, looking back over the window of the last
    ``strip_length`` epochs, at least ``min_agree`` indicators fired
    and the per-indicator firing patterns correlate by more than
    ``correlation_threshold``. The indicators do not need to fire on
    exactly the same epoch -- the window introduces a limited
    asynchronicity that makes the condition flexible rather than
    brittle. Correlation is measured with Pearson's coefficient, which
    is appropriate here because each indicator is the aggregate of many
    small additive effects.

    Compared to :class:`.EarlyStopping`, this rule is harder to trigger
    by a single noisy epoch but reacts to more than one symptom of
    overfitting, so it can fire earlier than a pure patience rule when
    several signals deteriorate at once.

    Parameters
    ----------
    monitor_valid : str (default='valid_loss')
      Name of the history entry holding the validation loss.

    monitor_train : str (default='train_loss')
      Name of the history entry holding the training loss.

    strip_length : int (default=5)
      Length of the training strip, in epochs, over which progress is
      measured and over which the indicators are correlated.

    min_agree : int (default=2)
      Number of indicators that must fire within the current strip
      before a stop is considered. The paper's rule requires at least
      two agreeing indicators.

    correlation_threshold : float (default=0.5)
      Lower bound on the correlation between the firing patterns of two
      agreeing indicators for the stop to trigger.

    gl_threshold : float (default=1.0)
      Relative increase, in percent, of the valid loss over its running
      minimum that makes the generalization-loss indicator fire.

    p_threshold : float (default=1.0)
      Ratio between the mean and the minimum train loss within a strip
      that makes the progress indicator fire (e.g. 1.001 means the mean
      exceeds the minimum by 0.1%).

    og_threshold : float (default=0.0)
      Increase of the train/valid loss gap over its running minimum
      that makes the overfitting-gap indicator fire.

    sink : callable (default=print)
      The target that the information about early stopping is sent to.
      Use :func:`skorch.utils.noop` to silence it.

    Examples
    --------
    >>> net = NeuralNetClassifier(
    ...     my_module,
    ...     max_epochs=100,
    ...     callbacks=[ConsensusStopping()],
    ... )
    >>> net.fit(X, y)

    Notes
    -----
    Adapted from "Early Stopping by Correlating Online Indicators in
    Neural Networks" (arXiv:2402.02513). The paper evaluates the
    consensus over a larger family of indicators and validates it
    through a cross-validation canary; the canary is cut here and the
    indicators are computed on the losses skorch already records.

    """
    def __init__(
            self,
            monitor_valid='valid_loss',
            monitor_train='train_loss',
            strip_length=5,
            min_agree=2,
            correlation_threshold=0.5,
            gl_threshold=1.0,
            p_threshold=1.0,
            og_threshold=0.0,
            sink=print,
    ):
        self.monitor_valid = monitor_valid
        self.monitor_train = monitor_train
        self.strip_length = strip_length
        self.min_agree = min_agree
        self.correlation_threshold = correlation_threshold
        self.gl_threshold = gl_threshold
        self.p_threshold = p_threshold
        self.og_threshold = og_threshold
        self.sink = sink

    def on_epoch_end(self, net, **kwargs):
        epoch = len(net.history)
        if epoch < self.strip_length + 1:
            # Not enough epochs recorded yet to fill one strip.
            return
        indicators = self._indicator_values(net)
        fired = {
            name: values[-self.strip_length:]
            for name, values in indicators.items()
        }

        agreeing = self._agreeing_indicators(fired)
        correlation = self._max_correlation(fired, agreeing)
        self.latest_indicators_ = {
            name: values[-1] for name, values in indicators.items()}
        self.latest_correlation_ = correlation
        stop = (
            len(agreeing) >= self.min_agree
            and correlation > self.correlation_threshold
        )

        if stop:
            if net.verbose:
                self._sink(
                    "Stopping since {} online indicators correlated above "
                    "{} within the last {} epochs.".format(
                        ', '.join(agreeing),
                        self.correlation_threshold,
                        self.strip_length,
                    ),
                    verbose=net.verbose,
                )
            raise KeyboardInterrupt

    def on_train_begin(self, net, **kwargs):
        self.latest_indicators_ = {}
        self.latest_correlation_ = 0.0

    def _indicator_values(self, net):
        """Evaluate each online indicator over the whole history.

        Each indicator maps to a list of 0/1 floats, one entry per
        epoch; the entries before the indicator has enough data are 0.

        """
        train_loss = net.history[:, self.monitor_train]
        valid_loss = net.history[:, self.monitor_valid]
        n_epochs = len(valid_loss)
        indicators = {name: [0.0] * n_epochs
                      for name in ('generalization_loss', 'progress',
                                   'overfitting_gap', 'valid_increase')}

        valid_min = np.inf
        gap_min = np.inf
        for epoch in range(n_epochs):
            valid_min = min(valid_min, valid_loss[epoch])
            gap = abs(train_loss[epoch] - valid_loss[epoch])
            gap_min = min(gap_min, gap)

            indicators['generalization_loss'][epoch] = float(
                valid_loss[epoch] > valid_min * (1 + self.gl_threshold / 100))

            start = epoch - self.strip_length + 1
            if start >= 0:
                strip = train_loss[start:epoch + 1]
                mean_loss = sum(strip) / len(strip)
                indicators['progress'][epoch] = float(
                    mean_loss > self.p_threshold * min(strip))
            else:
                indicators['progress'][epoch] = 0.0

            indicators['overfitting_gap'][epoch] = float(
                gap - gap_min > self.og_threshold)

            if epoch >= self.strip_length:
                indicators['valid_increase'][epoch] = float(
                    valid_loss[epoch] > valid_loss[epoch - self.strip_length])

        return indicators

    def _agreeing_indicators(self, fired):
        """Return the names of the indicators that fired in the strip."""
        return sorted(name for name, values in fired.items()
                      if any(values))

    @staticmethod
    def _max_correlation(fired, names):
        """Correlation between the firing patterns of agreeing indicators.

        Returns the maximum Pearson correlation over all pairs of
        agreeing indicators, each shifted against the other by up to
        one epoch less than the strip length, so that indicators firing
        a few epochs apart still count as agreeing. A pair that never
        disagrees is treated as perfectly correlated.

        """
        correlations = []
        for i, name_i in enumerate(names):
            for name_j in names[i + 1:]:
                left, right = fired[name_i], fired[name_j]
                if not any(left) or not any(right):
                    continue
                if all(not (a or b) for a, b in zip(left, right)):
                    correlations.append(1.0)
                    continue
                correlations.append(_pearson(left, right))
                correlations.extend(
                    _pearson(shifted, right)
                    for shifted in _shifted_patterns(left))
        return max(correlations, default=0.0)

    def _sink(self, text, verbose):
        # We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)


def _pearson(left, right):
    """Pearson correlation coefficient between two equal-length lists."""
    left = np.asarray(left, dtype='float64')
    right = np.asarray(right, dtype='float64')
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt((left_centered ** 2).sum()
                          * (right_centered ** 2).sum())
    if denominator == 0:
        # One pattern is constant within the strip; it either agrees
        # with the other on every epoch or not at all.
        return 0.0
    return float((left_centered * right_centered).sum() / denominator)


def _shifted_patterns(pattern):
    """Yield ``pattern`` delayed by 1, 2, ... epochs.

    A delayed copy stays 0 before its first firing epoch and keeps its
    last value afterwards, so its length never changes.

    """
    for shift in range(1, len(pattern)):
        yield [0.0] * shift + pattern[:-shift]
        yield pattern[shift:] + [pattern[-1]] * shift
