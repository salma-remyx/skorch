"""Schedule-free optimizers that need no learning rate schedule.

Adapted from "The Road Less Scheduled" (Defazio et al., 2024),
https://arxiv.org/abs/2405.15682, and from the reference Apache-2.0
implementation at https://github.com/facebookresearch/schedule_free.

Schedule-free optimizers interpolate between two sequences: gradients
are evaluated at the iterate ``y`` while evaluation/prediction uses the
weighted average ``x``. The two are exchanged whenever the optimizer is
switched between train and eval mode, which is why
:meth:`skorch.net.NeuralNet._set_training` propagates the mode to any
optimizer that has a ``train`` method -- this is what makes the
optimizers below usable from skorch without any extra user code.

Because the evaluation weights are only guaranteed to be in place in
eval mode, checkpoints should be written in eval mode (see
:meth:`SkorchLRSchedulerPassthrough` for the matching note on schedulers).
"""

from typing import Tuple

import torch

from skorch.callbacks import LRScheduler

__all__ = ['SGDScheduleFree', 'AdamWScheduleFree', 'SkorchLRSchedulerPassthrough']


class _ScheduleFreeMixin:
    """Shared logic of the schedule-free optimizers.

    Tracks the step count ``k`` and the running sum of the averaging
    weights in a single "schedule-free" state entry so that
    ``state_dict``/``load_state_dict`` keep working through skorch's
    ``save_params``/``load_params``.
    """

    def _init_sf_state(self) -> None:
        # pylint: disable=use-dict-literal
        self.sf_state = dict(k=0, weight_sum=0.0, lr_max=-1.0)

    @property
    def supports_train_mode(self) -> bool:
        """Schedule-free optimizers must be switched between train and
        eval mode to exchange the gradient and evaluation weights."""
        return True

    def _schedule(self) -> float:
        """Linear warmup factor of the learning rate, capped at 1."""
        warmup_steps = self.defaults.get('warmup_steps', 0)
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (self.sf_state['k'] + 1) / warmup_steps)

    def _update_weights(self, lr: float, weight_lr_power: float) -> Tuple[float, float]:
        """Return the averaging weight and the interpolation coefficient.

        The averaging weights are ``(k+1)**r * lr_max**weight_lr_power``,
        which corresponds to a polynomially-weighted average over the
        trajectory; ``ckp1`` is the weight of the new iterate in that
        average.
        """
        state = self.sf_state
        k = state['k'] + 1
        lr_max = max(state['lr_max'], lr)
        state['lr_max'] = lr_max

        weight = (k ** self.defaults.get('r', 0)) * (lr_max ** weight_lr_power)
        state['weight_sum'] += weight
        return weight, weight / state['weight_sum']

    @torch.no_grad()
    def _swap_weights(self, train: bool) -> None:
        """Swap parameters between the gradient iterate ``y`` and the
        evaluation average ``x``.

        Calling this twice with the same argument is a no-op; the
        ``train_mode`` flag on each param group guards against it.
        """
        for group in self.param_groups:
            if group.get('train_mode', True) == train:
                continue
            momentum = group.get('momentum', group.get('betas', (0.9,))[0])
            weight = (1 - momentum) if train else (1 - 1 / momentum)
            for param in group['params']:
                state = self.state[param]
                z = state.get('z')
                if z is not None:
                    param.lerp_(end=z.to(param.device), weight=weight)
            group['train_mode'] = train

    def train(self, mode: bool = True) -> "_ScheduleFreeMixin":
        """Set train (gradient) or eval (averaged) weights in place."""
        self._swap_weights(train=mode)
        return self

    def eval(self) -> "_ScheduleFreeMixin":
        """Set the evaluation (averaged) weights in place."""
        return self.train(False)

    def _check_train_mode(self) -> None:
        if any(not group.get('train_mode', True) for group in self.param_groups):
            raise RuntimeError(
                "The optimizer is in eval mode. Call .train() (or let skorch "
                "do it, e.g. by calling net.fit) before stepping it."
            )

    def _lazy_init_param(self, param: torch.Tensor) -> torch.Tensor:
        """Cache the auxiliary sequence ``z`` for a parameter on first use."""
        state = self.state[param]
        if 'z' not in state:
            state['z'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        return state['z']

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict['sf_state'] = {**self.sf_state}
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        sf_state = state_dict.pop('sf_state', None)
        super().load_state_dict(state_dict)
        if sf_state is not None:
            self.sf_state.update(sf_state)


def _validate_common(lr: float, weight_decay: float) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")


class SGDScheduleFree(_ScheduleFreeMixin, torch.optim.Optimizer):
    """Schedule-free SGD with momentum, no learning rate schedule needed.

    Wraps cleanly into skorch::

        net = NeuralNetClassifier(
            MyModule, optimizer=SGDScheduleFree, lr=0.1, momentum=0.9,
        )

    Since the optimizer interpolates between the gradient iterate and the
    averaged evaluation weights, skorch switches it between train and
    eval mode together with the module. No ``LRScheduler`` callback is
    required -- attach :class:`SkorchLRSchedulerPassthrough` only if you
    need the ``lr`` entry in the history.

    Parameters
    ----------
    params : iterable
      Parameters to optimize, as for any other ``torch.optim`` optimizer.

    lr : float (default=0.01)
      Peak learning rate; warmup reaches it after ``warmup_steps`` steps.

    momentum : float (default=0.9)
      Momentum in ``[0, 1)``.

    weight_decay : float (default=0.0)
      Weight decay (L2 penalty), added to the gradient.

    warmup_steps : int (default=0)
      Number of steps of linear learning rate warmup.

    r : float (default=0)
      Exponent of the polynomial averaging weight ``(k+1)**r``; ``r=0``
      gives a uniform average.

    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
            self,
            params,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 0.0,
            warmup_steps: int = 0,
            r: float = 0,
    ):
        _validate_common(lr, weight_decay)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'r': r,
            'train_mode': True,
        }
        super().__init__(params, defaults)
        self._init_sf_state()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._check_train_mode()
        sched = self._schedule()

        for group in self.param_groups:
            lr = group['lr'] * sched
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            _, ckp1 = self._update_weights(lr, weight_lr_power=2)

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)

                z = self._lazy_init_param(param)
                # These operations update y (= param) in place, without
                # computing x explicitly.
                param.lerp_(end=z, weight=ckp1)
                param.add_(grad, alpha=lr * (momentum * (1 - ckp1) - 1))
                z.sub_(grad, alpha=lr)

        self.sf_state['k'] += 1
        return loss


class AdamWScheduleFree(_ScheduleFreeMixin, torch.optim.Optimizer):
    """Schedule-free AdamW, no learning rate schedule needed.

    Decoupled weight decay is applied directly to the iterate, not to
    the gradient. Use it in skorch exactly like
    :class:`SGDScheduleFree`::

        net = NeuralNetClassifier(
            MyModule, optimizer=AdamWScheduleFree, lr=0.002,
        )

    Parameters
    ----------
    params : iterable
      Parameters to optimize, as for any other ``torch.optim`` optimizer.

    lr : float (default=0.0025)
      Peak learning rate; warmup reaches it after ``warmup_steps`` steps.

    betas : tuple of float (default=(0.9, 0.999))
      Coefficients for the momentum and second-moment estimates.

    eps : float (default=1e-8)
      Numerical stability term added to the denominator.

    weight_decay : float (default=0.0)
      Decoupled weight decay, applied to the iterate ``y``.

    warmup_steps : int (default=0)
      Number of steps of linear learning rate warmup.

    r : float (default=0)
      Exponent of the polynomial averaging weight ``(k+1)**r``; ``r=0``
      gives a uniform average.

    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
            self,
            params,
            lr: float = 0.0025,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            warmup_steps: int = 0,
            r: float = 0,
    ):
        beta1, beta2 = betas
        _validate_common(lr, weight_decay)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")

        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'r': r,
            'train_mode': True,
        }
        super().__init__(params, defaults)
        self._init_sf_state()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._check_train_mode()
        sched = self._schedule()

        for group in self.param_groups:
            lr = group['lr'] * sched
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            ckp1 = self._update_weights(lr, weight_lr_power=2)[1]

            k = self.sf_state['k']
            bias_correction2 = 1 - beta2 ** (k + 1)

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad

                state = self.state[param]
                if len(state) == 0:
                    state['z'] = torch.zeros_like(
                        param, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(
                        param, memory_format=torch.preserve_format)

                z = state['z']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                # Reuse the grad buffer for the normalized gradient
                grad_normalized = grad.div_(denom)
                if weight_decay != 0:
                    grad_normalized.add_(param, alpha=weight_decay)

                param.lerp_(end=z, weight=ckp1)
                param.add_(
                    grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))
                z.sub_(grad_normalized, alpha=lr)

        self.sf_state['k'] += 1
        return loss


class SkorchLRSchedulerPassthrough(LRScheduler):
    """Placeholder ``LRScheduler`` that does not change the learning rate.

    Schedule-free optimizers replace the learning rate schedule, not
    complement it, so attaching a real scheduler is a mistake. This
    passthrough exists so that tooling that expects an ``LRScheduler``
    callback (e.g. to record the learning rate in the history) keeps
    working; it records the unchanged learning rate and never steps a
    scheduler.

    Parameters
    ----------
    event_name : str or None (default='event_lr')
      Name of the event recording the learning rate in the history, or
      ``None`` to disable recording.

    """

    def __init__(self, event_name: str = 'event_lr'):
        super().__init__(policy='WarmRestartLR', event_name=event_name)
        self.policy_ = self._get_policy_cls()

    @property
    def kwargs(self):
        excluded = ('policy', 'monitor', 'event_name', 'step_every')
        return {key: val for key, val in vars(self).items()
                if not (key in excluded or key.endswith('_'))}

    def _get_scheduler(self, net, policy, **scheduler_kwargs):
        # no scheduler is created: the optimizer is the schedule
        # pylint: disable=unused-argument, no-self-use
        return None

    def _step(self, net, lr_scheduler, score=None):
        # Schedule-free optimizers handle the learning rate themselves;
        # stepping any scheduler here would fight the optimizer.
        # pylint: disable=unused-argument
        return None

    def _record_last_lr(self, net, kind):
        if self.event_name is None:
            return
        lrs = [group['lr'] for group in net.optimizer_.param_groups]
        if kind == 'epoch':
            net.history.record(self.event_name, lrs[0])
        else:
            net.history.record_batch(self.event_name, lrs[0])
