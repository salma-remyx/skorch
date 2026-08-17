"""Schedule-free optimization, following Defazio et al., "The Road Less
Scheduled" (2024), https://arxiv.org/abs/2405.15682.

The paper replaces the learning rate schedule with an interpolated
iterate. Instead of stepping the parameters along a schedule-decayed
learning rate, the model is evaluated at ``x``, a weighted average of
past iterates, while gradients are computed at an interpolation ``y``
between ``x`` and the raw iterate. Because the average is formed on the
fly, the learning rate never depends on the stopping step ``T``, which
is what makes a schedule unnecessary.

Use together with the :class:`.ScheduleFreeMode` callback, which
switches the optimizer between train and eval mode. skorch only calls
``train()``/``eval()`` on modules, never on optimizers, so without that
callback the optimizer never switches out of train mode and validation
and prediction would see the wrong iterate.

"""

import torch
from torch.optim.optimizer import Optimizer


__all__ = ['ScheduleFreeAdamW']


class ScheduleFreeAdamW(Optimizer):
    """AdamW variant that does not require a learning rate schedule.

    The module's parameters are kept at the interpolated iterate ``y``
    during training, and at the averaged iterate ``x`` for evaluation.
    Both are recovered from the parameter tensor and the ``z`` sequence
    stored in the optimizer state, so the method costs one extra buffer
    per parameter compared to AdamW.

    Use together with the :class:`.ScheduleFreeMode` callback, which
    switches the optimizer between train and eval mode:

    >>> net = NeuralNetClassifier(
    ...     module, optimizer=ScheduleFreeAdamW,
    ...     callbacks=[ScheduleFreeMode()],
    ... )

    ``step()`` raises if the optimizer is in eval mode, since a gradient
    step taken at the averaged iterate would invalidate the average.

    Parameters
    ----------
    params : iterable
      Parameters to optimize, as for any torch optimizer.

    lr : float (default=0.0025)
      Learning rate. Unlike with scheduled optimizers this value is used
      unchanged for the whole run; the effective step size is shaped by
      the averaging weights, not by decay.

    betas : tuple of float (default=(0.9, 0.999))
      ``betas[0]`` is the outer momentum, the interpolation weight
      between the averaged and the raw iterate. ``betas[1]`` is the
      coefficient for the running average of the squared gradient.

    eps : float (default=1e-8)
      Term added to the denominator to improve numerical stability.

    weight_decay : float (default=0)
      Weight decay, applied at ``y`` before the gradient step.

    warmup_steps : int (default=0)
      Number of steps over which the learning rate is linearly ramped
      up from 0. This warmup does not require knowing the stopping step,
      so it does not reintroduce a schedule; set it to 0 to disable.

    """

    def __init__(
            self,
            params,
            lr=0.0025,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            warmup_steps=0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0 <= warmup_steps:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            weight_lr_power=2.0,
            r=0,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            # Tracks how much of the average each past iterate
            # contributed to, so the running mean stays closed-form.
            group['weight_sum'] = group.get('weight_sum', 0.0)
            group['lr_max'] = group.get('lr_max', 0.0)
            group['k'] = group.get('k', 0)
            group['train_mode'] = group.get('train_mode', True)
            for param in group['params']:
                # z is the only extra buffer the method needs.
                self.state[param]['z'] = param.detach().clone()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Requires the optimizer to be in train mode; use
        :class:`.ScheduleFreeMode` to manage that.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group['train_mode']:
                raise RuntimeError(
                    "step() was called while the optimizer was in eval mode. "
                    "Call .train() on the optimizer first, e.g. through the "
                    "ScheduleFreeMode callback."
                )

            beta1, beta2 = group['betas']
            eps = group['eps']
            base_lr = group['lr']
            weight_decay = group['weight_decay']
            warmup_steps = group['warmup_steps']

            k = group['k']
            # Linear warmup, then a constant learning rate.
            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            lr = base_lr * sched
            lr_max = group['lr_max'] = max(lr, group['lr_max'])

            weight = ((k + 1) ** group['r']) * (lr_max ** group['weight_lr_power'])
            weight_sum = group['weight_sum'] = group['weight_sum'] + weight
            # Fraction of the running average contributed by the latest
            # iterate.
            ckp1 = weight / weight_sum if weight_sum > 0 else 0.0

            bias_correction2 = 1 - beta2 ** (k + 1)

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]
                if 'exp_avg_sq' not in state:
                    # z was already initialized in __init__ as a copy of
                    # the parameter; keep that value.
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denominator = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)
                grad_normalized = exp_avg.div(denominator)
                if weight_decay != 0:
                    # Decay is applied at y, i.e. at the parameters as
                    # they currently are.
                    grad_normalized.add_(param, alpha=weight_decay)

                # The two updates below are written to avoid keeping x
                # around: both y and x are linear combinations of the
                # parameter and z.
                #   y = y*(1-ckp1) + z*ckp1 - lr*(1-beta1*(1-ckp1)) * g
                param.lerp_(state['z'], weight=ckp1)
                param.add_(
                    grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))

                #   z = z - lr * g
                state['z'].sub_(grad_normalized, alpha=lr)

            group['k'] = k + 1

        return loss

    @torch.no_grad()
    def train(self):
        """Move the parameters from the averaged iterate ``x`` to the
        interpolated iterate ``y`` at which gradients are computed.

        """
        self._set_train_mode(True)

    @torch.no_grad()
    def eval(self):
        """Move the parameters to the averaged iterate ``x``.

        This is the iterate the paper's evaluation is reported at, and
        the one to checkpoint.

        """
        self._set_train_mode(False)

    def _set_train_mode(self, training):
        """Switch between holding ``y`` (train) and ``x`` (eval) in the
        parameter tensor, by interpolating towards ``z``.

        """
        for group in self.param_groups:
            if group['train_mode'] == training:
                continue
            group['train_mode'] = training
            # y = (1 - beta1) * x + beta1 * z is inverted into
            # x = (y - beta1 * z) / (1 - beta1), which torch expresses
            # as a single lerp_ with a weight outside [0, 1].
            weight = 1 - group['betas'][0] if training else 1 - 1 / group['betas'][0]
            for param in group['params']:
                state = self.state[param]
                if 'z' not in state:
                    continue
                param.lerp_(state['z'], weight=weight)
