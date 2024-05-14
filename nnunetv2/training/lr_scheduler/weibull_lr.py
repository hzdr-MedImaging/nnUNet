import math

from torch.optim.lr_scheduler import _LRScheduler


def weibull(x, k, scale):
    t = x / scale
    p = (k - 1) / k
    return t**(k-1) * math.exp(-t**k) / (p**p * math.exp(-p))


class WeibullLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_steps: int, max_lr: float, k: float = 2, scale_mul: float = 1/3, current_step: int = None):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.k = k
        self.scale = max_steps * scale_mul
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # avoiding lr = 0 in the beginning
        current_step = max(current_step, 0.1)
        new_lr = self.max_lr * weibull(current_step, self.k, self.scale)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
