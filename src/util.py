import importlib
import numpy as np

def mu_law(x, mu = 255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def inv_mu_law(x, mu = 255):
    return np.sign(x) * ((1 + mu) ** np.abs(x) - 1) / mu
    
def staged_mu_law(x, mu = 255, scale=1):
    x = scale * x
    _x = mu_law(x, mu=mu)
    x[x > 1] = _x[x > 1]
    x[x < -1] = _x[x < -1]
    return x / scale

def inv_staged_mu_law(x, mu = 255, scale=1):
    x = scale * x
    _x = inv_mu_law(x, mu=mu)
    x[x > 1] = _x[x > 1]
    x[x < -1] = _x[x < -1]
    return x / scale

def minus_one(x):
    return x - 1

def div_100_staged_mu_law(x, mu=255):
    return staged_mu_law(x / 100, mu=mu)

def dynamic_load(item):
    item = item.split(".")
    package = ".".join(item[:-1])
    item_name = item[-1]
    return getattr(importlib.import_module(package), item_name)

# Modified from
# https://github.com/Lightning-AI/pytorch-lightning/issues/2644
from lightning.pytorch.callbacks import EarlyStopping
class EarlyStoppingWithWarmup(EarlyStopping):
    """
    EarlyStopping, except don't watch the first `warmup` epochs.
    """

    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            not self._check_on_train_epoch_end
            or self._should_skip_check(trainer)
            or trainer.current_epoch < self.warmup
        ):
            return
        self._run_early_stopping_check(trainer)

    def on_validation_end(self, trainer, pl_module):
        if (
            self._check_on_train_epoch_end
            or self._should_skip_check(trainer)
            or trainer.current_epoch < self.warmup
        ):
            return
        self._run_early_stopping_check(trainer)