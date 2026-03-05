import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer

from ..models import PerturbationModel
from .batch_speed_monitor import BatchSpeedMonitorCallback
from .model_flops_utilization import ModelFLOPSUtilizationCallback
from .cumulative_flops import CumulativeFLOPSCallback
from .data_load_profiler import DataLoadProfilerCallback

__all__ = ["PerturbationModel", "BatchSpeedMonitorCallback", "ModelFLOPSUtilizationCallback", "CumulativeFLOPSCallback", "DataLoadProfilerCallback"]


class GradNormCallback(Callback):
    """
    Logs the gradient norm every N steps (default 50).
    Uses a single fused norm to avoid per-parameter CUDA syncs.
    """

    def __init__(self, log_interval: int = 50):
        super().__init__()
        self.log_interval = log_interval
        self._step_count = 0

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        self._step_count += 1
        if self._step_count % self.log_interval == 0:
            pl_module.log("train/gradient_norm", gradient_norm(pl_module))


def gradient_norm(model):
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.cat(grads).norm(2)
    return total_norm.item()
