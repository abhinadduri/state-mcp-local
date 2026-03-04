"""
Lightning callback that measures data loading time vs compute time per step.

Logs:
- train/data_time: seconds spent waiting for the next batch
- train/compute_time: seconds spent in forward+backward+optimizer
- train/data_fraction: data_time / (data_time + compute_time)
- train/steps_per_sec: throughput
"""

import logging
import time

from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


class DataLoadProfilerCallback(Callback):
    """Measures data loading vs GPU compute time split."""

    def __init__(self, log_interval: int = 50):
        super().__init__()
        self.log_interval = log_interval
        self._batch_end_time: float | None = None
        self._batch_start_time: float | None = None
        # Accumulators
        self._data_time_acc = 0.0
        self._compute_time_acc = 0.0
        self._step_count = 0
        self._window_start = 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        if self._batch_end_time is not None:
            # Time between end of last batch and start of this one = data loading time
            self._data_time_acc += now - self._batch_end_time
        else:
            self._window_start = now
        self._batch_start_time = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.perf_counter()
        self._batch_end_time = now
        if self._batch_start_time is not None:
            self._compute_time_acc += now - self._batch_start_time
        self._step_count += 1

        if self._step_count % self.log_interval == 0:
            total = self._data_time_acc + self._compute_time_acc
            if total > 0:
                data_frac = self._data_time_acc / total
                steps_per_sec = self._step_count / (now - self._window_start) if (now - self._window_start) > 0 else 0
                avg_data = self._data_time_acc / self._step_count
                avg_compute = self._compute_time_acc / self._step_count

                pl_module.log("train/data_time_avg", avg_data, prog_bar=False)
                pl_module.log("train/compute_time_avg", avg_compute, prog_bar=False)
                pl_module.log("train/data_fraction", data_frac, prog_bar=True)
                pl_module.log("train/steps_per_sec", steps_per_sec, prog_bar=True)

                logger.info(
                    "PERF [step %d] data=%.4fs compute=%.4fs data_frac=%.1f%% steps/s=%.2f",
                    trainer.global_step,
                    avg_data,
                    avg_compute,
                    data_frac * 100,
                    steps_per_sec,
                )

            # Reset accumulators
            self._data_time_acc = 0.0
            self._compute_time_acc = 0.0
            self._step_count = 0
            self._window_start = now
