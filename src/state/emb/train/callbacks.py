"""Utility functions extracted from former Lightning callbacks."""

import logging
import torch

logger = logging.getLogger(__name__)


def compute_forward_flops(model, batch, use_backward=True):
    """Measure FLOPs for a single forward (+ optional backward) pass.

    Uses torch.utils.flop_counter.FlopCounterMode which is the backend
    behind lightning.fabric.utilities.throughput.measure_flops.

    Returns total FLOPs as an int.
    """
    from torch.utils.flop_counter import FlopCounterMode

    flop_counter = FlopCounterMode(display=False)

    model.zero_grad(set_to_none=True)
    with flop_counter:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        if use_backward:
            loss.backward()

    total = flop_counter.get_total_flops()
    model.zero_grad(set_to_none=True)
    logger.info(f"Measured FLOPs per batch: {total:,}")
    return int(total)
