from __future__ import annotations

from typing import Any

import torch


def move_batch_to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if hasattr(batch, "to") and callable(batch.to):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [move_batch_to_device(v, device, non_blocking) for v in batch]
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    return batch


def assert_cuda_batch(batch, warn_only: bool = False) -> None:
    items = batch.items() if isinstance(batch, dict) else batch
    for key, value in items:
        if torch.is_tensor(value) and value.is_floating_point():
            if not value.is_cuda:
                msg = f"{key} is still on CPU"
                if warn_only:
                    import warnings
                    warnings.warn(msg)
                else:
                    raise AssertionError(msg)
