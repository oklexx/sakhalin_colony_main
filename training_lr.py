"""Small optimizer/scheduler helper shared by CLI and UI fine-tuning."""
from __future__ import annotations


def apply_configured_learning_rate(optimizer, learning_rate: float,
                                   scheduler=None) -> list[float]:
    """Set the new run's LR without discarding restored optimizer moments.

    ``Optimizer.load_state_dict`` restores checkpoint param-group learning
    rates too. Keep its Adam moments, but reset current rates and the freshly
    created scheduler's base rates to the explicit LR selected for this run.
    The scheduler itself remains fresh; no scheduler state is restored from
    the checkpoint.
    """
    requested = float(learning_rate)
    previous = []
    for group in optimizer.param_groups:
        previous.append(float(group.get("lr", requested)))
        group["lr"] = requested
        if "initial_lr" in group:
            group["initial_lr"] = requested
    if scheduler is not None:
        scheduler.base_lrs = [requested] * len(optimizer.param_groups)
        if hasattr(scheduler, "_last_lr"):
            scheduler._last_lr = [float(group["lr"]) for group in optimizer.param_groups]
    return previous
