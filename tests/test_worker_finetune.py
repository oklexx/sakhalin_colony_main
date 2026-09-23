"""Fine-tuning must honor the new run's LR after optimizer-state restoration."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_lr import apply_configured_learning_rate


class _FakeOptimizer:
    def __init__(self, learning_rates):
        self.param_groups = [{"lr": lr} for lr in learning_rates]


class _FakeScheduler:
    def __init__(self):
        self.base_lrs = [3e-4, 1e-4]
        self._last_lr = [3e-4, 1e-4]


def test_configured_lr_replaces_checkpoint_lr_but_keeps_other_state():
    optimizer = _FakeOptimizer([3e-4, 1e-4])
    optimizer.param_groups[0]["initial_lr"] = 3e-4
    optimizer.param_groups[0]["momentum_marker"] = "kept"
    scheduler = _FakeScheduler()
    old = apply_configured_learning_rate(optimizer, 3e-5, scheduler=scheduler)
    assert old == [3e-4, 1e-4]
    assert [g["lr"] for g in optimizer.param_groups] == [3e-5, 3e-5]
    assert optimizer.param_groups[0]["initial_lr"] == 3e-5
    assert scheduler.base_lrs == [3e-5, 3e-5]
    assert scheduler._last_lr == [3e-5, 3e-5]
    assert optimizer.param_groups[0]["momentum_marker"] == "kept"
