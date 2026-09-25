"""Регресс LR-шедулера PPO (ревью 2026-09-25).

В `rl/ppo.py` косинусный спад считался как `0.1 + 0.5·(1+cos)`: множитель
стартовал с 1.1, а не 1.0. `LambdaLR` применяет `lr_lambda(0)` уже в
`__init__`, поэтому обучение шло на 10% выше заданного LR (3.3e-4 вместо
3e-4) — и `training_lr.apply_configured_learning_rate` после resume
оставлял `_last_lr`, не совпадающий с формулой шедулера.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.actor_critic import ActorCritic
from rl.ppo import PPO
from rl.rollout_buffer import RolloutBuffer
from training_lr import apply_configured_learning_rate

DEVICE = torch.device("cpu")
LR = 3e-4
TOTAL = 1000


def _make_ppo(**kw):
    model = ActorCritic(4, 3, [8], DEVICE)
    buf = RolloutBuffer(4, 2, 4, 3, 0.99, 0.95, DEVICE)
    kw.setdefault("total_training_steps", TOTAL)
    return PPO(model=model, buffer=buf, lr=LR, device=DEVICE,
               use_amp=False, **kw)


def _lr(ppo):
    return ppo.optimizer.param_groups[0]["lr"]


def test_lr_starts_at_configured_value():
    """Множитель в нуле — ровно 1.0, а не 1.1."""
    ppo = _make_ppo()
    assert _lr(ppo) == LR


def test_lr_cosine_endpoints():
    """Спад 1.0 → 0.1 к концу обучения, дальше — пол."""
    ppo = _make_ppo()
    assert ppo.scheduler is not None
    for _ in range(TOTAL):
        ppo.scheduler.step()
    assert _lr(ppo) == LR * 0.1
    # Дальше прогресс зафиксирован на 1.0 — LR стоит на полу.
    for _ in range(10):
        ppo.scheduler.step()
    assert _lr(ppo) == LR * 0.1


def test_lr_cosine_midpoint():
    """Середина косинуса — среднее между 1.0 и 0.1."""
    ppo = _make_ppo()
    assert ppo.scheduler is not None
    for _ in range(TOTAL // 2):
        ppo.scheduler.step()
    expected = LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * 0.5)))
    assert _lr(ppo) == expected


def test_lr_no_decay_without_total_steps():
    """Без total_training_steps шедулер держит 1.0 (как раньше)."""
    ppo = _make_ppo(total_training_steps=0)
    assert ppo.scheduler is not None
    for _ in range(5):
        ppo.scheduler.step()
    assert _lr(ppo) == LR


def test_resume_lr_consistent_with_schedule():
    """После resume base_lrs и _last_lr согласуются с lr_lambda(0) = 1.0."""
    ppo = _make_ppo()
    assert ppo.scheduler is not None
    new_lr = 3e-5
    apply_configured_learning_rate(ppo.optimizer, new_lr,
                                   scheduler=ppo.scheduler)
    assert ppo.scheduler.base_lrs == [new_lr]
    assert _lr(ppo) == new_lr
