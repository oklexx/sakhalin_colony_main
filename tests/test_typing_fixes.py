"""Регрессии, найденные при включении mypy в CI (2026-09-25).

Каждый тест — баг, который mypy показал как ошибку типов, а в рантайме он
проявлялся бы только на редком пути.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _add_kwargs(n_envs: int = 2) -> dict:
    return dict(
        obs=torch.zeros(n_envs, 4),
        action=torch.zeros(n_envs, dtype=torch.long),
        reward=torch.zeros(n_envs),
        log_prob=torch.zeros(n_envs),
        value=torch.zeros(n_envs),
        done=torch.zeros(n_envs, dtype=torch.bool),
    )


def test_worker_requires_output():
    """Без --output run_train/run_eval падали на `None.write(...)` (mf = None)."""
    from train_ui2 import worker as W

    with pytest.raises(SystemExit):
        W._build_parser().parse_args(["--config", "c.json"])


def test_flat_buffer_rejects_flat_obs():
    """У плоского буфера хранить flat негде: раньше kwarg молча терялся."""
    from rl.rollout_buffer import RolloutBuffer

    buf = RolloutBuffer(n_steps=2, n_envs=2, obs_size=4, n_actions=3,
                        gamma=0.99, gae_lambda=0.95, device=torch.device("cpu"))
    with pytest.raises(TypeError, match="flat"):
        buf.add(**_add_kwargs(), flat=torch.zeros(2, 4))
    assert buf.pos == 0


def test_hybrid_buffer_positional_masks_are_masks():
    """`flat` стоял 8-м позиционным у подкласса и перехватывал action_masks.

    Сигнатура подкласса теперь совпадает с базой (flat — keyword-only), так
    что позиционный вызов «как у RolloutBuffer» кладёт маски в маски.
    """
    from rl.rollout_buffer import _TensorRolloutBuffer

    buf = _TensorRolloutBuffer(n_steps=2, n_envs=2, obs_shape=(4,), n_actions=3,
                               gamma=0.99, gae_lambda=0.95,
                               device=torch.device("cpu"))
    kw = _add_kwargs()
    masks = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    buf.add(kw["obs"], kw["action"], kw["reward"], kw["log_prob"], kw["value"],
            kw["done"], None, masks)
    assert torch.equal(buf.action_masks[0:2], masks)


def test_single_module_identity_for_python_dir():
    """`python.cpp_vecenv` и `cpp_vecenv` были двумя разными модулями.

    Monkeypatch одного (scripts/water_ab_run.py) не был виден через другой.
    Весь код импортирует плоское имя; пакетного больше нет.
    """
    offenders = []
    for path in [*ROOT.glob("*.py"), *ROOT.glob("rl/*.py"), *ROOT.glob("python/*.py"),
                 *ROOT.glob("train_ui2/*.py"), *ROOT.glob("scripts/*.py"),
                 *ROOT.glob("tests/*.py")]:
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        if "from python." in text or "import python." in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"импорт через пакет `python.`: {offenders}"
