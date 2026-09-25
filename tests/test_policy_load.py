"""P1-3 ревью 2026-09-24: `load_policy_state()` — единая загрузка весов.

`train.py --resume-model` грузил strict=True и падал на любом чекпойнте
старше 2026-09-22 (нет actor/critic_mask_proj), а остальные места грузили
strict=False без проверки и молча оставляли случайные веса при чужой
архитектуре. Контракт теперь один на всех.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from rl._nn_common import OPTIONAL_POLICY_KEY_PREFIXES, load_policy_state
from rl.actor_critic import ActorCritic

REPO = Path(__file__).resolve().parent.parent
CPU = torch.device("cpu")


def _legacy_state(model: ActorCritic, compiled: bool = False) -> dict:
    """state_dict «как до 09-22»: без mask-проекций (+ опц. префикс compile)."""
    pre = "_orig_mod." if compiled else ""
    return {pre + k: v.clone() for k, v in model.state_dict().items()
            if not k.startswith(OPTIONAL_POLICY_KEY_PREFIXES)}


@pytest.mark.parametrize("compiled", [False, True])
def test_legacy_checkpoint_loads_and_projections_stay_zero(compiled):
    torch.manual_seed(1)
    src = ActorCritic(10, 5, [16], CPU)
    with torch.no_grad():
        src.critic_head.weight.add_(1.0)
    dst = ActorCritic(10, 5, [16], CPU)
    logs: list[str] = []
    report = load_policy_state(dst, _legacy_state(src, compiled), "old.pt", log=logs.append)
    assert torch.equal(dst.critic_head.weight, src.critic_head.weight)
    assert torch.count_nonzero(dst.actor_mask_proj.weight) == 0
    assert torch.count_nonzero(dst.critic_mask_proj.weight) == 0
    assert set(report.missing_optional) == {"actor_mask_proj.weight", "critic_mask_proj.weight"}
    assert any("legacy checkpoint" in m for m in logs)


def test_wrong_obs_size_is_value_error_with_path():
    src = ActorCritic(10, 5, [16], CPU)
    dst = ActorCritic(12, 5, [16], CPU)
    with pytest.raises(ValueError, match=r"old\.pt.*trunk\.0\.weight"):
        load_policy_state(dst, src.state_dict(), "old.pt")


def test_missing_required_parameter_is_rejected():
    src = ActorCritic(10, 5, [16], CPU)
    state = {k: v for k, v in src.state_dict().items() if not k.startswith("critic_head.")}
    with pytest.raises(ValueError, match="critic_head"):
        load_policy_state(ActorCritic(10, 5, [16], CPU), state, "x.pt")


def test_unexpected_keys_are_warned_not_fatal():
    src = ActorCritic(10, 5, [16], CPU)
    state = dict(src.state_dict())
    state["legacy_extra.weight"] = torch.zeros(1)
    logs: list[str] = []
    rep = load_policy_state(ActorCritic(10, 5, [16], CPU), state, "x.pt", log=logs.append)
    assert rep.unexpected == ["legacy_extra.weight"]
    assert any("unexpected" in m for m in logs)


def test_ppo_load_accepts_legacy_checkpoint(tmp_path):
    from rl.ppo import PPO
    from rl.rollout_buffer import RolloutBuffer

    def make() -> PPO:
        m = ActorCritic(10, 5, [16], CPU)
        buf = RolloutBuffer(n_steps=2, n_envs=1, obs_size=10, n_actions=5,
                            gamma=0.99, gae_lambda=0.95, device=CPU)
        return PPO(model=m, buffer=buf, lr=1e-3, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                   n_epochs=1, batch_size=1, use_amp=False, device=CPU)

    src = make()
    path = tmp_path / "old.pt"
    torch.save({"model_state": _legacy_state(src.model)}, path)
    dst = make()
    dst.load(str(path))
    assert torch.equal(dst.model.critic_head.weight, src.model.critic_head.weight)


def test_all_checkpoint_loaders_use_the_shared_helper():
    """Никаких прямых model.load_state_dict(...) в путях загрузки политики."""
    files = ["train.py", "train_ui2/worker.py", "train_ui2/evaluator.py", "rl/ppo.py"]
    for rel in files:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        direct = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "load_state_dict"
            and "optimizer" not in ast.unparse(n.func.value)
        ]
        assert not direct, f"{rel}: прямой load_state_dict в строках {direct}"
