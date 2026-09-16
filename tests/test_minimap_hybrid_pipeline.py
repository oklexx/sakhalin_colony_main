"""End-to-end smoke of the shipped minimap/hybrid training path.

Uses the real factories (`rl.env_manager._make_model/_make_buffer/_make_ppo`)
and the real 8x32x32 minimap geometry that `ColonyEnvCpp::minimap()` emits, so
it exercises the same code the trainer runs -- no stand-in policy or buffer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from rl.actor_critic_hybrid import ActorCriticHybrid
from rl.config import Config
from rl.env_manager import _make_buffer, _make_model, _make_ppo

DEVICE = torch.device("cpu")
GRID = 32  # what ColonyEnvCpp::minimap() actually returns (8*32*32)
CH = 8
OBS_SIZE = 289
N_ACTIONS = 2 + 32 + 11 + 4  # DAY/WEEK + builds + managers + ROAD_E/W/S/N


def _cfg(obs_mode):
    cfg = Config()
    cfg.obs_mode = obs_mode
    cfg.n_envs = 4
    cfg.n_steps = 8
    cfg.batch_size = 8
    cfg.n_epochs = 1
    cfg.net_arch = [32]
    cfg.use_amp = False
    cfg.torch_compile = False
    return cfg


def test_minimap_factory_forward_and_ppo_update():
    cfg = _cfg("minimap")
    model = _make_model(cfg, OBS_SIZE, N_ACTIONS, DEVICE)
    assert model.grid_size == GRID, model.grid_size

    buffer = _make_buffer(cfg, cfg.n_envs, OBS_SIZE, N_ACTIONS, DEVICE)
    assert tuple(buffer.obs.shape) == (cfg.n_steps * cfg.n_envs, CH, GRID, GRID)

    ppo = _make_ppo(cfg, model, buffer)
    assert not ppo.is_hybrid

    masks = torch.ones(cfg.n_envs, N_ACTIONS)
    obs = torch.rand(cfg.n_envs, CH, GRID, GRID)
    for _ in range(cfg.n_steps):
        out = ppo.collect_step(obs, action_masks=masks)
        buffer.add(obs=obs, action=out["action"], reward=torch.rand(cfg.n_envs),
                   log_prob=out["log_prob"], value=out["value"],
                   done=torch.zeros(cfg.n_envs, dtype=torch.bool),
                   terminated=torch.zeros(cfg.n_envs, dtype=torch.bool),
                   action_masks=masks)
        obs = torch.rand(cfg.n_envs, CH, GRID, GRID)

    with torch.no_grad():  # AsyncTrainer._rollout computes last_value under no_grad
        last_value = model.get_value(obs)
    stats = ppo.update(last_value=last_value,
                       last_done=torch.zeros(cfg.n_envs, dtype=torch.bool))
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])


def test_hybrid_factory_forward_and_ppo_update():
    cfg = _cfg("hybrid")
    model = _make_model(cfg, OBS_SIZE, N_ACTIONS, DEVICE)
    assert model.grid_size == GRID, model.grid_size
    assert isinstance(model, ActorCriticHybrid)

    buffer = _make_buffer(cfg, cfg.n_envs, OBS_SIZE, N_ACTIONS, DEVICE)
    assert tuple(buffer.obs.shape) == (cfg.n_steps * cfg.n_envs, CH, GRID, GRID)
    assert tuple(buffer.flat_obs.shape) == (cfg.n_steps * cfg.n_envs, OBS_SIZE)

    ppo = _make_ppo(cfg, model, buffer)
    assert ppo.is_hybrid

    masks = torch.ones(cfg.n_envs, N_ACTIONS)
    flat = torch.rand(cfg.n_envs, OBS_SIZE)
    mm = torch.rand(cfg.n_envs, CH, GRID, GRID)
    for _ in range(cfg.n_steps):
        out = ppo.collect_step(flat, mm, action_masks=masks)
        buffer.add(obs=mm, flat=flat, action=out["action"],
                   reward=torch.rand(cfg.n_envs), log_prob=out["log_prob"],
                   value=out["value"],
                   done=torch.zeros(cfg.n_envs, dtype=torch.bool),
                   terminated=torch.zeros(cfg.n_envs, dtype=torch.bool),
                   action_masks=masks)
        flat = torch.rand(cfg.n_envs, OBS_SIZE)
        mm = torch.rand(cfg.n_envs, CH, GRID, GRID)

    with torch.no_grad():
        last_value = model.get_value(flat, mm)
    stats = ppo.update(last_value=last_value,
                       last_done=torch.zeros(cfg.n_envs, dtype=torch.bool))
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])


def test_stale_minimap_radius_config_is_ignored():
    """`minimap_radius` is still in every config, but nets are built at 32."""
    cfg = _cfg("minimap")
    cfg.minimap_radius = 14  # exactly what configs/exp_25_minimap.json sets
    model = _make_model(cfg, OBS_SIZE, N_ACTIONS, DEVICE)
    assert model.grid_size == 32
    assert model.minimap_radius == (32 - 1) // 2  # silently rewritten to 15


def test_compute_gae_survives_a_last_value_that_requires_grad():
    """Regression: advantages were whitened in place from their own mean/std.

    If `last_value` carries a grad_fn the in-place write bumped the version of
    a tensor the backward graph still needed, and PPO.update() died with
    "modified by an inplace operation". AsyncTrainer only avoided this by
    calling get_value() under no_grad -- compute_gae must not depend on that.
    """
    cfg = _cfg("minimap")
    model = _make_model(cfg, OBS_SIZE, N_ACTIONS, DEVICE)
    buffer = _make_buffer(cfg, cfg.n_envs, OBS_SIZE, N_ACTIONS, DEVICE)
    ppo = _make_ppo(cfg, model, buffer)

    masks = torch.ones(cfg.n_envs, N_ACTIONS)
    obs = torch.rand(cfg.n_envs, CH, GRID, GRID)
    for _ in range(cfg.n_steps):
        out = ppo.collect_step(obs, action_masks=masks)
        buffer.add(obs=obs, action=out["action"], reward=torch.rand(cfg.n_envs),
                   log_prob=out["log_prob"], value=out["value"],
                   done=torch.zeros(cfg.n_envs, dtype=torch.bool),
                   terminated=torch.zeros(cfg.n_envs, dtype=torch.bool),
                   action_masks=masks)
        obs = torch.rand(cfg.n_envs, CH, GRID, GRID)

    # compute_gae must not care whether the caller remembered no_grad: it used
    # to raise the version-counter error from the in-place whitening write.
    grad_value = model.get_value(obs)          # deliberately NOT under no_grad
    assert grad_value.requires_grad
    buffer.compute_gae(grad_value, torch.zeros(cfg.n_envs, dtype=torch.bool))
    # ...and the stored targets must come out detached from the policy graph.
    assert not buffer.advantages.requires_grad
    assert not buffer.returns.requires_grad


def test_returns_use_unnormalised_advantages():
    """`returns` is the value-head target and must not be whitened."""
    cfg = _cfg("minimap")
    buffer = _make_buffer(cfg, cfg.n_envs, OBS_SIZE, N_ACTIONS, DEVICE)
    n = cfg.n_steps * cfg.n_envs

    torch.manual_seed(0)
    buffer.rewards.copy_(torch.randn(n))
    buffer.values.copy_(torch.randn(n))
    buffer.dones.fill_(False)
    buffer.terminated.fill_(False)

    buffer.compute_gae(torch.zeros(cfg.n_envs), torch.zeros(cfg.n_envs, dtype=torch.bool))
    raw_adv = buffer.advantages[:n].clone()

    # recompute GAE into a second buffer and rebuild returns by hand
    cfg2 = _cfg("minimap")
    b2 = _make_buffer(cfg2, cfg.n_envs, OBS_SIZE, N_ACTIONS, DEVICE)
    b2.rewards.copy_(buffer.rewards)
    b2.values.copy_(buffer.values)
    b2.dones.fill_(False)
    b2.terminated.fill_(False)
    # bypass whitening by asking for the pre-normalisation advantages:
    # returns - values must equal the *unwhitened* advantage, so its mean is
    # generally not ~0 while the stored advantage mean is.
    implied = buffer.returns[:n] - buffer.values[:n]
    assert float(buffer.advantages[:n].mean().abs()) < 1e-5, "advantages should be whitened"
    assert float(implied.mean().abs()) > 1e-3, "returns must not be whitened"
    # raw_adv is the whitened advantage (std ~1); the advantage implied by
    # returns-values is the raw GAE estimate and must NOT have unit std.
    assert abs(float(raw_adv.std()) - 1.0) < 1e-4
    assert abs(float(implied.std()) - 1.0) > 1e-2
