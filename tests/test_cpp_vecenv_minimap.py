import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import torch
from rl.config import Config
from rl.env_manager import EnvManager


def test_env_manager_minimap_mode():
    cfg = Config()
    cfg.n_envs = 2
    cfg.obs_mode = "minimap"
    cfg.minimap_radius = 14
    device = torch.device("cpu")

    em = EnvManager(cfg, device)
    assert em.n_envs == 2
    obs = em.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (2, 8, 32, 32)

    actions = np.zeros(2, dtype=np.int64)
    next_obs, rewards, dones, infos = em.step(actions)
    assert next_obs.shape == (2, 8, 32, 32)
    assert rewards.shape == (2,)
    assert dones.shape == (2,)
    em.close()


def test_env_manager_hybrid_mode():
    cfg = Config()
    cfg.n_envs = 2
    cfg.obs_mode = "hybrid"
    cfg.minimap_radius = 14
    device = torch.device("cpu")

    em = EnvManager(cfg, device)
    assert em.n_envs == 2
    obs = em.reset()
    assert isinstance(obs, tuple)
    flat_obs, minimap_obs = obs
    assert isinstance(flat_obs, torch.Tensor)
    assert isinstance(minimap_obs, torch.Tensor)
    assert minimap_obs.shape == (2, 8, 32, 32)

    # Test collect_step in hybrid mode
    next_obs, infos = em.collect_step()
    assert isinstance(next_obs, tuple)
    next_flat, next_mm = next_obs
    assert next_mm.shape == (2, 8, 32, 32)
    em.close()
