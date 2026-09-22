import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
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


# ── Регрессия: пара (flat, minimap) обязана быть из одного состояния ─────────
# «Фикс B1» читал миникарту ДО super().step_wait() и на каждом завершении
# эпизода склеивал пару из двух эпизодов: возвращаемая flat после авто-reset —
# это s'_0 нового эпизода (SB3-контракт ColonyVecEnvCpp::step_wait_batch:
# терминальный obs уходит в infos["terminal_observation"]), а миникарта
# оставалась от старого. См. docs/WATER_HYBRID_AB_2026_09.md.


def _make_done_proxy(env, obs_size):
    """Прокси над cpp_vec: step_wait_batch моделирует done с авто-reset
    (result.obs = наблюдение НОВОГО эпизода), minimap_batch помечает,
    карта какого эпизода читается (строка map_id в канале 1)."""
    import json
    from types import SimpleNamespace

    state = {"map_id": 0}

    class _Proxy:
        def __init__(self, real):
            self._real = real

        def step_wait_batch(self):
            state["map_id"] = 1  # авто-reset: следующие чтения — карта нового
            r = SimpleNamespace()
            r.obs = np.zeros((env.num_envs, obs_size), dtype=np.float32)
            r.rewards = [0.0] * env.num_envs
            r.terminateds = [True] * env.num_envs
            r.trunceds = [False] * env.num_envs
            r.infos = [json.dumps(
                {"terminal_observation": [7.0] * obs_size})] * env.num_envs
            return r

        def minimap_batch(self):
            mm = np.zeros((env.num_envs, 8, 32, 32), dtype=np.float32)
            mm[:, 1, state["map_id"], 0] = 1.0  # маркер эпизода
            return mm

        def __getattr__(self, name):
            return getattr(self._real, name)

    return _Proxy


def test_minimap_pair_uses_post_reset_state_mock():
    """После done пара обязана быть картой НОВОГО эпизода (той же, что flat)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    pytest.importorskip("colony_cpp", reason="colony_cpp не собран (make build)")
    from python.cpp_vecenv_minimap import CppVecEnvMinimap

    for obs_mode in ("hybrid", "minimap"):
        env = CppVecEnvMinimap(n_envs=1, map_size=64, obs_mode=obs_mode, seed=0)
        try:
            env.reset()
            obs_size = int(env.cpp_vec.obs_size())
            env.cpp_vec = _make_done_proxy(env, obs_size)(env.cpp_vec)
            obs, rewards, dones, infos = env.step(np.array([0]))
            assert dones[0]
            mm = obs[1] if obs_mode == "hybrid" else obs
            mm = np.asarray(mm).reshape(1, 8, 32, 32)
            assert mm[0, 1, 1, 0] == 1.0, (
                f"{obs_mode}: пара несёт карту СТАРОГО эпизода — миникарта "
                f"читается до авто-reset (step_wait_batch уже вернул "
                f"пост-reset наблюдение)")
            assert mm[0, 1, 0, 0] == 0.0, f"{obs_mode}: маркера два — битый зонд"
            term = infos[0]["terminal_observation"]
            assert float(np.asarray(term)[0]) == 7.0, (
                "терминальный obs из infos должен сохраниться")
        finally:
            env.close()


def test_hybrid_pair_matches_current_state_live():
    """Живая проверка на реальном C++: парная миникарта равна minimap_batch()
    того состояния, в котором среда оказалась после шага (включая done)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    pytest.importorskip("colony_cpp", reason="colony_cpp не собран (make build)")
    from python.cpp_vecenv_minimap import CppVecEnvMinimap

    env = CppVecEnvMinimap(n_envs=1, map_size=64, obs_mode="hybrid", seed=3)
    try:
        env.reset()
        n_done = 0
        for t in range(8000):
            obs, _rewards, dones, infos = env.step(np.array([0]))
            _flat, mm = obs
            mm_now = env.minimap_obs()
            assert np.array_equal(np.asarray(mm), np.asarray(mm_now)), (
                f"t={t}: пара (flat, minimap) не из текущего состояния")
            if dones[0]:
                assert "terminal_observation" in infos[0]
                assert "terminal_minimap" in infos[0], (
                    "C++ must stash s_T minimap before auto-reset")
                tmm = np.asarray(infos[0]["terminal_minimap"])
                assert tmm.shape == (8, 32, 32)
                # Post-reset pair is a NEW map; terminal minimap is the old one
                # (they may coincide on tiny maps, so only shape/presence here).
                n_done += 1
                if n_done >= 1:
                    break
    finally:
        env.close()
