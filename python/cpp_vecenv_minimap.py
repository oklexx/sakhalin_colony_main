from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from cpp_vecenv import CppVecEnv


class CppVecEnvMinimap(CppVecEnv):
    """
    CppVecEnv extension that adds minimap observations (for 'minimap' and 'hybrid' obs_mode).
    """

    def __init__(
        self,
        *args: Any,
        minimap_radius: int = 14,
        obs_mode: str = "minimap",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.obs_mode = obs_mode
        self._configure_terminal_minimap()  # obs_mode известен только сейчас
        self.minimap_radius = minimap_radius
        self.channels = 8
        self.grid = 32

        if obs_mode == "minimap":
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.channels, self.grid, self.grid),
                dtype=np.float32,
            )

    # hybrid отдаёт пару (flat, minimap) — сознательно шире, чем np.ndarray у базы.
    def _get_obs(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        flat_obs = super()._get_obs()
        minimap_obs = np.ascontiguousarray(self.cpp_vec.minimap_batch(), dtype=np.float32)
        if self.obs_mode == "minimap":
            return minimap_obs
        elif self.obs_mode == "hybrid":
            return (flat_obs, minimap_obs)
        return flat_obs

    def step_wait(  # type: ignore[override]
        self,
    ) -> tuple[np.ndarray | tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        # ПОРЯДОК ВАЖЕН: миникарта читается ПОСЛЕ super().step_wait().
        # `step_wait_batch()` при done ВОЗВРАЩАЕТ пост-reset наблюдение (контракт
        # SB3: obs — это s'_0 нового эпизода, терминальный obs уезжает в
        # infos["terminal_observation"]), поэтому парой для этой flat обязана
        # быть карта НОВОГО эпизода — ровно та, что в minimap_batch() уже после
        # step_wait. Прежний порядок («прочитать до», так называемый «фикс B1»)
        # склеивал на каждом завершении эпизода пару из двух эпизодов:
        # flat = s'_0 нового, миникарта = s_T старого (живой замер: max|Δ
        # каналов| = 1.0) — а это ровно те шаги, где агент ищет воду по карте.
        # Регрессия: tests/test_cpp_vecenv_minimap.py (пара после авто-reset).
        obs, rewards, dones, infos = super().step_wait()
        minimap_obs = np.ascontiguousarray(self.cpp_vec.minimap_batch(), dtype=np.float32)
        if self.obs_mode == "minimap":
            return minimap_obs, rewards, dones, infos
        elif self.obs_mode == "hybrid":
            return (obs, minimap_obs), rewards, dones, infos
        return obs, rewards, dones, infos

    def minimap_obs(self) -> np.ndarray:
        return np.ascontiguousarray(self.cpp_vec.minimap_batch(), dtype=np.float32)
