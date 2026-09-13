from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any, Tuple, List
from python.cpp_vecenv import CppVecEnv


class CppVecEnvMinimap(CppVecEnv):
    """
    CppVecEnv extension that adds minimap observations (for 'minimap' and 'hybrid' obs_mode).
    """

    def __init__(
        self,
        *args,
        minimap_radius: int = 14,
        obs_mode: str = "minimap",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.obs_mode = obs_mode
        self.minimap_radius = minimap_radius
        
        # Set minimap radius in C++ venv
        if hasattr(self.cpp_vec, "set_minimap_radius"):
            self.cpp_vec.set_minimap_radius(minimap_radius)

        self.channels = 8
        self.grid = 2 * minimap_radius + 1

        if obs_mode == "minimap":
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.channels, self.grid, self.grid),
                dtype=np.float32,
            )

    def _get_obs(self):
        flat_obs = super()._get_obs()
        minimap_obs = np.ascontiguousarray(self.cpp_vec.minimap_batch(), dtype=np.float32)
        if self.obs_mode == "minimap":
            return minimap_obs
        elif self.obs_mode == "hybrid":
            return (flat_obs, minimap_obs)
        return flat_obs

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()
        minimap_obs = np.ascontiguousarray(self.cpp_vec.minimap_batch(), dtype=np.float32)
        if self.obs_mode == "minimap":
            return minimap_obs, rewards, dones, infos
        elif self.obs_mode == "hybrid":
            return (obs, minimap_obs), rewards, dones, infos
        return obs, rewards, dones, infos

    def minimap_obs(self) -> np.ndarray:
        return np.ascontiguousarray(self.cpp_vec.minimap_batch(), dtype=np.float32)
