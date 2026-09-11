import os
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple
import colony_cpp
from pathlib import Path

# Project root is parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Normalizer:
    """Single-env observation normalizer using C++ RunningMeanStd.

    Wraps colony_cpp.RunningMeanStd for use with CppColonyEnv (single env),
    matching the normalization applied by CppVecEnv during training.
    """

    def __init__(self, obs_size: int):
        self._rms = colony_cpp.RunningMeanStd(obs_size)
        self._obs_size = obs_size
        self._clip = 10.0
        self._update_enabled = True

    def set_update(self, enable: bool):
        """Enable/disable running statistics updates.

        Set to False during eval to prevent statistics drift.
        """
        self._update_enabled = enable

    def update(self, obs: np.ndarray):
        """Update running statistics with a single observation (if enabled)."""
        if not self._update_enabled:
            return
        obs_flat = np.asarray(obs, dtype=np.float32).flatten().copy()
        self._rms.update(obs_flat, 1, self._obs_size)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a single observation using current statistics."""
        obs_flat = np.asarray(obs, dtype=np.float32).flatten().copy()
        self._rms.normalize(obs_flat, 1, self._obs_size, self._clip)
        return obs_flat

    def save(self, path: str):
        """Save normalization stats to JSON file."""
        import json
        d = self.to_dict()
        with open(path, "w") as f:
            json.dump(d, f)

    def load(self, path: str):
        """Load normalization stats from JSON file.

        Accepts both the flat shape produced by Normalizer.save (mean/var/count
        at top level) and the C++ ColonyVecEnvCpp::save_normalization nested
        shape where stats live under an "obs_rms" key.
        """
        import json
        with open(path) as f:
            d = json.load(f)
        rms = d.get("obs_rms", d)
        self._rms.set_mean(rms["mean"])
        self._rms.set_var(rms["var"])
        self._rms.set_count(rms["count"])
        self._obs_size = d.get("obs_size", self._obs_size)
        self._clip = d.get("clip", d.get("clip_obs", 10.0))

    def to_dict(self) -> dict:
        return {
            "mean": list(self._rms.mean()),
            "var": list(self._rms.var()),
            "count": int(self._rms.count()),
            "obs_size": self._obs_size,
            "clip": self._clip,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        rms = d.get("obs_rms", d)
        n = cls(obs_size=int(d.get("obs_size", 0)))
        n._rms.set_mean(rms["mean"])
        n._rms.set_var(rms["var"])
        n._rms.set_count(rms["count"])
        n._clip = d.get("clip", d.get("clip_obs", 10.0))
        return n


class CppColonyEnv(gym.Env):
    """
    Gymnasium wrapper for C++ ColonyEnvCpp.
    Observation space: 207-dim float32 vector
    Action space: Discrete(45) - 0=DAY, 1=WEEK, 2-33=BUILD, 34-44=MANAGER
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        map_size: int = 280,
        curriculum_stage: int = 0,
        unlock_ids: Optional[str] = None,
        disable_net_worth: bool = False,
        disable_daily_income: bool = False,
        reward_config: Optional[Dict[str, float]] = None,
        difficulty: str = "normal",
    ):
        super().__init__()
        
        # Load static data
        self.base_data = colony_cpp.load_base_data(str(PROJECT_ROOT / "configs" / "bases.json"))
        self.events_data = colony_cpp.load_events(str(PROJECT_ROOT / "configs" / "events.json"))
        
        # Reward configuration — DRY: apply whatever caller provides
        rc = colony_cpp.RewardConfig()
        _INT_KEYS = {"idle_build_threshold_days"}
        _missing: list[str] = []
        if reward_config:
            for key, value in reward_config.items():
                if key in _INT_KEYS:
                    try:
                        value = int(value)  # type: ignore[assignment]
                    except Exception:
                        pass
                if not hasattr(rc, key):
                    _missing.append(key)
                    continue
                setattr(rc, key, value)
        if _missing:
            print(f"[CppColonyEnv] WARNING: colony_cpp.pyd stale — keys "
                  f"{_missing} not settable; rebuild.", flush=True)
        # CLI flags override only if explicitly True (keep JSON value otherwise)
        if disable_net_worth:
            rc.disable_net_worth = True
        if disable_daily_income:
            rc.disable_daily_income = True
        
        if os.environ.get("COLONY_DEBUG", ""):
            print("=" * 60, flush=True)
            print("[DEBUG CppColonyEnv] C++ RewardConfig after setup:", flush=True)
            if reward_config:
                for k, v in sorted(reward_config.items()):
                    print(f"  rc.{k:25s} = {v} (requested)", flush=True)
            print(f"  rc.disable_net_worth      = {rc.disable_net_worth}", flush=True)
            print(f"  rc.disable_daily_income   = {rc.disable_daily_income}", flush=True)
            print("=" * 60, flush=True)
        
        # Parse unlock_ids
        unlock_list = []
        if unlock_ids:
            unlock_list = [s.strip() for s in unlock_ids.split(",") if s.strip()]
        
        # Create C++ environment
        self.cpp_env = colony_cpp.ColonyEnvCpp(
            self.base_data,
            self.events_data,
            seed=0,  # will be set in reset
            map_size=map_size,
            curriculum_stage=curriculum_stage,
            unlock_ids=unlock_list,
            reward=rc,
            difficulty=difficulty,
        )
        
        # Spaces (use clipped observation space as SB3 expects)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.cpp_env.obs_size(),),
            dtype=np.float32
        )
        # Fix: VecNormalize expects clip_obs to be set, default None causes TypeError
        # Set clip_obs=10.0 which matches the VecNormalize clip_obs parameter in train.py
        self.clip_obs = 10.0
        self.action_space = gym.spaces.Discrete(self.cpp_env.n_actions())
        self.normalizer = Normalizer(obs_size=self.cpp_env.obs_size())
        
        # Action name mapping (for debugging)
        self._action_names = ["DAY", "WEEK"] + self.cpp_env.build_ids() + [
            "IMPROVE_LAND", "REPAIR", "REPAIR_ALL", "DEMOLISH", "PRESERVE",
            "UNPRESERVE", "SELL_SURPLUS", "BUY_FOOD", "TAKE_LOAN", "REPAY_LOAN", "PAY_TAX"
        ]
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        s = seed if seed is not None else np.random.randint(1 << 31)
        self.cpp_env.reset(s)
        raw_obs = np.array(self.cpp_env.obs(), dtype=np.float32)
        self.normalizer.update(raw_obs)
        obs = self.normalizer.normalize(raw_obs)
        return obs, {"seed": s, "days": 0}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        result = self.cpp_env.step(int(action))
        raw_obs = np.array(result["obs"], dtype=np.float32)
        self.normalizer.update(raw_obs)
        obs = self.normalizer.normalize(raw_obs)
        reward = float(result["reward"])
        terminated = bool(result["terminated"])
        truncated = bool(result["truncated"])
        info = {
            "days": int(result["days"]),
            "people": int(result["people"]),
            "money": int(result["money"]),
            "bases": int(result["bases"]),
            "tax_due_days": int(result["tax_due_days"]),
            "ep_return": float(result["ep_return"]),
            "steps": int(result["steps"]),
            "action_name": self._action_names[action] if action < len(self._action_names) else str(action),
        }
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            # Return a simple representation - could be extended to show actual map
            return np.zeros((400, 400, 3), dtype=np.uint8)
        return None
    
    def set_step_log(self, path: str):
        self.cpp_env.set_step_log(path)

    def action_mask(self) -> np.ndarray:
        """Return boolean mask of available actions [n_actions]."""
        return np.asarray(self.cpp_env.action_mask(), dtype=bool)

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self

def make_env(
    map_size: int = 280,
    curriculum_stage: int = 0,
    unlock_ids: Optional[str] = None,
    disable_net_worth: bool = False,
    disable_daily_income: bool = False,
    reward_config: Optional[Dict[str, float]] = None,
    seed: int = 0,
    difficulty: str = "normal",
) -> gym.Env:
    """Factory function for SubprocVecEnv compatibility."""
    def _init():
        env = CppColonyEnv(
            map_size=map_size,
            curriculum_stage=curriculum_stage,
            unlock_ids=unlock_ids,
            disable_net_worth=disable_net_worth,
            disable_daily_income=disable_daily_income,
            reward_config=reward_config,
            difficulty=difficulty,
        )
        env.reset(seed=seed)
        return env
    return _init