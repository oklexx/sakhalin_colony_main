import json
import os
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from stable_baselines3.common.vec_env import VecEnv
import colony_cpp
from colony_cpp_api import require_colony, stale_allowed, StaleExtensionError
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CppVecEnv(VecEnv):
    """
    Batched vectorized environment using C++ ColonyVecEnvCpp.
    N environments run in one process with a thread pool.
    VecNormalize is handled in C++ (no Python VecNormalize needed).

    Inherits from stable_baselines3.common.vec_env.VecEnv for SB3 compatibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_envs: int,
        map_size: int = 280,
        curriculum=None,  # CurriculumState | dict | None (None = unrestricted)
        disable_net_worth: Optional[bool] = None,
        disable_daily_income: Optional[bool] = None,
        reward_config: Optional[Dict[str, float]] = None,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        seed: int = 0,
        n_threads: int = 0,
        render_mode: Optional[str] = None,
        difficulty: str = "normal",
    ):
        # PR 3: fail fast on a stale binary (escape: COLONY_ALLOW_STALE_PYD=1).
        require_colony()
        self._seed = seed
        self.difficulty = difficulty
        self.render_mode = render_mode

        # Load static data
        base_data = colony_cpp.load_base_data(
            str(PROJECT_ROOT / "configs" / "bases.json")
        )
        events_data = colony_cpp.load_events(
            str(PROJECT_ROOT / "configs" / "events.json")
        )

        # Reward configuration — DRY: use whatever keys caller provides,
        # no hardcoded list. Unknown keys are ignored (forward compat), missing
        # fields keep C++ defaults.
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
            # The handshake above already rejected stale binaries, so unknown
            # keys here mean a typo in the reward config (or a config newer
            # than the extension) — fail loudly instead of silently ignoring.
            _msg = (f"unknown RewardConfig keys {_missing} — typo in reward "
                    f"config or stale colony_cpp binary; rebuild with "
                    f"build_pyext.bat (or set COLONY_ALLOW_STALE_PYD=1 to ignore)")
            if stale_allowed():
                print(f"[CppVecEnv] WARNING: {_msg}", flush=True)
            else:
                raise StaleExtensionError(f"[CppVecEnv] {_msg}")
        # Apply CLI flags only if explicitly provided (None = use JSON/reward_config value)
        if disable_net_worth is not None:
            rc.disable_net_worth = disable_net_worth
        if disable_daily_income is not None:
            rc.disable_daily_income = disable_daily_income

        if os.environ.get("COLONY_DEBUG", "").lower() in ("1", "true", "yes", "on"):
            print("=" * 60, flush=True)
            print("[DEBUG CppVecEnv] C++ RewardConfig after setup:", flush=True)
            if reward_config:
                for k, v in sorted(reward_config.items()):
                    print(f"  rc.{k:25s} = {v} (requested)", flush=True)
            print(f"  rc.disable_net_worth      = {rc.disable_net_worth}", flush=True)
            print(f"  rc.disable_daily_income   = {rc.disable_daily_income}", flush=True)
            print("=" * 60, flush=True)

        # PR 1: single curriculum contract (duck-typed — no rl import here, so
        # this module stays importable without torch: rl depends on python/, not vice versa).
        if curriculum is None:
            curr_dict = {"all_builds": True, "allowed_builds": [], "stage": 0}
            self.curriculum_state = None
        elif isinstance(curriculum, dict):
            curr_dict = curriculum
            self.curriculum_state = None
        else:
            curr_dict = curriculum.to_dict()
            self.curriculum_state = curriculum
        print(f"[CppVecEnv] init: n_envs={n_envs}, curriculum_all={curr_dict.get('all_builds', True)}, "
              f"allowed_builds={len(curr_dict.get('allowed_builds', []))}", flush=True)

        # Create C++ batched environment
        self.cpp_vec = colony_cpp.ColonyVecEnvCpp(
            base_data, events_data,
            n_envs=n_envs,
            base_seed=seed,
            map_size=map_size,
            curriculum=curr_dict,
            reward=rc,
            n_threads=n_threads,
            difficulty=difficulty,
        )

        obs_size = self.cpp_vec.obs_size()
        n_actions = self.cpp_vec.n_actions()

        observation_space = gym.spaces.Box(
            low=-clip_obs, high=clip_obs,
            shape=(obs_size,),
            dtype=np.float32,
        )
        action_space = gym.spaces.Discrete(n_actions)

        # Action masks: [n_envs, n_actions] float32 (1.0=available, 0.0=blocked)
        self._action_masks = np.ones((n_envs, n_actions), dtype=np.float32)

        # Init SB3 VecEnv (sets self.num_envs, self.observation_space, self.action_space)
        super().__init__(n_envs, observation_space, action_space)

        # Action names for display purposes.
        # Layout (constants.h): [DAY, WEEK] + BUILD_SUBSET (31 buildings, in
        # BUILD_SUBSET order) + 11 manager actions.
        # Building names: prefer C++ build order, fall back to BUILD_SUBSET order
        try:
            build_ids = list(self.cpp_vec.build_ids())
        except AttributeError:
            build_ids = [
                "WaterChannel", "Farm", "Garden", "House", "SmallHouse",
                "Sawmill", "Coalmine", "Ironmine", "Refinery", "Goldmine",
                "PowerStation", "HydroStation", "Road", "Fish", "CoalCut",
                "HuntingLand", "CowFarm", "Mushroom", "BigHouse", "BigFarm",
                "Apiary", "Torchlight", "Hothouse", "SuperHouse", "BigSawmill",
                "WaterMill", "BigRefinary", "Puerperal", "BigIronmine",
                "AirStation", "SmallAtomStation", "AtomStation",
            ]
            if len(build_ids) > self.cpp_vec.n_build():
                build_ids = build_ids[: self.cpp_vec.n_build()]
        self._build_names: List[str] = [
            "BUILD_" + b.upper().replace(" ", "_") for b in build_ids
        ]
        self._manager_names: List[str] = [
            "IMPROVE_LAND", "REPAIR", "REPAIR_ALL", "DEMOLISH", "PRESERVE",
            "UNPRESERVE", "SELL_SURPLUS", "BUY_FOOD", "TAKE_LOAN", "REPAY_LOAN", "PAY_TAX",
        ]
        self._action_names: List[str] = ["DAY", "WEEK"] + self._build_names + self._manager_names
        # Trim to actual n_actions if C++ has fewer
        if n_actions < len(self._action_names):
            self._action_names = self._action_names[:n_actions]
        # Pad with generic names if C++ has more
        while len(self._action_names) < n_actions:
            self._action_names.append(f"ACTION_{len(self._action_names)}")

    def reset(self):
        seeds = [self._seed + i * 10000 for i in range(self.num_envs)]
        self.cpp_vec.reset_batch(seeds)
        self._action_masks = np.asarray(
            self.cpp_vec.action_masks_batch(), dtype=np.float32
        )
        self.reset_infos = [{} for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return self._get_obs()

    def step_async(self, actions: Any) -> None:
        actions_list = actions.tolist() if hasattr(actions, "tolist") else list(actions)
        self.cpp_vec.step_async_batch(actions_list)

    def step_wait(self):
        result = self.cpp_vec.step_wait_batch()
        obs = self._reshape_obs(result.obs)
        rewards = np.array(result.rewards, dtype=np.float64)
        terminateds = np.array(result.terminateds, dtype=bool)
        trunceds = np.array(result.trunceds, dtype=bool)
        dones = terminateds | trunceds

        # Update action masks after step
        self._action_masks = np.asarray(
            self.cpp_vec.action_masks_batch(), dtype=np.float32
        )

        infos = []
        for i in range(self.num_envs):
            info_str = result.infos[i]
            if info_str and info_str != "{}":
                info = json.loads(info_str)
            else:
                info = {}
            if dones[i]:
                info["terminal_observation"] = obs[i].copy()
                info["TimeLimit.truncated"] = bool(trunceds[i] and not terminateds[i])
            infos.append(info)

        # Expose the raw termination flag separately from the (terminated|truncated)
        # `dones` so the RL layer can bootstrap GAE correctly on truncation.
        self._last_terminateds = terminateds
        return obs, rewards, dones, infos

    @property
    def action_masks(self) -> np.ndarray:
        """Return current action masks [n_envs, n_actions]."""
        return self._action_masks

    @property
    def action_names(self) -> List[str]:
        """Return list of action names matching action indices."""
        return list(self._action_names)

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None):
        target_envs = [self] * self.num_envs if indices is None else [self]
        return [getattr(env, attr_name) for env in target_envs]

    def set_attr(self, attr_name: str, value, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    @property
    def venv(self):
        return self.cpp_vec

    def _get_obs(self) -> np.ndarray:
        raw = self.cpp_vec.obs_buffer()
        return self._reshape_obs(raw)

    def _reshape_obs(self, raw) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        return arr.reshape(self.num_envs, -1)

    def dump_obs(self, env_idx: int = 0) -> str:
        """Return a human-readable observation dump for one sub-env."""
        return self.cpp_vec.dump_obs(env_idx)

    def get_images(self):
        return [None] * self.num_envs

    def render(self):
        if self.render_mode == "rgb_array":
            return np.zeros((self.num_envs, 400, 400, 3), dtype=np.uint8)
        return None


def make_cpp_vec_env(
    n_envs: int = 8,
    map_size: int = 280,
    curriculum=None,
    disable_net_worth: bool = False,
    disable_daily_income: bool = False,
    reward_config: Optional[Dict[str, float]] = None,
    seed: int = 0,
    n_threads: int = 0,
    render_mode: Optional[str] = None,
    difficulty: str = "normal",
) -> CppVecEnv:
    """Factory for creating CppVecEnv."""
    return CppVecEnv(
        n_envs=n_envs,
        map_size=map_size,
        curriculum=curriculum,
        disable_net_worth=disable_net_worth,
        disable_daily_income=disable_daily_income,
        reward_config=reward_config,
        seed=seed,
        n_threads=n_threads,
        render_mode=render_mode,
        difficulty=difficulty,
    )
