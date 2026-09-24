import json
import os
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, List
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
        tax_to_debt: bool = True,
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
            curr_dict = {"all_builds": True, "allowed_builds": [], "stage": 0,
                         "all_resources": True, "obs_version": 2}
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
            tax_to_debt=tax_to_debt,
        )

        # P3-6: миникарта s_T нужна только minimap/hybrid-политикам;
        # CppVecEnvMinimap повторно вызывает это после установки obs_mode.
        self._terminal_minimap_wanted = False
        self._configure_terminal_minimap()

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
        # Причины закрытых бит той же маски (MaskReason, uint8) — обновляются
        # одним вызовом action_masks_batch() внутри C++ (docs/MONITOR_ACTIONS_2026_09.md §4)
        self._mask_reasons: Optional[np.ndarray] = None

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
        # Directional road actions appended after the managers (constants.h
        # N_ROAD_DIRS). They extend the road frontier towards a compass
        # direction instead of the BFS-first cell BUILD_ROAD uses.
        self._road_dir_names: List[str] = ["ROAD_E", "ROAD_W", "ROAD_S", "ROAD_N"]
        self._action_names: List[str] = (
            ["DAY", "WEEK"] + self._build_names + self._manager_names + self._road_dir_names
        )
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
        self._mask_reasons = self._read_mask_reasons()
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

        # Update action masks after step (+ причины закрытия тем же проходом)
        self._action_masks = np.asarray(
            self.cpp_vec.action_masks_batch(), dtype=np.float32
        )
        self._mask_reasons = self._read_mask_reasons()

        infos = []
        for i in range(self.num_envs):
            info_str = result.infos[i]
            if info_str and info_str != "{}":
                info = json.loads(info_str)
            else:
                info = {}
            if dones[i]:
                if "terminal_observation" in info:
                    info["terminal_observation"] = np.asarray(
                        info["terminal_observation"], dtype=np.float32
                    )
                else:
                    # НЕ подставляем obs[i]: после авто-reset это первая
                    # наблюдка НОВОГО эпизода, и EnvManager посчитал бы по ней
                    # V(s_T) на усечении — тихо и неверно (P2-3 ревью
                    # 2026-09-24). Без ключа бутстрап честно пропускается;
                    # флаг и разовое предупреждение делают это видимым.
                    info["terminal_observation_missing"] = True
                    self._warn_missing_terminal_obs(i)
                if "terminal_observation_norm" in info:
                    info["terminal_observation_norm"] = np.asarray(
                        info["terminal_observation_norm"], dtype=np.float32
                    )
                info["TimeLimit.truncated"] = bool(trunceds[i] and not terminateds[i])
            infos.append(info)

        tmm_fn = getattr(self.cpp_vec, "terminal_minimap_batch", None)
        if getattr(self, "_terminal_minimap_wanted", False) and tmm_fn is not None and dones.any():
            tmm = np.asarray(tmm_fn(), dtype=np.float32).reshape(self.num_envs, 8, 32, 32)
            for i in range(self.num_envs):
                # terminal_minimap_missing (C++: размер minimap() не 8x32x32) —
                # в буфере нули; без ключа EnvManager честно пропустит бутстрап
                # вместо V(s_T) по пустой карте.
                if dones[i] and not infos[i].get("terminal_minimap_missing"):
                    infos[i]["terminal_minimap"] = tmm[i].copy()

        # Expose the raw termination flag separately from the (terminated|truncated)
        # `dones` so the RL layer can bootstrap GAE correctly on truncation.
        self._last_terminateds = terminateds
        self._last_trunceds = trunceds
        return obs, rewards, dones, infos

    def _configure_terminal_minimap(self) -> None:
        """Вкл/выкл расчёт миникарты s_T в C++ по obs_mode (P3-6 ревью 2026-09-24).

        flat-режим миникарту не использует: раньше C++ всё равно считал
        minimap() на каждом done, а Python копировал её в info. Старый бинарь
        без set_terminal_minimap_enabled — считаем, как раньше, но в flat
        в info не кладём.
        """
        want = getattr(self, "obs_mode", "flat") in ("minimap", "hybrid")
        self._terminal_minimap_wanted = want
        setter = getattr(self.cpp_vec, "set_terminal_minimap_enabled", None)
        if setter is not None:
            setter(bool(want))

    @property
    def action_masks(self) -> np.ndarray:
        """Return current action masks [n_envs, n_actions]."""
        return self._action_masks

    def _read_mask_reasons(self) -> Optional[np.ndarray]:
        """Коды MaskReason [n_envs, n_actions] к свежим маскам или None.

        None — старый бинарь под COLONY_ALLOW_STALE_PYD (нет биндинга):
        наблюдательная метрика, шаг тренировки из-за неё не падает.
        """
        fn = getattr(self.cpp_vec, "action_mask_reasons_batch", None)
        if fn is None:
            return None
        try:
            raw = fn()
        except Exception:  # noqa: BLE001 — наблюдательная метрика не валит шаг
            return None
        arr = np.asarray(raw, dtype=np.uint8)
        return arr.reshape(self.num_envs, -1)

    @property
    def mask_reasons(self) -> Optional[np.ndarray]:
        """Причины закрытых бит текущей маски [n_envs, n_actions] (или None).

        Тот же вызов action_masks_batch(), что и `action_masks` — состояние
        согласовано: EnvManager читает оба свойства до step (см. collect_step).
        """
        return self._mask_reasons

    @property
    def action_names(self) -> List[str]:
        """Return list of action names matching action indices."""
        return list(self._action_names)

    def close(self) -> None:
        pass

    def _warn_missing_terminal_obs(self, env_index: int) -> None:
        """Одно предупреждение на процесс: C++ не прислал terminal_observation."""
        if getattr(self, "_terminal_obs_warned", False):
            return
        self._terminal_obs_warned = True
        print(f"[CppVecEnv] WARNING: done без terminal_observation (env {env_index}) — "
              f"бутстрап V(s_T) на усечении пропускается; устаревший colony_cpp "
              f"(COLONY_ALLOW_STALE_PYD)? Пересоберите расширение.", flush=True)

    def _checked_indices(self, indices) -> List[int]:
        """Индексы по контракту SB3 VecEnv (None | int | iterable) с проверкой границ."""
        idx = [int(i) for i in self._get_indices(indices)]
        bad = [i for i in idx if not 0 <= i < self.num_envs]
        if bad:
            raise IndexError(f"env indices {bad} out of range 0..{self.num_envs - 1}")
        return idx

    def get_attr(self, attr_name: str, indices=None):
        # Одна батч-среда отвечает за все N подсред: атрибут общий, но длина и
        # порядок ответа — как у VecEnv (по одному значению на индекс). Раньше
        # при любом indices возвращался список длины 1 (P2-2).
        value = getattr(self, attr_name)
        return [value for _ in self._checked_indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None):
        # Атрибут общий на батч: частичная установка изменила бы ВСЕ подсреды.
        idx = self._checked_indices(indices)
        if len(idx) != self.num_envs:
            raise NotImplementedError(
                f"set_attr({attr_name!r}) для части подсред ({idx}) не поддерживается: "
                f"CppVecEnv хранит атрибуты общими на все {self.num_envs} сред")
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        # Метод батча вызывается ОДИН раз (раньше — N раз: env_method("reset")
        # сбросил бы все среды N раз), результат раздаётся по индексам.
        idx = self._checked_indices(indices)
        if len(idx) != self.num_envs:
            raise NotImplementedError(
                f"env_method({method_name!r}) для части подсред ({idx}) не поддерживается: "
                f"методы CppVecEnv действуют на весь батч из {self.num_envs} сред")
        result = getattr(self, method_name)(*method_args, **method_kwargs)
        return [result for _ in idx]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._checked_indices(indices)]

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
