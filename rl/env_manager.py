from __future__ import annotations

"""Environment manager — bridges C++ vectorized env and PyTorch policy.

Extracted factories (_ensure_python_path, _make_vec_env, _make_model,
_make_buffer, _make_ppo) reduce `__init__` duplication and centralise
observation-mode branching. Public API is unchanged for Config/EnvManager
consumers.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl.action_monitor import ActionLegalityMonitor
from rl.config import Config

# ── helpers ─────────────────────────────────────────────────────────────────

def _ensure_python_path() -> None:
    """Make sure `python/` package is importable."""
    python_dir = str(Path(__file__).resolve().parent.parent / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)


def _make_vec_env(cfg: Config):
    """Create CppVecEnv (with or without minimap observations)."""
    from python.cpp_vecenv import CppVecEnv  # lazy import after path fix

    reward_cfg = cfg.reward.to_dict()
    common = dict(
        n_envs=cfg.n_envs,
        map_size=cfg.map_size,
        # PR 1: один вычисленный контракт вместо (stage, unlock_ids)
        curriculum=cfg.curriculum_state(),
        reward_config=reward_cfg,
        seed=cfg.seed,
        difficulty=cfg.difficulty,
        tax_to_debt=cfg.tax_to_debt,
    )
    # minimap / hybrid need minimap observation plumbing
    if cfg.obs_mode in ("minimap", "hybrid"):
        from python.cpp_vecenv_minimap import CppVecEnvMinimap  # type: ignore

        return CppVecEnvMinimap(
            **common,
            minimap_radius=cfg.minimap_radius,
            obs_mode=cfg.obs_mode,
        )
    return CppVecEnv(**common)


def _make_model(cfg: Config, obs_size: int, n_actions: int, device: torch.device):
    """Instantiate correct policy class for cfg.obs_mode."""
    if cfg.obs_mode == "flat":
        from rl.actor_critic import ActorCritic

        return ActorCritic(obs_size, n_actions, cfg.net_arch, device)
    if cfg.obs_mode == "minimap":
        from rl.actor_critic_cnn import ActorCriticCNN

        return ActorCriticCNN(
            n_channels=8,
            grid_size=32,
            n_actions=n_actions,
            hidden_sizes=cfg.net_arch,
            device=device,
        )
    # hybrid
    from rl.actor_critic_hybrid import ActorCriticHybrid

    return ActorCriticHybrid(
        obs_size=obs_size,
        grid_size=32,
        minimap_channels=8,
        n_actions=n_actions,
        hidden_sizes=cfg.net_arch,
        device=device,
    )


def _make_buffer(
    cfg: Config,
    n_envs: int,
    obs_size: int,
    n_actions: int,
    device: torch.device,
):
    """Create rollout buffer (flat tensor or N-D tensor variant for minimap/hybrid)."""
    from rl.rollout_buffer import RolloutBuffer, _TensorRolloutBuffer

    grid = 32
    if cfg.obs_mode == "minimap":
        return _TensorRolloutBuffer(
            n_steps=cfg.n_steps,
            n_envs=n_envs,
            obs_shape=(8, grid, grid),
            n_actions=n_actions,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=device,
        )
    if cfg.obs_mode == "hybrid":
        return _TensorRolloutBuffer(
            n_steps=cfg.n_steps,
            n_envs=n_envs,
            obs_shape=(8, grid, grid),
            n_actions=n_actions,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=device,
            flat_dim=obs_size,
        )
    return RolloutBuffer(
        n_steps=cfg.n_steps,
        n_envs=n_envs,
        obs_size=obs_size,
        n_actions=n_actions,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        device=device,
    )


def _make_ppo(cfg: Config, model, buffer):
    from rl.ppo import PPO

    steps_per_rollout = cfg.n_steps * cfg.n_envs
    n_rollouts = max(1, int(math.ceil(cfg.total_timesteps / max(1, steps_per_rollout))))
    batches_per_epoch = max(1, steps_per_rollout // max(1, cfg.batch_size))
    total_optimizer_steps = n_rollouts * cfg.n_epochs * batches_per_epoch

    return PPO(
        model=model,
        buffer=buffer,
        device=buffer.device if hasattr(buffer, "device") else torch.device("cpu"),
        lr=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        use_amp=cfg.use_amp,
        amp_dtype=cfg.amp_dtype,
        torch_compile=cfg.torch_compile,
        total_training_steps=total_optimizer_steps,
        target_kl=cfg.target_kl,
        obs_version=cfg.obs_version,
    )


# ── EnvManager ──────────────────────────────────────────────────────────────

class EnvManager:
    """High-level façade used by AsyncTrainer / train.py.

    Public surface kept stable:
      n_envs, obs_size, n_actions, mm_env, model, buffer, ppo,
      reset, step, collect_step, finish_episode, get_allowed_buildings,
      set_curriculum_stage, close
    """

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        # Public alias for the observation mode. Consumers (AsyncTrainer) branch
        # on it to unpack `(flat, minimap)`; reading it as
        # `getattr(em, "obs_mode", "flat")` used to fall back to "flat" for every
        # hybrid run because the attribute did not exist — the raw tuple then went
        # into the policy as one tensor and the first Linear raised
        # "TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple".
        self.obs_mode: str = cfg.obs_mode
        self._curriculum_progress_step = 0
        self.device = device

        _ensure_python_path()
        from colony_cpp_api import require_colony  # lazy import after path fix
        # PR 3: fail fast on a stale colony_cpp binary, before any env exists.
        # Escape hatch: train.py --allow-stale-pyd / COLONY_ALLOW_STALE_PYD=1.
        require_colony()
        self.vec_env = _make_vec_env(cfg)
        # PR 1 parity: the env must report back exactly the curriculum Python
        # computed — silent «not applied» is the bug class this contract kills.
        self._assert_curriculum_parity(cfg.curriculum_state())

        # expose commonly accessed attributes for backward compat
        self.n_envs: int = self.vec_env.num_envs
        self.obs_size: int = getattr(self.vec_env, "obs_size", lambda: 0)() if callable(getattr(self.vec_env, "obs_size", None)) else getattr(self.vec_env, "obs_size", 0)  # type: ignore
        # fallback if obs_size not directly available — infer from observation_space
        if not self.obs_size:
            try:
                self.obs_size = int(self.vec_env.observation_space.shape[0])  # type: ignore
            except Exception:
                self.obs_size = 0

        try:
            self.n_actions: int = int(self.vec_env.action_space.n)  # type: ignore
        except Exception:
            self.n_actions = getattr(self.vec_env, "n_actions", 0)  # type: ignore

        # minimap env alias (for minimap-specific helpers)
        self.mm_env = self.vec_env if cfg.is_minimap else None

        self.model = _make_model(cfg, self.obs_size, self.n_actions, device)
        self.buffer = _make_buffer(cfg, self.n_envs, self.obs_size, self.n_actions, device)
        self.ppo = _make_ppo(cfg, self.model, self.buffer)

        self._obs: Optional[torch.Tensor | Dict[str, torch.Tensor]] = None
        self._last_dones = np.zeros(self.n_envs, dtype=bool)

        # Мониторинг легальности действий (см. rl/action_monitor.py). Среда
        # обязана отдавать маски — collect_step читает self.vec_env.action_masks
        # безусловно, — поэтому отдельная проверка «есть ли маски» тут только
        # молча выключала бы метрику на пустом месте.
        self._action_legality: Optional[ActionLegalityMonitor] = (
            ActionLegalityMonitor(self.n_actions)
            if bool(getattr(cfg, "monitor_action_legality", True))
            else None
        )

    # ── observation helpers ──

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), device=self.device, dtype=torch.float32)

    def _policy_obs(self, obs: Any):
        """Convert env obs to policy input (tensor / dict / tuple)."""
        if isinstance(obs, dict):
            return {k: self._to_tensor(v) for k, v in obs.items()}
        if isinstance(obs, tuple):
            return tuple(self._to_tensor(x) for x in obs)
        return self._to_tensor(obs)

    # ── lifecycle ──

    def reset(self) -> Any:
        obs = self.vec_env.reset()
        self._obs = self._policy_obs(obs)
        self._last_dones[:] = False
        return self._obs

    def step(self, actions: np.ndarray):
        """Step vec env (sync). Returns (obs, rewards, dones, infos)."""
        obs, rewards, dones, infos = self.vec_env.step(actions)
        self._obs = self._policy_obs(obs)
        self._last_dones = dones
        return self._obs, rewards, dones, infos

    # ── rollout integration ──

    def collect_step(self, obs: Any = None) -> Tuple[Any, List[Dict[str, Any]]]:
        """Collect one step into rollout buffer (policy sampling).

        Returns (new_obs, infos) for the trainer's episode tracking.
        """
        if obs is not None:
            self._obs = self._policy_obs(obs)
        if self._obs is None:
            self.reset()
        assert self._obs is not None

        action_masks_np = np.asarray(self.vec_env.action_masks, dtype=np.float32)
        action_masks_t = torch.as_tensor(action_masks_np, device=self.device, dtype=torch.float32)
        # Мониторинг: та самая маска, из которой сэмплируется действие (не следующая!),
        # поэтому доля легальных шагов считается честно (rl/action_monitor.py).
        if self._action_legality is not None:
            self._action_legality.add_step(action_masks_np)

        if self.cfg.obs_mode == "hybrid":
            flat_obs, minimap_obs = self._obs
            prev_flat = flat_obs
            prev_minimap = minimap_obs
        else:
            flat_obs = self._obs
            minimap_obs = None
            prev_flat = flat_obs

        sampled = self.ppo.collect_step(
            flat_obs,
            minimap=minimap_obs,
            action_masks=action_masks_t,
        )
        action = sampled["action"]
        log_prob = sampled["log_prob"]
        value = sampled["value"]

        actions_np = action.cpu().numpy().astype(np.int64)
        next_obs, rewards, dones, infos = self.vec_env.step(actions_np)
        next_obs_t = self._policy_obs(next_obs)
        self._last_dones = dones

        rewards_t = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device)
        terminated_np = np.asarray(getattr(self.vec_env, "_last_terminateds", dones), dtype=bool)
        terminated_t = torch.as_tensor(terminated_np, device=self.device)
        dones_np = np.asarray(dones, dtype=bool)
        dones_t = torch.as_tensor(dones_np, device=self.device)
        truncated_np = dones_np & ~terminated_np
        trunc_value_t = self._truncation_bootstrap_values(
            infos, truncated_np, action_masks_t
        )

        if self.cfg.obs_mode == "hybrid":
            next_flat, next_minimap = next_obs_t
            self._obs = next_obs_t
            self.buffer.add(
                prev_minimap,
                action,
                rewards_t,
                log_prob,
                value,
                dones_t,
                terminated=terminated_t,
                flat=prev_flat,
                action_masks=action_masks_t,
                trunc_value=trunc_value_t,
            )
        else:
            self._obs = next_obs_t
            self.buffer.add(
                prev_flat,
                action,
                rewards_t,
                log_prob,
                value,
                dones_t,
                terminated=terminated_t,
                action_masks=action_masks_t,
                trunc_value=trunc_value_t,
            )
        return self._obs, infos

    # ── мониторинг (UI: вкладка «Мониторинг») ──

    def pop_action_legality(self) -> Dict[str, float]:
        """Доля шагов роллаута, на которых действие было легальным (0..100).

        Окно — ровно один только что собранный роллаут: счётчики обнуляются,
        поэтому соседние обновления монитора не описывают один и тот же хвост
        лога. Пустой dict = легальность не измерялась (флаг выключен), и UI
        обязан показать «нет данных», а не «заблокировано маской».
        """
        if self._action_legality is None:
            return {}
        return self._action_legality.pop_percent(self.action_names)

    def get_curriculum_progress(self, step: int) -> Dict[str, object]:
        """Прогресс текущего этапа курикулума в шагах (для подписи в UI).

        Пересчитывать этапы здесь нельзя — единый источник `rl/curriculum.py`
        (RULES.md, п. 2 золотого правила). До 2026-09-23 этого метода не было,
        а вызов в AsyncTrainer падал в `except Exception`, из-за чего монитор
        вечно показывал «этап N (0%)».
        """
        from rl.curriculum import stage_progress

        return stage_progress(
            int(step),
            list(getattr(self.cfg, "curriculum_schedule", []) or []),
            int(getattr(self.cfg, "curriculum_stage", 0) or 0),
        )

    def _truncation_bootstrap_values(
        self,
        infos: List[Dict[str, Any]],
        truncated_np: np.ndarray,
        action_masks_t: torch.Tensor,
    ) -> torch.Tensor:
        """V(s_T) for truncated envs; zeros elsewhere."""
        n = self.n_envs
        out = torch.zeros(n, dtype=torch.float32, device=self.device)
        if not np.any(truncated_np):
            return out
        idx = np.flatnonzero(truncated_np)
        model = self.model
        with torch.no_grad():
            if self.cfg.obs_mode == "hybrid":
                flats = []
                mms = []
                ok_idx = []
                for i in idx:
                    info = infos[int(i)]
                    flat = info.get("terminal_observation_norm", info.get("terminal_observation"))
                    mm = info.get("terminal_minimap")
                    if flat is None or mm is None:
                        continue
                    flats.append(np.asarray(flat, dtype=np.float32))
                    mms.append(np.asarray(mm, dtype=np.float32))
                    ok_idx.append(int(i))
                if not ok_idx:
                    return out
                flat_t = torch.as_tensor(np.stack(flats), device=self.device, dtype=torch.float32)
                mm_t = torch.as_tensor(np.stack(mms), device=self.device, dtype=torch.float32)
                vals = model.get_value(flat_t, mm_t, action_masks=action_masks_t[ok_idx])
                out[ok_idx] = vals.float().reshape(-1)
                return out
            if self.cfg.obs_mode == "minimap":
                mms = []
                ok_idx = []
                for i in idx:
                    mm = infos[int(i)].get("terminal_minimap")
                    if mm is None:
                        continue
                    mms.append(np.asarray(mm, dtype=np.float32))
                    ok_idx.append(int(i))
                if not ok_idx:
                    return out
                mm_t = torch.as_tensor(np.stack(mms), device=self.device, dtype=torch.float32)
                vals = model.get_value(mm_t, action_masks=action_masks_t[ok_idx])
                out[ok_idx] = vals.float().reshape(-1)
                return out
            flats = []
            ok_idx = []
            for i in idx:
                info = infos[int(i)]
                flat = info.get("terminal_observation_norm", info.get("terminal_observation"))
                if flat is None:
                    continue
                flats.append(np.asarray(flat, dtype=np.float32))
                ok_idx.append(int(i))
            if not ok_idx:
                return out
            flat_t = torch.as_tensor(np.stack(flats), device=self.device, dtype=torch.float32)
            vals = model.get_value(flat_t, action_masks=action_masks_t[ok_idx])
            out[ok_idx] = vals.float().reshape(-1)
        return out

    # ── curriculum ──

    def _assert_curriculum_parity(self, st) -> None:
        """EnvManager-side parity check: env.curriculum() == computed state."""
        try:
            got = self.vec_env.venv.curriculum()  # type: ignore[attr-defined]
        except AttributeError:
            return  # exotic wrapper without .venv — nothing to check against
        # Buildings and resources are independent axes (open buildings +
        # weighted resources is valid) — both are always checked, no early out.
        if st.all_builds:
            assert got.get("all_builds", False), (
                f"курикулум не применён: expected unrestricted, env={got}")
        else:
            want = set(st.allowed_builds)
            have = set(got.get("allowed_builds", []))
            assert want == have, (
                f"курикулум не применён: env={sorted(have)[:5]}…({len(have)}) "
                f"expected={sorted(want)[:5]}…({len(want)})")
        # PR 5: obs layout version — an old binary has no "obs_version" key
        # and reads as v0, so a stale .so fails here with a clear message.
        have_v = got.get("obs_version", 0)
        assert int(have_v) == int(st.obs_version), (
            f"курикулум не применён: obs_version env={have_v} "
            f"expected={st.obs_version} (stale colony_cpp? rebuild the extension)")
        # PR 4: resource weights ride the same transport — same parity rule.
        # Exact compare: doubles end-to-end (C++ stores double since PR 4).
        if st.all_resources:
            assert got.get("all_resources", False), (
                f"курикулум не применён: expected all_resources, env={got}")
        else:
            want_w = [float(w) for w in st.resource_weights]
            have_w = [float(w) for w in got.get("resource_weights", [])]
            assert want_w == have_w, (
                f"курикулум не применён: resource_weights env={have_w} "
                f"expected={want_w}")
        # Mechanic gating is an allow-list over fixed manager logits. A stale
        # extension that drops this field would silently reopen early actions.
        want_mechanics = list(st.enabled_mechanics)
        have_mechanics = list(got.get("enabled_mechanics", []))
        assert have_mechanics == want_mechanics, (
            f"курикулум не применён: enabled_mechanics env={have_mechanics} "
            f"expected={want_mechanics} (rebuild colony_cpp)")

    @property
    def action_names(self) -> List[str]:
        """Action labels in env order (trainer top_actions, loop detector).

        Sourced from the vec env so monitoring labels can never drift from
        the real action indices (the trainer's hardcoded fallback did:
        env action 12 = Road was reported as BUILD_GOLDMINE).
        """
        get = getattr(self.vec_env, "action_names", None)
        if get is None:
            return []
        try:
            names = get() if callable(get) else get
            return [str(n) for n in names]
        except Exception:
            return []

    def get_allowed_buildings(self) -> List[str]:
        """Return ids allowed by current curriculum stage + manual set."""
        return self.get_allowed_buildings_for_stage(self.cfg.curriculum_stage)

    def get_allowed_buildings_for_stage(self, stage: int) -> List[str]:
        """Ids allowed at `stage` (stage preset ∪ manual set from the UI tab)."""
        try:
            from rl.curriculum import allowed_ids

            return allowed_ids(
                stage,
                unlock_ids=self.cfg.unlock_ids,
                use_curriculum_tab=self.cfg.use_curriculum_tab,
            )
        except Exception:
            # fallback: ask env
            try:
                return list(self.vec_env.build_ids())  # type: ignore
            except Exception:
                return []

    def set_curriculum_progress(self, step: int) -> None:
        """Apply building stage and additive mechanic unlocks atomically."""
        self._curriculum_progress_step = max(0, int(step))
        st = self.cfg.curriculum_state(self._curriculum_progress_step)
        try:
            venv = self.vec_env.venv  # type: ignore[attr-defined]
        except AttributeError:
            return
        venv.set_curriculum(st.to_dict())
        self._assert_curriculum_parity(st)

    def set_curriculum_stage(self, stage: int) -> None:
        self.cfg.curriculum_stage = stage
        # PR 1 + mechanic gating: recompute ONE state (stage/manual/resources
        # plus the monotonic mechanic allow-list) and apply it in one call.
        self.set_curriculum_progress(getattr(self, "_curriculum_progress_step", 0))

    def close(self) -> None:
        try:
            self.vec_env.close()
        except Exception:
            pass
