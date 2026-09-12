from __future__ import annotations

"""Environment manager — bridges C++ vectorized env and PyTorch policy.

Extracted factories (_ensure_python_path, _make_vec_env, _make_model,
_make_buffer, _make_ppo) reduce `__init__` duplication and centralise
observation-mode branching. Public API is unchanged for Config/EnvManager
consumers.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

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
        curriculum_stage=cfg.curriculum_stage,
        # ручной набор зданий применяется только при включённом чекбоксе
        unlock_ids=(cfg.unlock_ids if cfg.use_curriculum_tab else ""),
        reward_config=reward_cfg,
        seed=cfg.seed,
        difficulty=cfg.difficulty,
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
            minimap_radius=cfg.minimap_radius,
            n_actions=n_actions,
            hidden_sizes=cfg.net_arch,
            device=device,
        )
    # hybrid
    from rl.actor_critic_hybrid import ActorCriticHybrid

    return ActorCriticHybrid(
        obs_size=obs_size,
        minimap_radius=cfg.minimap_radius,
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
    """Create rollout buffer (tensor or dict variant)."""
    from rl.rollout_buffer import RolloutBuffer

    # hybrid needs dict buffer, others flat
    if cfg.obs_mode == "hybrid":
        from rl.rollout_buffer import DictRolloutBuffer  # type: ignore

        return DictRolloutBuffer(
            n_steps=cfg.n_steps,
            n_envs=n_envs,
            obs_size=obs_size,
            device=device,
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
        total_training_steps=cfg.total_timesteps,
        target_kl=cfg.target_kl,
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
        self.device = device

        _ensure_python_path()
        self.vec_env = _make_vec_env(cfg)

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

    # ── observation helpers ──

    def _policy_obs(self, obs: Any):
        """Convert env obs to policy input (tensor / dict)."""
        if isinstance(obs, dict):
            return {k: torch.as_tensor(v, device=self.device, dtype=torch.float32) for k, v in obs.items()}
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

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
        dones_t = torch.as_tensor(np.asarray(dones, dtype=bool), device=self.device)

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
            )
        return self._obs, infos

    # ── curriculum ──

    def get_allowed_buildings(self) -> List[str]:
        """Return ids allowed by current curriculum stage."""
        try:
            from rl.curriculum import ids_for_stage

            return ids_for_stage(
                self.cfg.curriculum_stage,
                unlock_ids=(self.cfg.unlock_ids if self.cfg.use_curriculum_tab else ""),
            )
        except Exception:
            # fallback: ask env
            try:
                return list(self.vec_env.build_ids())  # type: ignore
            except Exception:
                return []

    def set_curriculum_stage(self, stage: int) -> None:
        self.cfg.curriculum_stage = stage
        # vec env may need to be recreated or notified
        try:
            self.vec_env.venv.set_curriculum_stage(stage)  # type: ignore
        except AttributeError:
            pass

    def close(self) -> None:
        try:
            self.vec_env.close()
        except Exception:
            pass
