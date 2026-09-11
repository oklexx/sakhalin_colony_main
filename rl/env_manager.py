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
        unlock_ids=cfg.unlock_ids,
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

        # minimap observations are [B, C, R, R]; n_channels inferred from game
        # For now reuse flat obs_size as channel count if needed
        return ActorCriticCNN(
            n_channels=obs_size,  # placeholder — real channel count from env
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
        minimap_channels=3,  # placeholder
        n_actions=n_actions,
        hidden_sizes=cfg.net_arch,
        device=device,
    )


def _make_buffer(
    cfg: Config,
    n_envs: int,
    obs_size: int,
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
        device=device,
    )


def _make_ppo(cfg: Config, model, buffer):
    from rl.ppo import PPO

    return PPO(
        model=model,
        buffer=buffer,
        device=buffer.device if hasattr(buffer, "device") else torch.device("cpu"),
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
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
        self.buffer = _make_buffer(cfg, self.n_envs, self.obs_size, device)
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

    def collect_step(self) -> None:
        """Collect one step into rollout buffer (policy sampling)."""
        if self._obs is None:
            self.reset()
        assert self._obs is not None

        with torch.no_grad():
            # handle dict vs tensor obs
            if isinstance(self._obs, dict):
                action, log_prob, value = self.model.get_action_and_value(self._obs)  # type: ignore
            else:
                action, log_prob, value = self.model.get_action_and_value(self._obs)

        # action masks are not used here directly — env will block invalid actions
        actions_np = action.cpu().numpy()
        next_obs, rewards, dones, infos = self.step(actions_np)

        # convert rewards to tensor for buffer
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        terminateds = torch.as_tensor(dones, device=self.device, dtype=torch.float32)

        # buffer expects (obs, actions, log_probs, rewards, dones, values)
        # obs is the *previous* obs (before step)
        self.buffer.add(
            self._obs,  # type: ignore
            action,
            log_prob,
            rewards_t,
            terminateds,
            value,
        )
        self._obs = next_obs

    def finish_episode(self) -> None:
        """Finish rollout and run PPO update."""
        # bootstrap value for last obs
        with torch.no_grad():
            if isinstance(self._obs, dict):
                last_value = self.model.get_value(self._obs)  # type: ignore
            else:
                # _obs may be tensor
                last_value = self.model.get_value(self._obs)  # type: ignore
        self.buffer.compute_returns_and_advantages(last_value, self._last_dones)
        self.ppo.update(self.buffer)
        self.buffer.clear()

    # ── curriculum ──

    def get_allowed_buildings(self) -> List[str]:
        """Return ids allowed by current curriculum stage."""
        try:
            from rl.curriculum import ids_for_stage

            return ids_for_stage(self.cfg.curriculum_stage, unlock_ids=self.cfg.unlock_ids)
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
            self.vec_env.set_curriculum_stage(stage)  # type: ignore
        except AttributeError:
            pass

    def close(self) -> None:
        try:
            self.vec_env.close()
        except Exception:
            pass
