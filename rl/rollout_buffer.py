from __future__ import annotations

import torch
import numpy as np
from typing import Optional


class RolloutBuffer:
    """GPU-resident rollout buffer with GAE.

    Stores obs, actions, rewards, log_probs, values, dones as GPU tensors.
    GAE computed on GPU.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_size: int,
        n_actions: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        total = n_steps * n_envs
        self.obs = torch.empty(total, obs_size, dtype=torch.float32, device=device)
        self.actions = torch.empty(total, dtype=torch.long, device=device)
        self.rewards = torch.empty(total, dtype=torch.float32, device=device)
        self.log_probs = torch.empty(total, dtype=torch.float32, device=device)
        self.values = torch.empty(total, dtype=torch.float32, device=device)
        self.dones = torch.empty(total, dtype=torch.bool, device=device)
        # True termination flag (excludes time-limit truncation). Used as the
        # GAE bootstrap mask so that truncated episodes keep their value
        # bootstrap. Defaults to `done` when not supplied (backwards compat).
        self.terminated = torch.empty(total, dtype=torch.bool, device=device)
        # Action masks: [total, n_actions] — 1.0=available, 0.0=blocked
        self.action_masks = torch.empty(total, n_actions, dtype=torch.float32, device=device)

        self.advantages = torch.empty(total, dtype=torch.float32, device=device)
        self.returns = torch.empty(total, dtype=torch.float32, device=device)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
        terminated: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ):
        """Add one step of data. All tensors shape [n_envs].

        `done` is the episode-end flag (terminated | truncated) used for
        reward/length accounting. `terminated` is the true terminal flag used
        for the GAE bootstrap mask; if None it defaults to `done`.
        `action_masks` is [n_envs, n_actions] — 1.0=available, 0.0=blocked.
        """
        if self.pos >= self.n_steps:
            raise RuntimeError("Buffer full, call reset() first")
        start = self.pos * self.n_envs
        end = start + self.n_envs
        self.obs[start:end] = obs
        self.actions[start:end] = action
        self.rewards[start:end] = reward
        self.log_probs[start:end] = log_prob
        self.values[start:end] = value
        self.dones[start:end] = done
        self.terminated[start:end] = done if terminated is None else terminated
        if action_masks is not None:
            self.action_masks[start:end] = action_masks
        self.pos += 1
        if self.pos == self.n_steps:
            self.full = True

    def reset(self):
        self.pos = 0
        self.full = False

    def compute_gae(self, last_value: torch.Tensor, last_done: torch.Tensor):
        """Compute GAE advantages and returns.

        last_value: [n_envs] value of the state after the last step
        last_done:  [n_envs] whether the last step terminated
        """
        n = self.n_steps * self.n_envs
        adv = torch.zeros_like(self.advantages)
        last_gae = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)

        next_adv = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)

        for t in range(self.n_steps - 1, -1, -1):
            start = t * self.n_envs
            end = start + self.n_envs
            if t == self.n_steps - 1:
                next_values = last_value
                next_dones = last_done
            else:
                next_start = (t + 1) * self.n_envs
                next_values = self.values[next_start:next_start + self.n_envs]
                # terminated[t] is transition-based: terminated[t] == terminal(s_{t+1}).
                # The bootstrap mask for step t must use terminal(s_{t+1}), and must
                # NOT treat time-limit truncation as terminal (so the value
                # bootstrap is preserved across truncated episodes).
                next_dones = self.terminated[start:end]

            delta = (
                self.rewards[start:end]
                + self.gamma * next_values * (~next_dones)
                - self.values[start:end]
            )
            next_adv = delta + self.gamma * self.gae_lambda * (~next_dones) * next_adv
            adv[start:end] = next_adv

        # Advantages and returns are regression TARGETS, never part of the
        # policy graph. Detach at the source: `last_value` may legitimately
        # carry a grad_fn, and letting it propagate into self.advantages made
        # the whitening below an in-place write on a graph-tracked tensor --
        # backward() then died with "modified by an inplace operation". It also
        # kept the whole GAE graph alive across the update.
        adv = adv.detach()

        # `returns` must keep using the UNNORMALISED advantages (GAE targets for
        # the value head); only `advantages` gets whitened for the policy loss.
        self.advantages[:n] = adv
        self.returns[:n] = adv + self.values[:n]

        mean_adv = adv[:n].mean()
        std_adv = adv[:n].std()
        if std_adv > 1e-8:
            adv = (adv - mean_adv) / (std_adv + 1e-8)
        self.advantages[:n] = adv

    def get_batches(self, batch_size: int):
        """Yield mini-batches. Each batch: dict of tensors."""
        n = self.n_steps * self.n_envs
        indices = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "obs": self.obs[idx],
                "actions": self.actions[idx],
                "rewards": self.rewards[idx],
                "old_log_probs": self.log_probs[idx],
                "values": self.values[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
                "dones": self.dones[idx],
                "action_masks": self.action_masks[idx],
            }

    def clear_gpu_memory(self):
        torch.cuda.empty_cache()


class _TensorRolloutBuffer(RolloutBuffer):
    """RolloutBuffer with N-D observation tensors (e.g. minimap [C, H, W]).

    Same GAE logic as the base class, but `obs` is allocated with an explicit
    shape instead of a flat size.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple[int, ...],
        n_actions: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
        flat_dim: int = 0,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self._obs_shape = tuple(obs_shape)
        self.obs_size = int(np.prod(obs_shape))
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.flat_dim = int(flat_dim)

        total = n_steps * n_envs
        self.obs = torch.empty(total, *obs_shape, dtype=torch.float32, device=device)
        if self.flat_dim > 0:
            self.flat_obs = torch.empty(total, self.flat_dim, dtype=torch.float32, device=device)
        else:
            self.flat_obs = None
        self.actions = torch.empty(total, dtype=torch.long, device=device)
        self.rewards = torch.empty(total, dtype=torch.float32, device=device)
        self.log_probs = torch.empty(total, dtype=torch.float32, device=device)
        self.values = torch.empty(total, dtype=torch.float32, device=device)
        self.dones = torch.empty(total, dtype=torch.bool, device=device)
        self.terminated = torch.empty(total, dtype=torch.bool, device=device)
        self.action_masks = torch.empty(total, n_actions, dtype=torch.float32, device=device)
        self.advantages = torch.empty(total, dtype=torch.float32, device=device)
        self.returns = torch.empty(total, dtype=torch.float32, device=device)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, log_prob, value, done, terminated=None, flat=None, action_masks=None):
        """Add one step of data. All tensors shape [n_envs].

        `flat` is the flat observation tensor (shape [n_envs, flat_dim]) stored
        only when the buffer was created with `flat_dim > 0` (hybrid mode).
        `action_masks` is [n_envs, n_actions] — 1.0=available, 0.0=blocked.
        """
        super().add(obs, action, reward, log_prob, value, done, terminated, action_masks)
        if self.flat_obs is not None:
            if flat is None:
                raise RuntimeError("flat observation required but not provided (hybrid buffer)")
            start = (self.pos - 1) * self.n_envs
            end = self.pos * self.n_envs
            self.flat_obs[start:end] = flat

    def get_batches(self, batch_size: int):
        """Yield mini-batches. Each batch: dict of tensors."""
        n = self.n_steps * self.n_envs
        indices = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            batch = {
                "obs": self.obs[idx],
                "actions": self.actions[idx],
                "rewards": self.rewards[idx],
                "old_log_probs": self.log_probs[idx],
                "values": self.values[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
                "dones": self.dones[idx],
                "action_masks": self.action_masks[idx],
            }
            if self.flat_obs is not None:
                batch["flat"] = self.flat_obs[idx]
            yield batch
