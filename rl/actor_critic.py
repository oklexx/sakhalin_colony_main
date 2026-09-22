from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from rl._nn_common import ActorCriticBase, orthogonal_init


class ActorCritic(ActorCriticBase):
    """MLP actor-critic for discrete actions.

    Input:  [B, obs_size] float32
    Output: (logits [B, n_actions], values [B, 1])
    """

    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        hidden_sizes: List[int],
        device: torch.device,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = device

        layers: List[nn.Module] = []
        prev = obs_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        self.trunk = nn.Sequential(*layers).to(device)
        self.actor_head = nn.Linear(prev, n_actions).to(device)
        self.critic_head = nn.Linear(prev, 1).to(device)
        # The actor already receives masks in PPO. This tiny value-only branch
        # gives the critic the same availability context without changing the
        # observation width or the 49-action checkpoint head. It is zeroed so
        # old behavior is the initialization; old checkpoints load strict=False.
        self.critic_mask_proj = nn.Linear(n_actions, 1, bias=False).to(device)
        # Additive actor mask (variant B): logits += W·mask, W=0 so old
        # behaviour until training moves the threshold.
        self.actor_mask_proj = nn.Linear(n_actions, n_actions, bias=False).to(device)

        self._init_weights()
        nn.init.constant_(self.critic_mask_proj.weight, 0.0)
        nn.init.constant_(self.actor_mask_proj.weight, 0.0)

    def _init_weights(self) -> None:
        for m in self.modules():
            orthogonal_init(m, gain=1.0)

    def forward(
        self, obs: torch.Tensor, action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        values = self.critic_head(h)
        logits = self.actor_head(h)
        if action_masks is not None:
            m = action_masks.float()
            values = values + self.critic_mask_proj(m)
            logits = logits + self.actor_mask_proj(m)
        return logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs, action_masks)
        action, log_probs, _ = self._categorical_log_prob(logits, action)
        return action, log_probs, values.squeeze(-1)

    def get_value(
        self, obs: torch.Tensor, action_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, values = self.forward(obs, action_masks)
        return values.squeeze(-1)
