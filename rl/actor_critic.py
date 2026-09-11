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

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            orthogonal_init(m, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.actor_head(h), self.critic_head(h)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        action, log_probs, _ = self._categorical_log_prob(logits, action)
        return action, log_probs, values.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, values = self.forward(obs)
        return values.squeeze(-1)
