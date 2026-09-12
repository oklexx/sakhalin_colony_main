from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from rl._nn_common import ActorCriticBase, orthogonal_init


class ActorCriticCNN(ActorCriticBase):
    """CNN actor-critic for minimap observations.

    Input: [B, C, R, R] where R = 2*minimap_radius+1.
    Flattens after conv, then MLP heads.
    """

    def __init__(
        self,
        n_channels: int = 8,
        minimap_radius: Optional[int] = None,
        grid_size: Optional[int] = None,
        n_actions: int = 45,
        hidden_sizes: Optional[List[int]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if grid_size is not None:
            minimap_radius = (grid_size - 1) // 2
        elif minimap_radius is None:
            minimap_radius = 14
        self.n_channels = n_channels
        self.minimap_radius = minimap_radius
        self.grid_size = 2 * minimap_radius + 1
        self.n_actions = n_actions
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ).to(device)

        map_size = 2 * minimap_radius + 1
        conv_out = 64 * map_size * map_size

        layers: List[nn.Module] = []
        prev = conv_out
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
        # obs: [B, C, R, R]
        h = self.conv(obs)
        h = h.flatten(1)
        h = self.trunk(h)
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
