from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from rl._nn_common import ActorCriticBase, orthogonal_init


class ActorCriticCNN(ActorCriticBase):
    """CNN actor-critic for minimap observations.

    Input: [B, C, G, G] where G = grid_size (the env emits a fixed global 32x32;
    `minimap_radius` is a legacy alias kept only for old call sites).
    Flattens after conv, then MLP heads.
    """

    def __init__(
        self,
        n_channels: int = 8,
        minimap_radius: Optional[int] = None,
        grid_size: Optional[int] = None,
        n_actions: int = 49,  # 2 + 32 builds + 11 managers + 4 road dirs
        hidden_sizes: Optional[List[int]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if grid_size is not None:
            self.grid_size = grid_size
            minimap_radius = (grid_size - 1) // 2
        else:
            if minimap_radius is None:
                minimap_radius = 14
            self.grid_size = 2 * minimap_radius + 1
        self.n_channels = n_channels
        self.minimap_radius = minimap_radius
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

        map_size = self.grid_size
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
        self.critic_mask_proj = nn.Linear(n_actions, 1, bias=False).to(device)

        self._init_weights()
        nn.init.constant_(self.critic_mask_proj.weight, 0.0)

    def _init_weights(self) -> None:
        for m in self.modules():
            orthogonal_init(m, gain=1.0)

    def forward(
        self, obs: torch.Tensor, action_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs: [B, C, R, R]
        h = self.conv(obs)
        h = h.flatten(1)
        h = self.trunk(h)
        values = self.critic_head(h)
        if action_masks is not None:
            values = values + self.critic_mask_proj(action_masks.float())
        return self.actor_head(h), values

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
