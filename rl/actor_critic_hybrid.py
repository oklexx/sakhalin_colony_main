from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

from rl._nn_common import ActorCriticBase, orthogonal_init


class ActorCriticHybrid(ActorCriticBase):
    """Hybrid actor-critic: flat vector + minimap CNN branches concatenated.

    Observation is a dict with keys 'flat' [B, F] and 'minimap' [B, C, R, R].
    If observation is a single tensor, it is treated as flat-only.
    """

    def __init__(
        self,
        obs_size: int,
        minimap_channels: int = 8,
        minimap_radius: Optional[int] = None,
        n_channels: Optional[int] = None,
        grid_size: Optional[int] = None,
        n_actions: int = 49,  # 2 + 32 builds + 11 managers + 4 road dirs
        hidden_sizes: Optional[List[int]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if n_channels is not None:
            minimap_channels = n_channels
        if grid_size is not None:
            self.grid_size = grid_size
            minimap_radius = (grid_size - 1) // 2
        else:
            if minimap_radius is None:
                minimap_radius = 14
            self.grid_size = 2 * minimap_radius + 1
        self.obs_size = obs_size
        self.minimap_radius = minimap_radius
        self.minimap_channels = minimap_channels
        self.n_channels = minimap_channels
        self.n_actions = n_actions
        self.device = device

        self.flat_trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_sizes[0]),
            nn.ReLU(),
        ).to(device) if hidden_sizes else nn.Identity().to(device)

        map_size = self.grid_size
        self.cnn = nn.Sequential(
            nn.Conv2d(minimap_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        ).to(device)

        cnn_out = 64 * map_size * map_size
        flat_out = hidden_sizes[0] if hidden_sizes else obs_size
        combined = flat_out + cnn_out

        # remaining hidden layers after concatenation
        layers: List[nn.Module] = []
        prev = combined
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h

        self.joint = nn.Sequential(*layers).to(device) if layers else nn.Identity().to(device)
        joint_out = prev if layers else combined
        self.actor_head = nn.Linear(joint_out, n_actions).to(device)
        self.critic_head = nn.Linear(joint_out, 1).to(device)
        # Legacy aliases expected by some tests / older checkpoints.
        self.flat_proj = self.flat_trunk[0] if isinstance(self.flat_trunk, nn.Sequential) else self.flat_trunk
        self.actor = self.actor_head
        self.critic = self.critic_head

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            # CNN benefits from sqrt(2) gain for ReLU
            if isinstance(m, nn.Conv2d):
                orthogonal_init(m, gain=1.41421356)
            else:
                orthogonal_init(m, gain=1.0)

    def forward(self, flat: Any, minimap: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(flat, dict):
            flat_in = flat.get("flat", flat)
            mini = flat.get("minimap")
        elif isinstance(flat, (tuple, list)) and len(flat) == 2 and minimap is None:
            flat_in, mini = flat
        else:
            flat_in = flat
            mini = minimap
        if flat_in is not None and mini is not None:
            h_flat = self.flat_trunk(flat_in)
            h_cnn = self.cnn(mini).flatten(1)
            h = torch.cat([h_flat, h_cnn], dim=1)
        elif flat_in is not None:
            h = self.flat_trunk(flat_in)
        elif mini is not None:
            h = self.cnn(mini).flatten(1)
        else:
            raise ValueError("Hybrid obs has no 'flat' or 'minimap'")

        if hasattr(self, "joint"):
            h = self.joint(h)
        return self.actor_head(h), self.critic_head(h)

    def get_action_and_value(
        self,
        flat: Any,
        minimap: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(flat, minimap)
        action, log_probs, _ = self._categorical_log_prob(logits, action)
        return action, log_probs, values.squeeze(-1)

    def get_value(self, flat: Any, minimap: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, values = self.forward(flat, minimap)
        return values.squeeze(-1)

    def act(self, flat: Any, minimap: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedy action (mirrors actor_critic.py helper)."""
        logits, values = self.forward(flat, minimap)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        log_prob = torch.distributions.Categorical(logits=logits).log_prob(action)
        return action, log_prob, values.squeeze(-1)
