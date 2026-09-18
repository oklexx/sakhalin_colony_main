"""Shared NN utilities — orthogonal init, distribution helpers, base mixin."""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal init for Linear/Conv2d — single source of truth."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_module_list(modules: List[nn.Module], gain: float = 1.0, sqrt2: bool = False) -> None:
    g = math.sqrt(2) if sqrt2 else gain
    for m in modules:
        orthogonal_init(m, gain=g)


class ActorCriticBase(nn.Module):
    """Mixin with shared helpers: params, state_dict handling, distribution."""

    @property
    def params(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def state_dict_for_env(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_state_dict_from_env(self, state: dict) -> None:
        # Legacy checkpoints predate the critic's value-only action-mask
        # projection; the actor/action head and observation contract are still
        # identical, so missing optional keys are safe.
        self.load_state_dict(state, strict=False)

    @staticmethod
    def _categorical_log_prob(logits: torch.Tensor, action: torch.Tensor | None):
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist
