from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rl._nn_common import ActorCriticBase, orthogonal_init


class ActorCriticHybrid(ActorCriticBase):
    """Hybrid actor-critic: flat vector + minimap CNN branches concatenated.

    Observation is a dict with keys 'flat' [B, F] and 'minimap' [B, C, R, R].
    If observation is a single tensor, it is treated as flat-only.

    Branch balance (2026-09-22, docs/WATER_HYBRID_AB_2026_09.md): the CNN
    flatten used to contribute 64*G*G = 65 536 features against the flat
    trunk's 256 and drowned every flat hint in the joint layer (flat branch
    energy ~0.3-0.4%, water-logit sensitivity 13x below flat-only). Now the
    CNN is pooled to 2x2, projected to the same width as the flat trunk and
    the joint layer starts with LayerNorm — widths are 256 vs 256. This
    CHANGES joint/cnn_head shapes: checkpoints saved before 2026-09-22 do
    not load (state_dict size mismatch); conv feature keys cnn.* are
    unchanged and transfer.
    """

    def __init__(
        self,
        obs_size: int,
        minimap_channels: int = 8,
        minimap_radius: int | None = None,
        n_channels: int | None = None,
        grid_size: int | None = None,
        n_actions: int = 49,  # 2 + 32 builds + 11 managers + 4 road dirs
        hidden_sizes: list[int] | None = None,
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

        self.cnn = nn.Sequential(
            nn.Conv2d(minimap_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        ).to(device)

        # Branch balance: pool the conv map down to 2x2 (64*2*2 = 256) and
        # project it to the flat trunk's width — so the joint layer sees two
        # equally-sized branches instead of 65 536 CNN features vs 256 flat
        # (the flat signal drowned, see the class docstring). Adaptive pool
        # also makes the head width grid-size independent.
        self.cnn_pool = nn.AdaptiveAvgPool2d((2, 2)).to(device)
        self.cnn_head = nn.Sequential(
            nn.Linear(64 * 2 * 2, hidden_sizes[0]),
            nn.ReLU(),
        ).to(device) if hidden_sizes else nn.Identity().to(device)

        cnn_out = hidden_sizes[0] if hidden_sizes else 64 * 2 * 2
        flat_out = hidden_sizes[0] if hidden_sizes else obs_size
        combined = flat_out + cnn_out

        # remaining hidden layers after concatenation; LayerNorm equalises
        # the branch scales on top of the width balance
        layers: list[nn.Module] = []
        prev = combined
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h

        self.joint = nn.Sequential(*layers).to(device) if layers else nn.Identity().to(device)
        joint_out = prev if layers else combined
        self.actor_head = nn.Linear(joint_out, n_actions).to(device)
        self.critic_head = nn.Linear(joint_out, 1).to(device)
        self.critic_mask_proj = nn.Linear(n_actions, 1, bias=False).to(device)
        self.actor_mask_proj = nn.Linear(n_actions, n_actions, bias=False).to(device)
        # Legacy aliases expected by some tests / older checkpoints.
        self.flat_proj = self.flat_trunk[0] if isinstance(self.flat_trunk, nn.Sequential) else self.flat_trunk
        self.actor = self.actor_head
        self.critic = self.critic_head

        self._init_weights()
        nn.init.constant_(self.critic_mask_proj.weight, 0.0)
        nn.init.constant_(self.actor_mask_proj.weight, 0.0)

    def _init_weights(self) -> None:
        for m in self.modules():
            # CNN benefits from sqrt(2) gain for ReLU
            if isinstance(m, nn.Conv2d):
                orthogonal_init(m, gain=1.41421356)
            else:
                orthogonal_init(m, gain=1.0)

    def _looks_like_masks(self, t: Any) -> bool:
        """True when `t` is a [B, n_actions] tensor — i.e. masks, not a minimap."""
        return bool(torch.is_tensor(t) and t.dim() == 2 and t.shape[-1] == self.n_actions)

    def _split_obs(
        self, flat: Any, minimap: Any, action_masks: torch.Tensor | None,
    ) -> tuple[Any, torch.Tensor | None, torch.Tensor | None]:
        """Normalise every accepted observation form to (flat, minimap, masks).

        `flat` may be:
          * a tensor ``[B, F]``                       — flat branch only;
          * a ``(flat, minimap)`` tuple/list          — what EnvManager.reset()/
            step() return in hybrid mode;
          * a ``{"flat": …, "minimap": …}`` mapping   — the same pair as a dict.

        A 2-D second positional is action masks, NOT a minimap. ``get_value(obs,
        masks)`` is the flat/CNN policy signature, and hybrid runs used to reach
        this module through it: the ``(flat, minimap)`` tuple landed in the first
        ``nn.Linear`` unchanged and training died on the first rollout with
        ``TypeError: linear(): argument 'input' (position 1) must be Tensor, not
        tuple``. Unpacking here makes the call site mistake impossible to repeat
        silently — and the errors below name the real problem instead of a matmul.
        """
        if isinstance(flat, dict):
            if "flat" not in flat:
                raise TypeError(
                    f"hybrid obs dict must contain a 'flat' key, got {sorted(flat)}")
            if minimap is None:
                minimap = flat.get("minimap")
            flat = flat["flat"]
        elif isinstance(flat, (tuple, list)):
            if len(flat) != 2:
                raise TypeError(
                    f"hybrid obs tuple must be a (flat, minimap) pair, "
                    f"got {len(flat)} items")
            pair_flat, pair_minimap = flat
            if minimap is not None and not torch.is_tensor(minimap):
                raise TypeError(
                    f"minimap must be a Tensor, got {type(minimap).__name__}")
            if minimap is not None and minimap.dim() != 4:
                # the only non-minimap a caller can pass here is action masks
                if action_masks is None and self._looks_like_masks(minimap):
                    action_masks = minimap
                else:
                    raise TypeError(
                        f"second positional must be a [B, C, G, G] minimap or "
                        f"[B, {self.n_actions}] action masks, got shape "
                        f"{tuple(minimap.shape)}")
                minimap = None
            if minimap is None:
                minimap = pair_minimap
            flat = pair_flat

        if flat is not None and not torch.is_tensor(flat):
            raise TypeError(
                f"hybrid flat obs must be a Tensor, got {type(flat).__name__} "
                f"(a (flat, minimap) pair is unpacked automatically — pass the "
                f"tensors, not the container)")
        if minimap is not None:
            if not torch.is_tensor(minimap):
                raise TypeError(
                    f"minimap must be a Tensor, got {type(minimap).__name__}")
            if minimap.dim() != 4:
                raise TypeError(
                    f"minimap must be [B, C, G, G], got {tuple(minimap.shape)}")
        return flat, minimap, action_masks

    def forward(
        self, flat: Any, minimap: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_in, mini, action_masks = self._split_obs(flat, minimap, action_masks)
        if flat_in is not None and mini is not None:
            h_flat = self.flat_trunk(flat_in)
            h_cnn = self.cnn_head(self.cnn_pool(self.cnn(mini)).flatten(1))
            h = torch.cat([h_flat, h_cnn], dim=1)
        elif flat_in is not None:
            h = self.flat_trunk(flat_in)
        elif mini is not None:
            # The joint trunk is sized flat_out + cnn_out: a minimap-only call
            # cannot match it. Say so instead of failing inside a matmul.
            raise ValueError(
                "hybrid policy needs the flat branch: got minimap only "
                "(use ActorCriticCNN for a minimap-only obs_mode)")
        else:
            raise ValueError("Hybrid obs has no 'flat' or 'minimap'")

        if hasattr(self, "joint"):
            h = self.joint(h)
        values = self.critic_head(h)
        logits = self.actor_head(h)
        if action_masks is not None:
            m = action_masks.float()
            values = values + self.critic_mask_proj(m)
            logits = logits + self.actor_mask_proj(m)
        return logits, values

    def get_action_and_value(
        self,
        flat: Any,
        minimap: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(flat, minimap, action_masks)
        action, log_probs, _ = self._categorical_log_prob(logits, action)
        return action, log_probs, values.squeeze(-1)

    def get_value(
        self, flat: Any, minimap: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, values = self.forward(flat, minimap, action_masks)
        return values.squeeze(-1)

    def act(self, flat: Any, minimap: torch.Tensor | None = None, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedy action (mirrors actor_critic.py helper)."""
        logits, values = self.forward(flat, minimap, action_masks=None)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        log_prob = torch.distributions.Categorical(logits=logits).log_prob(action)
        return action, log_prob, values.squeeze(-1)
