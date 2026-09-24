"""Shared NN utilities — orthogonal init, distribution helpers, base mixin."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional

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
        load_policy_state(self, state)

    @staticmethod
    def _categorical_log_prob(logits: torch.Tensor, action: torch.Tensor | None):
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist


# ── загрузка весов политики из чекпойнта ─────────────────────────────────────

#: Параметры, которых нет в чекпойнтах старше 2026-09-22. Они инициализированы
#: нулями, поэтому их отсутствие сохраняет поведение старой модели.
OPTIONAL_POLICY_KEY_PREFIXES: tuple = ("actor_mask_proj.", "critic_mask_proj.")

_COMPILE_PREFIX = "_orig_mod."


@dataclass
class PolicyLoadReport:
    """Что пропущено/лишнее при загрузке весов (для лога вызывающего)."""

    missing_optional: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)


def clean_policy_state(state: Mapping[str, torch.Tensor]) -> dict:
    """Снять префикс `_orig_mod.` (state_dict модели после torch.compile)."""
    return {
        (k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k): v
        for k, v in state.items()
    }


def load_policy_state(model: nn.Module, state: Mapping[str, torch.Tensor],
                      ckpt_path: str = "<state_dict>",
                      log: Optional[Callable[[str], None]] = None) -> PolicyLoadReport:
    """Единая загрузка весов политики (P1-3 ревью 2026-09-24).

    Раньше в проекте было четыре копии этого кода с разными правилами:
    `train.py` грузил strict=True и падал на любом чекпойнте старше 09-22
    («Missing key(s): actor_mask_proj…»), а worker/ppo/evaluator — strict=False
    без проверки, т.е. молча оставляли случайные веса при чужой архитектуре.

    Контракт:
    * `_orig_mod.` снимается;
    * отсутствие ТОЛЬКО `OPTIONAL_POLICY_KEY_PREFIXES` — норма (старый чекпойнт),
      сообщается в `log`;
    * отсутствие любого другого параметра или несовпадение формы — ValueError
      с путём чекпойнта (архитектура/obs-размер не те);
    * лишние ключи в чекпойнте — предупреждение в `log`, не ошибка.
    """
    clean = clean_policy_state(state)
    expected = model.state_dict()
    shape_errors = [
        f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(expected[k].shape)}"
        for k, v in clean.items()
        if k in expected and hasattr(v, "shape") and tuple(v.shape) != tuple(expected[k].shape)
    ]
    if shape_errors:
        raise ValueError(
            f"checkpoint {ckpt_path} does not fit the model (obs/action size or "
            f"net_arch differ): " + "; ".join(shape_errors[:5]))
    missing = [k for k in expected if k not in clean]
    required_missing = [k for k in missing if not k.startswith(OPTIONAL_POLICY_KEY_PREFIXES)]
    if required_missing:
        raise ValueError(
            f"checkpoint {ckpt_path} is missing policy parameters "
            f"{required_missing[:5]}{' …' if len(required_missing) > 5 else ''} "
            f"— другая архитектура (flat/minimap/hybrid или net_arch)?")
    result = model.load_state_dict(clean, strict=False)
    report = PolicyLoadReport(
        missing_optional=sorted(result.missing_keys),
        unexpected=sorted(result.unexpected_keys),
    )
    if log is not None:
        if report.missing_optional:
            log(f"[Load] {ckpt_path}: legacy checkpoint without "
                f"{report.missing_optional} — zero-initialized (behaviour unchanged)")
        if report.unexpected:
            log(f"[Load] WARNING {ckpt_path}: unexpected keys ignored: {report.unexpected[:5]}")
    return report
