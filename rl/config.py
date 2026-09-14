"""Central configuration — single source of truth for training."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Канонический профиль наград
#   configs/reward_v3.json — единственный источник значений по умолчанию.
#   Дефолты RewardConfig зеркалят его; при наличии файла он побеждает.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REWARD_PROFILE = "reward_v3.json"


def default_reward_profile_path() -> Optional[Path]:
    """Путь к каноническому профилю наград (или None)."""
    p = Path(__file__).resolve().parent.parent / "configs" / DEFAULT_REWARD_PROFILE
    return p if p.exists() else None


# ── helpers ─────────────────────────────────────────────────────────────────

def _as_dict(obj: Any) -> Dict[str, Any]:
    """Dataclass → dict via introspection (DRY, no hardcoded field lists)."""
    if not is_dataclass(obj):
        raise TypeError(f"expected dataclass, got {type(obj)}")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def _update_dataclass_from_dict(obj: Any, data: Dict[str, Any], *, coerce_int: set[str] = frozenset()) -> None:
    """In-place update of dataclass fields from dict, ignoring unknown keys."""
    allowed = {f.name for f in fields(obj)}
    for k, v in data.items():
        if k not in allowed:
            continue
        if k in coerce_int:
            try:
                v = int(v)
            except Exception:
                pass
        setattr(obj, k, v)


# ── RewardConfig ────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """All reward coefficients. Defaults mirror configs/reward_v3.json."""

    # base bonuses
    build_bonus: float = 2.0
    chain_bonus: float = 1.0
    chain_daily: float = 0.5

    # v3: resource extraction / action costs
    first_extraction_bonus: float = 3.0
    extraction_daily: float = 0.3
    need_fill_bonus: float = 1.5
    loan_penalty: float = 0.5

    novelty: float = 5.0
    daily_income: float = 1.0
    sale_bonus: float = 0.5
    tax_daily_bonus: float = 0.3
    survival_bonus: float = 0.0
    game_over_penalty: float = 10.0
    diversity_bonus: float = 3.0

    # penalties / special actions
    error_penalty: float = -2.0
    preserve_penalty: float = 0.3
    demolish_penalty: float = -3.0
    manual_tax_penalty: float = -0.5
    build_cost_penalty: float = 0.0001
    idle_build_penalty: float = -2.0
    idle_build_threshold_days: int = 7
    survival_coeff: float = 0.0

    # milestones
    milestone_base_bonus: float = 30.0
    milestone_people_bonus: float = 2.0
    milestone_day_bonus: float = 2.0
    milestone_year_bonus: float = 5.0

    # spatial / clipping
    proximity_bonus: float = 0.5
    clip_reward_min: float = -50.0
    clip_reward_max: float = 50.0

    # flags (ablations)
    disable_net_worth: bool = False
    disable_daily_income: bool = False
    disable_provider_bonus: bool = False

    # formerly hardcoded C++ weights — must be exported or C++ keeps defaults
    tax_fail_penalty: float = 5.0
    death_penalty: float = 20.0
    base_lost_penalty: float = 30.0
    born_bonus: float = 1.0
    debt_coeff: float = 0.1
    home_overflow_penalty: float = 2.0
    housing_need_bonus: float = 3.0
    food_need_bonus: float = 0.8
    water_need_bonus: float = 0.8
    buy_food_penalty: float = 3.0

    # ── serialization ──

    def to_dict(self) -> Dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base: Optional["RewardConfig"] = None) -> "RewardConfig":
        """Create from dict; only explicitly present keys override base.
        Accepts nested {"reward": {...}} form. Base defaults to canonical profile.
        """
        if isinstance(data.get("reward"), dict):
            data = data["reward"]
        if base is None:
            base = load_default_reward_config()
        # work on a copy to avoid mutating the canonical instance when base is shared
        import copy
        out = copy.copy(base)
        _update_dataclass_from_dict(out, data, coerce_int={"idle_build_threshold_days"})
        return out


def load_default_reward_config() -> RewardConfig:
    """Return RewardConfig from canonical JSON or dataclass defaults."""
    p = default_reward_profile_path()
    if p is not None:
        try:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and raw:
                return RewardConfig.from_dict(raw, base=RewardConfig())
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return RewardConfig()


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Top-level training configuration.

    Fields are intentionally flat for backward compatibility (JSON, UI, CLI).
    For readability they are grouped by comments; validation is in __post_init__.
    """

    # ── environment ──
    map_size: int = 280
    difficulty: str = "normal"  # normal | light
    n_envs: int = 8
    seed: int = 42
    curriculum_stage: int = 0
    unlock_ids: str = ""
    # True = ручной набор зданий из вкладки «Курикулум» (unlock_ids) применяется к среде;
    # False = unlock_ids игнорируется, действует только curriculum_stage.
    use_curriculum_tab: bool = False
    curriculum_resources: str = ""  # csv, e.g. "water,wood,coal"; "" = all
    obs_version: int = 1  # PR 5: 0 = legacy 246-dim obs, 1 = 287-dim frame
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ── PPO ──
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 8192
    n_epochs: int = 10
    gamma: float = 0.999
    gae_lambda: float = 0.98
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02  # 0 disables KL early-stop

    # ── network / observation ──
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    obs_mode: str = "flat"  # flat | minimap | hybrid
    minimap_radius: int = 14

    # ── training / eval ──
    total_timesteps: int = 1_000_000
    save_freq: int = 500_000
    eval_freq: int = 100_000
    eval_episodes: int = 20
    eval_score_weights: Tuple[float, float, float, float] = (0.10, 1.0, 0.10, 0.0001)
    eval_min_bases: int = 5
    eval_min_days: float = 730.0
    eval_use_median: bool = True
    eval_seeds: List[int] = field(default_factory=lambda: [42])
    early_stopping_patience: int = 0
    curriculum_schedule: List[List[int]] = field(default_factory=list)  # [[step, stage], ...]

    # ── performance ──
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 | float16
    torch_compile: bool = False
    cpp_threads: int = 0
    torch_threads: int = 0

    # ── paths ──
    log_dir: str = ""
    model_dir: str = ""

    # ── async / loop detection ──
    async_train: bool = False
    queue_size: int = 2
    loop_detection_enabled: bool = False
    loop_consecutive_threshold: int = 10

    # ── validation & defaults ──
    def __post_init__(self) -> None:
        self._apply_path_defaults()
        self._validate()

    def _apply_path_defaults(self) -> None:
        if not self.log_dir:
            self.log_dir = str(Path.home() / "colony_runs" / "logs")
        if not self.model_dir:
            self.model_dir = str(Path.home() / "colony_runs" / "models")

    def _validate(self) -> None:
        if self.amp_dtype not in ("bfloat16", "float16"):
            raise ValueError(f"amp_dtype must be bfloat16 or float16, got {self.amp_dtype}")
        if self.obs_mode not in ("flat", "minimap", "hybrid"):
            raise ValueError(f"obs_mode must be 'flat', 'minimap', or 'hybrid', got {self.obs_mode}")

        # torch availability warnings — keep non-fatal for CPU-only / test envs
        try:
            import torch  # type: ignore
        except Exception:
            return
        if self.torch_compile and not torch.cuda.is_available():
            warnings.warn(
                "torch_compile=True but CUDA is not available — will run eager.",
                UserWarning,
                stacklevel=3,
            )
        if self.use_amp and self.amp_dtype == "bfloat16" and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():  # type: ignore[attr-defined]
                warnings.warn(
                    "amp_dtype='bfloat16' but GPU has no BF16 — AMP falls back to float32.",
                    UserWarning,
                    stacklevel=3,
                )

    # ── serialization ──
    def to_dict(self) -> Dict[str, Any]:
        """Flat dict + nested reward (backward compatible)."""
        data: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "reward":
                continue
            data[f.name] = getattr(self, f.name)
        # keep eval_score_weights as tuple for code, but JSON will store as list; both accepted
        data["reward"] = self.reward.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config from flat dict (handles legacy nested reward & curriculum aliases)."""
        if not isinstance(data, dict):
            raise TypeError(f"Config.from_dict expected dict, got {type(data)}")

        # avoid mutating caller
        src = dict(data)
        reward_raw = src.pop("reward", None)

        # known fields only — ignore unknown keys for forward compat
        allowed = {f.name for f in fields(cls)} - {"reward"}
        filtered = {k: v for k, v in src.items() if k in allowed}

        # coerce eval_score_weights tuple
        if "eval_score_weights" in filtered:
            try:
                filtered["eval_score_weights"] = tuple(float(x) for x in filtered["eval_score_weights"])
            except Exception:
                pass
        if "eval_min_days" in filtered:
            try:
                filtered["eval_min_days"] = float(filtered["eval_min_days"])
            except Exception:
                pass

        cfg = cls(**filtered)  # type: ignore[arg-type]

        if reward_raw is not None:
            # reward_raw may be dict or nested
            if isinstance(reward_raw, dict):
                cfg.reward = RewardConfig.from_dict(reward_raw)
            else:
                warnings.warn(f"ignoring non-dict reward config: {type(reward_raw)}", UserWarning, stacklevel=2)

        # legacy: curriculum keys sometimes nested inside reward dict or top-level aliases
        for key in ("unlock_ids", "curriculum_resources", "curriculum_stage", "curriculum_schedule"):
            if key in src and not getattr(cfg, key):
                val = src[key]
                if val not in ("", None, []):
                    setattr(cfg, key, val)

        # legacy configs have no `use_curriculum_tab`: если в них уже записан
        # ручной набор зданий — считаем, что он должен применяться.
        if "use_curriculum_tab" not in src and cfg.unlock_ids:
            cfg.use_curriculum_tab = True

        return cfg

    def load_from_file(self, path: str | Path) -> "Config":
        """In-place update from JSON file (partial). Returns self for chaining."""
        p = Path(path)
        if not p.exists():
            return self
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"config file {p} must contain a JSON object")
        allowed = {f.name for f in fields(self)} - {"reward"}
        for k, v in raw.items():
            if k in allowed:
                setattr(self, k, v)
        if "reward" in raw and isinstance(raw["reward"], dict):
            self.reward = RewardConfig.from_dict(raw["reward"])
        # re-validate after load
        self._validate()
        return self

    # ── curriculum ──
    def effective_unlock_ids(self) -> str:
        """Manual building set as CSV, or "" when the «Курикулум» checkbox is off.

        Every env instance (training, eval, watch) must build its curriculum from
        this — otherwise a locked building (e.g. Goldmine) leaks into a scenario
        where only WaterChannel was requested.
        """
        from rl.curriculum import manual_ids_csv

        return manual_ids_csv(self.unlock_ids, self.use_curriculum_tab)

    def curriculum_state(self) -> CurriculumState:
        """Computed curriculum for every env instance (PR 1 single contract)."""
        from rl.curriculum import build_state

        return build_state(
            self.curriculum_stage,
            self.unlock_ids,
            self.use_curriculum_tab,
            self.curriculum_resources,
            self.obs_version,
        )

    def curriculum_meta(self) -> Dict[str, Any]:
        """Curriculum fields to persist next to a checkpoint (meta.json)."""
        return {
            "curriculum_stage_at_best": int(self.curriculum_stage),
            "unlock_ids": self.effective_unlock_ids(),
            "use_curriculum_tab": bool(self.use_curriculum_tab),
            "curriculum_resources": str(self.curriculum_resources or ""),
            "obs_version": int(self.obs_version),
        }

    # ── convenience ──
    @property
    def is_hybrid(self) -> bool:
        return self.obs_mode == "hybrid"

    @property
    def is_minimap(self) -> bool:
        return self.obs_mode in ("minimap", "hybrid")
