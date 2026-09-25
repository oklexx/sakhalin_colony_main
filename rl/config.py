"""Central configuration — single source of truth for training."""
from __future__ import annotations

import json
import warnings
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# Для аннотации Config.curriculum_state (ruff F821, get_type_hints). Цикла нет:
# rl.curriculum зависит только от stdlib и rl.config не импортирует.
from rl.curriculum import CurriculumState

# ─────────────────────────────────────────────────────────────────────────────
# Канонический профиль наград
#   configs/reward_v4.json — единственный источник значений по умолчанию.
#   Дефолты RewardConfig зеркалят его; при наличии файла он побеждает.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REWARD_PROFILE = "reward_v4.json"


def default_reward_profile_path() -> Path | None:
    """Путь к каноническому профилю наград (или None)."""
    p = Path(__file__).resolve().parent.parent / "configs" / DEFAULT_REWARD_PROFILE
    return p if p.exists() else None


# ── helpers ─────────────────────────────────────────────────────────────────

def _as_dict(obj: Any) -> dict[str, Any]:
    """Dataclass → dict via introspection (DRY, no hardcoded field lists)."""
    if not is_dataclass(obj):
        raise TypeError(f"expected dataclass, got {type(obj)}")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def _update_dataclass_from_dict(obj: Any, data: dict[str, Any], *, coerce_int: AbstractSet[str] = frozenset()) -> None:
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
    """All reward coefficients. Defaults mirror configs/reward_v4.json."""

    # base bonuses (v4: build_bonus = 1.2, build_cost_penalty = 0.00004)
    build_bonus: float = 1.2
    chain_bonus: float = 1.0
    chain_daily: float = 0.5

    # v3/v4: resource extraction / action costs
    first_extraction_bonus: float = 3.0
    extraction_daily: float = 0.3
    need_fill_bonus: float = 1.5
    loan_penalty: float = 2.0
    # P0: штраф за переоформление неоплаченного налога в долг банку
    # (log1p(borrowed/1000) * вес). См. Game::settle_tax_with_debt.
    tax_debt_penalty: float = 2.0

    novelty: float = 3.0
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
    build_cost_penalty: float = 0.00004
    idle_build_penalty: float = -2.0
    idle_build_threshold_days: int = 7
    survival_coeff: float = 0.0005

    # milestones (v4: milestone_base_bonus = 10.0)
    milestone_base_bonus: float = 10.0
    milestone_people_bonus: float = 2.0
    milestone_day_bonus: float = 2.0
    milestone_year_bonus: float = 5.0

    # spatial / clipping (v4: clip +-100)
    proximity_bonus: float = 0.5
    clip_reward_min: float = -100.0
    clip_reward_max: float = 100.0

    # P2-9 (2026-09-19): potential-based road shaping к воде. Раньше числа были
    # захардкожены в ColonyEnvCpp::step()/::reset() (1.5 / 1.0 / +3.0 / -0.1),
    # теперь это такие же поля профиля, как остальные. Дефолты = прежний хардкод,
    # поэтому старые прогоны не меняются.
    road_shaping_cap: float = 1.5          # потолок бонуса за одну дорогу
    road_shaping_per_cell: float = 1.0     # бонус за клетку приближения к воде
    water_reach_bonus: float = 3.0         # разовый бонус «дорога дошла до воды»
    water_reach_radius: float = 1.5        # что считать «дошли» (клеток до воды)
    road_no_progress_penalty: float = 0.1  # штраф за дорогу без приближения
    road_progress_epsilon: float = 0.25    # гистерезис: считать ли приближение

    # flags (ablations)
    disable_net_worth: bool = False
    disable_daily_income: bool = False
    disable_provider_bonus: bool = False
    # P1: маскировать менеджеров по применимости (не по формальной легальности)
    mask_managers_by_applicability: bool = True
    # Абляции, которые раньше жили ТОЛЬКО в C++ (include/colony/reward_config.h)
    # и потому были недостижимы из Python: RewardConfig.from_dict молча выбрасывал
    # неизвестные ключи, и значение из reward-JSON не долетало до среды.
    # priority_count_over_allowed — считать потребителей ресурсов в compute_catalog
    #   только по разрешённым курикулумом постройкам (false = по всем 32).
    # obs_mask_locked_catalog — занулять строки каталога закрытых построек в obs
    #   (false = каталог гейт игнорирует; counts/idle_by_type не фильтруются).
    priority_count_over_allowed: bool = False
    obs_mask_locked_catalog: bool = False

    # formerly hardcoded C++ weights — must be exported or C++ keeps defaults
    tax_fail_penalty: float = 5.0
    death_penalty: float = 12.0
    base_lost_penalty: float = 30.0
    born_bonus: float = 2.0
    debt_coeff: float = 0.0
    home_overflow_penalty: float = 2.0
    housing_need_bonus: float = 3.0
    food_need_bonus: float = 0.8
    water_need_bonus: float = 0.8
    buy_food_penalty: float = 3.0

    # v4 (2026-09): цель на выживание, бонус за оплату 500k и предналоговое давление
    goal_survival_coeff: float = 200.0
    main_tax_cash_bonus: float = 100.0
    main_tax_pressure_coeff: float = 0.002

    # ── serialization ──

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], base: RewardConfig | None = None) -> RewardConfig:
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
    # P0 (2026-09-17): 0 = 248-dim, 1 = 289-dim (frame), 2 = 299-dim (+dx,dy
    # к ближайшим wood/coal/iron/oil/gold). 2 — дефолт: без направлений 12
    # из 32 построек недостижимы политикой (карты в flat-obs нет).
    obs_version: int = 2  # = rl.curriculum.CURRENT_OBS_VERSION (держать в синхроне)
    # Automatic tax settlement is deliberately independent from the manual
    # TAKE_LOAN/REPAY_LOAN mechanic gate. True converts unpaid tax to bank debt;
    # false keeps the explicit/dialogue tax policy (no hidden manual loan).
    tax_to_debt: bool = True
    # Mechanic gating never changes the 49-logit action head. Defaults keep a
    # NEW run in the minimal «stage 1» mode (see rl/curriculum.py
    # STAGE1_PRESET and docs/TWO_STAGE_TRAINING_2026_09.md): only «sell the
    # surplus» and «credit» are enabled — the manager slots that the first
    # survival loop needs. Old configs without these fields keep the legacy
    # all-enabled behaviour (Config.from_dict resets them to []).
    disabled_mechanics: list[str] = field(
        default_factory=lambda: [
            "improve_land", "repair", "destroy", "preservation",
            "buy_food", "manual_tax",
        ]
    )
    # Unlock entries are additive and applied by absolute environment steps.
    # Empty by default: stage 2 is a separate run with its own config (or the
    # user edits the schedule manually in the UI).
    mechanics_unlock_schedule: list = field(default_factory=list)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ── PPO ──
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 8192
    n_epochs: int = 10
    # Баланс дисконта (2026-09): gamma=0.999 даёт горизонт ~1000 дней (полураспад 693 дня).
    # Это полностью покрывает годовой налоговый цикл (365 дней) и окупаемость зданий,
    # избегая при этом катастрофической дисперсии критика при 0.99999.
    gamma: float = 0.999
    gae_lambda: float = 0.99
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02  # 0 disables KL early-stop

    # ── network / observation ──
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    obs_mode: str = "flat"  # flat | minimap | hybrid
    # DEPRECATED / dead: ColonyEnvCpp::minimap() emits a fixed global 32x32 and
    # _make_model() passes grid_size=32 explicitly, so this value is ignored
    # (the nets rewrite it to 15). Kept so old configs still load.
    # DEPRECATED (P2-8): миникарта давно глобальная 32x32 (ColonyEnvCpp::
    # minimap_grid_size()), поле ни на что не влияет и сохранено только чтобы
    # старые конфиги/чекпойнты читались. Сеттер в C++ выдаёт DeprecationWarning.
    minimap_radius: int = 14

    # ── training / eval ──
    total_timesteps: int = 1_000_000
    save_freq: int = 500_000
    eval_freq: int = 100_000
    eval_episodes: int = 20
    eval_score_weights: tuple[float, float, float, float] = (0.10, 1.0, 0.10, 0.0001)
    eval_min_bases: int = 5
    eval_min_days: float = 730.0
    eval_use_median: bool = True
    eval_seeds: list[int] = field(default_factory=lambda: [42])
    early_stopping_patience: int = 0
    curriculum_schedule: list[list[int]] = field(default_factory=list)  # [[step, stage], ...]

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

    # ── мониторинг действий (вкладка «Мониторинг» в UI) ──
    # Что означает «процент» в панели: доля шагов роллаута, в которых политика
    # выбрала действие. Счётчики копятся на весь роллаут, а не по последним
    # 1000 действий: окно в 1000 при n_steps*n_envs = 32768 показывало ~3%
    # роллаута, и строка «Водоканал» гасла только потому, что в ХВОСТЕ роллаута
    # не осталось нажатий.
    # 0 = не обрезать список (все действия с count > 0 уезжают в UI, а сколько
    # строк показать решает панель). >0 — жёсткий потолок строк в сообщении.
    monitor_action_top: int = 0
    # Считать долю шагов, на которых действие было легальным (маска == 1).
    # Без неё невозможно отличить «политика не хочет» от «маска закрыла» —
    # именно этот вопрос и возникал на водоканале (нужен свободный LT_WATER,
    # см. tests/cpp/water_mask_check.cpp W3/W6). Стоит одно сложение по строке
    # маски на шаг, поэтому включено по умолчанию.
    monitor_action_legality: bool = True

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
            import torch
        except Exception:
            return
        if self.torch_compile and not torch.cuda.is_available():
            warnings.warn(
                "torch_compile=True but CUDA is not available — will run eager.",
                UserWarning,
                stacklevel=3,
            )
        if self.use_amp and self.amp_dtype == "bfloat16" and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                warnings.warn(
                    "amp_dtype='bfloat16' but GPU has no BF16 — AMP falls back to float32.",
                    UserWarning,
                    stacklevel=3,
                )

    # ── serialization ──
    def to_dict(self) -> dict[str, Any]:
        """Flat dict + nested reward (backward compatible)."""
        data: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "reward":
                continue
            data[f.name] = getattr(self, f.name)
        # keep eval_score_weights as tuple for code, but JSON will store as list; both accepted
        data["reward"] = self.reward.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
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

        cfg = cls(**filtered)

        # A pre-mechanic config/checkpoint had no gating contract. Preserve its
        # legacy all-enabled behaviour; newly created Config() instances use
        # the early-economy defaults above. Once serialized, the new fields are
        # explicit and therefore survive round-trips.
        if "disabled_mechanics" not in src and "mechanics_unlock_schedule" not in src:
            cfg.disabled_mechanics = []
            cfg.mechanics_unlock_schedule = []

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

    def load_from_file(self, path: str | Path) -> Config:
        """In-place update from JSON file (partial). Returns self for chaining."""
        p = Path(path)
        if not p.exists():
            return self
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"config file {p} must contain a JSON object")
        if "disabled_mechanics" not in raw and "mechanics_unlock_schedule" not in raw:
            # Same compatibility rule as from_dict for old flat config files.
            self.disabled_mechanics = []
            self.mechanics_unlock_schedule = []
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

    def curriculum_state(self, step: int = 0) -> CurriculumState:
        """Compute one transport contract for a training/eval environment.

        ``step`` is absolute training progress. Eval/watch callers use the
        checkpoint's persisted allow-list or explicitly pass a step; a fresh
        run therefore starts with the configured mechanics disabled.
        """
        from rl.curriculum import build_state, mechanics_enabled_at_step

        enabled = mechanics_enabled_at_step(
            step, self.disabled_mechanics, self.mechanics_unlock_schedule
        )
        return build_state(
            self.curriculum_stage,
            self.unlock_ids,
            self.use_curriculum_tab,
            self.curriculum_resources,
            self.obs_version,
            enabled,
        )

    def enabled_mechanics_at(self, step: int = 0) -> tuple[str, ...]:
        """Canonical allow-list at ``step`` (useful for metadata/UI/tests)."""
        from rl.curriculum import mechanics_enabled_at_step
        return mechanics_enabled_at_step(
            step, self.disabled_mechanics, self.mechanics_unlock_schedule
        )

    def curriculum_meta(self) -> dict[str, Any]:
        """Curriculum fields to persist next to a checkpoint (meta.json)."""
        return {
            "curriculum_stage_at_best": int(self.curriculum_stage),
            "unlock_ids": self.effective_unlock_ids(),
            "use_curriculum_tab": bool(self.use_curriculum_tab),
            "curriculum_resources": str(self.curriculum_resources or ""),
            "obs_version": int(self.obs_version),
            "tax_to_debt": bool(self.tax_to_debt),
            "disabled_mechanics": list(self.disabled_mechanics),
            "mechanics_unlock_schedule": self.mechanics_unlock_schedule,
            "enabled_mechanics": list(self.enabled_mechanics_at(0)),
        }

    # ── convenience ──
    @property
    def is_hybrid(self) -> bool:
        return self.obs_mode == "hybrid"

    @property
    def is_minimap(self) -> bool:
        return self.obs_mode in ("minimap", "hybrid")
