from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Единственный канонический профиль наград по умолчанию.
#
#   configs/reward_v3.json  —  ОДИН источник значений по умолчанию для
#   наград во ВСЕХ местах: UI (дефолт полей «Награды»), train.py (CLI без
#   --reward-config), worker (отсутствие ключа в конфиге), C++ gui-бинарь.
#
# Дефолты датакласса RewardConfig НИЖЕ зеркалят этот профиль (keep in sync).
# Если профиль-файл существует и валиден — он побеждает (load_default_reward_config).
# reward.json / reward_v2.json — исторические профили старых экспериментов,
# автозагрузка их НИГДЕ не происходит (только явный --reward-config).
# ─────────────────────────────────────────────────────────────────────────────

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
DEFAULT_REWARD_PROFILE = "reward_v3.json"


def default_reward_profile_path() -> Optional[Path]:
    """Путь к каноническому профилю наград (или None, если файла нет)."""
    p = Path(__file__).resolve().parent.parent / "configs" / DEFAULT_REWARD_PROFILE
    return p if p.exists() else None


@dataclass
class RewardConfig:
    # ─── Defaults = зеркало канонического профиля configs/reward_v3.json ───
    # (единый источник — профиль-файл; при его наличии load_default_reward_config
    #  отдаёт его значения, а отсюда берутся только «запасные» дефолты).
    # v1 values (build 1.0 / novelty 15 / diversity 8 / idle -10@3d /
    # survival_coeff 0.01) let the agent drain cash into one-off bonuses and
    # die at the first annual tax (day 365) — see REPORT_2026_09.md §2.
    # --- base bonuses ---
    build_bonus: float = 2.0
    chain_bonus: float = 1.0
    chain_daily: float = 0.5
    # --- v3: добыча ресурсов / стоимость действий (см. ANALYSIS_TRAINING_REPORT.md §5-P1) ---
    # first_extraction_bonus: разово за эпизод, первый раз добыт тип ресурса
    #   (×вес ресурса = min(1, 0.25*число типов-потребителей); еда/золото = 0)
    # extraction_daily: ежедневно за ресурс с активной добычей
    #   (×вес×насыщение: 1 производитель = 100%, 3+ = всё те же 100%)
    # need_fill_bonus: бонус за постройку производителя ресурса, которого
    #   прямо сейчас «голодает» работающее здание
    # loan_penalty: стоимость успешного TAKE_LOAN (ломает кредитный луп)
    # preserve_penalty ниже поднят до 0.3: PRESERVE/UNPRESERVE становятся
    #   строго хуже «просто прожить день» (ключ раньше не применялся в C++)
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
    # --- penalties for errors / special actions ---
    error_penalty: float = -2.0  # v3 профиль: −2.0 (было −1.0 в v2)
    # v3: preserve/unpreserve теперь реально штрафуется (C++ применяет ключ):
    # «жонглирование» сохранением строго дороже, чем просто прожить день
    preserve_penalty: float = 0.3
    demolish_penalty: float = -3.0
    manual_tax_penalty: float = -0.5
    build_cost_penalty: float = 0.0001
    idle_build_penalty: float = -2.0
    idle_build_threshold_days: int = 7
    survival_coeff: float = 0.0
    # --- milestone bonuses ---
    milestone_base_bonus: float = 30.0
    milestone_people_bonus: float = 2.0
    milestone_day_bonus: float = 2.0
    milestone_year_bonus: float = 5.0
    # --- spatial bonuses ---
    proximity_bonus: float = 0.5
    # --- clip raw reward ---
    clip_reward_min: float = -50.0
    clip_reward_max: float = 50.0
    # --- flags ---
    disable_net_worth: bool = False
    disable_daily_income: bool = False
    disable_provider_bonus: bool = False
    # --- hardcoded weights (from env.cpp, now configurable) ---
    tax_fail_penalty: float = 5.0
    death_penalty: float = 20.0
    base_lost_penalty: float = 30.0
    born_bonus: float = 1.0
    debt_coeff: float = 0.1  # v3 профиль: 0.1 (было 0.02 в v2) — ломает кредитный луп
    home_overflow_penalty: float = 2.0
    housing_need_bonus: float = 3.0
    food_need_bonus: float = 0.8   # снижено чтобы не перевешивало buy_food_penalty (3.0)
    water_need_bonus: float = 0.8
    buy_food_penalty: float = 3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_bonus": self.build_bonus,
            "chain_bonus": self.chain_bonus,
            "chain_daily": self.chain_daily,
            "first_extraction_bonus": self.first_extraction_bonus,
            "extraction_daily": self.extraction_daily,
            "need_fill_bonus": self.need_fill_bonus,
            "loan_penalty": self.loan_penalty,
            "novelty": self.novelty,
            "daily_income": self.daily_income,
            "sale_bonus": self.sale_bonus,
            "tax_daily_bonus": self.tax_daily_bonus,
            "survival_bonus": self.survival_bonus,
            "game_over_penalty": self.game_over_penalty,
            "diversity_bonus": self.diversity_bonus,
            "error_penalty": self.error_penalty,
            "preserve_penalty": self.preserve_penalty,
            "demolish_penalty": self.demolish_penalty,
            "manual_tax_penalty": self.manual_tax_penalty,
            "build_cost_penalty": self.build_cost_penalty,
            "idle_build_penalty": self.idle_build_penalty,
            "idle_build_threshold_days": self.idle_build_threshold_days,
            "survival_coeff": self.survival_coeff,
            "milestone_base_bonus": self.milestone_base_bonus,
            "milestone_people_bonus": self.milestone_people_bonus,
            "milestone_day_bonus": self.milestone_day_bonus,
            "milestone_year_bonus": self.milestone_year_bonus,
            "proximity_bonus": self.proximity_bonus,
            "clip_reward_min": self.clip_reward_min,
            "clip_reward_max": self.clip_reward_max,
            "disable_net_worth": self.disable_net_worth,
            "disable_daily_income": self.disable_daily_income,
            "disable_provider_bonus": self.disable_provider_bonus,
            # --- formerly-hardcoded C++ weights (env.h). They MUST be exported,
            # otherwise the C++ side silently keeps its own defaults and any
            # UI/JSON edits of these fields are ignored. ---
            "tax_fail_penalty": self.tax_fail_penalty,
            "death_penalty": self.death_penalty,
            "base_lost_penalty": self.base_lost_penalty,
            "born_bonus": self.born_bonus,
            "debt_coeff": self.debt_coeff,
            "home_overflow_penalty": self.home_overflow_penalty,
            "housing_need_bonus": self.housing_need_bonus,
            "food_need_bonus": self.food_need_bonus,
            "water_need_bonus": self.water_need_bonus,
            "buy_food_penalty": self.buy_food_penalty,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], base: Optional["RewardConfig"] = None) -> "RewardConfig":
        """JSON переопределяет лишь явно заданные ключи.

        База (дефолты для отсутствующих ключей) — канонический профиль
        configs/reward_v3.json (load_default_reward_config); передаётся
        аргументом base, чтобы избежать рекурсии при загрузке самого профиля.

        Принимает и вложенный формат (d["reward"]) — тогда берётся он.
        """
        if isinstance(d.get("reward"), dict):
            d = d["reward"]
        if base is None:
            base = load_default_reward_config()
        for k, v in d.items():
            if not hasattr(base, k):
                continue
            if k == "idle_build_threshold_days":
                v = int(v)
            setattr(base, k, v)
        return base


def load_default_reward_config() -> "RewardConfig":
    """Канонический профиль наград по умолчанию.

    Возвращает RewardConfig, заполненный из configs/reward_v3.json
    (единственный источник дефолтов). Если файла нет или он невалиден —
    дефолты датакласса (зеркало того же профиля).

    Используется везде, где нужен «дефолт наград»: UI, train.py без
    --reward-config, RewardConfig.from_dict при отсутствующих ключах.
    """
    p = default_reward_profile_path()
    if p is not None:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and d:
                return RewardConfig.from_dict(d, base=RewardConfig())
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return RewardConfig()


@dataclass
class Config:
    # Environment
    map_size: int = 280
    difficulty: str = "normal"  # "normal" or "light" (light = 2x money, no main tax)
    n_envs: int = 8
    seed: int = 42
    curriculum_stage: int = 0
    unlock_ids: str = ""
    # Curriculum: фильтр ресурсов (comma-separated, e.g. "water,wood,coal,iron,oil"). Пусто = все ресурсы.
    curriculum_resources: str = ""
    reward: RewardConfig = field(default_factory=RewardConfig)

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 8192
    n_epochs: int = 10
    gamma: float = 0.999
    gae_lambda: float = 0.98
    clip_range: float = 0.2
    ent_coef: float = 0.01  # P1: ent_coef 0.01 with annealing down to 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02          # 0 = early-stop по KL выключен

    # Network
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    # Observation mode: "flat" = 209-dim vector + MLP (legacy),
    # "minimap" = 2D spatial tensor (8, 2R+1, 2R+1) + CNN trunk
    obs_mode: str = "flat"
    minimap_radius: int = 14

    # Training
    total_timesteps: int = 1_000_000
    save_freq: int = 500_000
    eval_freq: int = 100_000
    eval_episodes: int = 20

    # Eval scoring (composite champion selection)
    # v2 (SWEEP_ANALYSIS 4.1): score = days*0.10 + bases*1.0 + people*0.10 + 1e-4*max(0,return)
    # threshold: bases>=5 AND days>=730 (survived 2+ annual taxes)
    eval_score_weights: tuple = (0.10, 1.0, 0.10, 0.0001)
    eval_min_bases: int = 5
    eval_min_days: float = 730.0
    eval_use_median: bool = True
    eval_seeds: list = field(default_factory=lambda: [42])
    early_stopping_patience: int = 0

    # Curriculum schedule: list of (step_threshold, stage) pairs
    curriculum_schedule: list = field(default_factory=list)

    # GPU / performance
    device: str = "cuda"
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    torch_compile: bool = False
    cpp_threads: int = 0
    torch_threads: int = 0

    # Paths
    log_dir: str = ""
    model_dir: str = ""

    # Async
    async_train: bool = False
    queue_size: int = 2

    # Loop detection
    loop_detection_enabled: bool = False
    loop_consecutive_threshold: int = 10

    def __post_init__(self):
        if not self.log_dir:
            home = Path.home()
            self.log_dir = str(home / "colony_runs" / "logs")
        if not self.model_dir:
            home = Path.home()
            self.model_dir = str(home / "colony_runs" / "models")
        if self.amp_dtype not in ("bfloat16", "float16"):
            raise ValueError(f"amp_dtype must be bfloat16 or float16, got {self.amp_dtype}")
        if self.obs_mode not in ("flat", "minimap", "hybrid"):
            raise ValueError(f"obs_mode must be 'flat', 'minimap', or 'hybrid', got {self.obs_mode}")

        import torch
        if self.torch_compile and not torch.cuda.is_available():
            import warnings
            warnings.warn(
                "torch_compile=True but CUDA is not available. "
                "torch.compile will be disabled at runtime.",
                UserWarning,
                stacklevel=2,
            )

        if self.use_amp and self.amp_dtype == "bfloat16" and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                import warnings
                warnings.warn(
                    "amp_dtype='bfloat16' but GPU does not support BF16. "
                    "AMP will silently fall back to float32.",
                    UserWarning,
                    stacklevel=2,
                )

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k in (
            "map_size", "difficulty", "n_envs", "seed", "curriculum_stage", "unlock_ids",
            "curriculum_resources",
            "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
            "target_kl",
            "net_arch", "obs_mode", "minimap_radius",
            "total_timesteps", "save_freq", "eval_freq", "eval_episodes",
            "eval_score_weights", "eval_min_bases", "eval_min_days",
            "eval_use_median", "eval_seeds", "early_stopping_patience", "curriculum_schedule",
            "device", "use_amp", "amp_dtype", "torch_compile",
            "cpp_threads", "torch_threads", "async_train", "queue_size",
            "loop_detection_enabled", "loop_consecutive_threshold",
            "log_dir", "model_dir",
        ):
            d[k] = getattr(self, k)
        d["reward"] = self.reward.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        reward = d.pop("reward", None)
        # copy so caller dict not mutated unexpectedly when 'reward' popped
        d = dict(d)
        cfg = cls(
            map_size=d.get("map_size", 280),
            difficulty=d.get("difficulty", "normal"),
            n_envs=d.get("n_envs", 8),
            seed=d.get("seed", 42),
            curriculum_stage=d.get("curriculum_stage", 0),
            unlock_ids=d.get("unlock_ids", ""),
            curriculum_resources=d.get("curriculum_resources", ""),
            learning_rate=d.get("learning_rate", 3e-4),
            n_steps=d.get("n_steps", 4096),
            batch_size=d.get("batch_size", 8192),
            n_epochs=d.get("n_epochs", 10),
            gamma=d.get("gamma", 0.999),
            gae_lambda=d.get("gae_lambda", 0.98),
            clip_range=d.get("clip_range", 0.2),
            ent_coef=d.get("ent_coef", 0.01),
            vf_coef=d.get("vf_coef", 0.5),
            max_grad_norm=d.get("max_grad_norm", 0.5),
            target_kl=d.get("target_kl", 0.02),
            net_arch=d.get("net_arch", [256, 256]),
            obs_mode=d.get("obs_mode", "flat"),
            minimap_radius=d.get("minimap_radius", 14),
            total_timesteps=d.get("total_timesteps", 1_000_000),
            save_freq=d.get("save_freq", 500_000),
            eval_freq=d.get("eval_freq", 100_000),
            eval_episodes=d.get("eval_episodes", 20),
            eval_score_weights=tuple(float(x) for x in d.get("eval_score_weights", (0.10, 1.0, 0.10, 0.0001))),
            eval_min_bases=d.get("eval_min_bases", 5),
            eval_min_days=float(d.get("eval_min_days", 730.0)),
            eval_use_median=d.get("eval_use_median", True),
            eval_seeds=d.get("eval_seeds", [42]),
            early_stopping_patience=d.get("early_stopping_patience", 0),
            curriculum_schedule=d.get("curriculum_schedule", []),
            device=d.get("device", "cuda"),
            use_amp=d.get("use_amp", True),
            amp_dtype=d.get("amp_dtype", "bfloat16"),
            torch_compile=d.get("torch_compile", False),
            cpp_threads=d.get("cpp_threads", 0),
            torch_threads=d.get("torch_threads", 0),
            async_train=d.get("async_train", False),
            queue_size=d.get("queue_size", 2),
            loop_detection_enabled=d.get("loop_detection_enabled", False),
            loop_consecutive_threshold=d.get("loop_consecutive_threshold", 10),
            log_dir=d.get("log_dir", ""),
            model_dir=d.get("model_dir", ""),
        )
        if reward is not None:
            cfg.reward = RewardConfig.from_dict(reward)
        # также поддержать вложенные ключи unlock_ids / curriculum_* внутри reward-словаря (совместимость)
        for k in ("unlock_ids", "curriculum_resources", "curriculum_stage", "curriculum_schedule"):
            if k in d and getattr(cfg, k, None) == "" and d[k] not in ("", None, []):
                setattr(cfg, k, d[k])
        return cfg

    def load_from_file(self, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            return self
        with open(p) as f:
            data = json.load(f)
        import dataclasses
        allowed = {f.name for f in dataclasses.fields(Config)} - {"reward"}
        self.__dict__.update({k: v for k, v in data.items() if k in allowed})
        if "reward" in data:
            self.reward = RewardConfig.from_dict(data["reward"])
        return self
