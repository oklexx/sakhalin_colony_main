"""UI constants — groups, curriculum presets, captions, resource lists.

Extracted from main_window.py for maintainability. All values are re-exported
via train_ui2.main_window for backward compatibility (tests import from there).
"""
from __future__ import annotations

# ── parameter groups (training tab) ──────────────────────────────────────────
PARAM_GROUPS = {
    "Среда": ["n_envs", "map_size", "seed", "total_timesteps"],
    "PPO": [
        "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
        "vf_coef", "max_grad_norm", "target_kl", "n_steps", "batch_size", "n_epochs",
    ],
    "Оценка и сохранение": [
        "eval_freq", "eval_episodes", "eval_min_days",
        "eval_min_bases", "save_freq", "early_stopping_patience",
    ],
}

# ── reward groups (rewards tab) ──────────────────────────────────────────────
REWARD_GROUPS = {
    "Стройка": ["build_bonus", "chain_bonus", "chain_daily", "novelty",
                "diversity_bonus", "proximity_bonus", "build_cost_penalty"],
    "Добыча (v3)": ["first_extraction_bonus", "extraction_daily",
                    "need_fill_bonus", "loan_penalty"],
    "Экономика": ["daily_income", "sale_bonus", "tax_daily_bonus",
                  "manual_tax_penalty", "debt_coeff", "tax_debt_penalty",
                  "main_tax_cash_bonus", "main_tax_pressure_coeff"],
    "Выживание": ["survival_bonus", "survival_coeff", "game_over_penalty",
                  "death_penalty", "tax_fail_penalty", "base_lost_penalty",
                  "born_bonus", "home_overflow_penalty", "goal_survival_coeff"],
    # P2-9 (2026-09-19): potential-based shaping дороги к воде — раньше эти
    # числа были хардкодом в src/env.cpp и в UI их нельзя было ни увидеть, ни
    # подобрать.
    "Дороги к воде": ["road_shaping_cap", "road_shaping_per_cell",
                      "water_reach_bonus", "water_reach_radius",
                      "road_no_progress_penalty", "road_progress_epsilon"],
    "Потребности": ["housing_need_bonus", "food_need_bonus", "water_need_bonus",
                    "buy_food_penalty"],
    "Дисциплина": ["error_penalty", "preserve_penalty", "demolish_penalty",
                   "idle_build_penalty", "idle_build_threshold_days"],
    "Milestones и клип": ["milestone_base_bonus", "milestone_people_bonus",
                          "milestone_day_bonus", "milestone_year_bonus",
                          "clip_reward_min", "clip_reward_max"],
}

REWARD_FLAGS = [
    ("disable_daily_income", "Выкл. daily_income", "Не начислять ежедневный доход (абляция)"),
    ("disable_net_worth", "Выкл. net worth бонус", "Убрать бонус за чистую стоимость (абляция)"),
    ("disable_provider_bonus", "Выкл. provider бонус", "Убрать бонус провайдера потребностей (абляция)"),
    ("mask_managers_by_applicability", "Маска по применимости",
     "Маскировать менеджеров по применимости (BUY_FOOD без нужды, REPAY без долга, "
     "IMPROVE без денег и т.п. — иначе это чистый error_penalty)"),
    ("priority_count_over_allowed", "Потребители только по разрешённым",
     "Считать потребителей ресурсов (n_consumers в каталоге) только по постройкам, "
     "разрешённым курикулумом. Выключено = считать по всем 32 (старые прогоны)"),
    ("obs_mask_locked_catalog", "Занулять каталог закрытых построек",
     "В наблюдении занулять строки каталога (4 числа на постройку) для построек, "
     "закрытых курикулумом. Выключено = каталог гейт игнорирует"),
]

# ── curriculum (single source: rl/curriculum.py) ─────────────────────────────
try:
    from rl.curriculum import STAGE_MAP as CURRICULUM_STAGE_MAP, ALL_IDS as ALL_BUILD_IDS
except Exception:  # fallback for environments without torch / rl importable
    CURRICULUM_STAGE_MAP = {
        1: ["House", "SmallHouse", "Farm", "Garden", "Mushroom", "WaterChannel",
            "Refinery", "Fish", "HuntingLand", "CowFarm", "Apiary", "Hothouse",
            "Puerperal", "Road"],
        2: ["Sawmill", "Coalmine", "CoalCut", "Ironmine", "PowerStation",
            "HydroStation", "AirStation", "Torchlight", "Goldmine", "BigHouse",
            "BigFarm"],
        3: ["BigSawmill", "BigRefinary", "BigIronmine", "WaterMill",
            "SmallAtomStation", "AtomStation", "SuperHouse"],
    }
    ALL_BUILD_IDS = [
        "Farm", "Garden", "WaterChannel", "Sawmill", "Coalmine", "Ironmine", "Refinery", "Goldmine",
        "PowerStation", "HydroStation", "Road", "House", "SmallHouse", "Fish", "CoalCut",
        "HuntingLand", "CowFarm", "Mushroom", "BigHouse", "BigFarm", "Apiary", "Torchlight", "Hothouse",
        "SuperHouse", "BigSawmill", "WaterMill", "BigRefinary", "Puerperal", "BigIronmine",
        "AirStation", "SmallAtomStation", "AtomStation",
    ]

BUILD_CAPTIONS = {
    "Farm": "Ферма", "Garden": "Сад", "WaterChannel": "Водоканал", "Sawmill": "Лесопилка",
    "Coalmine": "Шахта", "Ironmine": "Карьер", "Refinery": "Нефтедобыча", "Goldmine": "Золотой прииск",
    "PowerStation": "Электростанция", "HydroStation": "Гидростанция", "Road": "Дорога", "House": "Жилой дом",
    "SmallHouse": "Хижина", "Fish": "Рыбный промысел", "CoalCut": "Угольный разрез",
    "HuntingLand": "Охотничьи угодья", "CowFarm": "Животноводческая ферма", "Mushroom": "Грибная плантация",
    "BigHouse": "Жилой район", "BigFarm": "Хозяйство", "Apiary": "Пасека", "Torchlight": "Факел",
    "Hothouse": "Теплица", "SuperHouse": "Жилой центр", "BigSawmill": "Лесоповал",
    "WaterMill": "Водокачка", "BigRefinary": "Нефтенасос", "Puerperal": "Дом матери и ребенка",
    "BigIronmine": "Катакомбы", "AirStation": "Ветряная ЭС", "SmallAtomStation": "Малая АЭС",
    "AtomStation": "АЭС",
}

RESOURCE_IDS = ["gold", "food", "coal", "iron", "oil", "stone", "water", "wood", "energy"]
RESOURCE_CAPTIONS = {
    "gold": "Золото", "food": "Еда", "coal": "Уголь", "iron": "Железо", "oil": "Нефть",
    "stone": "Камень", "water": "Вода", "wood": "Дерево", "energy": "Энергия",
}

BUILD_IMAGE_INDEX = {
    "Farm": 0, "Garden": 1, "WaterChannel": 2, "Sawmill": 3, "Coalmine": 4, "Ironmine": 5,
    "Refinery": 6, "Goldmine": 7, "PowerStation": 8, "HydroStation": 9, "Road": 12,
    "House": 13, "SmallHouse": 14, "Fish": 15, "CoalCut": 11, "HuntingLand": 16,
    "CowFarm": 17, "Mushroom": 18, "BigHouse": 19, "BigFarm": 20, "Apiary": 21,
    "Torchlight": 22, "Hothouse": 23, "SuperHouse": 24, "BigSawmill": 25, "WaterMill": 26,
    "BigRefinary": 27, "Puerperal": 28, "BigIronmine": 29, "AirStation": 30, "SmallAtomStation": 31,
    "AtomStation": 32,
}

# ── config persistence ──────────────────────────────────────────────────────
CONFIG_VERSION = 2
