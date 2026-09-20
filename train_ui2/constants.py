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
    ("disable_daily_income", "Отключить бонус за прибыль",
     "Не платить бонус за дни, когда бюджет колонии растёт."),
    ("disable_net_worth", "Отключить бонус за капитал",
     "Не учитывать общий капитал (деньги + стоимость зданий) в награде."),
    ("disable_provider_bonus", "Отключить бонус «покровителя»",
     "Не платить бонус за постройку здания, которое закрывает чужую нужду "
     "(например, водоканал для фермы)."),
    ("mask_managers_by_applicability", "Прятать бессмысленные действия",
     "Если у действия заведомо нет смысла прямо сейчас (купить еду, когда склад полон; "
     "заплатить налог, когда нечего платить) — не показывать его ИИ вообще. "
     "Иначе ИИ будет наступать на эти грабли и терять очки."),
    ("priority_count_over_allowed", "Считать нужды только открытых зданий",
     "Считать «сколько зданий ждёт ресурс» только по зданиям, разрешённым на этом этапе. "
     "Выключено = считать по всем 32 зданиям игры, даже закрытым (как в старых прогонах)."),
    ("obs_mask_locked_catalog", "Скрывать закрытые здания из обзора",
     "ИИ не видит информацию о зданиях, закрытых на этом этапе (как будто их пока "
     "не существует). Выключено = ИИ видит всё меню, но строить закрытое нельзя."),
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

# ── Гейтинг механик: 11 действий-менеджеров, сгруппированных в 8 механик ────
# Единый источник порядка — rl/curriculum.py MECHANIC_NAMES (= C++
# MECHANIC_NAMES в include/colony/constants.h); фолбэк повторяет его для
# окружений без импорта rl. Регресс: test_stage1_preset.py.
try:
    from rl.curriculum import MECHANIC_NAMES as MECHANIC_IDS
except Exception:
    MECHANIC_IDS = (
        "improve_land", "repair", "destroy", "preservation",
        "sell", "buy_food", "credit", "manual_tax",
    )

# Человеческие подписи + пояснения для вкладки «Курикулум».
MECHANIC_CAPTIONS = {
    "improve_land": ("Улучшить землю", "Кнопка ИИ «улучшить участок земли» (дорого, ~33 000)."),
    "repair": ("Ремонт зданий", "Кнопки ИИ «починить худшее здание» и «отремонтировать всё». Здания изнашиваются — без ремонта разрушаются."),
    "destroy": ("Снос здания", "Кнопка ИИ «снести худшее здание» (со штрафом)."),
    "preservation": ("Консервация зданий", "Кнопки ИИ «законсервировать здание» и «расконсервировать». Консервация замораживает износ, но здание не работает."),
    "sell": ("Продать излишки", "Кнопка ИИ «продать всё лишнее со склада на рынок» — главный источник денег."),
    "buy_food": ("Купить еду", "Кнопка ИИ «докупить еды». Жители в этой игре еду не едят — почти всегда бессмысленная трата."),
    "credit": ("Кредиты банка", "Кнопки ИИ «взять 50 000 в долг» и «вернуть долг». Спасение, если бюджет съеден, но долг растёт под проценты."),
    "manual_tax": ("Оплатить налог вручную", "Кнопка ИИ «заплатить налог». Налог и так списывается автоматически — вручную платить незачем."),
}

# ── Пресеты обучения (вкладка «Курикулум», кнопка «Пресет») ─────────────────
# «stage1» — тот же источник, что и CLI --preset stage1 (rl/curriculum.py
# STAGE1_PRESET); здесь остаётся только презентация.
try:
    from rl.curriculum import STAGE1_PRESET as _S1
    _S1_BUILDINGS = [b for b in str(_S1["unlock_ids"]).split(",") if b]
    _S1_RESOURCES = [r for r in str(_S1["curriculum_resources"]).split(",") if r]
    _S1_MECHANICS = ("sell", "credit")
except Exception:
    _S1_BUILDINGS = ["Road", "WaterChannel", "Garden", "Farm", "Mushroom",
                     "Fish", "SmallHouse", "House"]
    _S1_RESOURCES = ["water", "food", "wood"]
    _S1_MECHANICS = ("sell", "credit")

PRESETS = {
    "stage1": {
        "title": "Этап 1 · База и ресурсы",
        "summary": "Минимум для выживания: дорога к воде → водоканал → еда → продажа излишков.",
        "detail": (
            "Что открыто ИИ:\n"
            "  Здания (8): дорога, водоканал, сад, ферма, грибная плантация,\n"
            "              рыбный промысел, хижина, жилой дом.\n"
            "  Действия (2): продать излишки, кредит банка.\n"
            "  Бонусы добычи: вода, еда, дерево.\n"
            "Всё остальное (шахты, лесопилка, энергетика, ремонт, снос, налоги) закрыто.\n"
            "Задача ИИ: построить первую рабочую цепочку и прожить год.\n"
            "Следующий шаг: «Этап 2 · Вся экономика» — дообучение от лучшей модели этапа 1."
        ),
        "buildings": _S1_BUILDINGS,
        "resources": _S1_RESOURCES,
        "mechanics": list(_S1_MECHANICS),
    },
    "full": {
        "title": "Этап 2 · Вся экономика",
        "summary": "Полная игра: все 32 здания, все действия. Дообучайте от лучшей модели этапа 1.",
        "detail": (
            "Все здания и механики открыты, приоритетных ресурсов нет.\n"
            "Запускать кнопкой «Дообучить» на вкладке «Модели» — тогда стартуют веса\n"
            "лучшей модели этапа 1 (обычно с lr в ~10 раз ниже)."
        ),
        "buildings": [],   # пусто = все
        "resources": [],   # пусто = все
        "mechanics": list(MECHANIC_IDS),
    },
}
PRESET_ORDER = ["stage1", "full"]

# ── Глоссарий наград простыми словами (вкладка «Награды») ────────────────────
# Ключ = поле RewardConfig. Показывается под подписью параметра серым.
HUMAN_REWARD_HELP = {
    # Стройка
    "build_bonus": "Сколько очков ИИ получает за каждое новое здание.",
    "chain_bonus": "Разовая премия за первую рабочую цепочку (добыл → переработал).",
    "chain_daily": "Ежедневная премия за каждую работающую цепочку.",
    "novelty": "Премия за здание нового типа — чтобы не строил сто ферм подряд.",
    "diversity_bonus": "Доплата за разнообразие: чем больше разных типов зданий, тем больше.",
    "proximity_bonus": "Доплата, если новое здание стоит рядом с полезным соседом.",
    "build_cost_penalty": "Небольшой вычет со стоимости постройки — чтобы ИИ не сорил деньгами.",
    # Добыча
    "first_extraction_bonus": "Разовая премия, когда ресурс добыт впервые за партию.",
    "extraction_daily": "Ежедневная премия, пока добыча реально работает (насыщается к 3-му зданию).",
    "need_fill_bonus": "Премия за здание, которое закрывает чужой голод (ферме не хватало воды — построил водоканал).",
    "loan_penalty": "Штраф за каждый взятый кредит — против «жить вечно в долгах».",
    # Экономика
    "daily_income": "Бонус за каждый день, когда бюджет колонии растёт.",
    "sale_bonus": "Бонус за продажу излишков на рынке (растёт с суммой сделки).",
    "tax_daily_bonus": "Маленький бонус за спокойный день без налоговых проблем.",
    "manual_tax_penalty": "Штраф за ручную оплату налога (норм. режим делает это сам).",
    "debt_coeff": "Ежедневный штраф, пока есть долг — чем больше долг, тем сильнее.",
    "tax_debt_penalty": "Штраф, когда неоплаченный налог переводится в долг банку.",
    "main_tax_cash_bonus": "Крупная премия, если главный налог (500 000) оплачен своими деньгами, без долга.",
    "main_tax_pressure_coeff": "Нарастающий штраф по мере приближения главного налога — учит копить заранее.",
    # Выживание
    "survival_bonus": "Премия за каждый просто прожитый день.",
    "survival_coeff": "Насколько рост капитала колонии добавляет к ежедневной награде.",
    "game_over_penalty": "Огромный штраф за полную гибель колонии.",
    "death_penalty": "Штраф за каждого погибшего жителя.",
    "tax_fail_penalty": "Штраф за просрочку годового налога (в диалоговом режиме).",
    "base_lost_penalty": "Штраф за здание, разрушившееся от износа.",
    "born_bonus": "Премия за каждого новорождённого жителя.",
    "home_overflow_penalty": "Штраф, когда людей больше, чем мест в домах.",
    "goal_survival_coeff": "Финальная премия за долгую жизнь — главный противовес «быстро умереть выгодно».",
    # Дороги к воде
    "road_shaping_cap": "Потолок бонуса за одну дорогу, приближающую колонию к воде.",
    "road_shaping_per_cell": "Сколько бонуса даёт каждая клетка приближения к воде.",
    "water_reach_bonus": "Разовая премия, когда дорога дошла до воды.",
    "water_reach_radius": "На каком расстоянии от воды дорога считается «дошедшей».",
    "road_no_progress_penalty": "Штраф за дорогу, которая к воде НЕ приблизилась.",
    "road_progress_epsilon": "Минимальное приближение, которое засчитывается как прогресс.",
    # Потребности
    "housing_need_bonus": "Премия, когда всем хватает жилья.",
    "food_need_bonus": "Премия, когда на складе достаточно еды.",
    "water_need_bonus": "Премия, когда на складе достаточно воды.",
    "buy_food_penalty": "Штраф за покупку еды (жители её не едят — деньги на ветер).",
    # Дисциплина
    "error_penalty": "Штраф за ошибочное действие (нет денег, нет места и т.п.). Отрицательное число.",
    "preserve_penalty": "Цена консервации здания (лёгкий вычет).",
    "demolish_penalty": "Штраф за снос здания. Отрицательное число.",
    "idle_build_penalty": "Штраф, когда рабочие простаивают N дней без дела. Отрицательное число.",
    "idle_build_threshold_days": "Сколько дней простоя считать «бездельем».",
    # Milestones и клип
    "milestone_base_bonus": "Премия за каждое новое здание-достижение.",
    "milestone_people_bonus": "Премия за рост населения.",
    "milestone_day_bonus": "Премия за прожитые дни.",
    "milestone_year_bonus": "Премия за каждый прожитый год.",
    "clip_reward_min": "Нижний предел разового штрафа (защита от выбросов).",
    "clip_reward_max": "Верхний предел разовой премии (защита от выбросов).",
}
