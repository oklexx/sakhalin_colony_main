"""Двухэтапное обучение: пресет «Стадия 1» и полный гейтинг механик (2026-09-21).

Три контракта (docs/TWO_STAGE_TRAINING_2026_09.md):
1. Таблицы механик синхронны в трёх местах: rl/curriculum.py MECHANIC_NAMES
   == train_ui2/constants.py MECHANIC_IDS == C++ MECHANIC_NAMES
   (include/colony/constants.h). Порядок — часть контракта транспорта.
2. STAGE1_PRESET валиден и применяется к Config без остатка: build_state
   даёт ровно 8 зданий, allow-list механик = {sell, credit}.
3. Дефолт НОВОГО прогона — минимальный режим; старый конфиг без полей
   гейтинга сохраняет легаси-поведение (все механики включены).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.config import Config  # noqa: E402
from rl.curriculum import (  # noqa: E402
    ALL_IDS,
    MECHANIC_NAMES,
    RESOURCE_NAMES,
    STAGE1_PRESET,
    STAGE1_PRESET_NAME,
    CurriculumState,
    apply_stage1_preset,
    mechanics_enabled_at_step,
    normalize_enabled_mechanics,
    parse_mechanic_ids,
    parse_resources,
    parse_unlock_ids,
)
from train_ui2.constants import (  # noqa: E402
    HUMAN_REWARD_HELP,
    MECHANIC_CAPTIONS,
    MECHANIC_IDS,
    PRESET_ORDER,
    PRESETS,
    REWARD_FLAGS,
    REWARD_GROUPS,
)

# ── 1. Синхронность таблиц механик (Python UI / Python RL / C++) ─────────────

def _cpp_mechanic_names() -> list[str]:
    """MECHANIC_NAMES из include/colony/constants.h (регресс против рассинхрона)."""
    src = (ROOT / "include" / "colony" / "constants.h").read_text(encoding="utf-8")
    m = re.search(
        r"MECHANIC_NAMES\[N_MECHANICS\]\s*=\s*\{(.*?)\}", src, re.DOTALL)
    assert m, "MECHANIC_NAMES не найден в include/colony/constants.h"
    return re.findall(r'"([a-z_]+)"', m.group(1))


def _cpp_manager_mechanic() -> list[int]:
    """MANAGER_MECHANIC из include/colony/constants.h."""
    src = (ROOT / "include" / "colony" / "constants.h").read_text(encoding="utf-8")
    m = re.search(r"MANAGER_MECHANIC\[N_MANAGERS\]\s*=\s*\{(.*?)\}", src, re.DOTALL)
    assert m, "MANAGER_MECHANIC не найден в include/colony/constants.h"
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def test_mechanic_tables_in_sync():
    cpp = _cpp_mechanic_names()
    assert MECHANIC_NAMES == tuple(cpp), (
        f"рассинхрон Python/C++ механик: {MECHANIC_NAMES} != {tuple(cpp)}")
    assert list(MECHANIC_IDS) == list(MECHANIC_NAMES), (
        "train_ui2 MECHANIC_IDS != rl MECHANIC_NAMES")
    # Легаси-имена остаются на своих местах (старые мета-файлы).
    assert MECHANIC_NAMES[:3] == ("improve_land", "repair", "destroy")
    assert "preservation" in MECHANIC_NAMES and "credit" in MECHANIC_NAMES


def test_manager_mechanic_table_covers_all_slots():
    mm = _cpp_manager_mechanic()
    assert len(mm) == 11, f"MANAGER_MECHANIC должен покрывать 11 слотов, {len(mm)}"
    assert all(0 <= x < len(MECHANIC_NAMES) for x in mm)
    # Плотность покрытия: каждая механика используется хотя бы одним слотом.
    assert set(mm) == set(range(len(MECHANIC_NAMES))), (
        "не каждая механика привязана к слоту менеджера")


# ── 2. Пресет «Стадия 1» ─────────────────────────────────────────────────────

def test_stage1_preset_fields_are_valid():
    buildings = parse_unlock_ids(STAGE1_PRESET["unlock_ids"])  # неизвестные — raise
    assert len(buildings) == 8
    assert set(buildings) <= set(ALL_IDS)
    # Экономика пресета: без зданий с недостижимыми входами.
    for must in ("Road", "WaterChannel", "Mushroom", "Garden", "Farm"):
        assert must in buildings, f"{must} обязан быть в пресете"
    weights = parse_resources(STAGE1_PRESET["curriculum_resources"])
    assert weights is not None and len(weights) == len(RESOURCE_NAMES)
    disabled = STAGE1_PRESET["disabled_mechanics"]
    assert set(disabled) <= set(MECHANIC_NAMES)
    # allow-list = MECHANIC_NAMES − disabled ровно {sell, credit}
    enabled = tuple(m for m in MECHANIC_NAMES if m not in set(disabled))
    assert enabled == ("sell", "credit")


def test_apply_stage1_preset_on_config():
    cfg = Config()
    apply_stage1_preset(cfg)
    state = cfg.curriculum_state()
    assert state.all_builds is False
    assert set(state.allowed_builds) == set(parse_unlock_ids(STAGE1_PRESET["unlock_ids"]))
    assert state.enabled_mechanics == ("sell", "credit")
    assert state.all_resources is False
    # веса: вода/еда/дерево = 1, остальное 0
    w = dict(zip(RESOURCE_NAMES, state.resource_weights, strict=True))
    assert w["water"] == 1.0 and w["food"] == 1.0 and w["wood"] == 1.0
    assert w["gold"] == 0.0 and w["coal"] == 0.0
    # критерии eval под задачу этапа
    assert cfg.eval_min_days == 365.0
    assert cfg.eval_min_bases == 3
    #round-trip через транспортную форму
    rt = CurriculumState.from_dict(state.to_dict())
    assert rt.enabled_mechanics == ("sell", "credit")
    assert set(rt.allowed_builds) == set(state.allowed_builds)


def test_stage1_preset_name():
    assert STAGE1_PRESET_NAME == "stage1"
    assert PRESETS["stage1"]["buildings"] == parse_unlock_ids(
        STAGE1_PRESET["unlock_ids"]), "UI-пресет stage1 разошёлся с rl STAGE1_PRESET"
    assert set(PRESETS["stage1"]["mechanics"]) == {"sell", "credit"}
    assert PRESETS["stage1"]["learning_rate"] == 3e-4
    assert PRESETS["stage1"]["eval_min_days"] == 365.0
    assert PRESETS["stage1"]["eval_min_bases"] == 3
    assert PRESETS["full"]["learning_rate"] == 3e-5
    assert PRESETS["full"]["eval_min_days"] == 730.0
    assert PRESETS["full"]["eval_min_bases"] == 5
    assert PRESET_ORDER == ["stage1", "full"]


# ── 3. Дефолты Config: новый прогон минимальный, старый конфиг — легаси ─────

def test_default_config_is_stage1_mechanics():
    cfg = Config()
    assert cfg.enabled_mechanics_at(0) == ("sell", "credit")
    # расписание по умолчанию пустое: этап 2 — отдельный прогон
    assert cfg.mechanics_unlock_schedule == []


def test_old_config_without_fields_keeps_legacy_all_enabled():
    cfg = Config.from_dict({"curriculum_stage": 0})  # ни disabled, ни schedule
    assert cfg.disabled_mechanics == []
    assert cfg.mechanics_unlock_schedule == []
    assert cfg.enabled_mechanics_at(0) == MECHANIC_NAMES


def test_full_preset_config_enables_everything():
    cfg = Config.from_dict({"disabled_mechanics": []})
    assert cfg.enabled_mechanics_at(0) == MECHANIC_NAMES


# ── 4. Парсер механик: новые имена, псевдонимы, ошибки ──────────────────────

def test_parse_mechanic_ids_new_names_and_aliases():
    assert parse_mechanic_ids("sell,credit") == ["sell", "credit"]
    assert parse_mechanic_ids(["repair", "restore_all"]) == ["repair"]
    assert parse_mechanic_ids("all") == list(MECHANIC_NAMES)
    assert parse_mechanic_ids("none") == []
    with pytest.raises(ValueError):
        parse_mechanic_ids("teleport")
    assert normalize_enabled_mechanics(None) == MECHANIC_NAMES
    assert normalize_enabled_mechanics("sell , credit") == ("sell", "credit")


def test_mechanics_schedule_with_full_set():
    # ничего не выключено → все включены на любом шаге
    assert mechanics_enabled_at_step(0, [], [[100, ["repair"]]]) == tuple(MECHANIC_NAMES)
    # выключены repair и sell → включены остальные шесть
    enabled6 = mechanics_enabled_at_step(100, ["repair", "sell"], [])
    assert enabled6 == tuple(m for m in MECHANIC_NAMES if m not in ("repair", "sell"))
    # расписание аддитивно: base = только sell, repair открывается на шаге 100
    rest = [m for m in MECHANIC_NAMES if m != "sell"]
    assert mechanics_enabled_at_step(0, rest, [[100, ["repair"]]]) == ("sell",)
    assert mechanics_enabled_at_step(100, rest, [[100, ["repair"]]]) == ("repair", "sell")


# ── 5. UI-словари понятности (глоссарий наград и подписи механик) ───────────

def test_human_reward_help_covers_all_reward_keys():
    keys = {k for group in REWARD_GROUPS.values() for k in group}
    missing = keys - set(HUMAN_REWARD_HELP)
    assert not missing, f"награды без человеческого описания: {sorted(missing)}"
    # Каждая подпись — не жаргон: не совпадает с ключом и не пустая.
    for key, text in HUMAN_REWARD_HELP.items():
        assert text.strip() and text != key


def test_mechanic_captions_complete():
    assert set(MECHANIC_CAPTIONS) == set(MECHANIC_NAMES)
    for cap, tip in MECHANIC_CAPTIONS.values():
        assert cap.strip() and tip.strip()


def test_reward_flags_have_plain_labels():
    for _key, label, tip in REWARD_FLAGS:
        assert "выкл." not in label.lower(), f"программистская подпись: {label}"
        assert label.strip() and tip.strip()


# ── 6. Полный контракт через C++ (если собран extension) ────────────────────

def test_stage1_state_accepted_by_cpp():
    colony_cpp = pytest.importorskip("colony_cpp", reason="нет собранного colony_cpp")
    cfg = Config()
    apply_stage1_preset(cfg)
    state = cfg.curriculum_state()
    env = colony_cpp.ColonyEnvCpp(
        colony_cpp.load_base_data(str(ROOT / "configs" / "bases.json")),
        colony_cpp.load_events(str(ROOT / "configs" / "events.json")),
        42, 280, state.to_dict(),
    )
    back = env.curriculum()
    # curriculum_to_dict отдаёт механики в порядке MECHANIC_NAMES
    assert list(back["enabled_mechanics"]) == ["sell", "credit"]
    assert not back["all_builds"]
    assert set(back["allowed_builds"]) == set(state.allowed_builds)
