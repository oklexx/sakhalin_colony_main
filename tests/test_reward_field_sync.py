"""Синхронизация полей RewardConfig по всем слоям (золотое правило RULES.md).

Слой, который отстал, ломается молча: pybind11-биндинг без `def_readwrite`
означает, что Python не может передать значение в C++ (и C++ считает со своим
дефолтом), а поле без ключа в профиле — что A/B двух профилей на самом деле
сравнивает гибрид (см. P0-2 в docs/REMAINING_WORK_2026_09.md: configs/
reward_v3.json не содержит ключей v4, и они остаются дефолтами структуры).

Проверяются исходники текстом, поэтому тест не требует ни torch, ни собранного
colony_cpp — он ловит рассинхрон до сборки.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HEADER = ROOT / "include" / "colony" / "reward_config.h"
BINDINGS = ROOT / "src" / "bindings.cpp"
PROFILE = ROOT / "configs" / "reward_v4.json"
UI_SPECS = ROOT / "train_ui2" / "parameter_widget.py"
UI_GROUPS = ROOT / "train_ui2" / "constants.py"


def _cpp_fields() -> set[str]:
    text = HEADER.read_text(encoding="utf-8")
    body = text[text.index("struct RewardConfig {"):text.index("};", text.index("struct RewardConfig {"))]
    return set(re.findall(r"^\s*(?:double|bool|int)\s+(\w+)\s*=", body, re.M))


def _binding_fields() -> set[str]:
    text = BINDINGS.read_text(encoding="utf-8")
    block = text[text.index('py::class_<RewardConfig>'):]
    block = block[:block.index("py::class_<", 10)]
    return set(re.findall(r'def_readwrite\(\s*"(\w+)"', block))


def _py_fields() -> set[str]:
    spec = importlib.util.spec_from_file_location(
        "colony_rl_config_sync_test", str(ROOT / "rl" / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colony_rl_config_sync_test"] = mod
    spec.loader.exec_module(mod)
    return set(mod.RewardConfig().to_dict().keys())


def _profile_keys() -> set[str]:
    data = json.loads(PROFILE.read_text(encoding="utf-8"))
    return {k for k in data if not k.startswith("_")}


def test_cpp_header_matches_python_dataclass():
    cpp, py = _cpp_fields(), _py_fields()
    assert cpp == py, (
        f"C++ RewardConfig и rl.config.RewardConfig разошлись: "
        f"только в C++ {sorted(cpp - py)}, только в Python {sorted(py - cpp)}")


def test_bindings_expose_every_cpp_field():
    cpp, bound = _cpp_fields(), _binding_fields()
    missing = sorted(cpp - bound)
    assert not missing, (
        f"pybind11 не отдаёт в Python поля {missing} — значение из профиля "
        f"до C++ не долетит, среда посчитает со своим дефолтом")


def test_profile_is_a_subset_of_cpp_fields():
    cpp, prof = _cpp_fields(), _profile_keys()
    unknown = sorted(prof - cpp)
    assert not unknown, f"в configs/reward_v4.json неизвестные ключи: {unknown}"


def test_v4_only_fields_are_explicit_in_v3_profile():
    """configs/reward_v3.json — исторический профиль: ключей v4 в нём нет, и они
    остаются дефолтами структуры (= значения v4). A/B «v3 против v4» на таком
    файле сравнивает гибрид. Проверяем, что расхождение известно и ограничено
    терминальными членами v4, а не любыми полями."""
    v3 = json.loads((ROOT / "configs" / "reward_v3.json").read_text(encoding="utf-8"))
    cpp = _cpp_fields()
    unset = sorted(cpp - {k for k in v3 if not k.startswith("_")})
    # Флаги-абляции не входят ни в один профиль (дефолт false = старое поведение);
    # v4-терминалы в v3 теперь заданы ЯВНО нулями — иначе файл был бы гибридом.
    known_v4_only = {"priority_count_over_allowed", "obs_mask_locked_catalog"}
    # дорожный shaping (P2-9) в v3 не задаётся намеренно: дефолты равны
    # прежнему хардкоду, т.е. поведение v3 не меняется.
    road_shaping = {"road_shaping_cap", "road_shaping_per_cell", "water_reach_bonus",
                    "water_reach_radius", "road_no_progress_penalty",
                    "road_progress_epsilon"}
    unexpected = sorted(set(unset) - known_v4_only - road_shaping)
    assert not unexpected, (
        f"configs/reward_v3.json не задаёт {unexpected} — эти поля останутся "
        f"дефолтами структуры (v4), и профиль перестанет быть «чистым v3»")


def test_cpp_defaults_match_python_defaults():
    """Дефолты C++-структуры = дефолты dataclass (иначе консольная/GUI-сборка
    и RL-путь считают по-разному)."""
    header = HEADER.read_text(encoding="utf-8")
    body = header[header.index("struct RewardConfig {"):]
    cpp_defaults = {m.group(1): m.group(2) for m in
                    re.finditer(r"^\s*(?:double|bool|int)\s+(\w+)\s*=\s*([^;]+);", body, re.M)}
    spec = importlib.util.spec_from_file_location(
        "colony_rl_config_sync_test2", str(ROOT / "rl" / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colony_rl_config_sync_test2"] = mod
    spec.loader.exec_module(mod)
    py_defaults = mod.RewardConfig().to_dict()

    for key, raw in cpp_defaults.items():
        raw = raw.strip()
        py_val = py_defaults[key]
        if raw in ("true", "false"):
            assert isinstance(py_val, bool) and (py_val == (raw == "true")), \
                f"{key}: C++ {raw} != Python {py_val}"
        else:
            assert abs(float(raw) - float(py_val)) < 1e-12, \
                f"{key}: C++ {raw} != Python {py_val}"


def test_ui_exposes_cpp_fields():
    """Поле, которого нет во вкладке «Награды», нельзя ни увидеть, ни подобрать."""
    specs = UI_SPECS.read_text(encoding="utf-8")
    groups = UI_GROUPS.read_text(encoding="utf-8")
    spec_keys = set(re.findall(r'ParamSpec\(\s*"(\w+)"', specs))
    flag_keys = set(re.findall(r'\(\s*"(\w+)",\s*"[^"]+",\s*\n?\s*"', groups))
    cpp = _cpp_fields()
    # абляции-флаги живут в REWARD_FLAGS, остальные — в REWARD_SPECS
    missing = sorted(cpp - spec_keys - flag_keys)
    assert not missing, f"в UI (train_ui2) нет полей: {missing}"
