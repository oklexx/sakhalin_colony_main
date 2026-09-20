"""Единый источник дефолтов наград: канонический профиль configs/reward_v4.json.

Проверяет, что все «слои» согласованы (защита от «параметры не долетают»):
  1. профиль-файл существует и полный;
  2. load_default_reward_config() отдаёт его;
  3. дефолты dataclass rl.config.RewardConfig — зеркало профиля;
  4. RewardConfig.from_dict при отсутствующих ключах заполняет из профиля;
  5. C++-дефолты colony_cpp.RewardConfig — зеркало профиля;
  6. UI (REWARD_SPECS) содержит все ключи профиля — иначе _collect_config
     их выброшит и они «не долетят» до C++.

Без torch: слои 1–4; с torch + PySide6: +6; с собранным colony_cpp: +5.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False

V4_KEYS = ("first_extraction_bonus", "extraction_daily", "need_fill_bonus", "loan_penalty")
PROFILE = ROOT / "configs" / "reward_v4.json"


def _profile():
    d = json.loads(PROFILE.read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _config_module():
    """rl/config.py напрямую (rl/__init__.py тянет torch)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "colony_rl_config_test", str(ROOT / "rl" / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colony_rl_config_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_profile_file_exists_and_complete():
    assert PROFILE.exists(), "configs/reward_v4.json — канонический профиль по умолчанию"
    d = _profile()
    for k in V4_KEYS + ("error_penalty", "debt_coeff", "preserve_penalty",
                        "build_bonus", "game_over_penalty", "goal_survival_coeff",
                        "main_tax_cash_bonus", "main_tax_pressure_coeff"):
        assert k in d, f"{k} отсутствует в профиле"
    assert d["first_extraction_bonus"] == 3.0
    assert d["extraction_daily"] == 0.3
    assert d["need_fill_bonus"] == 1.5
    assert d["loan_penalty"] == 2.0
    assert d["error_penalty"] == -2.0
    assert d["debt_coeff"] == 0.0


def test_load_default_reward_config_returns_profile():
    mod = _config_module()
    rc = mod.load_default_reward_config()
    assert rc.first_extraction_bonus == 3.0
    assert rc.extraction_daily == 0.3
    assert rc.need_fill_bonus == 1.5
    assert rc.loan_penalty == 2.0
    assert rc.error_penalty == -2.0
    assert rc.debt_coeff == 0.0


def test_dataclass_defaults_mirror_profile():
    mod = _config_module()
    d = mod.RewardConfig().to_dict()
    prof = _profile()
    for k, v in prof.items():
        assert d.get(k) == v, f"dataclass-дефолт {k}={d.get(k)} != профиль {v}"


def test_from_dict_partial_fills_from_profile():
    mod = _config_module()
    rc = mod.RewardConfig.from_dict({"build_bonus": 99.0})
    assert rc.build_bonus == 99.0
    # ключи, которых в dict нет, — из профиля (не «голые» дефолты)
    assert rc.first_extraction_bonus == 3.0
    assert rc.loan_penalty == 2.0


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_cpp_defaults_mirror_profile():
    prof = _profile()
    rc = colony_cpp.RewardConfig()
    for k, v in prof.items():
        if k.startswith("disable_"):
            continue
        assert abs(getattr(rc, k) - v) < 1e-9, \
            f"C++-дефолт {k}={getattr(rc, k)} != профиль {v} (нужна пересборка pyd)"


def test_ui_reward_specs_cover_profile():
    """В REWARD_SPECS/REWARD_FLAGS — все ключи профиля; иначе UI их не покажет.

    Импорт из train_ui2 (актуальный UI): старый `train_ui` удалён, и тест молча
    скипался, хотя «поле не долетело до C++» — ровно тот класс багов, от которого
    он защищал. Boolean-ключи живут в REWARD_FLAGS (чекбоксы), а не в REWARD_SPECS.
    """
    try:
        from train_ui2.parameter_widget import REWARD_SPECS
        from train_ui2.constants import REWARD_FLAGS
    except Exception as e:  # нет torch в тестовой среде
        pytest.skip(f"UI недоступен: {e}")
    ui_keys = {s.key for s in REWARD_SPECS} | {k for k, _label, _tip in REWARD_FLAGS}
    for k in _profile():
        assert k in ui_keys, (
            f"{k} нет ни в REWARD_SPECS, ни в REWARD_FLAGS — поля нет в UI, "
            f"и его нельзя ни увидеть, ни подобрать из вкладки «Награды»")
    # новые поля профиля подписаны (непустая русская подпись)
    for s in REWARD_SPECS:
        if s.key in V4_KEYS:
            assert s.label and any("\u0400" <= ch <= "\u04ff" for ch in s.label), \
                f"у {s.key} нет русской подписи: {s.label!r}"
    # дефолт спецификации = значение профиля
    prof = _profile()
    for s in REWARD_SPECS:
        if s.key in prof:
            assert abs(float(s.default) - float(prof[s.key])) < 1e-9, \
                f"REWARD_SPECS[{s.key}].default={s.default} != профиль {prof[s.key]}"
