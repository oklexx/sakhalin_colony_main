"""Конфиги экспериментов (configs/exp_*.json): схема, VRAM-бюджет, batch_size.

Закрывает N12 / P0-3 (docs/REMAINING_WORK_2026_09.md): exp_25/26/27 с
n_envs=1024 требуют 56+ ГБ VRAM и не запускаются на типовом GPU, поэтому рядом
лежат уменьшенные варианты *_n64 (12 ГБ) и *_n128 (24 ГБ). Этот тест — защита от
того, что конфиг снова «уедет» за бюджет или получит ключ с опечаткой
(`_update_dataclass_from_dict` молча игнорирует неизвестные ключи).

torch и colony_cpp не нужны: проверяются только JSON и dataclass rl.config.Config.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"

# Размеры наблюдения в float32 (см. ColonyEnvCpp::obs_size и миникарту 8×32×32).
FLAT_OBS = 299
MINIMAP_OBS = 8 * 32 * 32
OBS_FLOATS = {"flat": FLAT_OBS, "minimap": MINIMAP_OBS, "hybrid": FLAT_OBS + MINIMAP_OBS}

# Служебные ключи конфига (не поля Config).
# `name` — человеческая метка прогона: в rl.config.Config такого поля нет,
# поэтому Config.load_from_file его молча игнорирует (имя прогона даёт
# `train.py --name`). Держим осознанно, как подпись файла.
META_KEYS = {"_comment", "_vram_budget_gb", "_buffer_gb_estimate", "name"}

REQUIRED = ("name", "n_envs", "n_steps", "batch_size", "obs_mode", "obs_version",
            "total_timesteps", "learning_rate", "seed")


def _config_dataclass():
    """rl/config.py напрямую (rl/__init__.py тянет torch)."""
    spec = importlib.util.spec_from_file_location(
        "colony_rl_config_exp_test", str(ROOT / "rl" / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colony_rl_config_exp_test"] = mod
    spec.loader.exec_module(mod)
    return mod.Config


def _exp_configs():
    return sorted(CONFIGS.glob("exp_*.json"))


def buffer_gb(cfg: dict) -> float:
    """Rollout-буфер obs в ГБ: n_envs × n_steps × obs_floats × 4B (rl/rollout_buffer.py)."""
    obs = OBS_FLOATS[cfg["obs_mode"]]
    return cfg["n_envs"] * cfg["n_steps"] * obs * 4 / 1e9


def test_exp_configs_exist():
    names = {p.name for p in _exp_configs()}
    # исходные (большие) три + уменьшенные варианты под 12/24 ГБ
    for expected in ("exp_25_minimap.json", "exp_26_hybrid.json",
                     "exp_27_minimap_lr_low.json", "exp_26_hybrid_n64.json",
                     "exp_26_hybrid_n128.json", "exp_25_minimap_n64.json",
                     "exp_27_minimap_lr_low_n64.json"):
        assert expected in names, f"нет {expected} в configs/"


@pytest.mark.parametrize("path", _exp_configs(), ids=lambda p: p.name)
def test_schema_and_keys(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in REQUIRED:
        assert key in cfg, f"{path.name}: отсутствует обязательный ключ {key}"
    assert cfg["obs_mode"] in OBS_FLOATS, f"{path.name}: obs_mode={cfg['obs_mode']}"
    assert cfg["obs_version"] in (0, 1, 2), f"{path.name}: obs_version={cfg['obs_version']}"
    assert cfg["n_envs"] > 0 and cfg["n_steps"] > 0

    # Неизвестные ключи = опечатка: Config.from_file их МОЛЧА игнорирует.
    allowed = {f.name for f in dc_fields(_config_dataclass())} | META_KEYS
    unknown = set(cfg) - allowed
    assert not unknown, f"{path.name}: неизвестные ключи {sorted(unknown)} (опечатка?)"


@pytest.mark.parametrize("path", _exp_configs(), ids=lambda p: p.name)
def test_batch_divides_buffer(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    total = cfg["n_envs"] * cfg["n_steps"]
    assert cfg["batch_size"] <= total, f"{path.name}: batch_size > n_envs*n_steps"
    assert total % cfg["batch_size"] == 0, \
        f"{path.name}: буфер {total} не делится на batch_size {cfg['batch_size']}"


def test_original_configs_really_need_more_than_24gb():
    """N12 подтверждён расчётом: исходные exp_25/26/27 в 24 ГБ не влезают."""
    for name in ("exp_25_minimap.json", "exp_26_hybrid.json",
                 "exp_27_minimap_lr_low.json"):
        cfg = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
        assert cfg["n_envs"] == 1024
        assert buffer_gb(cfg) > 24.0, \
            f"{name}: ожидалось >24 ГБ буфера (N12), получилось {buffer_gb(cfg):.1f}"


def test_reduced_configs_fit_their_budget():
    """Уменьшенные варианты: буфер ≤ 60% заявленного бюджета (остальное — модель,
    оптимизатор, активации CNN, фрагментация аллокатора)."""
    reduced = [p for p in _exp_configs() if "_vram_budget_gb" in
               json.loads(p.read_text(encoding="utf-8"))]
    assert len(reduced) >= 4, "нужны уменьшенные конфиги под 12 и 24 ГБ"
    budgets = set()
    for path in reduced:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        budget = float(cfg["_vram_budget_gb"])
        gb = buffer_gb(cfg)
        budgets.add(budget)
        assert gb <= 0.6 * budget, (
            f"{path.name}: буфер {gb:.1f} ГБ > 60% бюджета {budget:.0f} ГБ")
        # заявленная в комментарии оценка должна совпадать с расчётом
        if "_buffer_gb_estimate" in cfg:
            assert abs(float(cfg["_buffer_gb_estimate"]) - gb) < 0.05, \
                f"{path.name}: _buffer_gb_estimate={cfg['_buffer_gb_estimate']} != {gb:.2f}"
    assert 12 in budgets and 24 in budgets, "нужны варианты и под 12 ГБ, и под 24 ГБ"


def test_reduced_configs_keep_experiment_semantics():
    """Уменьшенный вариант отличается от исходного ТОЛЬКО масштабом батча."""
    pairs = [("exp_25_minimap.json", "exp_25_minimap_n64.json"),
             ("exp_26_hybrid.json", "exp_26_hybrid_n64.json"),
             ("exp_26_hybrid.json", "exp_26_hybrid_n128.json"),
             ("exp_27_minimap_lr_low.json", "exp_27_minimap_lr_low_n64.json")]
    for big_name, small_name in pairs:
        big = json.loads((CONFIGS / big_name).read_text(encoding="utf-8"))
        small = json.loads((CONFIGS / small_name).read_text(encoding="utf-8"))
        for key in ("obs_mode", "obs_version", "learning_rate", "gamma", "gae_lambda",
                    "n_epochs", "ent_coef", "vf_coef", "net_arch", "map_size",
                    "curriculum_stage", "n_steps"):
            assert big[key] == small[key], \
                f"{small_name}: {key}={small[key]} != {big[key]} из {big_name}"
        assert small["n_envs"] < big["n_envs"], f"{small_name}: n_envs не уменьшен"
