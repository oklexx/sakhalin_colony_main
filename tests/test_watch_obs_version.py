"""Наблюдение за моделью обязано идти в её собственной раскладке obs (P0 2026-09).

Регрессия, которую эти тесты закрывают: после смены дефолта на obs v2 (299)
UI-наблюдение передаёт `--obs-version` только если его задали руками — а значит
`watch_champion.py` обязан сам брать версию из meta модели. Иначе старая (289)
модель падает с «obs mismatch», хотя смотреть её надо ровно так, как обучали.

Файл не требует torch: `rl.curriculum` загружается напрямую (importlib), а
дефолт аргумента читается из AST `watch_champion.py`.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def _load_curriculum():
    """rl.curriculum без rl/__init__.py (тот импортирует torch-зависимые модули)."""
    name = "rl_curriculum_standalone"
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / "rl" / "curriculum.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclass() ищет модуль в sys.modules по __module__ — регистрируем до exec
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cur = _load_curriculum()


def _model_dir(tmp_path: Path, obs_version=None, name="meta.json") -> Path:
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    meta = {"curriculum_stage_at_best": 0, "unlock_ids": ""}
    if obs_version is not None:
        meta["obs_version"] = obs_version
    (d / name).write_text(json.dumps(meta), encoding="utf-8")
    return d


# ── разрешение версии ─────────────────────────────────────────────────────

def test_explicit_version_wins(tmp_path):
    d = _model_dir(tmp_path, obs_version=1)
    assert cur.resolve_obs_version(0, d) == 0
    assert cur.resolve_obs_version(2, d) == 2


def test_stored_version_is_used(tmp_path):
    assert cur.resolve_obs_version(None, _model_dir(tmp_path, obs_version=1)) == 1
    assert cur.resolve_obs_version(None, _model_dir(tmp_path, obs_version=0)) == 0
    assert cur.resolve_obs_version(None, _model_dir(tmp_path, obs_version=2)) == 2


def test_legacy_meta_without_key_reads_as_v0(tmp_path):
    # meta есть, ключа obs_version нет → это прогон до PR 5, т.е. v0
    assert cur.resolve_obs_version(None, _model_dir(tmp_path)) == 0


def test_missing_meta_falls_back_to_current_default(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cur.resolve_obs_version(None, empty) == cur.CURRENT_OBS_VERSION == 2


def test_meta_dict_form(tmp_path):
    assert cur.resolve_obs_version(None, None, {"obs_version": 1}) == 1
    assert cur.resolve_obs_version(None, None, {}) == cur.CURRENT_OBS_VERSION


def test_size_map_matches_watch_and_cpp():
    assert cur.obs_size_for_version(0) == 248
    assert cur.obs_size_for_version(1) == 289
    assert cur.obs_size_for_version(2) == 299


# ── дефолты не разъезжаются ───────────────────────────────────────────────

def test_current_obs_version_matches_config_default():
    text = (PROJECT_ROOT / "rl" / "config.py").read_text(encoding="utf-8")
    m = re.search(r"^\s*obs_version:\s*int\s*=\s*(\d+)", text, re.M)
    assert m, "в rl/config.py не найден дефолт obs_version"
    assert int(m.group(1)) == cur.CURRENT_OBS_VERSION


def _watch_arg_defaults(flag: str):
    """kwargs конкретного parser.add_argument('<flag>', ...) из watch_champion.py."""
    tree = ast.parse((PROJECT_ROOT / "watch_champion.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value == flag:
            return {kw.arg: kw.value for kw in node.keywords}
    raise AssertionError(f"{flag} не найден в watch_champion.py")


def test_watch_obs_version_default_is_none():
    kw = _watch_arg_defaults("--obs-version")
    assert "default" in kw
    assert isinstance(kw["default"], ast.Constant) and kw["default"].value is None, (
        "дефолт --obs-version обязан быть None: иначе UI-наблюдение старой модели "
        "снова упрётся в obs mismatch")


def test_watch_visual_tax_policy_flag_exists():
    # --tax-to-debt / --tax-dialog понимает gui.cpp (headless-ai по умолчанию —
    # долговая политика, как при обучении)
    text = (PROJECT_ROOT / "watch_champion.py").read_text(encoding="utf-8")
    assert "--tax-to-debt" in text and "tax_to_debt: bool = True" in text
