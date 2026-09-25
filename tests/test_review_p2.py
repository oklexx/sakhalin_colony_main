"""Этап 1 ревью 2026-09-24, пункты P2.

P2-1  таблица расписания курикулума: строка с опечаткой не исчезает молча —
      `parse_schedule_rows` возвращает причину, UI показывает её; голых
      `except:` в main_window больше нет.
P2-2  `CppVecEnv.get_attr/set_attr/env_method/env_is_wrapped` — по контракту
      SB3 VecEnv (длина и порядок по `indices`, метод батча вызывается один раз).
P2-3  `terminal_observation` не подменяется obs НОВОГО эпизода после авто-reset.

Тесты CppVecEnv не требуют собранного `colony_cpp`: модуль импортируется со
стабом расширения, объект создаётся без C++-конструктора. Стаб ставится только
на время теста и убирается (RULES.md, «Стабы в sys.modules»).
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from train_ui2.curriculum_table import parse_schedule_rows

REPO = Path(__file__).resolve().parent.parent


# ── P2-1 ────────────────────────────────────────────────────────────────────

def test_valid_rows_sorted_and_underscores_allowed():
    res = parse_schedule_rows([["400000", "2"], ["200_000", "1"], [" 0 ", "0"]], max_stage=3)
    assert res.rows == [[0, 0], [200000, 1], [400000, 2]]
    assert res.warnings == []


def test_bad_rows_reported_with_user_row_numbers():
    res = parse_schedule_rows(
        [["200000", "1"], ["200к", "2"], [None, "1"], ["300000", "7"], ["-5", "1"],
         ["500000", ""]],
        max_stage=3)
    assert res.rows == [[200000, 1]]
    joined = " | ".join(res.warnings)
    assert "строка 2" in joined and "'200к'" in joined
    assert "строка 3: пустое поле «шаг»" in joined
    assert "строка 4: этап 7 вне 0–3" in joined
    assert "строка 5: шаг -5 < 0" in joined
    assert "строка 6: пустое поле «этап»" in joined
    assert len(res.warnings) == 5


def test_duplicate_step_last_wins_and_is_reported():
    res = parse_schedule_rows([["100", "1"], ["100", "2"]], max_stage=3)
    assert res.rows == [[100, 2]]
    assert res.warnings and "повторяется" in res.warnings[0]


def test_max_stage_comes_from_curriculum_single_source():
    from rl.curriculum import STAGE_MAP
    from train_ui2.constants import CURRICULUM_STAGE_MAP
    assert max(CURRICULUM_STAGE_MAP) == max(STAGE_MAP)


def test_main_window_has_no_bare_except():
    tree = ast.parse((REPO / "train_ui2" / "main_window.py").read_text(encoding="utf-8"))
    bare = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.type is None]
    assert not bare, f"голый except в main_window.py: строки {bare}"


def test_curriculum_table_uses_parser_everywhere():
    """Все чтения таблицы — через _parse_curriculum_table (нет int(item.text()))."""
    src = (REPO / "train_ui2" / "main_window.py").read_text(encoding="utf-8")
    assert "int(self.tbl_curriculum.item(" not in src


# ── P2-2 / P2-3: CppVecEnv со стабом расширения ─────────────────────────────

@pytest.fixture
def cpp_vecenv_module(monkeypatch) -> Iterator[Any]:
    pytest.importorskip("stable_baselines3")
    stub = types.ModuleType("colony_cpp")
    monkeypatch.setitem(sys.modules, "colony_cpp", stub)
    saved = sys.modules.pop("cpp_vecenv", None)
    try:
        yield importlib.import_module("cpp_vecenv")
    finally:
        sys.modules.pop("cpp_vecenv", None)
        if saved is not None:
            sys.modules["cpp_vecenv"] = saved


def _bare_env(mod: Any, n: int = 3) -> Any:
    env = mod.CppVecEnv.__new__(mod.CppVecEnv)
    env.num_envs = n
    env.marker = "m"
    return env


def test_get_attr_follows_indices(cpp_vecenv_module):
    env = _bare_env(cpp_vecenv_module, n=3)
    assert env.get_attr("marker") == ["m", "m", "m"]
    assert env.get_attr("marker", indices=[2, 0]) == ["m", "m"]
    assert env.get_attr("marker", indices=1) == ["m"]
    assert env.env_is_wrapped(object, indices=[0, 1]) == [False, False]
    with pytest.raises(IndexError):
        env.get_attr("marker", indices=[3])


def test_set_attr_partial_is_explicit_error(cpp_vecenv_module):
    env = _bare_env(cpp_vecenv_module, n=2)
    env.set_attr("marker", "x")
    assert env.marker == "x"
    with pytest.raises(NotImplementedError):
        env.set_attr("marker", "y", indices=[0])
    assert env.marker == "x"


def test_env_method_calls_batch_method_once(cpp_vecenv_module):
    env = _bare_env(cpp_vecenv_module, n=4)
    calls: list[int] = []
    env.bump = lambda k: calls.append(k) or len(calls)
    assert env.env_method("bump", 5) == [1, 1, 1, 1]
    assert calls == [5]
    with pytest.raises(NotImplementedError):
        env.env_method("bump", 5, indices=[1])


class _FakeCpp:
    def __init__(self, infos: list[str], obs_size: int = 4) -> None:
        self._infos = infos
        self._obs_size = obs_size

    def step_wait_batch(self) -> Any:
        n = len(self._infos)
        return types.SimpleNamespace(
            obs=np.full((n, self._obs_size), 3.0, dtype=np.float32),  # obs НОВОГО эпизода
            rewards=[0.0] * n, terminateds=[False] * n, trunceds=[True] * n,
            infos=self._infos)

    def action_masks_batch(self) -> Any:
        return np.ones((len(self._infos), 5), dtype=np.float32)


def test_missing_terminal_observation_is_not_faked(cpp_vecenv_module, capsys):
    env = _bare_env(cpp_vecenv_module, n=2)
    env.cpp_vec = _FakeCpp(["{}", json.dumps({"terminal_observation": [7.0] * 4})])
    for _ in range(2):  # предупреждение — один раз на процесс
        _obs, _r, dones, infos = env.step_wait()
    assert dones.all()
    assert "terminal_observation" not in infos[0]
    assert infos[0]["terminal_observation_missing"] is True
    assert infos[0]["TimeLimit.truncated"] is True
    # присланное C++ значение сохраняется как есть
    assert infos[1]["terminal_observation"][0] == 7.0
    assert "terminal_observation_missing" not in infos[1]
    out = capsys.readouterr().out
    assert out.count("без terminal_observation") == 1


def test_env_manager_skips_bootstrap_without_terminal_obs():
    """Без ключа бутстрап V(s_T) пропускается (0), а не считается по чужому obs."""
    import torch

    from rl.env_manager import EnvManager

    em = EnvManager.__new__(EnvManager)
    em.n_envs = 2
    em.device = torch.device("cpu")
    em.cfg = types.SimpleNamespace(obs_mode="flat")

    class _M:
        def get_value(self, x, action_masks=None):
            return torch.full((x.shape[0],), 5.0)

    em.model = _M()
    infos = [{"terminal_observation_missing": True},
             {"terminal_observation": np.zeros(4, dtype=np.float32)}]
    out = em._truncation_bootstrap_values(infos, np.array([True, True]), torch.ones(2, 5))
    assert out.tolist() == [0.0, 5.0]
