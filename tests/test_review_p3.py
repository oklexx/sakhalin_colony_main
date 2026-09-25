"""Этап 2 ревью 2026-09-24, пункты P3 (docs/CODE_REVIEW_2026_09_24.md).

P3-1  аннотация `Config.curriculum_state() -> CurriculumState` разрешается в
      рантайме (`typing.get_type_hints`), а не только для ruff/mypy.
P3-4  окно наблюдения потребляет невалидное действие и отвечает state.json с
      "error" (C++-логика — tests/cpp/watch_ipc_check.cpp; здесь — что gui.cpp
      и watch_champion.py действительно ею пользуются).
P3-5  NaN/Inf в эпизодной диагностике → null, а не ValueError из json.dumps.
P3-6  терминальная миникарта: в flat-режиме не запрашивается, при
      `terminal_minimap_missing` нулевая карта в info не подставляется.

CppVecEnv — со стабом `colony_cpp` только на время теста (RULES.md).
"""
from __future__ import annotations

import importlib
import json
import math
import sys
import types
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


# ── P3-1 ────────────────────────────────────────────────────────────────────

def test_config_curriculum_state_hint_resolves_at_runtime() -> None:
    from rl.config import Config
    from rl.curriculum import CurriculumState

    hints = typing.get_type_hints(Config.curriculum_state)
    assert hints["return"] is CurriculumState


# ── P3-5 ────────────────────────────────────────────────────────────────────

def test_episode_diagnostics_nan_becomes_null(tmp_path: Path) -> None:
    from rl.episode_diagnostics import EpisodeDiagnosticsWriter

    path = tmp_path / "episodes.jsonl"
    writer = EpisodeDiagnosticsWriter(path, fsync=False,
                                      metadata={"lr": float("inf")})
    info = {"episode": {"r": float("nan"), "l": 3,
                        "metrics": {"net_worth": float("-inf"),
                                    "values": [1.0, np.float32("nan")]}}}
    assert writer.append_episode(info, total_timesteps=3, env_index=0,
                                 curriculum_stage=0)
    writer.close(total_timesteps=3, status="completed")

    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text  # строгий JSON
    start, row, _end = [json.loads(line) for line in text.splitlines()]
    assert start["metadata"]["lr"] is None
    assert row["episode_return"] is None
    assert row["episode_metrics"]["net_worth"] is None
    assert row["episode_metrics"]["values"] == [1.0, None]


def test_json_safe_keeps_finite_numbers() -> None:
    from rl.episode_diagnostics import _json_safe

    assert _json_safe(1.5) == 1.5
    assert _json_safe(np.float64(2.25)) == 2.25
    assert _json_safe(7) == 7 and _json_safe(True) is True
    assert _json_safe(math.nan) is None


# ── P3-4 ────────────────────────────────────────────────────────────────────

def test_gui_uses_shared_action_reader_and_answers_invalid() -> None:
    src = (REPO / "src" / "gui.cpp").read_text(encoding="utf-8")
    assert '#include "colony/watch_ipc.h"' in src
    assert "read_action_file(" in src
    assert "ActionRead::INVALID" in src
    # ответ на невалидное действие — state.json с полем error, без шага
    assert '\\"error\\"' in src
    cpp_checks = (REPO / "scripts" / "cpp_checks.sh").read_text(encoding="utf-8")
    assert "watch_ipc_check" in cpp_checks.split("CHECKS=", 1)[1].splitlines()[0]


def test_watch_champion_logs_window_error() -> None:
    src = (REPO / "watch_champion.py").read_text(encoding="utf-8")
    assert 'state.get("error")' in src


# ── P3-6: CppVecEnv со стабом расширения ────────────────────────────────────

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


class _FakeCpp:
    """step_wait_batch: все среды done; terminal_minimap_batch считает вызовы."""

    def __init__(self, infos: list[str], obs_size: int = 4) -> None:
        self._infos = infos
        self._obs_size = obs_size
        self.tmm_calls = 0
        self.enabled: list[bool] = []

    def step_wait_batch(self) -> Any:
        n = len(self._infos)
        return types.SimpleNamespace(
            obs=np.zeros((n, self._obs_size), dtype=np.float32),
            rewards=[0.0] * n, terminateds=[True] * n, trunceds=[False] * n,
            infos=self._infos)

    def action_masks_batch(self) -> Any:
        return np.ones((len(self._infos), 5), dtype=np.float32)

    def terminal_minimap_batch(self) -> Any:
        self.tmm_calls += 1
        return np.ones((len(self._infos), 8, 32, 32), dtype=np.float32)

    def set_terminal_minimap_enabled(self, on: bool) -> None:
        self.enabled.append(on)


def _env(mod: Any, fake: _FakeCpp, obs_mode: str | None) -> Any:
    env = mod.CppVecEnv.__new__(mod.CppVecEnv)
    env.num_envs = len(fake._infos)
    env.cpp_vec = fake
    if obs_mode is not None:
        env.obs_mode = obs_mode
    env._configure_terminal_minimap()
    return env


def _term(**extra: Any) -> str:
    return json.dumps({"terminal_observation": [1.0] * 4, **extra})


def test_flat_mode_disables_terminal_minimap(cpp_vecenv_module) -> None:
    fake = _FakeCpp([_term(), _term()])
    env = _env(cpp_vecenv_module, fake, obs_mode=None)  # базовый CppVecEnv = flat
    assert fake.enabled == [False]
    _obs, _r, dones, infos = env.step_wait()
    assert dones.all()
    assert fake.tmm_calls == 0
    assert all("terminal_minimap" not in info for info in infos)


@pytest.mark.parametrize("mode", ["minimap", "hybrid"])
def test_minimap_modes_skip_missing_terminal_minimap(cpp_vecenv_module, mode: str) -> None:
    fake = _FakeCpp([_term(terminal_minimap_missing=True), _term()])
    env = _env(cpp_vecenv_module, fake, obs_mode=mode)
    assert fake.enabled == [True]
    _obs, _r, _dones, infos = env.step_wait()
    assert fake.tmm_calls == 1
    # нулевая/чужая карта не подставляется → EnvManager пропустит бутстрап
    assert "terminal_minimap" not in infos[0]
    assert infos[0]["terminal_minimap_missing"] is True
    assert infos[1]["terminal_minimap"].shape == (8, 32, 32)


def test_old_binary_without_toggle_still_works(cpp_vecenv_module) -> None:
    fake = _FakeCpp([_term()])
    del_attr = types.SimpleNamespace(**{k: getattr(fake, k) for k in
                                        ("step_wait_batch", "action_masks_batch",
                                         "terminal_minimap_batch")})
    env = cpp_vecenv_module.CppVecEnv.__new__(cpp_vecenv_module.CppVecEnv)
    env.num_envs = 1
    env.cpp_vec = del_attr
    env.obs_mode = "hybrid"
    env._configure_terminal_minimap()  # нет set_terminal_minimap_enabled — не падает
    _obs, _r, _dones, infos = env.step_wait()
    assert "terminal_minimap" in infos[0]


def test_minimap_subclass_reconfigures_after_obs_mode() -> None:
    src = (REPO / "python" / "cpp_vecenv_minimap.py").read_text(encoding="utf-8")
    i_mode = src.index("self.obs_mode = obs_mode")
    i_conf = src.index("self._configure_terminal_minimap()")
    assert i_mode < i_conf


# ── P3-3/P3-8: детектор циклов с deque-историей ─────────────────────────────

def test_loop_detector_update_does_not_slice_deque() -> None:
    """`_action_history` — deque; срез deque[-n:] бросал TypeError, и
    `loop_detection_enabled=True` ронял обучение на первом роллауте."""
    from collections import deque

    from rl.async_trainer import AsyncTrainer
    from rl.loop_detector import LoopDetector

    trainer = AsyncTrainer.__new__(AsyncTrainer)
    trainer.loop_detector = LoopDetector({"consecutive_threshold": 3})
    trainer._action_history = deque(maxlen=1000)
    for _ in range(5):
        trainer._action_history.append((0, "day"))
    trainer._action_history.append((1, "week"))

    n_loops, name = trainer._update_loop_detector(window=64, total_done=0)
    assert n_loops == 1 and name == "day"

    trainer.loop_detector = None  # выключен (по умолчанию) — ноль без работы
    assert trainer._update_loop_detector(window=64, total_done=0) == (0, None)


def test_async_trainer_dead_code_removed() -> None:
    src = (REPO / "rl" / "async_trainer.py").read_text(encoding="utf-8")
    assert "_get_steps_in_curriculum_stage" not in src
    assert "rollout_time" not in src
    assert "self._action_history[-" not in src
