"""Визуальное наблюдение (галочка «GUI-окно»): raylib-exe + файловый IPC.

Регрессия, которую закрывает файл (2026-09-20): «не запускается GUI при
наблюдении за обученной моделью» — драйвер не проверял ни старт exe, ни
появление state.json, и в любой непонятной ситуации молча спал в
`time.sleep(0.05)` (или вечно рестартовал exe). Теперь у каждого ожидания есть
таймаут, heartbeat в лог UI и диагноз (`run_visual_watch`,
`describe_gui_failure`).

Тесты гоняют НАСТОЯЩИЙ `watch_champion.main()` в процессе pytest; вместо
raylib-exe — фейковое «окно» (тот же протокол: actions.txt → state.json), вместо
`colony_cpp` — стаб, если расширение не собрано. Поэтому прогон работает и в
песочнице без C++-бинаря, и в CI с настоящим.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

torch = pytest.importorskip("torch")

OBS_SIZE = 299          # obs v2 — канон (rl/curriculum.CURRENT_OBS_VERSION)
N_ACTIONS = 49
MM_LEN = 8 * 32 * 32    # ColonyEnvCpp::minimap() — глобальная сетка 8×32×32
BUILD_IDS = [f"B{i}" for i in range(32)]


# ── стаб colony_cpp (нужен, только если расширение не собрано) ────────────────

def _fake_colony_cpp() -> types.ModuleType:
    mod = types.ModuleType("colony_cpp")

    class RunningMeanStd:
        def __init__(self, n: int):
            self._m = [0.0] * n
            self._v = [1.0] * n
            self._c = 0

        def update(self, x, count, n):
            pass

        def normalize(self, x, n, obs_size, clip):
            pass  # in-place в настоящем RMS; здесь obs уже «нормализован»

        def mean(self):
            return self._m

        def var(self):
            return self._v

        def count(self):
            return self._c

        def set_mean(self, m):
            self._m = list(m)

        def set_var(self, v):
            self._v = list(v)

        def set_count(self, c):
            self._c = int(c)

    class RewardConfig:
        def __init__(self):
            self.disable_net_worth = False
            self.disable_daily_income = False

    class ColonyEnvCpp:
        def __init__(self, bd=None, ed=None, seed=0, map_size=280, curriculum=None,
                     reward=None, difficulty="normal", tax_to_debt=True, **kw):
            self._tax = bool(tax_to_debt)
            self._curriculum = curriculum

        def obs_size(self):
            return OBS_SIZE

        def n_actions(self):
            return N_ACTIONS

        def build_ids(self):
            return list(BUILD_IDS)

        def reset(self, seed=0):
            pass

        def obs(self):
            return [0.0] * OBS_SIZE

        def action_mask(self):
            return [1.0] * N_ACTIONS

        def minimap(self):
            return [0.0] * MM_LEN

        def set_minimap_radius(self, r):
            pass

        def minimap_radius(self):
            return 16

        def tax_to_debt(self):
            return self._tax

        def curriculum(self):
            return dict(self._curriculum or {})

        def step(self, action):
            return {"obs": [0.0] * OBS_SIZE, "reward": 0.0, "terminated": False,
                    "truncated": False, "days": 1, "people": 50, "money": 82000,
                    "bases": 1, "tax_due_days": 0, "ep_return": 0.0, "steps": 1}

        def set_step_log(self, path):
            pass

        def close(self):
            pass

    mod.RunningMeanStd = RunningMeanStd
    mod.RewardConfig = RewardConfig
    mod.ColonyEnvCpp = ColonyEnvCpp
    mod.load_base_data = lambda path: object()
    mod.load_events = lambda path: object()
    mod.extension_info = lambda: {
        "version": 99,
        "features": ["set_curriculum", "resource_curriculum", "obs_v2",
                     "tax_to_debt", "mechanic_curriculum", "minimap"],
        "src_sha": "test-stub",
    }
    return mod


@pytest.fixture(scope="module")
def wc():
    """Модуль watch_champion: с настоящим colony_cpp, а без него — со стабом.

    Стаб живёт только на время этого модуля и убирается за собой: иначе он
    протекает в sys.modules и тесты, которые честно скипаются без расширения
    (например test_review_plan_regressions), начинают падать на неполном стабе.
    """
    saved = {n: sys.modules.get(n) for n in ("colony_cpp", "cpp_env", "watch_champion")}
    stubbed = False
    try:
        import watch_champion as mod
    except Exception:
        sys.modules["colony_cpp"] = _fake_colony_cpp()
        for name in ("cpp_env", "watch_champion"):
            sys.modules.pop(name, None)
        import watch_champion as mod
        stubbed = True
    yield mod
    if stubbed:
        for name in ("colony_cpp", "cpp_env", "watch_champion"):
            sys.modules.pop(name, None)
        for name, old_mod in saved.items():
            if old_mod is not None:
                sys.modules[name] = old_mod


# ── фейковое «окно» (протокол gui.cpp --headless-ai) ─────────────────────────

FAKE_GUI = r'''#!/usr/bin/env python3
"""Двойник sakhalin_colony_gui.exe --headless-ai (тот же файловый протокол)."""
import json, os, sys, time
from pathlib import Path

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

MODE = os.environ.get("FAKE_GUI_MODE", "ok")
ACTIONS_LOG = os.environ.get("FAKE_GUI_ACTIONS_LOG", "")
TERMINATE_AFTER = int(os.environ.get("FAKE_GUI_TERMINATE_AFTER", "4"))
LIFETIME = float(os.environ.get("FAKE_GUI_LIFETIME", "60"))
OBS_N = int(os.environ.get("FAKE_GUI_OBS", "299"))
MM_N = int(os.environ.get("FAKE_GUI_MM", "8192"))

actions_file = Path(arg("--actions-file"))
state_file = Path(arg("--state-file"))
assert "--headless-ai" in sys.argv, "драйвер обязан звать окно в --headless-ai"

if MODE == "die":
    sys.stderr.write("FAKE GUI: не смог открыть окно (имитация падения exe)\n")
    sys.stderr.flush()
    raise SystemExit(3)

day = 1
step = 0
terminated = False
t0 = time.monotonic()

def write_state(action, reward=0.0, term=False):
    payload = {
        "day": day, "month": 3, "year": 2026, "people": 50, "bases": 1 + step // 3,
        "money": 82000 - step * 100, "reward": reward, "action": action,
        "terminated": term,
        "obs": [0.01] * OBS_N,
        "action_mask": [1.0] * 49,
    }
    if MM_N:
        payload["minimap"] = [0.0] * MM_N
    tmp = state_file.with_name(state_file.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, state_file)

if MODE == "silent":
    # окно живо, но IPC не трогает вовсе (имитация «exe стартовал и завис»)
    while time.monotonic() - t0 < LIFETIME:
        time.sleep(0.05)
    raise SystemExit(0)

if MODE != "legacy":
    write_state(-1)            # новый протокол: состояние публикуется при старте

if MODE == "freeze":
    # стартовое состояние опубликовано, а на действия окно больше не отвечает
    while time.monotonic() - t0 < LIFETIME:
        time.sleep(0.05)
    raise SystemExit(0)

term_at = None
while time.monotonic() - t0 < LIFETIME:
    # авто-рестарт карты по таймеру (в gui.cpp — 5 с), действий для этого не нужно
    if terminated and term_at is not None and time.monotonic() - term_at > 0.3:
        terminated, term_at, step = False, None, 0
        write_state(-1)
        continue
    if actions_file.exists():
        try:
            action = int(actions_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            time.sleep(0.005)
            continue
        actions_file.unlink()          # consumed
        if ACTIONS_LOG:
            with open(ACTIONS_LOG, "a", encoding="utf-8") as f:
                f.write(f"{action}\n")
        step += 1
        day += 1
        terminated = step >= TERMINATE_AFTER
        if terminated:
            term_at = time.monotonic()
        write_state(action, reward=0.5, term=terminated)
    time.sleep(0.005)
'''


@pytest.fixture
def fake_gui(tmp_path) -> Path:
    script = tmp_path / "fake_gui.py"
    script.write_text(FAKE_GUI, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    if os.name == "nt":  # Windows: Popen не исполняет .py напрямую
        wrapper = tmp_path / "fake_gui.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
                           encoding="utf-8")
        return wrapper
    return script


@pytest.fixture
def model_dir(tmp_path) -> Path:
    """Гибридный чекпойнт + нормализатор + meta — как после настоящего обучения."""
    from rl.actor_critic_hybrid import ActorCriticHybrid

    d = tmp_path / "run_test"
    d.mkdir()
    model = ActorCriticHybrid(
        obs_size=OBS_SIZE, n_channels=8, grid_size=32, n_actions=N_ACTIONS,
        hidden_sizes=[16, 16], device=torch.device("cpu"),
    )
    torch.save({
        "model_state": model.state_dict(),
        "n_actions": N_ACTIONS,
        "hidden_sizes": [16, 16],
        "obs_size": OBS_SIZE,
        "n_channels": 8,
        "grid_size": 32,
        "obs_version": 2,
    }, str(d / "best_model.pt"))
    (d / "normalization.json").write_text(json.dumps({
        "mean": [0.0] * OBS_SIZE, "var": [1.0] * OBS_SIZE, "count": 1000,
        "obs_size": OBS_SIZE, "clip": 10.0,
    }), encoding="utf-8")
    (d / "best_model.meta.json").write_text(json.dumps({
        "obs_version": 2, "curriculum_stage_at_best": 0, "unlock_ids": "",
        "total_timesteps": 1000,
    }), encoding="utf-8")
    return d


@pytest.fixture
def watch(wc, monkeypatch, tmp_path, fake_gui, model_dir):
    """Готовый запуск watch_champion.main() в визуальном режиме."""
    monkeypatch.setenv("COLONY_GUI_EXE", str(fake_gui))
    # таймауты теста: не ждём по 25 с, как в бою
    monkeypatch.setattr(wc, "GUI_START_TIMEOUT", 3.0)
    monkeypatch.setattr(wc, "GUI_STEP_TIMEOUT", 3.0)
    monkeypatch.setattr(wc, "GUI_RESET_TIMEOUT", 3.0)
    monkeypatch.setattr(wc, "GUI_HEARTBEAT_EVERY", 0.4)
    log = tmp_path / "watch.log"
    actions_log = tmp_path / "actions.log"
    monkeypatch.setenv("FAKE_GUI_ACTIONS_LOG", str(actions_log))
    argv = ["watch_champion.py", "--model-dir", str(model_dir), "--visual",
            "--speed", "0", "--episodes", "1", "--max-steps", "50",
            "--seed", "12345", "--map-size", "280", "--log-file", str(log),
            "--curriculum-stage", "0", "--unlock-ids", ""]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdout", sys.stdout)  # TeeWriter подменит — вернём

    def run() -> int:
        try:
            wc.main()
        except SystemExit as e:  # main() выходит с кодом при diagnose-ветке
            return int(e.code or 0)
        return 0

    return types.SimpleNamespace(run=run, log=log, actions_log=actions_log,
                                 tmp=tmp_path)


def _log_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _steps(path: Path) -> "list[dict]":
    out = []
    for line in _log_text(path).splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "step":
                out.append(d)
    return out


# ── сценарии ─────────────────────────────────────────────────────────────────

def test_visual_watch_drives_gui_and_feeds_ui(watch):
    """Счастливый путь: окно поднялось, шаги идут, UI получает JSONL."""
    assert watch.run() == 0
    text = _log_text(watch.log)
    assert "GUI-окно запущено: pid=" in text
    assert "Visual watch running" in text
    assert "Game Over at day" in text
    assert "Visual watch stopped." in text

    steps = _steps(watch.log)
    assert len(steps) >= 4, steps
    assert {"day", "money", "people", "bases", "action", "reward"} <= set(steps[0])

    sent = [int(x) for x in watch.actions_log.read_text().split()]
    assert len(sent) >= 4
    assert all(0 <= a < N_ACTIONS for a in sent)


def test_visual_watch_bootstraps_legacy_exe(watch, monkeypatch):
    """Старый exe не пишет стартовое состояние — драйвер толкает его действием 0."""
    monkeypatch.setenv("FAKE_GUI_MODE", "legacy")
    assert watch.run() == 0
    sent = [int(x) for x in watch.actions_log.read_text().split()]
    assert sent[0] == 0, "первым действием обязан быть DAY-бутстрап"
    # бутстрап-действие окно считает своим первым шагом, поэтому на один шаг меньше
    assert len(_steps(watch.log)) >= 3


def test_visual_watch_reports_exe_that_dies_at_once(wc, watch, monkeypatch):
    """exe падает при старте → внятный диагноз и код 1, а не вечный рестарт."""
    monkeypatch.setenv("FAKE_GUI_MODE", "die")
    assert watch.run() == 1
    text = _log_text(watch.log)
    assert "ERROR: GUI-окно не запустилось" in text
    assert "exit code 3" in text
    assert "имитация падения exe" in text      # stderr окна перехвачен в gui_output.log
    assert text.count("restarting") <= wc.MAX_GUI_RESTARTS


def test_visual_watch_reports_silent_window(watch, monkeypatch):
    """Окно живо, но state.json не появляется → heartbeat + диагноз, а не тишина."""
    monkeypatch.setenv("FAKE_GUI_MODE", "silent")
    monkeypatch.setenv("FAKE_GUI_LIFETIME", "15")
    assert watch.run() == 1
    text = _log_text(watch.log)
    assert "жду первое состояние от GUI-окна" in text   # heartbeat виден в логе UI
    assert "нет state.json" in text
    assert "процесс жив" in text


def test_visual_watch_reports_frozen_window(watch, monkeypatch):
    """Окно стартовало, но на действие не отвечает → диагноз вместо вечного ожидания."""
    monkeypatch.setenv("FAKE_GUI_MODE", "freeze")
    monkeypatch.setenv("FAKE_GUI_LIFETIME", "15")
    assert watch.run() == 1
    text = _log_text(watch.log)
    assert "жду ответ от GUI-окна" in text
    assert "окно не ответило на действие" in text


def test_visual_watch_reports_missing_minimap_for_hybrid(watch, monkeypatch):
    """Гибрид без миникарты в state.json — явная ошибка, а не тихий flat-инференс."""
    monkeypatch.setenv("FAKE_GUI_MODE", "no-minimap")
    monkeypatch.setenv("FAKE_GUI_MM", "0")
    assert watch.run() == 0        # прогон не роняем: окно-то работает
    text = _log_text(watch.log)
    assert "не прислало миникарту" in text
    assert "build_gui.bat" in text


def test_visual_watch_uses_per_run_ipc_dir(wc, watch):
    """IPC-каталог свой на каждый запуск и убирается за собой."""
    root = Path(wc.tempfile.gettempdir()) / "colony_watch"
    before = {p.name for p in root.glob("*")} if root.exists() else set()
    assert watch.run() == 0
    created = {p.name for p in root.glob("*")} - before if root.exists() else set()
    assert not created, f"IPC-каталог не убран: {created}"


def test_visual_watch_two_runs_do_not_share_ipc(watch, monkeypatch):
    """Два запуска подряд не мешают друг другу (сироты прошлых прогонов)."""
    monkeypatch.setenv("FAKE_GUI_TERMINATE_AFTER", "2")
    assert watch.run() == 0
    first = _steps(watch.log)
    assert watch.run() == 0
    assert len(_steps(watch.log)) >= len(first)


def test_make_ipc_dir_is_unique(wc, monkeypatch, tmp_path):
    monkeypatch.setattr(wc.tempfile, "gettempdir", lambda: str(tmp_path))
    a = wc.make_ipc_dir()
    b = wc.make_ipc_dir()
    assert a != b and a.is_dir() and b.is_dir()


def test_write_action_is_atomic_and_reports_unread(wc, tmp_path):
    f = tmp_path / "actions.txt"
    assert wc.write_action(f, 7, timeout=0.2) is True
    assert f.read_text(encoding="utf-8").strip() == "7"
    assert not (tmp_path / "actions.txt.tmp").exists()
    # файл не забрали → False (раньше драйвер просто перезаписывал действие)
    assert wc.write_action(f, 8, timeout=0.1) is False


def test_wait_for_state_requires_fresh_answer(wc, tmp_path):
    """Lock-step: ответ на действие обязан быть НОВЫМ state.json."""
    sf = tmp_path / "state.json"
    sf.write_text(json.dumps({"day": 1}), encoding="utf-8")
    stamp = wc.state_stamp(sf)
    assert wc.wait_for_state(sf, since=stamp, timeout=0.2) is None
    beats = []
    sf.write_text(json.dumps({"day": 2}), encoding="utf-8")
    got = wc.wait_for_state(sf, since=stamp, timeout=1.0,
                            on_wait=beats.append)
    assert got is not None and got["day"] == 2


def test_gui_exe_stale_sources_flags_old_binary(wc, tmp_path, monkeypatch):
    """exe старше исходников → предупреждение (старый exe не понимает протокол)."""
    exe = tmp_path / "sakhalin_colony_gui.exe"
    exe.write_bytes(b"MZ")
    old = os.path.getmtime(exe) - 10_000
    os.utime(exe, (old, old))
    monkeypatch.setattr(wc, "PROJECT_ROOT", PROJECT_ROOT)
    newer = wc.gui_exe_stale_sources(exe)
    assert any(name.startswith("src/") or name.startswith("include/")
               for name in newer), newer
    # свежий exe — предупреждений нет
    os.utime(exe, None)
    assert wc.gui_exe_stale_sources(exe) == []


def test_launch_visual_watch_captures_gui_output(wc, tmp_path, monkeypatch):
    """stdout/stderr окна пишутся в gui_output.log (иначе причина не видна)."""
    calls = {}

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    log = tmp_path / "gui_output.log"
    wc.launch_visual_watch(
        model_dir=tmp_path, exe_path="gui.exe",
        actions_file=tmp_path / "actions.txt", state_file=tmp_path / "state.json",
        seed=1, map_size=280, curriculum=None, gui_log=log)
    assert calls["args"][0] == "gui.exe"
    assert "--headless-ai" in calls["args"][0 + 1:]
    assert calls["kwargs"]["stdout"].name == str(log)
    assert calls["kwargs"]["stdout"].closed       # ручку родитель закрывает


def test_visual_watch_waits_for_map_autoreset(watch, monkeypatch):
    """Эпизод закончился → ждём свежий state от авто-рестарта карты в окне."""
    monkeypatch.setenv("FAKE_GUI_TERMINATE_AFTER", "3")
    argv = list(sys.argv)
    argv[argv.index("--episodes") + 1] = "2"
    monkeypatch.setattr(sys, "argv", argv)
    assert watch.run() == 0
    text = _log_text(watch.log)
    game_overs = [ln for ln in text.splitlines() if "Game Over at day" in ln]
    assert len(game_overs) == 2, game_overs
    assert "episode 2/2" in text


# ── какой файл весов грузит CLI ──────────────────────────────────────────────

def test_resolve_model_file_prefers_requested(wc, tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "final_model.pt").write_bytes(b"f")
    (d / "best_model.pt").write_bytes(b"b")
    assert wc.resolve_model_file(d, "best_model.pt") == d / "best_model.pt"
    assert wc.resolve_model_file(d, "final_model.pt") == d / "final_model.pt"


def test_resolve_model_file_falls_back_to_best_before_checkpoints(wc, tmp_path):
    """Прерванный прогон: final нет, но есть чемпион турнира и чекпойнты.

    Старая цепочка (final → чекпойнт) пропустила бы `best_model.pt` и грузила
    произвольный чекпойнт; порядок теперь совпадает с
    `train_ui2.models.pick_model_file`.
    """
    d = tmp_path / "interrupted"
    d.mkdir()
    (d / "best_model.pt").write_bytes(b"b")
    (d / "checkpoint_1000_steps.pt").write_bytes(b"c")
    assert wc.resolve_model_file(d, "final_model.pt") == d / "best_model.pt"


def test_resolve_model_file_last_checkpoint_and_missing(wc, tmp_path):
    d = tmp_path / "only_ckpts"
    d.mkdir()
    (d / "checkpoint_999000_steps.pt").write_bytes(b"c1")
    (d / "checkpoint_1000000_steps.pt").write_bytes(b"c2")
    assert wc.resolve_model_file(d, "best_model.pt") == d / "checkpoint_1000000_steps.pt"

    empty = tmp_path / "empty"
    empty.mkdir()
    # ничего нет → возвращаем запрошенный путь, чтобы ошибка назвала его же
    assert wc.resolve_model_file(empty, "best_model.pt") == empty / "best_model.pt"


def test_make_ipc_dir_unique_even_in_same_millisecond(wc, monkeypatch, tmp_path):
    """Два запуска в одном процессе не обязаны попадать в разные миллисекунды.

    Прежняя метка `<pid>_<мс>` совпадала, если предыдущий каталог уже убран:
    второй запуск получал тот же путь, а значит — те же actions.txt/state.json.
    """
    monkeypatch.setattr(wc.tempfile, "gettempdir", lambda: str(tmp_path))
    frozen = 1_700_000_000.0
    monkeypatch.setattr(wc.time, "time", lambda: frozen)
    first = wc.make_ipc_dir()
    first.rmdir()                      # как после штатной уборки за запуском
    second = wc.make_ipc_dir()
    assert first != second
    assert second.is_dir()
