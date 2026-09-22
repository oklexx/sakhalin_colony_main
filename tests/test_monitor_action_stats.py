"""Регрессии панели «Активности / Строительство зданий» (вкладка «Мониторинг»).

Повод (2026-09-23): строка «Водоканал» появлялась, её процент рос, а потом
строка исчезала. Причина — не в политике, а в способе измерения:

1. `rl/async_trainer.py` считал доли по `deque(maxlen=1000)`, тогда как роллаут
   = `n_steps * n_envs` (дефолт 4096 * 8 = 32768 действий). Панель показывала
   ~3% роллаута: «в хвосте не было нажатий» = «действие пропало».
2. Отсечка top-15 в тренере: при равных count стабильная сортировка отдавала
   приоритет меньшему индексу действия — строка вылетала из списка, хотя её
   доля не падала.
3. Маска: BUILD-действие легально, только если оно открыто курикулумом, денег
   хватает и `find_lot` нашёл свободный участок (`tests/cpp/water_mask_check.cpp`
   W3/W6). У водоканала окно легальности короткое, а монитор о нём молчал.

Отсюда контракты ниже: счётчики — по всему роллауту; нулевые и вытесненные
строки остаются «липкими»; легальность считается и доезжает до UI; прогресс
этапа не врёт нулём.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _require_torch() -> None:
    """`rl/*` тянет torch: без него скипаем, а не падаем (как с colony_cpp)."""
    pytest.importorskip("torch")


# ── 1. окно счётчика = весь роллаут, а не хвост в 1000 действий ──────────────

def test_rollout_counters_reset_each_rollout():
    """_record_rollout_actions копит по роллауту, pop обнуляет окно."""
    _require_torch()
    import numpy as np

    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    names = ["DAY", "BUILD_WATERCHANNEL", "BUILD_FARM"]
    trainer = AsyncTrainer(cfg=cfg, env_manager=SimpleNamespace(
        device="cpu", action_names=names))

    # 6 шагов роллаута (n_steps=3 * n_envs=2): BUILD_WATERCHANNEL 6 раз, DAY 2
    for _ in range(3):
        trainer._record_rollout_actions(np.array([1, 1], dtype=np.int64))
    trainer._record_rollout_actions(np.array([0, 0], dtype=np.int64))
    counts = trainer._calculate_action_distribution()
    assert counts == [2, 6, 0]

    # 6 из 8 = 75% — т.е. доля «сколько шагов роллаута ушло на действие»,
    # а не 0.1%-кванты окна в 1000 записей
    assert trainer._top_actions_dict(counts, sum(counts)) == \
        {"BUILD_WATERCHANNEL": 75.0, "DAY": 25.0}

    trainer._reset_rollout_actions()
    assert trainer._calculate_action_distribution() == [0, 0, 0]
    assert trainer._top_actions_dict(trainer._calculate_action_distribution(), 0) == {}


def test_calculate_action_distribution_window_is_explicit():
    """Окно по истории осталось, но только как явный отладочный режим."""
    _require_torch()
    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    names = ["DAY", "BUILD_WATERCHANNEL", "BUILD_FARM"]
    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    trainer = AsyncTrainer(cfg=cfg, env_manager=SimpleNamespace(
        device="cpu", action_names=names))

    trainer._action_history.extend([(0, names[1]), (1, names[1]), (0, names[0])])
    # окно = последние N записей: [names[1], names[0]]
    assert trainer._calculate_action_distribution(window=2) == [1, 1, 0]
    assert trainer._calculate_action_distribution(window=3) == [1, 2, 0]
    # окно по умолчанию = счётчик роллаута (пустой, пока идёт сбор)
    assert trainer._calculate_action_distribution(window=0) == [0, 0, 0]


def test_monitor_action_top_zero_means_no_truncation():
    """monitor_action_top = 0 → все ненулевые действия, вытеснений нет."""
    _require_torch()
    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    names = [f"BUILD_{i}" for i in range(30)]
    counts = [1] * 30

    def make(top: int) -> AsyncTrainer:
        cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False,
                     monitor_action_top=top)
        return AsyncTrainer(cfg=cfg, env_manager=SimpleNamespace(
            device="cpu", action_names=names))

    assert len(make(0)._top_actions_dict(counts, 30)) == 30, \
        "при равных count строки не должны вылетать из-за top-N"
    assert len(make(5)._top_actions_dict(counts, 30)) == 5


def test_record_rollout_actions_ignores_out_of_range():
    """Индексы за пределами списка имён (битый fallback) не роняют счётчик."""
    _require_torch()
    import numpy as np

    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    trainer = AsyncTrainer(cfg=cfg, env_manager=SimpleNamespace(
        device="cpu", action_names=["DAY", "WEEK"]))
    trainer._record_rollout_actions(np.array([0, 1, 99, -1], dtype=np.int64))
    assert trainer._calculate_action_distribution() == [1, 1]


# ── 2. легальность действия по маскам ────────────────────────────────────────

def test_legality_monitor_percentages():
    _require_torch()
    import numpy as np

    from rl.action_monitor import ActionLegalityMonitor

    mon = ActionLegalityMonitor(n_actions=3)
    mon.add_step(np.array([[1, 0, 1], [1, 0, 1]], dtype=np.float32))
    mon.add_step(np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32))
    per = mon.pop_percent(["DAY", "BUILD_WATERCHANNEL", "BUILD_FARM"])
    assert per["BUILD_WATERCHANNEL"] == pytest.approx(25.0)
    assert per["DAY"] == pytest.approx(100.0)
    assert per["BUILD_FARM"] == pytest.approx(50.0)
    # pop закрывает окно: следующий роллаут считается заново, пустое окно = {}
    assert mon.pop_percent(["DAY"]) == {}


def test_legality_monitor_tolerates_shape_mismatch():
    """Старый .pyd с другим числом действий не должен ронять обучение."""
    _require_torch()
    import numpy as np

    from rl.action_monitor import ActionLegalityMonitor

    mon = ActionLegalityMonitor(n_actions=5)
    mon.add_step(np.ones((2, 3), dtype=np.float32))
    mon.add_step(None)
    mon.add_step(np.ones((2,), dtype=np.float32))
    per = mon.pop_percent(["A", "B", "C", "D", "E"])
    assert per["A"] == pytest.approx(100.0)
    assert per["D"] == pytest.approx(0.0)
    assert mon.steps == 0


def test_env_manager_exposes_legality_bridge():
    """EnvManager обязан быть мостом «маски → метрика», иначе тренеру брать легальность неоткуда."""
    _require_torch()
    from rl.env_manager import EnvManager

    assert hasattr(EnvManager, "pop_action_legality")


def test_track_step_actions_feeds_both_consumers():
    """Один срез буфера кормит и счётчик роллаута, и историю циклов."""
    _require_torch()
    import numpy as np

    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    names = ["DAY", "BUILD_WATERCHANNEL", "BUILD_FARM"]

    class _Arr:
        """Копия формы RolloutBuffer.actions: плоский массив с .cpu().numpy()."""

        def __init__(self, a):
            self._a = np.asarray(a)

        def __getitem__(self, item):
            return _Arr(self._a[item])

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    class _Buffer:
        def __init__(self, actions):
            self.actions = _Arr(actions)
            self.pos = 1  # последний записанный шаг = строки [0:n_envs]

    logs: List[str] = []
    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    em = SimpleNamespace(device="cpu", action_names=names, n_envs=2,
                         buffer=_Buffer(np.array([1, 1])))
    trainer = AsyncTrainer(cfg=cfg, env_manager=em,
                           logger=SimpleNamespace(info=logs.append))

    trainer._track_step_actions(2, names)
    assert trainer._calculate_action_distribution() == [0, 2, 0]
    assert list(trainer._action_history) == [(0, "BUILD_WATERCHANNEL"),
                                             (1, "BUILD_WATERCHANNEL")]

    # Сломанный доступ к буферу не валит обучение, но виден в логе (1 раз)
    em.buffer = None
    trainer._track_step_actions(2, names)
    trainer._track_step_actions(2, names)
    assert len([l for l in logs if "actions шага" in l]) == 1
    assert trainer._calculate_action_distribution() == [0, 2, 0]


def test_env_manager_pop_action_legality_uses_env_names():
    """Мост маски → метрика: имена берёт у среды, окно закрывает за один вызов."""
    _require_torch()
    import numpy as np

    from rl.action_monitor import ActionLegalityMonitor
    from rl.env_manager import EnvManager

    em = EnvManager.__new__(EnvManager)  # без __init__: реальная C++-среда не нужна
    em.vec_env = SimpleNamespace(action_names=lambda: ["DAY", "BUILD_WATERCHANNEL",
                                                        "BUILD_FARM"])
    em._action_legality = ActionLegalityMonitor(3)
    em._action_legality.add_step(np.array([[1, 0, 1]], dtype=np.float32))
    assert em.pop_action_legality() == {"DAY": 100.0, "BUILD_WATERCHANNEL": 0.0,
                                        "BUILD_FARM": 100.0}
    # окно закрыто: следующее = пусто (это «не измерялось», а не «0%»)
    assert em.pop_action_legality() == {}

    em._action_legality = None  # флаг monitor_action_legality = False
    assert em.pop_action_legality() == {}


# ── 3. прогресс этапа: метод должен существовать (раньше врал 0%) ────────────

def test_env_manager_has_get_curriculum_progress():
    """AsyncTrainer зовёт em.get_curriculum_progress — метода не было вовсе.

    Вызов падал в `except Exception`, прогресс всегда = 0.0, и подпись
    «курикулум: этап N (0%)» врела на каждом прогоне.
    """
    _require_torch()
    from rl.env_manager import EnvManager

    assert hasattr(EnvManager, "get_curriculum_progress")


def test_stage_progress_schedule_math():
    _require_torch()
    from rl.curriculum import stage_progress

    schedule = [[1_000_000, 1], [2_000_000, 2]]
    p = stage_progress(step=500_000, schedule=schedule, base_stage=0)
    assert p["mode"] == "schedule" and p["stage"] == 0
    assert p["progress"] == pytest.approx(0.5)
    assert p["next_at_step"] == 1_000_000 and p["next_stage"] == 1

    p = stage_progress(step=1_500_000, schedule=schedule, base_stage=1)
    assert p["progress"] == pytest.approx(0.5) and p["next_stage"] == 2
    assert p["steps_remaining"] == 500_000

    # Последний этап: ждать перехода некуда → «фикс.», а не 100%/0%
    p = stage_progress(step=3_000_000, schedule=schedule, base_stage=2)
    assert p["mode"] == "fixed" and p["progress"] is None

    p = stage_progress(step=10, schedule=[], base_stage=1)
    assert p["mode"] == "fixed" and p["progress"] is None
    assert p["progress_percent"] is None

    # Мусорная строка расписания из UI-таблицы не должна ронять монитор
    p = stage_progress(step=10, schedule=[["x", "y"]], base_stage=0)
    assert p["mode"] == "fixed"


def test_trainer_marks_progress_invalid_instead_of_zero():
    """Нет метода у em → valid=False («н/д» в UI), и ровно одно предупреждение."""
    _require_torch()
    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    logs: List[str] = []
    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    trainer = AsyncTrainer(cfg=cfg, env_manager=SimpleNamespace(device="cpu"),
                           logger=SimpleNamespace(info=logs.append))
    pct, valid = trainer._curriculum_progress_view(step=123)
    assert valid is False and pct == 0.0
    trainer._curriculum_progress_view(step=456)
    warns = [l for l in logs if "get_curriculum_progress" in l]
    assert len(warns) == 1, f"предупреждение должно быть однократным: {logs}"

    real = SimpleNamespace(device="cpu",
                           get_curriculum_progress=lambda step: {"progress": 0.25})
    assert AsyncTrainer(cfg=cfg, env_manager=real)._curriculum_progress_view(10) == \
        (0.25, True)


# ── 4. чистая логика панелей: без Qt → тестируется headless ──────────────────

def test_split_by_panel_is_case_insensitive_on_bare_ids():
    from train_ui2.monitor import split_by_panel

    acts, builds = split_by_panel(
        {"BUILD_WATERCHANNEL": 1.0, "WATERCHANNEL": 0.5, "FARM": 0.2, "DAY": 90.0},
        ["WaterChannel", "Farm"],
    )
    assert builds == {"BUILD_WATERCHANNEL": 1.0, "WATERCHANNEL": 0.5, "FARM": 0.2}
    assert acts == {"DAY": 90.0}


def test_rows_with_sticky_keeps_watch_list_and_legality():
    from train_ui2.monitor import rows_with_sticky

    items = {"BUILD_FARM": 3.0, "BUILD_GARDEN": 2.0}
    legality = {"BUILD_FARM": 90.0, "BUILD_GARDEN": 80.0, "BUILD_WATERCHANNEL": 12.0}
    rows = rows_with_sticky(items, legality, keep=["BUILD_WATERCHANNEL"], max_rows=10)
    assert [r[0] for r in rows] == ["BUILD_FARM", "BUILD_GARDEN", "BUILD_WATERCHANNEL"]
    assert rows[0] == ("BUILD_FARM", 3.0, 90.0, False)
    assert rows[2] == ("BUILD_WATERCHANNEL", 0.0, 12.0, True)


def test_rows_with_sticky_pins_watch_rows_and_cuts_the_tail():
    """Закреплённая строка видна всегда; отрезается хвост остальных, а не она."""
    from train_ui2.monitor import rows_with_sticky

    items = {f"BUILD_{i}": 10.0 - i for i in range(6)}
    rows = rows_with_sticky(items, {}, keep=["BUILD_WATCH"], max_rows=4)
    assert len(rows) == 4
    assert any(r[0] == "BUILD_WATCH" for r in rows), \
        "закреплённое здание не имеет права пропадать из-за переполнения панели"
    # порядок — по доле: нулевая закреплённая строка честно уезжает вниз,
    # но остаётся видимой (её и читают как диагноз)
    assert [r[0] for r in rows] == ["BUILD_0", "BUILD_1", "BUILD_2", "BUILD_WATCH"]
    # нулевая закреплённая строка помечена липкой (рисуется серым), реальные — нет
    assert rows[-1][3] is True and rows[0][3] is False

    # если места меньше, чем закреплённых строк, приоритет у большей доли
    tight = rows_with_sticky({"BUILD_A": 5.0, "BUILD_B": 1.0}, {},
                             keep=["BUILD_A", "BUILD_B"], max_rows=1)
    assert [r[0] for r in tight] == ["BUILD_A"]


def test_action_verdict_separates_cant_from_wont():
    """Главный диагностический вопрос: «нельзя» или «не хочет»."""
    from train_ui2.monitor import action_verdict

    assert action_verdict(0.0, 0.0) == "заблокировано маской"
    assert action_verdict(0.0, 0.5) == "заблокировано маской"
    assert action_verdict(0.0, 5.0) == "окно легальности узкое"
    assert action_verdict(0.0, 70.0) == "легально, но не выбирает"
    assert action_verdict(1.5, 70.0) == "выбирает"
    # Доля есть, легальность узкая — это уже «выбирает», а не жалоба на маску:
    # вердикт про маску интересует только нулевые строки.
    assert action_verdict(0.4, 12.0) == "выбирает"
    # Легальность неизвестна — вердикт про маску выдумывать нельзя
    assert action_verdict(1.5, None) == ""
    assert action_verdict(0.0, None) == "нет данных о легальности"


def test_watch_report_prefers_panel_shares_over_recount():
    from train_ui2.monitor import watch_report

    counts = {"BUILD_WATERCHANNEL": 4, "BUILD_ROAD": 400}
    legality = {"BUILD_WATERCHANNEL": 12.0, "BUILD_ROAD": 98.0}
    rows = watch_report(counts, legality, total_actions=1000,
                        names=["BUILD_WATERCHANNEL", "BUILD_ROAD", "BUILD_HOUSE"])
    assert len(rows) == 3
    assert rows[0] == {"name": "BUILD_WATERCHANNEL", "count": 4,
                       "pct": pytest.approx(0.4), "legal": 12.0, "verdict": "выбирает"}
    assert rows[1]["pct"] == pytest.approx(40.0)
    # Действие вне счётчиков = ноль, а не KeyError; легальности нет → честное
    # «нет данных», а не «заблокировано маской»
    assert rows[2] == {"name": "BUILD_HOUSE", "count": 0, "pct": 0.0, "legal": None,
                       "verdict": "нет данных о легальности"}

    # Тот самый сценарий жалобы: построил — окно легальности закрылось
    zero = watch_report({"BUILD_WATERCHANNEL": 0}, {"BUILD_WATERCHANNEL": 0.0},
                        total_actions=1000, names=["BUILD_WATERCHANNEL"])
    assert zero[0]["pct"] == 0.0 and zero[0]["verdict"] == "заблокировано маской"

    # Готовые доли панели имеют приоритет: подпись и шкала не расходятся
    rows = watch_report(counts, legality, total_actions=1000,
                        names=["BUILD_ROAD"], shares={"BUILD_ROAD": 39.5})
    assert rows[0]["pct"] == pytest.approx(39.5) and rows[0]["count"] == 400


def test_fmt_pct_does_not_flush_small_shares_to_zero():
    """На окне в 32768 шагов 4 постройки = 0.012%: «0.0%» читалось бы как «нет».
    """
    from train_ui2.monitor import fmt_pct

    assert fmt_pct(0.0) == "0%"
    assert fmt_pct(0.012) == "0.012%"
    assert fmt_pct(1.5) == "1.50%"
    assert fmt_pct(60.0) == "60.0%"


# ── 5. протокол и проводка метрик ────────────────────────────────────────────

def test_protocol_roundtrip_new_monitor_fields():
    from train_ui2 import protocol as P

    msg = P.ProgressMsg(
        done=1, total=2,
        top_actions={"BUILD_WATERCHANNEL": 0.4},
        action_counts={"BUILD_WATERCHANNEL": 12, "DAY": 900},
        action_legality={"BUILD_WATERCHANNEL": 37.5, "DAY": 100.0},
        action_total_steps=1000,
        curriculum_progress_percent=0.42,
        curriculum_progress_valid=False,
    )
    back = P.decode(P.encode(msg))
    assert back.action_counts == {"BUILD_WATERCHANNEL": 12, "DAY": 900}
    assert back.action_legality == {"BUILD_WATERCHANNEL": 37.5, "DAY": 100.0}
    assert back.action_total_steps == 1000
    assert back.curriculum_progress_valid is False


def test_protocol_decodes_legacy_progress_lines_without_new_fields():
    """Старый лог/старый воркер: ключей нет → дефолты, а не KeyError."""
    import json

    from train_ui2 import protocol as P

    legacy = json.dumps({"type": "progress", "done": 5, "total": 10,
                         "top_actions": {"DAY": 100.0}})
    back = P.decode(legacy)
    assert back.top_actions == {"DAY": 100.0}
    assert back.action_counts == {} and back.action_legality == {}
    assert back.action_total_steps == 0
    assert back.curriculum_progress_valid is True


def test_protocol_survives_nan_in_action_fields():
    """NaN/inf в метриках не должен валить allow_nan=False в encode()."""
    from train_ui2 import protocol as P

    msg = P.ProgressMsg(done=1, total=2,
                        action_counts={"DAY": float("nan")},
                        action_legality={"DAY": float("inf")})
    back = P.decode(P.encode(msg))
    assert back.action_counts == {"DAY": 0} and back.action_legality == {"DAY": 0.0}


def test_worker_forwards_monitor_fields():
    """Имена полей в воркере обязаны совпадать с TrainMetrics (опечатка = молчаливый 0)."""
    src = (ROOT / "train_ui2" / "worker.py").read_text(encoding="utf-8")
    for field in ("top_actions", "action_counts", "action_legality",
                  "action_total_steps", "curriculum_progress_valid"):
        pat = re.compile(rf"{field}\s*=\s*.*getattr\(metrics,\s*['\"]{field}['\"]")
        assert pat.search(src), f"воркер не прокидывает metrics.{field}"


def test_metrics_have_the_fields_worker_reads():
    """Worker читает getattr(metrics, ...) — поля обязаны существовать."""
    _require_torch()
    from dataclasses import fields as dc_fields

    from rl.async_trainer import TrainMetrics

    names = {f.name for f in dc_fields(TrainMetrics)}
    assert {"top_actions", "action_counts", "action_legality", "action_total_steps",
            "curriculum_progress_valid", "curriculum_progress_percent"} <= names


def test_bars_widget_sourced_without_qt_import_side_effects():
    """main_window обязан передавать легальность в Bars (иначе след не отрисуется)."""
    src = (ROOT / "train_ui2" / "main_window.py").read_text(encoding="utf-8")
    assert "set_items(actions_dict, legality)" in src
    assert "set_items(build_dict, legality)" in src
    assert "keep=KEY_ACTIONS_MONITOR" in src


def test_config_has_monitor_knobs():
    _require_torch()
    from rl.config import Config

    cfg = Config()
    assert cfg.monitor_action_top == 0, "по умолчанию — без отсечки top-N"
    assert cfg.monitor_action_legality is True
    cfg2 = Config.from_dict(cfg.to_dict())
    assert cfg2.monitor_action_top == 0 and cfg2.monitor_action_legality is True
