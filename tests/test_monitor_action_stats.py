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

    logs: list[str] = []
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
    assert len([line for line in logs if "actions шага" in line]) == 1
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

    logs: list[str] = []
    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    trainer = AsyncTrainer(cfg=cfg, env_manager=SimpleNamespace(device="cpu"),
                           logger=SimpleNamespace(info=logs.append))
    pct, valid = trainer._curriculum_progress_view(step=123)
    assert valid is False and pct == 0.0
    trainer._curriculum_progress_view(step=456)
    warns = [line for line in logs if "get_curriculum_progress" in line]
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
    assert action_verdict(0.0, 5.0) == "окно доступности узкое"
    assert action_verdict(0.0, 70.0) == "доступно, но не выбирает"
    assert action_verdict(1.5, 70.0) == "выбирает"
    # Доля есть, легальность узкая — это уже «выбирает», а не жалоба на маску:
    # вердикт про маску интересует только нулевые строки.
    assert action_verdict(0.4, 12.0) == "выбирает"
    # Легальность неизвестна — вердикт про маску выдумывать нельзя
    assert action_verdict(1.5, None) == ""
    assert action_verdict(0.0, None) == "нет данных о доступности"


def test_watch_report_prefers_panel_shares_over_recount():
    from train_ui2.monitor import watch_report

    counts = {"BUILD_WATERCHANNEL": 4, "BUILD_ROAD": 400}
    legality = {"BUILD_WATERCHANNEL": 12.0, "BUILD_ROAD": 98.0}
    rows = watch_report(counts, legality, total_actions=1000,
                        names=["BUILD_WATERCHANNEL", "BUILD_ROAD", "BUILD_HOUSE"])
    assert len(rows) == 3
    assert rows[0] == {"name": "BUILD_WATERCHANNEL", "count": 4,
                       "pct": pytest.approx(0.4), "legal": 12.0, "verdict": "выбирает",
                       "reason": None}
    assert rows[1]["pct"] == pytest.approx(40.0)
    # Действие вне счётчиков = ноль, а не KeyError; легальности нет → честное
    # «нет данных», а не «заблокировано маской»
    assert rows[2] == {"name": "BUILD_HOUSE", "count": 0, "pct": 0.0, "legal": None,
                       "verdict": "нет данных о доступности", "reason": None}

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
        action_mask_reasons={"BUILD_WATERCHANNEL": "no_lot"},
        action_total_steps=1000,
        curriculum_progress_percent=0.42,
        curriculum_progress_valid=False,
    )
    back = P.decode(P.encode(msg))
    assert back.action_counts == {"BUILD_WATERCHANNEL": 12, "DAY": 900}
    assert back.action_legality == {"BUILD_WATERCHANNEL": 37.5, "DAY": 100.0}
    assert back.action_mask_reasons == {"BUILD_WATERCHANNEL": "no_lot"}
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
    assert back.action_mask_reasons == {}
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
                  "action_mask_reasons", "action_total_steps",
                  "curriculum_progress_valid"):
        pat = re.compile(rf"{field}\s*=\s*.*getattr\(metrics,\s*['\"]{field}['\"]")
        assert pat.search(src), f"воркер не прокидывает metrics.{field}"


def test_metrics_have_the_fields_worker_reads():
    """Worker читает getattr(metrics, ...) — поля обязаны существовать."""
    _require_torch()
    from dataclasses import fields as dc_fields

    from rl.async_trainer import TrainMetrics

    names = {f.name for f in dc_fields(TrainMetrics)}
    assert {"top_actions", "action_counts", "action_legality", "action_mask_reasons",
            "action_total_steps",
            "curriculum_progress_valid", "curriculum_progress_percent"} <= names


def test_bars_widget_sourced_without_qt_import_side_effects():
    """main_window обязан передавать легальность и причины в Bars."""
    src = (ROOT / "train_ui2" / "main_window.py").read_text(encoding="utf-8")
    assert "set_items(actions_dict, legality, reasons)" in src
    assert "set_items(build_dict, legality, reasons)" in src
    assert "keep=KEY_ACTIONS_MONITOR" in src


def test_config_has_monitor_knobs():
    _require_torch()
    from rl.config import Config

    cfg = Config()
    assert cfg.monitor_action_top == 0, "по умолчанию — без отсечки top-N"
    assert cfg.monitor_action_legality is True
    cfg2 = Config.from_dict(cfg.to_dict())
    assert cfg2.monitor_action_top == 0 and cfg2.monitor_action_legality is True


# ── 6. атрибуция причины закрытой маски: деньги / нет участка / курикулум ────
# docs/MONITOR_ACTIONS_2026_09.md §4: биндинг action_mask_reason_counts() в
# src/bindings.cpp + «та же колонка в панели». Коды причин — это W2 из
# water_mask_check (first-fail: курикулум → деньги → нет клетки), но уже
# по каждому действию, а не только по водоканалу.

def _cpp_mask_reason_names() -> list[str]:
    """MASK_REASON_NAMES из include/colony/constants.h (регресс против рассинхрона)."""
    src = (ROOT / "include" / "colony" / "constants.h").read_text(encoding="utf-8")
    m = re.search(
        r"MASK_REASON_NAMES\[N_MASK_REASONS\]\s*=\s*\{(.*?)\}", src, re.DOTALL)
    assert m, "MASK_REASON_NAMES не найден в include/colony/constants.h"
    return re.findall(r'"([a-z_]+)"', m.group(1))


def test_mask_reason_keys_in_sync_with_cpp():
    """Python-ключи причин обязаны совпадать с таблицей в constants.h."""
    _require_torch()
    from rl.action_monitor import MASK_REASON_KEYS

    assert list(MASK_REASON_KEYS) == _cpp_mask_reason_names(), (
        f"рассинхрон Python/C++ причин маски: {MASK_REASON_KEYS}")
    # порядок first-fail из W2 (water_mask_check): курикулум → деньги → участок
    assert MASK_REASON_KEYS == ("open", "curriculum", "money", "no_lot", "other")


def test_mask_reason_monitor_dominant_reason_per_action():
    """Окно = роллаут; на каждый action — доминирующая причина закрытия."""
    _require_torch()
    import numpy as np

    from rl.action_monitor import MaskReasonMonitor

    mon = MaskReasonMonitor(n_actions=3)
    # коды: 0=open, 2=money, 3=no_lot — два env-шага подряд
    mon.add_step(np.array([[0, 2, 3],
                           [0, 2, 3]], dtype=np.uint8))
    per = mon.pop_dominant(["DAY", "BUILD_WATERCHANNEL", "BUILD_FARM"])
    assert per == {"BUILD_WATERCHANNEL": "money", "BUILD_FARM": "no_lot"}
    # действие, открытое все шаги, в ответ не попадает (причины нет — не врём)
    # pop закрывает окно: следующий роллаут считается заново
    assert mon.pop_dominant(["DAY"]) == {}

    # Всегда открытое окно → пустой ответ, а не «open» в каждой строке
    mon.add_step(np.array([[0, 0, 0]], dtype=np.uint8))
    assert mon.pop_dominant(["DAY", "WEEK", "X"]) == {}


def test_mask_reason_monitor_tie_prefers_earlier_key():
    """Тай-брейк = порядок MASK_REASON_KEYS[1:] (курикулум раньше денег)."""
    _require_torch()
    import numpy as np

    from rl.action_monitor import MaskReasonMonitor

    mon = MaskReasonMonitor(n_actions=2)
    # action1: равное число money(2) и curriculum(1) → curriculum (argmax первым)
    mon.add_step(np.array([[0, 2],
                           [0, 1]], dtype=np.uint8))
    per = mon.pop_dominant(["DAY", "BUILD_X"])
    assert per == {"BUILD_X": "curriculum"}


def test_mask_reason_monitor_tolerates_bad_shapes():
    """Старый .pyd без биндинга / 1D-мусор / коды вне корзин — не роняем шаг."""
    _require_torch()
    import numpy as np

    from rl.action_monitor import MaskReasonMonitor

    mon = MaskReasonMonitor(n_actions=3)
    mon.add_step(None)
    mon.add_step(np.zeros((2,), dtype=np.uint8))          # 1D — молча мимо
    mon.add_step(np.full((2, 5), 99, dtype=np.uint8))     # мусорные коды мимо корзин
    per = mon.pop_dominant(["A", "B", "C"])
    assert per == {}  # ни одного валидно-закрытого шага
    assert mon.steps == 0
    # форма шире окна — обрезаем, уже накопленное не теряем
    mon.add_step(np.array([[0, 2, 3, 7, 8]], dtype=np.uint8))
    assert mon.pop_dominant(["A", "B", "C"]) == {"B": "money", "C": "no_lot"}


def test_env_manager_exposes_reason_bridge():
    """Мост «причины из vec_env → метрика»: метод есть, окно закрывается за pop."""
    _require_torch()
    import numpy as np

    from rl.action_monitor import MaskReasonMonitor
    from rl.env_manager import EnvManager

    assert hasattr(EnvManager, "pop_action_mask_reasons")
    em = EnvManager.__new__(EnvManager)  # без __init__: реальная C++-среда не нужна
    em.vec_env = SimpleNamespace(action_names=lambda: ["DAY", "WEEK", "BUILD_X"])
    em._action_reasons = MaskReasonMonitor(3)
    em._action_reasons.add_step(np.array([[0, 2, 3]], dtype=np.uint8))
    assert em.pop_action_mask_reasons() == {"WEEK": "money", "BUILD_X": "no_lot"}
    assert em.pop_action_mask_reasons() == {}  # окно закрыто
    em._action_reasons = None  # флаг monitor_action_legality = False
    assert em.pop_action_mask_reasons() == {}


def test_cpp_vecenv_sources_cache_mask_reasons():
    """Кэш причин в CppVecEnv: обновляется там же, где _action_masks."""
    src = (ROOT / "python" / "cpp_vecenv.py").read_text(encoding="utf-8")
    assert "action_mask_reasons_batch" in src, \
        "step_wait/reset должны читать причины из того же action_masks_batch"
    assert re.search(r"def mask_reasons\s*\(", src), \
        "у CppVecEnv нужен свойство mask_reasons (читает EnvManager)"
    assert "_mask_reasons" in src


def test_action_verdict_with_reason_attribute():
    """«Заблокировано: деньги» / «нет участка» / «курикулум» вместо обобщённого."""
    from train_ui2.monitor import action_verdict

    assert action_verdict(0.0, 0.0, "money") == "заблокировано: деньги"
    assert action_verdict(0.0, 0.5, "no_lot") == "заблокировано: нет участка"
    assert action_verdict(0.0, 5.0, "curriculum") == "окно доступности узкое: курикулум"
    # Без причины — прежние строки (контракт 2026-09-23 не меняем задним числом)
    assert action_verdict(0.0, 0.0) == "заблокировано маской"
    assert action_verdict(0.0, 5.0) == "окно доступности узкое"
    # «выбирает» раньше причин: доля есть — вопрос о маске неактуален
    assert action_verdict(1.5, 12.0, "money") == "выбирает"
    # Легальность неизвестна — причина из маски не выдумывается
    assert action_verdict(0.0, None, "money") == "нет данных о доступности"


def test_watch_report_carries_reason_into_verdict():
    from train_ui2.monitor import watch_report

    rows = watch_report({"BUILD_WATERCHANNEL": 0}, {"BUILD_WATERCHANNEL": 0.0},
                        total_actions=1000, names=["BUILD_WATERCHANNEL"],
                        reasons={"BUILD_WATERCHANNEL": "no_lot"})
    assert rows[0]["reason"] == "no_lot"
    assert rows[0]["verdict"] == "заблокировано: нет участка"
    # Без переданной причины — None в строке и прежний вердикт
    rows = watch_report({}, {}, total_actions=0, names=["DAY"])
    assert rows[0]["reason"] is None
    assert rows[0]["verdict"] == "нет данных о доступности"


def test_protocol_roundtrip_action_mask_reasons():
    """Поле протокола: строковые значения, legacy-лог и мусор не валят encode."""
    import json

    from train_ui2 import protocol as P

    msg = P.ProgressMsg(done=1, total=2,
                        action_mask_reasons={"BUILD_WATERCHANNEL": "money"})
    back = P.decode(P.encode(msg))
    assert back.action_mask_reasons == {"BUILD_WATERCHANNEL": "money"}

    # Мусорные значения: None выкидываем, прочее приводим к строке (JSON-safe)
    msg2 = P.ProgressMsg(done=1, total=2,
                         action_mask_reasons={"DAY": None, "WEEK": 5,
                                              "BUILD_X": "no_lot"})
    back2 = P.decode(P.encode(msg2))
    assert back2.action_mask_reasons == {"WEEK": "5", "BUILD_X": "no_lot"}

    # decode: вместо dict — {}
    legacy = json.dumps({"type": "progress", "done": 1, "total": 2,
                         "action_mask_reasons": "oops"})
    assert P.decode(legacy).action_mask_reasons == {}


def test_bars_sources_render_reason_column():
    """Bars обязан уметь колонку причин (set_items + отрисовка)."""
    src = (ROOT / "train_ui2" / "charts.py").read_text(encoding="utf-8")
    assert "reasons" in src, "charts.Bars.set_items должен принимать reasons"
    assert re.search(r"def set_items\s*\(\s*self\s*,\s*items[^)]*reasons", src), \
        "set_items(..., reasons=None) — третий аргумент обязателен"
    assert "_reasons" in src, "причины должны кэшироваться на уровне label'ов"
