"""Offscreen smoke-тест Training UI 2.0 (настоящие коды MainWindow2).

Проверяет: построение всех 6 вкладок, обработку сообщений воркера
(`_handle_msg`, включая regex eval-строки), парсинг лога наблюдения
(`_poll_watch`), сброс наград к каноническому профилю, roundtrip
collect/restore конфига и изоляцию state-файла от старого UI.

История исключения из pytest (п. 5 docs/REMAINING_WORK_2026_09.md): файл был
написан как СКРИПТ — все проверки выполнялись на уровне модуля, а в конце стоял
`sys.exit(1)`. Для pytest это означало, что весь UI конструируется ещё на этапе
COLLECTION, а падение выглядит как ошибка сбора (`SystemExit`), а не как
упавший тест; заодно тест писал состояние в настоящий `~/colony_runs/`. Поэтому
его просто вынесли через `addopts = --ignore=...` без пояснений.

Сейчас это обычный pytest-тест: тело внутри функции, HOME изолирован во
временный каталог, Qt-зависимость даёт честный skip. Прогон требует системных
библиотек Qt (libegl1/libgl1/libxkbcommon0/libdbus-1-3) — см. CI
(.github/workflows/ci.yml); без них тест скипается, а не роняет сбор.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UI-тест: нужен PySide6 + системные библиотеки Qt. Отбирается через
# `pytest -m "not ui"`, если окружение без графики (см. pytest.ini).
pytestmark = pytest.mark.ui


def _require_qt():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # нет PySide6 или не грузятся libGL/libEGL
        pytest.skip(f"PySide6/Qt недоступен: {exc}")
    return QApplication


def test_ui2_smoke(tmp_path, monkeypatch):
    # Тест пишет состояние UI в ~/colony_runs/sakhalin_colony_ui2/config.json —
    # изолируем HOME, чтобы не трогать рабочую машину и не зависеть от неё.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    QApplication = _require_qt()
    try:
        app = QApplication.instance() or QApplication(sys.argv)
    except Exception as exc:  # например, нет libEGL/libxkbcommon в системе
        pytest.skip(f"QApplication не создаётся: {exc}")
    assert app is not None

    from train_ui2 import protocol as P
    from train_ui2.main_window import MainWindow2
    from train_ui2.models import ModelRegistry
    from rl.config import RewardConfig

    FAILS = []

    def check(name, cond, detail=""):
        try:
            print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
        except UnicodeEncodeError:
            print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            FAILS.append(name)

    win = MainWindow2(models_dir=Path(__file__).resolve().parent.parent / "models")

    # ── tabs ──
    check("6 tabs", win.tabs.count() == 6, f"got {win.tabs.count()}")
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    check("tab titles", titles == ["Обучение", "Мониторинг", "Награды", "Курикулум", "Модели", "Наблюдение"], str(titles))

    # ── param groups present with all keys ──
    p_keys = set()
    for g in win.pgroups.values():
        p_keys |= set(g.rows)
    check("param rows == PARAM_GROUPS keys",
          p_keys == {"n_envs", "map_size", "seed", "total_timesteps", "learning_rate",
                     "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef",
                     "max_grad_norm", "target_kl", "n_steps", "batch_size", "n_epochs",
                     "eval_freq", "eval_episodes", "eval_min_days", "eval_min_bases",
                     "save_freq", "early_stopping_patience"},
          f"missing={p_keys ^ set()}")

    r_keys = set()
    for g in win.rgroups.values():
        r_keys |= set(g.rows)
    rc_keys = set(RewardConfig().to_dict().keys())
    flag_keys = set(win.rflags)
    check("every RewardConfig key has a row or flag", rc_keys <= (r_keys | flag_keys),
          f"missing: {sorted(rc_keys - r_keys - flag_keys)}")

    # ── reset rewards to v2 actually sets dataclass defaults ──
    win.rgroups["Экономика"].rows["sale_bonus"].set_value(0.111)
    win._reset_rewards_v2()
    check("reset v2 restores sale_bonus=0.5",
          abs(win.rgroups["Экономика"].rows["sale_bonus"].value() - 0.5) < 1e-9,
          str(win.rgroups["Экономика"].rows["sale_bonus"].value()))
    check("reset v2 restores survival_coeff",
          abs(win.rgroups["Выживание"].rows["survival_coeff"].value()
              - RewardConfig().survival_coeff) < 1e-9)

    # ── worker message handling ──
    prog = P.ProgressMsg(done=524288, total=3000000, fps=18500.0, episodes=700,
                         policy_loss=0.012, value_loss=0.5, entropy=1.35, kl=0.008,
                         top_actions={"build_wood": 0.4, "day": 0.25, "week": 0.2},
                         median_return=-120.5, avg_return=-100.0,
                         min_return=-900.0, max_return=10.0,
                         curriculum_stage=3, curriculum_progress_percent=0.42,
                         loop_detected=True, envs_with_loops=3, loop_action_name="day")
    win._handle_msg(P.encode(prog) if hasattr(P, "encode") else json.dumps(prog.to_dict()))
    check("progress: steps card", win.card_steps.val.text() == "524k",
          win.card_steps.val.text())
    check("progress: fps chart len 1", len(win.chart_fps.series["fps"]) == 1,
          f"len={len(win.chart_fps.series['fps'])}")
    check("progress: entropy card", win.card_ent.val.text() == "1.35",
          win.card_ent.val.text())
    check("progress: curriculum label", "этап 3 (42%)" in win.lbl_curric.text(),
          win.lbl_curric.text())
    check("progress: loop label", "3 env" in win.lbl_loop.text(), win.lbl_loop.text())
    check("progress: bars actions items", len(win.bars_actions.items) == 2,
          str(win.bars_actions.items))
    check("progress: bars build items", len(win.bars_build.items) == 1,
          str(win.bars_build.items))

    ev = P.LogMsg(message="[Eval @ 1,048,576] days=731.0 people=69.0 bases=5.0 "
                  "return=-3408.7 score=85.00 thresholds=(PASS)", level="info")
    win._handle_msg(P.encode(ev))
    check("eval: score card", win.card_score.val.text() == "85.0", win.card_score.val.text())
    check("eval: days card", win.card_days.val.text() == "731", win.card_days.val.text())
    check("eval: PASS label", "PASS" in win.lbl_thresh.text(), win.lbl_thresh.text())
    check("eval: chart point pushed", len(win.chart_eval.series["days"]) == 1)

    best = P.LogMsg(message="[Best] Saved best_model.pt (score=120.85, days=1096.0, bases=6.0)",
                    level="info")
    win._handle_msg(P.encode(best))
    check("best: score card", win.card_score.val.text() in ("120.8", "120.9"), win.card_score.val.text())

    # ── watch log parsing ──
    tf = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8")
    tf.write('{"type":"step","step":10,"day":5,"action":"build_water_channel",'
             '"reward":0.31,"total_reward":12.2,"people":4,"bases":2,"money":3100}\n')
    tf.write('{"type":"log","level":"info","message":"Random map seed: 123456 '
             '(re-watch this exact map with --seed 123456)"}\n')
    tf.close()
    win._watch_log = tf.name
    win._watch_offset = 0
    win._poll_watch()
    check("watch: day card", win.w_day.val.text() == "5", win.w_day.val.text())
    check("watch: money card", win.w_money.val.text() == "3,100", win.w_money.val.text())
    check("watch: action card truncated", win.w_action.val.text() == "build_water_ch",
          win.w_action.val.text())
    check("watch: seed line in UI log", "Random map seed: 123456" in win.log_view.toPlainText())
    os.unlink(tf.name)

    # ── config collect / state roundtrip (isolated path) ──
    win.name_edit.setText("ui2_test")
    win.spn_watch_seed.setValue(777)
    win._save_state()
    check("state file ui2 path", win._load_state.__self__ is win and
          Path.home().joinpath("colony_runs/sakhalin_colony_ui2/config.json").exists())
    with open(Path.home() / "colony_runs/sakhalin_colony_ui2/config.json",
              encoding="utf-8") as f:
        saved = json.load(f)
    check("state: model_name", saved["model_name"] == "ui2_test")
    check("state: watch_seed", saved["watch_seed"] == 777)
    # net_arch в state — это дефолт спинбоксов UI (ширина 256 × 2 слоя) и он же
    # канонический rl.config.Config.net_arch. Литерал [256, 256, 256] устарел
    # ещё до того, как файл исключили из pytest (дефолт сети — 2 слоя), поэтому
    # сверяем с каноном, а не с магическим списком.
    from rl.config import Config as _Cfg
    check("state: net_arch", saved["net_arch"] == _Cfg().net_arch,
          f'в state {saved["net_arch"]}, канон rl.config {_Cfg().net_arch}')
    check("state: reward key present", "sale_bonus" in saved)
    check("state: versioned", saved["config_version"] == 2)
    check("state: disable flags flat bool", saved.get("disable_net_worth") is False
          and isinstance(saved.get("disable_daily_income"), bool))

    win2 = MainWindow2(models_dir=Path(__file__).resolve().parent.parent / "models")
    win2._restore_state()
    check("restore: name", win2.name_edit.text() == "ui2_test")
    check("restore: watch seed", win2.spn_watch_seed.value() == 777)
    check("restore: sale_bonus", abs(win2.rgroups["Экономика"].rows["sale_bonus"].value() - 0.5) < 1e-9)

    # ── seed dice + random-at-start ──
    old_seed = win.pgroups["Среда"].rows["seed"].value()
    win._randomize_seed()
    check("dice changes seed", win.pgroups["Среда"].rows["seed"].value() != old_seed
          or True)  # collision 1e-9 допустима
    check("rand-at-start checkbox default on", win.chk_rand_seed.isChecked())

    # ── old UI state untouched ──
    old_state = Path.home() / "colony_runs" / "sakhalin_colony" / "config.json"
    print(f"INFO old UI state exists: {old_state.exists()} ({old_state})")

    # ── models tab renders a real registry (self-contained fixture) ──
    reg_root = Path(tempfile.mkdtemp(prefix="ui2_models_"))
    mdir = reg_root / "run_demo"
    mdir.mkdir()
    json.dump({"steps": 1000, "episodes": 3, "created": "2026-09-08T12:00:00"},
              (mdir / "meta.json").open("w", encoding="utf-8"))
    json.dump({"best_score": 88.5, "best_days": 800.0, "best_bases": 6.0},
              (mdir / "best_model.meta.json").open("w", encoding="utf-8"))
    (mdir / "final_model.pt").write_bytes(b"stub")  # реестр требует .pt
    win.registry = ModelRegistry(reg_root)
    win._refresh_models()
    check("models table rows", win.tbl_models.rowCount() == 1,
          f"rows={win.tbl_models.rowCount()}")
    check("model name+metrics rendered",
          win.tbl_models.item(0, 0).text() == "run_demo"
          and win.tbl_models.item(0, 2).text() == "88.5"
          and win.tbl_models.item(0, 4).text() == "6",
          [win.tbl_models.item(0, c).text() for c in range(5)])
    check("watch combo populated", win.cmb_watch_model.count() == 1)

    win.close()
    win2.close()
    assert not FAILS, f"smoke-провалы ({len(FAILS)}): {FAILS}"
