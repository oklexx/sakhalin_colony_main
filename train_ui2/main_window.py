"""UI 2.0 main window — with Curriculum tab and unified config.

Layout principles (vs the old UI):
  * thematic tabs instead of one scrolling wall;
  * every metric has exactly ONE home (no duplicated readouts);
  * 22px controls, 11px font, 4px grids — nothing overlaps;
  * live charts + KPI cards + eval tracking;
  * separate state file → runs in parallel with the old UI.
  * NEW: dedicated Curriculum tab with building/resource selection via icons+checkbox
"""
from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from train_ui2 import protocol as P
from train_ui2.models import ModelInfo, ModelRegistry, pick_model_file
from rl.config import RewardConfig as _RC

from train_ui2 import theme as T
from train_ui2.charts import Bars, Chart, action_ru
from train_ui2.monitor import fmt_pct, split_by_panel, watch_report
from train_ui2.controls import ParamGroup, StatCard

CONFIG_PATH = Path.home() / "colony_runs" / "sakhalin_colony_ui2" / "config.json"

# centralised constants (backward compat re-export)
from train_ui2.constants import (
    PARAM_GROUPS,
    REWARD_GROUPS,
    REWARD_FLAGS,
    CURRICULUM_STAGE_MAP,
    ALL_BUILD_IDS,
    BUILD_CAPTIONS,
    RESOURCE_IDS,
    RESOURCE_CAPTIONS,
    BUILD_IMAGE_INDEX,
    KEY_ACTIONS_MONITOR,
    CONFIG_VERSION,
    MECHANIC_IDS,
    MECHANIC_CAPTIONS,
    PRESETS,
    PRESET_ORDER,
)
from train_ui2.icons import (
    building_icon as _building_icon,
    resource_icon as _resource_icon,
    format_steps as _fmt_steps,
    escape_html as _esc,
)

# eval lines look like:
# [Eval @ 1,048,576] days=731.0 people=69.0 bases=5.0 return=-3408.7 score=85.00 ...
_EVAL_RE = re.compile(
    r"\[Eval @ ([\d,]+)\] days=([\d.]+) people=([\d.]+) bases=([\d.]+) "
    r"return=(-?[\d.]+) score=(-?[\d.]+)")
_BEST_RE = re.compile(
    r"\[Best\] Saved best_model\.pt \(score=([\d.]+), days=([\d.]+), bases=([\d.]+)\)")


class MainWindow2(QMainWindow):
    def __init__(self, models_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("Sakhalin Colony — Training UI 2.0")
        self.resize(1240, 940)
        self.registry = ModelRegistry(models_dir)
        self.config: Dict[str, Any] = self._load_state()

        # worker state
        self._train_proc: Optional[subprocess.Popen] = None
        self._msg_file: Optional[str] = None
        self._msg_offset = 0
        self._cmd_file: Optional[str] = None
        self._cfg_tmp: Optional[str] = None
        self._msg_timer = QTimer(self)
        self._msg_timer.timeout.connect(self._poll_worker)
        # watch state
        self._watch_proc: Optional[subprocess.Popen] = None
        self._watch_log: Optional[str] = None
        self._watch_offset = 0
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._poll_watch)
        self._extra_cfg: Dict[str, Any] = {}
        # curriculum state
        self._building_checks: Dict[str, QCheckBox] = {}
        self._resource_checks: Dict[str, QCheckBox] = {}
        self._mechanic_checks: Dict[str, QCheckBox] = {}

        self._build_ui()
        self._restore_state()
        self._refresh_models()

    # ─────────────────────────── UI construction ───────────────────────────

    def _build_ui(self):
        split = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_training(), "Обучение")
        self.tabs.addTab(self._tab_monitor(), "Мониторинг")
        self.tabs.addTab(self._tab_rewards(), "Награды")
        self.tabs.addTab(self._tab_curriculum(), "Курикулум")
        self.tabs.addTab(self._tab_models(), "Модели")
        self.tabs.addTab(self._tab_watch(), "Наблюдение")
        split.addWidget(self.tabs)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(6000)
        self.log_view.setStyleSheet("font-family: Consolas, monospace; font-size:10px;")
        split.addWidget(self.log_view)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([600, 180])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Готово. Канонический профиль наград: v4 (configs/reward_v4.json).")

    # ── Tab: Обучение ──
    def _tab_training(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # run bar
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(T.label("Имя:", T.DIM))
        self.name_edit = QLineEdit()
        self.name_edit.setFixedWidth(150)
        self.name_edit.setPlaceholderText("run_001")
        bar.addWidget(self.name_edit)
        self.btn_start = T.button("▶ Старт", self._start_training, "primary",
                                  "Запустить обучение с текущими параметрами")
        self.btn_pause = T.button("⏸ Пауза", self._toggle_pause)
        self.btn_pause.setEnabled(False)
        self.btn_stop = T.button("■ Стоп", self._stop_training, "danger")
        self.btn_stop.setEnabled(False)
        self.btn_entropy = T.button("Энтропия ×2", self._boost_entropy,
                                    tooltip="Удвоить ent_coef (борьба со схлопыванием)")
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_pause)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_entropy)
        bar.addStretch(1)
        self.run_status = T.label("остановлено", T.DIM)
        bar.addWidget(self.run_status)
        root.addLayout(bar)

        # config save/load unified row
        cfg_bar = QHBoxLayout()
        cfg_bar.setSpacing(6)
        cfg_bar.addWidget(T.label("Конфиг:", T.DIM))
        cfg_bar.addWidget(T.button("💾 Сохранить…", self._save_full_config, tooltip="Сохранить полный конфиг (все вкладки) в JSON"))
        cfg_bar.addWidget(T.button("📂 Загрузить…", self._load_full_config, tooltip="Загрузить полный конфиг из JSON (перезапишет все вкладки)"))
        cfg_bar.addWidget(T.button("↺ Сбросить к v3", self._reset_full_to_v3, tooltip="Сбросить все параметры к дефолтам v3"))
        cfg_bar.addStretch(1)
        cfg_bar.addWidget(T.label("v4 — канонический профиль наград (configs/reward_v4.json)", T.DIM))
        root.addLayout(cfg_bar)

        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m шагов  (%p%)")
        root.addWidget(self.progress)

        # parameter groups in 2 columns
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        cols = QHBoxLayout()
        cols.setSpacing(8)
        col_w: List[QVBoxLayout] = [QVBoxLayout(), QVBoxLayout()]
        self.pgroups: Dict[str, ParamGroup] = {}
        for i, (title, keys) in enumerate(PARAM_GROUPS.items()):
            g = ParamGroup(title, keys)
            g.value_changed.connect(self._on_param_changed)
            self.pgroups[title] = g
            col_w[i % 2].addWidget(g)
        # difficulty / obs_mode / stage rows appended into the Среда group
        env_box = self.pgroups["Среда"]
        grp_grid = env_box.layout()  # QVBoxLayout (outer)
        inner = grp_grid.itemAt(0).widget().layout()  # QGridLayout of group
        row = inner.rowCount()
        inner.addWidget(T.field_label("Сложность", "light = ×2 деньги, без главного налога, до 1950"), row, 0)
        self.cmb_difficulty = T.combo(["normal", "light"], "normal")
        self.cmb_difficulty.currentTextChanged.connect(lambda _: self._on_param_changed())
        inner.addWidget(self.cmb_difficulty, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Режим obs", "flat=MLP 299, minimap=CNN, hybrid=оба"), row, 0)
        self.cmb_obs_mode = T.combo(["flat", "minimap", "hybrid"], "flat")
        self.cmb_obs_mode.currentTextChanged.connect(lambda _: self._on_param_changed())
        inner.addWidget(self.cmb_obs_mode, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Миникарта R", "Радиус миникарты (сетка 2R+1)"), row, 0)
        self.spn_mm_radius = QSpinBox(); self.spn_mm_radius.setRange(4, 32)
        self.spn_mm_radius.setValue(14); self.spn_mm_radius.setFixedWidth(92)
        self.spn_mm_radius.valueChanged.connect(lambda _: self._on_param_changed())
        inner.addWidget(self.spn_mm_radius, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Курикулум", "0 = все здания, 1–3 = ограниченные наборы (см. вкладку Курикулум)"), row, 0)
        stage_bar = QHBoxLayout()
        stage_bar.setSpacing(6)
        stage_bar.setContentsMargins(0, 0, 0, 0)
        self.cmb_stage = T.combo([0, 1, 2, 3], 0)
        self.cmb_stage.currentIndexChanged.connect(lambda _: self._on_param_changed())
        stage_bar.addWidget(self.cmb_stage)
        self.chk_use_curriculum_tab = T.check(
            "Ручной набор из Курикулума", False,
            "Применять здания, отмеченные галочками на вкладке «Курикулум» (unlock_ids).\n"
            "Выключено → работает только выбранный этап 0–3.")
        self.chk_use_curriculum_tab.stateChanged.connect(lambda _: self._on_param_changed())
        stage_bar.addWidget(self.chk_use_curriculum_tab)
        stage_bar.addStretch(1)
        stage_w = QWidget(); stage_w.setLayout(stage_bar)
        inner.addWidget(stage_w, row, 1, Qt.AlignLeft)
        row += 1
        inner.addWidget(T.field_label("Карта", "Каждый старт — новый seed (новая карта)"), row, 0)
        seed_bar = QHBoxLayout()
        seed_bar.setSpacing(4)
        self.btn_seed_dice = T.button("🎲", self._randomize_seed,
                                      tooltip="Случайный seed карты")
        self.chk_rand_seed = T.check("случайный при старте", True,
                                     "Генерировать новый seed при каждом запуске обучения")
        self.chk_rand_seed.stateChanged.connect(lambda _: self._on_param_changed())
        seed_bar.addWidget(self.btn_seed_dice)
        seed_bar.addWidget(self.chk_rand_seed)
        seed_bar.addStretch(1)
        seed_w = QWidget(); seed_w.setLayout(seed_bar)
        inner.addWidget(seed_w, row, 1, Qt.AlignLeft)

        # network + performance group (custom widgets)
        perf = T.group("Сеть и производительность")
        pg = perf.layout()
        pg.addWidget(T.field_label("Слоёв"), 0, 0)
        self.spn_layers = QSpinBox(); self.spn_layers.setRange(1, 4)
        self.spn_layers.setValue(2); self.spn_layers.setFixedWidth(92)
        self.spn_layers.valueChanged.connect(lambda _: self._on_param_changed())
        pg.addWidget(self.spn_layers, 0, 1, Qt.AlignLeft)
        pg.addWidget(T.field_label("Ширина слоя"), 1, 0)
        self.spn_width = QSpinBox(); self.spn_width.setRange(32, 2048)
        self.spn_width.setSingleStep(32); self.spn_width.setValue(256)
        self.spn_width.setFixedWidth(92)
        self.spn_width.valueChanged.connect(lambda _: self._on_param_changed())
        pg.addWidget(self.spn_width, 1, 1, Qt.AlignLeft)
        self.chk_amp = T.check("AMP (bfloat16)", True,
                               "Смешанная точность в update() на CUDA")
        self.chk_amp.stateChanged.connect(lambda _: self._on_param_changed())
        pg.addWidget(self.chk_amp, 2, 0, 1, 2)
        self.chk_compile = T.check("torch.compile", False,
                                   "Только Linux/CUDA; на Windows автооткат в eager")
        self.chk_compile.stateChanged.connect(lambda _: self._on_param_changed())
        pg.addWidget(self.chk_compile, 3, 0, 1, 2)
        pg.addWidget(T.field_label("Потоки C++", "0 = авто"), 4, 0)
        self.spn_cpp_threads = QSpinBox(); self.spn_cpp_threads.setRange(0, 64)
        self.spn_cpp_threads.setValue(0); self.spn_cpp_threads.setFixedWidth(92)
        self.spn_cpp_threads.valueChanged.connect(lambda _: self._on_param_changed())
        pg.addWidget(self.spn_cpp_threads, 4, 1, Qt.AlignLeft)
        col_w[1].addWidget(perf)
        for c in col_w:
            c.addStretch(1)
            w = QWidget(); w.setLayout(c)
            cols.addWidget(w, 1)
        scroll_w = QWidget(); scroll_w.setLayout(cols)
        scroll.setWidget(scroll_w)
        root.addWidget(scroll, 1)
        return page

    # ── Tab: Мониторинг ──
    def _tab_monitor(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        cards = QHBoxLayout()
        cards.setSpacing(6)
        self.card_steps = StatCard("Шаги", "0")
        self.card_fps = StatCard("FPS", "—", T.OK)
        self.card_eps = StatCard("Эпизоды", "0")
        self.card_score = StatCard("Лучший eval score", "—", "#c586c0")
        self.card_days = StatCard("Eval дни (мед.)", "—", T.OK)
        self.card_bases = StatCard("Eval базы (мед.)", "—", T.ACCENT)
        self.card_ent = StatCard("Энтропия", "—", "#9cdcfe")
        self.card_kl = StatCard("KL", "—", T.ERR)
        for c in (self.card_steps, self.card_fps, self.card_eps, self.card_score,
                  self.card_days, self.card_bases, self.card_ent, self.card_kl):
            cards.addWidget(c)
        root.addLayout(cards)

        from PySide6.QtWidgets import QGridLayout
        g = QGridLayout()
        g.setSpacing(6)
        self.chart_fps = Chart("шаги/сек", 600)
        self.chart_fps.add_series("fps")
        self.chart_ret = Chart("return эпизодов (последние 50)", 600, y_zero_line=True)
        for s in ("median", "avg", "min", "max"):
            self.chart_ret.add_series(s)
        self.chart_loss = Chart("loss", 600, y_zero_line=True)
        self.chart_loss.add_series("policy_loss")
        self.chart_loss.add_series("value_loss")
        self.chart_ent = Chart("энтропия / KL", 600)
        self.chart_ent.add_series("entropy")
        self.chart_ent.add_series("kl")
        self.chart_eval = Chart("eval: медиана дней (порог 730)", 200)
        self.chart_eval.add_series("days")
        self.chart_eval.add_series("bases")
        g.addWidget(self.chart_fps, 0, 0)
        g.addWidget(self.chart_ret, 0, 1)
        g.addWidget(self.chart_loss, 1, 0)
        g.addWidget(self.chart_ent, 1, 1)
        g.addWidget(self.chart_eval, 2, 0, 1, 2)
        root.addLayout(g, 1)

        bottom = QHBoxLayout()
        # Серый «след» под полоской = доля шагов, где действие было легально
        # (маска = 1): без неё короткое окно легальности водоканала выглядит
        # как зависание/потеря данных.
        legend = "след = % легальных шагов"
        self.bars_actions = Bars(title="Активности", legend=legend)
        self.bars_build = Bars(title="Строительство зданий", legend=legend,
                               keep=KEY_ACTIONS_MONITOR)
        bottom.addWidget(self.bars_actions, 2)
        bottom.addWidget(self.bars_build, 2)
        side = QVBoxLayout()
        self.lbl_loop = T.label("циклы: нет", T.DIM)
        self.lbl_curric = T.label("курикулум: —", T.DIM, word_wrap=True)
        self.lbl_thresh = T.label("пороги eval: —", T.DIM)
        self.lbl_key_actions = T.label(
            "ключевые действия: —", T.DIM, word_wrap=True, size=10)
        side.addWidget(self.lbl_loop)
        side.addWidget(self.lbl_curric)
        side.addWidget(self.lbl_thresh)
        side.addWidget(self.lbl_key_actions)
        side.addStretch(1)
        bottom.addLayout(side, 1)
        root.addLayout(bottom)
        return page

    # ── Tab: Награды ──
    def _tab_rewards(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.addWidget(T.label(
            "Канонический профиль — configs/reward_v4.json (дефолты rl.config.RewardConfig). "
            "Правки авто-сохраняются.", T.DIM))
        bar.addStretch(1)
        bar.addWidget(T.button("↺ Сбросить к v4", self._reset_rewards_canonical,
                               tooltip="Вернуть всем наградам дефолты канонического профиля v4 (configs/reward_v4.json)"))
        bar.addWidget(T.button("📂 Профиль…", self._load_reward_json, tooltip="Загрузить профиль наград (только награды)"))
        bar.addWidget(T.button("💾 Профиль…", self._save_reward_json, tooltip="Сохранить профиль наград"))
        root.addLayout(bar)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout(); grid.setSpacing(8)
        self.rgroups: Dict[str, ParamGroup] = {}
        for i, (title, keys) in enumerate(REWARD_GROUPS.items()):
            g = ParamGroup(title, keys)
            g.value_changed.connect(self._on_param_changed)
            self.rgroups[title] = g
            grid.addWidget(g, i // 2, i % 2)
        # boolean ablation switches
        flags_box = T.group("Отключение подсистем (абляции)")
        self.rflags: Dict[str, Any] = {}
        for i, (key, label, tip) in enumerate(REWARD_FLAGS):
            chk = T.check(label, False, tip)
            chk.stateChanged.connect(self._on_param_changed)
            self.rflags[key] = chk
            flags_box.layout().addWidget(chk, i, 0, 1, 2)
        grid.addWidget(flags_box, (len(REWARD_GROUPS) + 1) // 2, 1)
        grid.setRowStretch((len(REWARD_GROUPS) + 1) // 2 + 1, 1)
        w = QWidget(); w.setLayout(grid)
        scroll.setWidget(w)
        root.addWidget(scroll, 1)
        return page

    # ── Tab: Курикулум (NEW) ──
    def _tab_curriculum(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # header
        hdr = QLabel(
            "<b>Курикулум</b> — что из игры открыто ИИ на этом этапе обучения.<br>"
            "<span style='color:#8a8a8a'>Простое правило: на старте открыто только то, что нужно для выживания. "
            "Чем меньше лишних действий — тем быстрее ИИ находит правильные. Начните с кнопки «Этап 1» ниже, "
            "а когда он освоится — дообучите в режиме «Этап 2» (вся игра).</span>"
        )
        hdr.setWordWrap(True)
        hdr.setTextFormat(Qt.RichText)
        root.addWidget(hdr)

        # ── Готовые режимы: одно нажатие настраивает всю вкладку ──
        preset_box = T.group("Режим обучения — нажмите одну кнопку, остальное настроится само")
        # QGridLayout группы нельзя передавать родителем в QHBoxLayout —
        # добавляем layout в layout (как rgroup/agroup ниже); регресс:
        # tests/test_ui2_smoke.py (пойман CI, PySide6 ругался на тип).
        prow = QHBoxLayout()
        preset_box.layout().addLayout(prow, 0, 0)
        for pid in PRESET_ORDER:
            p = PRESETS[pid]
            b = T.button(p["title"],
                         lambda _=False, pid=pid: self._apply_preset(pid),
                         "primary" if pid == "stage1" else "")
            b.setToolTip(p["summary"] + "\n\n" + p["detail"])
            b.setMinimumHeight(34)
            prow.addWidget(b)
        self.lbl_preset_status = T.label("", T.DIM, word_wrap=True)
        prow.addWidget(self.lbl_preset_status, 1)
        root.addWidget(preset_box)

        # preset buttons
        preset_bar = QHBoxLayout()
        preset_bar.setSpacing(6)
        preset_bar.addWidget(T.label("Наборы зданий (старые этапы):", T.DIM))
        preset_bar.addWidget(T.button("Этап 0 (все)", lambda: self._curriculum_preset(0)))
        preset_bar.addWidget(T.button("Этап 1", lambda: self._curriculum_preset(1)))
        preset_bar.addWidget(T.button("Этап 2", lambda: self._curriculum_preset(2)))
        preset_bar.addWidget(T.button("Этап 3", lambda: self._curriculum_preset(3)))
        preset_bar.addWidget(T.button("Очистить", self._curriculum_clear))
        preset_bar.addWidget(T.button("Все", self._curriculum_select_all))
        preset_bar.addStretch(1)
        preset_bar.addWidget(T.label("Расписание: шаг → этап (curriculum_schedule)", T.DIM))
        root.addLayout(preset_bar)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget(); lay = QVBoxLayout(inner); lay.setSpacing(10); lay.setContentsMargins(4,4,4,4)

        # Buildings group
        bgroup = T.group(f"Здания курикулума — отмеченные попадут в unlock_ids ({len(ALL_BUILD_IDS)} шт.)")
        bgrid = QGridLayout(); bgrid.setSpacing(4)
        # header legend
        # 4 columns × 8 rows = 32
        cols = 4
        for idx, bid in enumerate(ALL_BUILD_IDS):
            r = idx // cols
            c = idx % cols
            cell = QWidget()
            h = QHBoxLayout(cell); h.setContentsMargins(2,1,2,1); h.setSpacing(4)
            chk = QCheckBox()
            chk.setToolTip(f"{bid} — {BUILD_CAPTIONS.get(bid, bid)}")
            chk.stateChanged.connect(self._on_curriculum_changed)
            self._building_checks[bid] = chk
            # icon
            pm = _building_icon(bid)
            icon_lbl = QLabel()
            if pm is not None:
                icon_lbl.setPixmap(pm)
                icon_lbl.setFixedSize(24,24)
            else:
                icon_lbl.setText("▣")
                icon_lbl.setFixedWidth(16)
            cap = BUILD_CAPTIONS.get(bid, bid)
            txt = QLabel(f"{cap} <span style='color:#8a8a8a'>({bid})</span>")
            txt.setTextFormat(Qt.RichText)
            txt.setToolTip(bid)
            h.addWidget(chk)
            h.addWidget(icon_lbl)
            h.addWidget(txt, 1)
            bgrid.addWidget(cell, r, c)
        bgroup.layout().addLayout(bgrid, 0, 0)
        lay.addWidget(bgroup)

        # Resources group
        rgroup = T.group("Ресурсы курикулума — добыча выбранных ресурсов даёт бонусы (curriculum_resources)")
        rgrid = QGridLayout(); rgrid.setSpacing(4)
        for idx, rid in enumerate(RESOURCE_IDS):
            r = idx // 3
            c = idx % 3
            cell = QWidget()
            h = QHBoxLayout(cell); h.setContentsMargins(2,1,2,1); h.setSpacing(4)
            chk = QCheckBox()
            chk.setChecked(True)
            chk.stateChanged.connect(self._on_curriculum_changed)
            self._resource_checks[rid] = chk
            pm = _resource_icon(rid)
            icon_lbl = QLabel()
            if pm is not None:
                icon_lbl.setPixmap(pm)
                icon_lbl.setFixedSize(20,20)
            else:
                icon_lbl.setText("●")
                icon_lbl.setFixedWidth(12)
            cap = RESOURCE_CAPTIONS.get(rid, rid)
            txt = QLabel(cap)
            h.addWidget(chk)
            h.addWidget(icon_lbl)
            h.addWidget(txt,1)
            rgrid.addWidget(cell, r, c)
        # extra buttons for resources
        res_btn_bar = QHBoxLayout()
        res_btn_bar.addWidget(T.button("Все ресурсы", self._curriculum_resources_all))
        res_btn_bar.addWidget(T.button("Сбросить", self._curriculum_resources_clear))
        res_btn_bar.addStretch(1)
        res_btn_bar.addWidget(T.label("Если ничего не отмечено — бонусы за все ресурсы (как сейчас).", T.DIM))
        rgroup.layout().addLayout(rgrid, 0, 0)
        rgroup.layout().addLayout(res_btn_bar, 1, 0)
        lay.addWidget(rgroup)

        # Actions group (механики): 11 менеджерских кнопок ИИ в 8 группах.
        agroup = T.group("Действия ИИ (кроме строительства) — что разрешено")
        agrid = QGridLayout(); agrid.setSpacing(4)
        for idx, mid in enumerate(MECHANIC_IDS):
            r, c = idx // 2, idx % 2
            cell = QWidget()
            h = QHBoxLayout(cell); h.setContentsMargins(2,1,2,1); h.setSpacing(4)
            chk = QCheckBox()
            cap, tip = MECHANIC_CAPTIONS.get(mid, (mid, mid))
            chk.setToolTip(tip)
            chk.stateChanged.connect(self._on_curriculum_changed)
            self._mechanic_checks[mid] = chk
            txt = QLabel(cap)
            txt.setToolTip(tip)
            h.addWidget(chk)
            h.addWidget(txt, 1)
            agrid.addWidget(cell, r, c)
        agroup.layout().addLayout(agrid, 0, 0)
        agroup.layout().addWidget(T.label(
            "Галочка = ИИ может пользоваться действием. «Продать излишки» и «Кредиты банка» нужны для базовой "
            "экономики; ремонт и консервация пригодятся на этапе 2, когда здания начнут изнашиваться.", 
            T.DIM, word_wrap=True), 1, 0)
        lay.addWidget(agroup)

        # Schedule group
        sgroup = T.group("Расписание курикулума — на каком шаге какой этап включается")
        sgrid = QGridLayout(); sgrid.setSpacing(6)
        self.tbl_curriculum = QTableWidget(0, 2)
        self.tbl_curriculum.setHorizontalHeaderLabels(["шаг (total_timesteps)", "этап (0–3)"])
        self.tbl_curriculum.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_curriculum.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_curriculum.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.tbl_curriculum.setFixedHeight(120)
        sgrid.addWidget(self.tbl_curriculum, 0, 0, 1, 4)
        sgrid.addWidget(T.label("Шаг:", T.DIM), 1, 0)
        self.spn_cur_step = QSpinBox(); self.spn_cur_step.setRange(0, 100_000_000); self.spn_cur_step.setSingleStep(50000); self.spn_cur_step.setValue(200000); self.spn_cur_step.setFixedWidth(110)
        sgrid.addWidget(self.spn_cur_step, 1, 1)
        sgrid.addWidget(T.label("Этап:", T.DIM), 1, 2)
        self.spn_cur_stage = QSpinBox(); self.spn_cur_stage.setRange(0, 3); self.spn_cur_stage.setValue(1); self.spn_cur_stage.setFixedWidth(60)
        sgrid.addWidget(self.spn_cur_stage, 1, 3)
        btn_add = T.button("＋ Добавить", self._curriculum_add_schedule)
        btn_del = T.button("－ Удалить", self._curriculum_del_schedule, "danger")
        sgrid.addWidget(btn_add, 2, 0, 1, 2)
        sgrid.addWidget(btn_del, 2, 2, 1, 2)
        sgrid.addWidget(T.label("Пример: 200000:1, 500000:2, 800000:3 — на 200k откроется этап1, на 500k этап2 и т.д.", T.DIM), 3, 0, 1, 4)
        sgroup.layout().addLayout(sgrid, 0, 0)
        lay.addWidget(sgroup)

        lay.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # footer hint
        root.addWidget(T.label(
            "Совет: не знаете, что выбрать — нажмите «Этап 1 · База и ресурсы» сверху. "
            "Этапы 0–3 ниже — это старые наборы зданий (0 = все 32, 1 = базовые 14, 2 = средние 11, 3 = тяжёлые 7); "
            "они только меняют галочки зданий. Ручной набор зданий действует при включённом чекбоксе "
            "«Ручной набор из Курикулума» на вкладке «Обучение».", T.DIM, word_wrap=True))

        return page

    def _curriculum_preset(self, stage: int):
        # stage 0 = all, 1..3 = cumulative
        target = set()
        if stage == 0:
            target = set(ALL_BUILD_IDS)
        else:
            for s in range(1, stage+1):
                target.update(CURRICULUM_STAGE_MAP.get(s, []))
        for bid, chk in self._building_checks.items():
            chk.blockSignals(True)
            chk.setChecked(bid in target)
            chk.blockSignals(False)
        # also set curriculum_stage combo
        self.cmb_stage.blockSignals(True)
        self.cmb_stage.setCurrentIndex(stage)
        self.cmb_stage.blockSignals(False)
        # пресет = осознанный выбор набора → включаем его применение
        self.chk_use_curriculum_tab.blockSignals(True)
        self.chk_use_curriculum_tab.setChecked(True)
        self.chk_use_curriculum_tab.blockSignals(False)
        self._on_curriculum_changed()
        self.log("info", f"Курикулум: пресет этап {stage} → {len(target)} зданий")

    # ── Готовые режимы («Этап 1 · База и ресурсы» / «Этап 2 · Вся экономика») ──
    def _apply_preset(self, pid: str):
        """Применить готовый режим: здания, ресурсы, действия ИИ, критерии eval.

        Пресеты описаны в train_ui2/constants.py (PRESETS); состав «Этапа 1»
        берётся из того же источника, что и CLI --preset stage1
        (rl/curriculum.py STAGE1_PRESET) — UI и CLI не могут разъехаться.
        """
        p = PRESETS.get(pid)
        if not p:
            return
        buildings = set(p["buildings"]) if p["buildings"] else set(ALL_BUILD_IDS)
        for bid, chk in self._building_checks.items():
            chk.blockSignals(True)
            chk.setChecked(bid in buildings)
            chk.blockSignals(False)
        resources = set(p["resources"]) if p["resources"] else set(RESOURCE_IDS)
        for rid, chk in self._resource_checks.items():
            chk.blockSignals(True)
            chk.setChecked(rid in resources)
            chk.blockSignals(False)
        mechanics = set(p["mechanics"])
        for mid, chk in self._mechanic_checks.items():
            chk.blockSignals(True)
            chk.setChecked(mid in mechanics)
            chk.blockSignals(False)
        # ручной набор применяется; этап 0 (пресет сам задаёт точный список)
        self.chk_use_curriculum_tab.blockSignals(True)
        self.chk_use_curriculum_tab.setChecked(True)
        self.chk_use_curriculum_tab.blockSignals(False)
        self.cmb_stage.blockSignals(True)
        self.cmb_stage.setCurrentIndex(0)
        self.cmb_stage.blockSignals(False)
        # критерии проверки подбираются под задачу этапа
        if pid == "stage1":
            eval_rows = self.pgroups.get("Оценка и сохранение")
            if eval_rows is not None:
                if "eval_min_days" in eval_rows.rows:
                    eval_rows.rows["eval_min_days"].set_value(365.0)
                if "eval_min_bases" in eval_rows.rows:
                    eval_rows.rows["eval_min_bases"].set_value(3)
        self._update_preset_status()
        self._on_curriculum_changed()
        self.log("info", f"Режим обучения: {p['title']} — {p['summary']}")

    def _detect_preset(self) -> Optional[str]:
        """Какому пресету соответствует текущее состояние галочек (или None)."""
        for pid in PRESET_ORDER:
            p = PRESETS[pid]
            want_b = set(p["buildings"]) if p["buildings"] else set(ALL_BUILD_IDS)
            want_r = set(p["resources"]) if p["resources"] else set(RESOURCE_IDS)
            want_m = set(p["mechanics"])
            cur_b = {bid for bid, chk in self._building_checks.items() if chk.isChecked()}
            cur_r = {rid for rid, chk in self._resource_checks.items() if chk.isChecked()}
            cur_m = {mid for mid, chk in self._mechanic_checks.items() if chk.isChecked()}
            if (cur_b == want_b and cur_r == want_r and cur_m == want_m
                    and self.chk_use_curriculum_tab.isChecked()):
                return pid
        return None

    def _update_preset_status(self):
        """Строка состояния под кнопками: что сейчас выбрано, простыми словами."""
        pid = self._detect_preset()
        if pid is not None:
            p = PRESETS[pid]
            self.lbl_preset_status.setText(f"✓ Включено: {p['title']}. {p['summary']}")
        else:
            nb = sum(1 for c in self._building_checks.values() if c.isChecked())
            nr = sum(1 for c in self._resource_checks.values() if c.isChecked())
            nm = sum(1 for c in self._mechanic_checks.values() if c.isChecked())
            self.lbl_preset_status.setText(
                f"Свой набор: {nb} зданий · {nr} ресурсов с бонусом · {nm} действий ИИ. "
                "Нажмите «Этап 1» или «Этап 2», чтобы вернуться к готовым настройкам.")

    def _curriculum_clear(self):
        for chk in self._building_checks.values():
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._on_curriculum_changed()

    def _curriculum_select_all(self):
        for chk in self._building_checks.values():
            chk.blockSignals(True)
            chk.setChecked(True)
            chk.blockSignals(False)
        self._on_curriculum_changed()

    def _curriculum_resources_all(self):
        for chk in self._resource_checks.values():
            chk.blockSignals(True)
            chk.setChecked(True)
            chk.blockSignals(False)
        self._on_curriculum_changed()

    def _curriculum_resources_clear(self):
        for chk in self._resource_checks.values():
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._on_curriculum_changed()

    def _curriculum_add_schedule(self):
        step = self.spn_cur_step.value()
        stage = self.spn_cur_stage.value()
        # insert sorted
        rows = self.tbl_curriculum.rowCount()
        # check duplicate
        for r in range(rows):
            if int(self.tbl_curriculum.item(r,0).text()) == step:
                self.tbl_curriculum.item(r,1).setText(str(stage))
                self._on_curriculum_changed()
                return
        self.tbl_curriculum.insertRow(rows)
        self.tbl_curriculum.setItem(rows, 0, QTableWidgetItem(str(step)))
        self.tbl_curriculum.setItem(rows, 1, QTableWidgetItem(str(stage)))
        # sort by step
        self._sort_curriculum_table()
        self._on_curriculum_changed()

    def _curriculum_del_schedule(self):
        rows = set(idx.row() for idx in self.tbl_curriculum.selectedIndexes())
        for r in sorted(rows, reverse=True):
            self.tbl_curriculum.removeRow(r)
        self._on_curriculum_changed()

    def _sort_curriculum_table(self):
        rows = []
        for r in range(self.tbl_curriculum.rowCount()):
            try:
                step = int(self.tbl_curriculum.item(r,0).text())
                stage = int(self.tbl_curriculum.item(r,1).text())
                rows.append((step, stage))
            except: pass
        rows.sort()
        self.tbl_curriculum.setRowCount(0)
        for step, stage in rows:
            r = self.tbl_curriculum.rowCount()
            self.tbl_curriculum.insertRow(r)
            self.tbl_curriculum.setItem(r,0, QTableWidgetItem(str(step)))
            self.tbl_curriculum.setItem(r,1, QTableWidgetItem(str(stage)))

    def _collect_curriculum_schedule(self) -> List[List[int]]:
        out = []
        for r in range(self.tbl_curriculum.rowCount()):
            try:
                step = int(self.tbl_curriculum.item(r,0).text())
                stage = int(self.tbl_curriculum.item(r,1).text())
                out.append([step, stage])
            except: pass
        out.sort(key=lambda x: x[0])
        return out

    def _set_curriculum_schedule(self, sched: List[List[int]]):
        self.tbl_curriculum.setRowCount(0)
        for step, stage in sorted(sched):
            r = self.tbl_curriculum.rowCount()
            self.tbl_curriculum.insertRow(r)
            self.tbl_curriculum.setItem(r,0, QTableWidgetItem(str(int(step))))
            self.tbl_curriculum.setItem(r,1, QTableWidgetItem(str(int(stage))))

    def _on_curriculum_changed(self, *_):
        # sync stage combo if building selection matches a preset
        # do not auto-change stage if user manually edits — just mark dirty
        checked = any(chk.isChecked() for chk in self._building_checks.values())
        if checked and not self.chk_use_curriculum_tab.isChecked():
            self.chk_use_curriculum_tab.blockSignals(True)
            self.chk_use_curriculum_tab.setChecked(True)
            self.chk_use_curriculum_tab.blockSignals(False)
        self._update_preset_status()
        QTimer.singleShot(100, self._save_state)

    # ── Tab: Модели ──
    def _tab_models(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        bar = QHBoxLayout()
        bar.addWidget(T.button("⟳ Обновить", self._refresh_models))
        bar.addWidget(T.button("👁 Наблюдать", self._watch_selected_model, "primary"))
        bar.addWidget(T.button("🎓 Дообучить", self._finetune_selected,
                               tooltip="Старт с весами выбранной модели (--resume-model)"))
        bar.addWidget(T.button("📂 Папка", self._open_selected_folder))
        bar.addWidget(T.button("🗑 Удалить", self._delete_selected, "danger"))
        bar.addStretch(1)
        root.addLayout(bar)
        self.tbl_models = QTableWidget(0, 7)
        self.tbl_models.setHorizontalHeaderLabels(
            ["имя", "шаги", "eval score", "eval дни", "eval базы", "эпизоды", "создана"])
        self.tbl_models.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_models.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_models.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_models.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_models.verticalHeader().setVisible(False)
        self.tbl_models.itemSelectionChanged.connect(self._sync_watch_model)
        root.addWidget(self.tbl_models, 1)
        return page

    # ── Tab: Наблюдение ──
    def _tab_watch(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.addWidget(T.label("Модель:", T.DIM))
        self.cmb_watch_model = QComboBox()
        self.cmb_watch_model.setMinimumWidth(160)
        bar.addWidget(self.cmb_watch_model)
        bar.addWidget(T.label("Карта (seed):", T.DIM))
        self.spn_watch_seed = QSpinBox()
        self.spn_watch_seed.setRange(0, 999_999_999)
        self.spn_watch_seed.setValue(0)
        self.spn_watch_seed.setFixedWidth(110)
        self.spn_watch_seed.setToolTip("0 = случайная карта при каждом запуске")
        bar.addWidget(self.spn_watch_seed)
        bar.addWidget(T.button("🎲", self._random_watch_seed, tooltip="Случайный seed"))
        bar.addWidget(T.label("Размер:", T.DIM))
        self.spn_watch_map = QSpinBox(); self.spn_watch_map.setRange(100, 500)
        self.spn_watch_map.setValue(280); self.spn_watch_map.setFixedWidth(70)
        self.spn_watch_map.setToolTip("Должен совпадать с тренировочным map_size!")
        bar.addWidget(self.spn_watch_map)
        bar.addWidget(T.label("Скорость:", T.DIM))
        self.cmb_watch_speed = T.combo(["1", "3", "5", "10", "max"], "5")
        self.cmb_watch_speed.setFixedWidth(60)
        bar.addWidget(self.cmb_watch_speed)
        self.chk_watch_visual = T.check("GUI-окно", True,
                                        "raylib-окно игры (нужен sakhalin_colony_gui.exe)")
        bar.addWidget(self.chk_watch_visual)
        self.chk_watch_sample = T.check("Сэмплирование", False,
                                        "Выбирать действия стохастично по вероятностям вместо строгого argmax (как во время обучения)")
        bar.addWidget(self.chk_watch_sample)
        self.btn_watch = T.button("👁 Наблюдать", self._toggle_watch, "primary")
        bar.addWidget(self.btn_watch)
        bar.addStretch(1)
        root.addLayout(bar)

        cards = QHBoxLayout()
        self.w_day = StatCard("День", "—")
        self.w_money = StatCard("Касса", "—", T.OK)
        self.w_people = StatCard("Люди", "—")
        self.w_bases = StatCard("Базы", "—", T.ACCENT)
        self.w_action = StatCard("Действие", "—", "#dcdcaa")
        for c in (self.w_day, self.w_money, self.w_people, self.w_bases, self.w_action):
            cards.addWidget(c)
        root.addLayout(cards)

        self.chart_watch = Chart("награда за шаг", 400, y_zero_line=True)
        self.chart_watch.add_series("reward")
        root.addWidget(self.chart_watch, 1)
        return page

    # ─────────────────────────── state ───────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                from rl.config import Config
                default_cfg = Config().to_dict()
                # preserve curriculum_resources if missing (old config version)
                merged = {**default_cfg, **d}
                if "curriculum_resources" not in merged:
                    merged["curriculum_resources"] = ""
                if "curriculum_schedule" not in merged:
                    merged["curriculum_schedule"] = []
                if "unlock_ids" not in merged:
                    merged["unlock_ids"] = ""
                if "use_curriculum_tab" not in d:
                    # legacy state file: ручной набор уже был → применяем его
                    merged["use_curriculum_tab"] = bool(str(merged["unlock_ids"]).strip())
                # P0: старые прогоны шли с gamma 0.99/0.995/0.997/0.999 — при
                # горизонте эпизода 10 000 шагов это «близоруко» (см. diagnostic).
                if merged.get("gamma") in (0.99, 0.995, 0.997, 0.999):
                    merged["gamma"] = 0.99999
                if merged.get("ent_coef") == 0.05:
                    merged["ent_coef"] = 0.01
                # v3 migration: if file predates v3, ensure food/water 0.8
                if d.get("config_version", 0) < 2:
                    # will be overwritten by Config defaults (0.8) if missing; no need
                    pass
                return merged
            return {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self):
        cfg = self._collect_config()
        cfg["config_version"] = CONFIG_VERSION
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def _restore_state(self):
        cfg = self.config
        self.name_edit.setText(str(cfg.get("model_name", "")))
        for g in list(self.pgroups.values()) + list(self.rgroups.values()):
            g.set_values(cfg)
        for key, chk in self.rflags.items():
            chk.setChecked(bool(cfg.get(key, False)))
        idx = self.cmb_difficulty.findText(str(cfg.get("difficulty", "normal")))
        if idx >= 0:
            self.cmb_difficulty.setCurrentIndex(idx)
        idx = self.cmb_obs_mode.findText(str(cfg.get("obs_mode", "flat")))
        if idx >= 0:
            self.cmb_obs_mode.setCurrentIndex(idx)
        self.spn_mm_radius.setValue(int(cfg.get("minimap_radius", 14)))
        self.cmb_stage.setCurrentIndex(int(cfg.get("curriculum_stage", 0)))
        # legacy-конфиг без флага: включаем, если ручной набор уже был сохранён
        self.chk_use_curriculum_tab.blockSignals(True)
        self.chk_use_curriculum_tab.setChecked(bool(cfg.get(
            "use_curriculum_tab", bool(str(cfg.get("unlock_ids", "")).strip()))))
        self.chk_use_curriculum_tab.blockSignals(False)
        net = cfg.get("net_arch", [256, 256])
        if isinstance(net, list) and net:
            self.spn_layers.setValue(len(net))
            self.spn_width.setValue(int(net[0]))
        self.chk_amp.setChecked(bool(cfg.get("use_amp", True)))
        self.chk_compile.setChecked(bool(cfg.get("torch_compile", False)))
        self.spn_cpp_threads.setValue(int(cfg.get("cpp_threads", 0)))
        self.spn_watch_map.setValue(int(cfg.get("watch_map_size", 280)))
        self.spn_watch_seed.setValue(int(cfg.get("watch_seed", 0)))
        self.chk_watch_visual.setChecked(bool(cfg.get("watch_visual", True)))
        self.chk_watch_sample.setChecked(bool(cfg.get("watch_sample", False)))
        idx = self.cmb_watch_speed.findText(str(cfg.get("watch_speed", "5")))
        if idx >= 0:
            self.cmb_watch_speed.setCurrentIndex(idx)
        # curriculum: buildings
        unlock = str(cfg.get("unlock_ids", "")).strip()
        unlocked = set(s.strip() for s in unlock.split(",") if s.strip())
        # if unlocked empty but stage>0, show preset stage instead of empty
        stage = int(cfg.get("curriculum_stage", 0))
        if not unlocked and stage > 0:
            # populate from preset for visual, but not yet written to unlock_ids
            preset = set()
            for s in range(1, stage+1):
                preset.update(CURRICULUM_STAGE_MAP.get(s, []))
            unlocked = preset
        for bid, chk in self._building_checks.items():
            chk.blockSignals(True)
            chk.setChecked(bid in unlocked)
            chk.blockSignals(False)
        # resources
        cur_res = str(cfg.get("curriculum_resources", "")).strip()
        if cur_res == "":
            # empty = all resources (legacy) → check all
            for chk in self._resource_checks.values():
                chk.blockSignals(True); chk.setChecked(True); chk.blockSignals(False)
        else:
            selected = set(s.strip() for s in cur_res.split(",") if s.strip())
            for rid, chk in self._resource_checks.items():
                chk.blockSignals(True)
                chk.setChecked(rid in selected)
                chk.blockSignals(False)
        # механики: disabled_mechanics → галочки. Отсутствующее поле = легаси
        # (все включены), ровно как в Config.from_dict.
        mech_raw = cfg.get("disabled_mechanics")
        disabled_mech = {str(m) for m in mech_raw} if isinstance(mech_raw, list) else set()
        for mid, chk in self._mechanic_checks.items():
            chk.blockSignals(True)
            chk.setChecked(mid not in disabled_mech)
            chk.blockSignals(False)
        # schedule
        sched = cfg.get("curriculum_schedule", [])
        if isinstance(sched, list):
            # sched is list of [step, stage] or tuples
            clean = []
            for item in sched:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    clean.append([int(item[0]), int(item[1])])
            self._set_curriculum_schedule(clean)
        self._extra_cfg = {k: v for k, v in cfg.items() if k not in (
            "model_name","difficulty","obs_mode","minimap_radius","curriculum_stage",
            "unlock_ids","use_curriculum_tab","curriculum_resources","curriculum_schedule",
            "disabled_mechanics","mechanics_unlock_schedule",
            "net_arch","use_amp","torch_compile","cpp_threads",
            "watch_map_size","watch_seed","watch_visual","watch_speed",
            "config_version"
        ) and k not in self._all_param_keys()}
        self._update_preset_status()

    def _all_param_keys(self):
        keys = set()
        for g in list(self.pgroups.values())+list(self.rgroups.values()):
            keys.update(g.rows.keys())
        keys.update(self.rflags.keys())
        return keys

    def _collect_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = dict(self._extra_cfg)
        params: Dict[str, Any] = {}
        for g in self.pgroups.values():
            params.update(g.values())
        for g in self.rgroups.values():
            params.update(g.values())
        cfg.update(params)
        for key, chk in self.rflags.items():
            cfg[key] = chk.isChecked()
        # curriculum: buildings → unlock_ids
        checked_buildings = [bid for bid, chk in self._building_checks.items() if chk.isChecked()]
        # if all buildings checked → empty means "all" (stage 0 semantics) → store "" to keep unlock_ids clean
        # but we keep explicit list if not all and not empty
        if len(checked_buildings) == len(ALL_BUILD_IDS):
            unlock_ids = ""  # means all unlocked (stage 0)
        else:
            unlock_ids = ",".join(checked_buildings)
        # curriculum resources
        checked_res = [rid for rid, chk in self._resource_checks.items() if chk.isChecked()]
        if len(checked_res) == len(RESOURCE_IDS):
            curriculum_resources = ""  # all resources
        else:
            curriculum_resources = ",".join(checked_res)
        # механики: невыставленные галочки становятся disabled_mechanics
        # (декларативная форма; allow-list выводится в Config.curriculum_state)
        checked_mech = [mid for mid, chk in self._mechanic_checks.items() if chk.isChecked()]
        # schedule
        sched = self._collect_curriculum_schedule()

        cfg.update({
            "model_name": self.name_edit.text().strip(),
            "difficulty": self.cmb_difficulty.currentText(),
            "obs_mode": self.cmb_obs_mode.currentText(),
            "minimap_radius": self.spn_mm_radius.value(),
            "curriculum_stage": self.cmb_stage.currentIndex(),
            "unlock_ids": unlock_ids,
            "use_curriculum_tab": self.chk_use_curriculum_tab.isChecked(),
            "curriculum_resources": curriculum_resources,
            "curriculum_schedule": sched,
            "disabled_mechanics": [m for m in MECHANIC_IDS if m not in checked_mech],
            # UI не редактирует расписание разблокировки механик: пусто =
            # «не трогать» (worker/Config применит свой дефолт для новых прогонов).
            "mechanics_unlock_schedule": [],
            "net_arch": [self.spn_width.value()] * self.spn_layers.value(),
            "use_amp": self.chk_amp.isChecked(),
            "torch_compile": self.chk_compile.isChecked(),
            "cpp_threads": self.spn_cpp_threads.value(),
        })
        # watch prefs live in the same state file but are not sent to the worker
        cfg["watch_map_size"] = self.spn_watch_map.value()
        cfg["watch_seed"] = self.spn_watch_seed.value()
        cfg["watch_visual"] = self.chk_watch_visual.isChecked()
        cfg["watch_sample"] = self.chk_watch_sample.isChecked()
        cfg["watch_speed"] = self.cmb_watch_speed.currentText()
        return cfg

    def _on_param_changed(self, *_a):
        QTimer.singleShot(800, self._save_state)

    def closeEvent(self, ev):
        self._save_state()
        if self._train_proc and self._train_proc.poll() is None:
            ret = QMessageBox.question(
                self, "Обучение идёт",
                "Обучение ещё работает. Остановить и выйти?",
                QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                ev.ignore()
                return
            self._stop_training()
        if self._watch_proc and self._watch_proc.poll() is None:
            # Дерево целиком: raylib-окно — ребёнок watch_champion.py и иначе
            # остаётся висеть после закрытия UI (см. _stop_watch_proc).
            self._stop_watch_proc()
        super().closeEvent(ev)

    # ─────────────────────────── log ───────────────────────────

    def log(self, level: str, text: str):
        color = {"info": T.TXT, "warn": T.WARN, "warning": T.WARN,
                 "error": T.ERR}.get(level, T.TXT)
        ts = time.strftime("%H:%M:%S")
        self.log_view.appendHtml(
            f'<span style="color:{T.DIM}">{ts}</span> '
            f'<span style="color:{color}">{_esc(text)}</span>')
        self.log_view.moveCursor(QTextCursor.End)

    # ─────────────────────────── training ───────────────────────────

    def _randomize_seed(self):
        seed = random.randint(1, 999_999_999)
        self.pgroups["Среда"].rows["seed"].set_value(seed)
        self.log("info", f"Seed карты: {seed}")

    def _start_training(self, resume_model: Optional[Path] = None):
        if self._train_proc and self._train_proc.poll() is None:
            self.log("warn", "Обучение уже запущено")
            return
        if self.chk_rand_seed.isChecked():
            self._randomize_seed()
        cfg = self._collect_config()
        name = cfg.get("model_name") or self._next_run_name()
        cfg["model_name"] = name
        self.name_edit.setText(name)
        if resume_model:
            name = name + "_ft"
            cfg["model_name"] = name
            self.name_edit.setText(name)

        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                          encoding="utf-8")
        json.dump(cfg, tmp, ensure_ascii=False)
        tmp.close()
        self._cfg_tmp = tmp.name
        msg_file = os.path.join(tempfile.gettempdir(),
                                f"colony_ui2_{int(time.time()*1000)}.jsonl")
        cmd_file = os.path.join(tempfile.gettempdir(),
                                f"colony_cmd2_{int(time.time()*1000)}.jsonl")
        self._msg_file, self._msg_offset = msg_file, 0
        self._cmd_file = cmd_file

        args = [sys.executable, "-u",
                str(_PROJECT / "train_ui2" / "worker.py"),
                "--config", tmp.name, "--name", name,
                "--output", msg_file, "--command-file", cmd_file]
        if resume_model:
            args += ["--resume-model", str(resume_model)]
        try:
            self._train_proc = subprocess.Popen(
                args, cwd=str(_PROJECT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self.log("error", f"не удалось запустить worker: {e}")
            return
        self._msg_timer.start(250)
        self._set_running(True)
        for ch in (self.chart_fps, self.chart_ret, self.chart_loss,
                   self.chart_ent, self.chart_eval):
            ch.clear()
        self.log("info", f"Обучение: {name} (pid={self._train_proc.pid})")

    def _next_run_name(self) -> str:
        max_num = 0
        for m in self.registry.scan():
            if m.name.startswith("run_"):
                try:
                    max_num = max(max_num, int(m.name.split("_", 1)[1]))
                except (ValueError, IndexError):
                    pass
        return f"run_{max_num + 1:03d}"

    def _set_running(self, on: bool):
        self.btn_start.setEnabled(not on)
        self.btn_pause.setEnabled(on)
        self.btn_stop.setEnabled(on)
        self.run_status.setText("обучение…" if on else "остановлено")
        self.run_status.setStyleSheet(
            f"color:{T.OK if on else T.DIM}; font-weight:bold; background:transparent;")
        for g in list(self.pgroups.values()) + list(self.rgroups.values()):
            g.set_enabled(not on)

    def _stop_training(self):
        if self._cmd_file and os.path.exists(self._cmd_file):
            try:
                with open(self._cmd_file, "a", encoding="utf-8") as f:
                    f.write(P.encode_command("stop_training") + "\n")
            except OSError:
                pass
        if self._train_proc and self._train_proc.poll() is None:
            self._train_proc.terminate()
        self.log("info", "Остановка обучения…")

    def _toggle_pause(self):
        if not (self._train_proc and self._train_proc.poll() is None):
            return
        paused = self.btn_pause.text().startswith("▶")
        cmd = "resume_training" if paused else "pause_training"
        self._send_command(cmd)
        self.btn_pause.setText("⏸ Пауза" if paused else "▶ Продолжить")

    def _boost_entropy(self):
        self._send_command("boost_entropy", {"factor": 2.0})
        self.log("info", "Команда: ent_coef ×2")

    def _send_command(self, cmd: str, payload: dict = None):
        if not self._cmd_file:
            return
        try:
            with open(self._cmd_file, "a", encoding="utf-8") as f:
                f.write(P.encode_command(cmd, payload or {}) + "\n")
        except OSError as e:
            self.log("error", f"команда не отправлена: {e}")

    def _poll_worker(self):
        # read messages
        if self._msg_file:
            try:
                size = os.path.getsize(self._msg_file)
            except OSError:
                size = -1
            if size > self._msg_offset:
                try:
                    with open(self._msg_file, "r", encoding="utf-8") as f:
                        f.seek(self._msg_offset)
                        data = f.read()
                        self._msg_offset = f.tell()
                except (OSError, UnicodeDecodeError):
                    data = ""
                for line in data.splitlines():
                    line = line.strip()
                    if line:
                        self._handle_msg(line)
        # liveness
        if self._train_proc and self._train_proc.poll() is not None:
            self._msg_timer.stop()
            self._set_running(False)
            self.log("info", f"Worker завершён (code={self._train_proc.returncode})")
            self._refresh_models()
            self._save_state()

    def _handle_msg(self, line: str):
        try:
            msg = P.decode(line)
        except Exception:
            self.log("info", line)
            return
        d = msg.to_dict() if hasattr(msg, "to_dict") else {}
        t = d.get("type")
        if t == "log":
            text = d.get("message", "")
            level = d.get("level", "info")
            self.log(level, text)
            m = _EVAL_RE.search(text)
            if m:
                step = int(m.group(1).replace(",", ""))
                days, people, bases = float(m.group(2)), float(m.group(3)), float(m.group(4))
                score = float(m.group(6))
                self.chart_eval.push({"days": days, "bases": bases})
                self.card_days.set_value(f"{days:.0f}")
                self.card_bases.set_value(f"{bases:.0f}")
                self.card_score.set_value(f"{score:.1f}")
                ok = "PASS" in text
                self.lbl_thresh.setText(f"пороги eval: {'PASS ✅' if ok else 'FAIL ❌'}")
                self.lbl_thresh.setStyleSheet(
                    f"color:{T.OK if ok else T.ERR}; background:transparent;")
            mb = _BEST_RE.search(text)
            if mb:
                self.card_score.set_value(f"{float(mb.group(1)):.1f}", "#c586c0")
        elif t == "progress":
            done, total = d.get("done", 0), d.get("total", 1)
            self.progress.setMaximum(max(1, int(total)))
            self.progress.setValue(int(done))
            self.card_steps.set_value(_fmt_steps(done))
            fps = d.get("fps", 0)
            self.card_fps.set_value(f"{fps:,.0f}")
            self.chart_fps.push({"fps": fps})
            self.card_eps.set_value(str(d.get("episodes", 0)))
            self.chart_ret.push({
                "median": d.get("median_return"), "avg": d.get("avg_return"),
                "min": d.get("min_return"), "max": d.get("max_return")})
            self.chart_loss.push({"policy_loss": d.get("policy_loss"),
                                  "value_loss": d.get("value_loss")})
            ent, kl = d.get("entropy", 0), d.get("kl", 0)
            self.card_ent.set_value(f"{ent:.2f}")
            self.card_kl.set_value(f"{kl:.4f}")
            self.chart_ent.push({"entropy": ent, "kl": kl})
            # Разбор долей — в train_ui2/monitor.py (без Qt, значит с тестами);
            # легальность приходит отдельным полем, чтобы «пропавшая» строка
            # читалась как «закрыто маской», а не как «данных нет».
            actions_dict, build_dict = split_by_panel(d.get("top_actions", {}),
                                                       ALL_BUILD_IDS)
            legality = d.get("action_legality", {}) or {}
            reasons = d.get("action_mask_reasons", {}) or {}
            self.bars_actions.set_items(actions_dict, legality, reasons)
            self.bars_build.set_items(build_dict, legality, reasons)
            self._update_key_actions(d)
            if d.get("loop_detected"):
                self.lbl_loop.setText(
                    f"циклы: {d.get('envs_with_loops', 0)} env ({d.get('loop_action_name') or '?'})")
                self.lbl_loop.setStyleSheet(f"color:{T.ERR}; background:transparent;")
            else:
                self.lbl_loop.setText("циклы: нет")
                self.lbl_loop.setStyleSheet(f"color:{T.DIM}; background:transparent;")
            stage = d.get("curriculum_stage", 0)
            prog = d.get("curriculum_progress_percent", 0) or 0
            if d.get("curriculum_progress_valid", True):
                txt = f"курикулум: этап {stage} ({prog*100:.0f}%)"
            else:
                # Раньше сюда прилетал вечно нулевой прогресс (тренер звал
                # несуществующий метод); «прогресс не измеряется» честнее, чем
                # «0%», а причину (нет расписания этапов / среда не отдаёт
                # метрику) сообщает однократное предупреждение в логе.
                txt = f"курикулум: этап {stage} (прогресс не измеряется)"
            avail = d.get("curriculum_available_actions", "")
            if avail:
                txt += f"\nоткрыто зданий: {avail}"
            self.lbl_curric.setText(txt)
        elif t == "error":
            self.log("error", d.get("message", "ошибка worker"))
        elif t == "done":
            self.log("info", f"Готово: {d.get('total',0):,} шагов за {d.get('time_s',0):.0f}с")
        elif t == "saved":
            self.log("info", f"Сохранено: {d.get('path')}")

    def _update_key_actions(self, d: dict) -> None:
        """Подпись «ключевые действия»: счётчик, доля роллаута, легальность, вердикт.

        Отвечает на ровно тот вопрос, из-за которого панель вообще считали
        сломанной: если «Водоканал» пропал из списка, то его закрыла маска
        (нет денег / нет свободного участка с водой, к которому доехала дорога)
        или политика перестала строить. Строки берутся из KEY_ACTIONS_MONITOR,
        поэтому они не зависят от того, влезло ли действие в видимый список.
        """
        rows = watch_report(
            d.get("action_counts", {}) or {},
            d.get("action_legality", {}) or {},
            KEY_ACTIONS_MONITOR,
            total_actions=int(d.get("action_total_steps", 0) or 0),
            shares=d.get("top_actions", {}) or {},
            reasons=d.get("action_mask_reasons", {}) or {},
        )
        total = int(d.get("action_total_steps", 0) or 0)
        head = (f"ключевые действия (доля из {total:,} шагов роллаута):"
                if total > 0 else "ключевые действия:")
        lines = []
        for row in rows:
            label = action_ru(str(row["name"]))
            share = fmt_pct(float(row["pct"]))
            cnt = row["count"]
            legal = row["legal"]
            if legal is None:
                lines.append(f"{label}: {cnt} ({share}) · легальность не измерялась")
            else:
                lines.append(f"{label}: {cnt} ({share}) · легально {float(legal):.0f}% "
                             f"· {row['verdict']}")
        self.lbl_key_actions.setText(
            head + ("\n" + "\n".join(lines) if lines else ""))

    # ─────────────────────────── rewards tab ───────────────────────────

    def _reset_rewards_canonical(self):
        """Сброс наград к дефолтам rl.config.RewardConfig.

        Дефолты dataclass по золотому правилу совпадают с каноническим профилем
        (сейчас configs/reward_v4.json), так что «канонический» — это не номер
        версии, а DEFAULT_REWARD_PROFILE из rl.config.
        """
        rc = _RC()
        data = rc.to_dict()
        for g in self.rgroups.values():
            g.set_values(data)
        for key, chk in self.rflags.items():
            chk.setChecked(bool(data.get(key, False)))
        self._save_state()
        self.log("info", "Награды сброшены к каноническому профилю v4 (configs/reward_v4.json)")

    # keep aliases for old callers/tests (v2/v3 — исторические названия профиля)
    def _reset_rewards_v3(self):
        self._reset_rewards_canonical()

    def _reset_rewards_v2(self):
        self._reset_rewards_canonical()

    def _load_reward_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Профиль наград", "",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("reward"), dict):
                data = data["reward"]
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Ошибка", str(e))
            return
        for g in self.rgroups.values():
            g.set_values(data)
        for key, chk in self.rflags.items():
            if key in data:
                chk.setChecked(bool(data[key]))
        self._save_state()
        self.log("info", f"Награды загружены: {path}")

    def _save_reward_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить профиль",
                                              "reward_profile.json", "JSON (*.json)")
        if not path:
            return
        data = {}
        for g in self.rgroups.values():
            data.update(g.values())
        for key, chk in self.rflags.items():
            data[key] = chk.isChecked()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            self.log("info", f"Профиль сохранён: {path}")
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    # ─────────────────────────── unified config ───────────────────────────
    def _save_full_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить полный конфиг",
                                              "sakhalin_config.json", "JSON (*.json)")
        if not path:
            return
        cfg = self._collect_config()
        cfg["config_version"] = CONFIG_VERSION
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            self.log("info", f"Конфиг сохранён: {path} (v{CONFIG_VERSION}, {len(cfg)} ключей)")
        except OSError as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _load_full_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить полный конфиг", "",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # merge into current config and restore UI
            self.config = {**self._collect_config(), **data}
            # ensure curriculum keys exist
            if "curriculum_resources" not in self.config:
                self.config["curriculum_resources"] = ""
            if "curriculum_schedule" not in self.config:
                self.config["curriculum_schedule"] = []
            if "unlock_ids" not in self.config:
                self.config["unlock_ids"] = ""
            if "use_curriculum_tab" not in self.config:
                self.config["use_curriculum_tab"] = bool(str(self.config["unlock_ids"]).strip())
            self._restore_state()
            self._save_state()
            self.log("info", f"Конфиг загружен: {path} → все вкладки обновлены")
        except (OSError, json.JSONDecodeError) as e:
            QMessageBox.warning(self, "Ошибка", str(e))

    def _reset_full_to_v3(self):
        ret = QMessageBox.question(self, "Сброс", "Сбросить ВСЕ параметры (обучение+награды+курикулум) к дефолтам v3?\nТекущие настройки будут потеряны.", QMessageBox.Yes | QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        from rl.config import Config
        cfg = Config().to_dict()
        cfg["config_version"] = CONFIG_VERSION
        self.config = cfg
        self._restore_state()
        self._save_state()
        self.log("info", "Все параметры сброшены к v4 (configs/reward_v4.json + дефолты обучения)")

    # ─────────────────────────── models tab ───────────────────────────

    def _refresh_models(self):
        models = self.registry.scan()
        self.tbl_models.setRowCount(0)
        self.cmb_watch_model.clear()
        for m in models:
            best_meta = {}
            bm = m.path / "best_model.meta.json"
            if bm.exists():
                try:
                    best_meta = json.loads(bm.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    best_meta = {}
            row = self.tbl_models.rowCount()
            self.tbl_models.insertRow(row)
            vals = [
                m.name,
                _fmt_steps(m.steps),
                f"{best_meta.get('best_score', 0):.1f}" if best_meta else "—",
                f"{best_meta.get('best_days', 0):.0f}" if best_meta else "—",
                f"{best_meta.get('best_bases', 0):.0f}" if best_meta else "—",
                str(m.episodes),
                m.created.strftime("%d.%m %H:%M") if m.created else "—",
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 0:
                    item.setData(Qt.UserRole, str(m.path))
                self.tbl_models.setItem(row, c, item)
            self.cmb_watch_model.addItem(m.name, str(m.path))

    def _selected_model_path(self) -> Optional[Path]:
        row = self.tbl_models.currentRow()
        if row < 0:
            return None
        item = self.tbl_models.item(row, 0)
        return Path(item.data(Qt.UserRole)) if item else None

    def _sync_watch_model(self):
        p = self._selected_model_path()
        if p:
            idx = self.cmb_watch_model.findData(str(p))
            if idx >= 0:
                self.cmb_watch_model.setCurrentIndex(idx)
            # Auto-sync watch map size with model's trained map_size from meta
            for meta_name in ("best_model.meta.json", "meta.json"):
                mf = p / meta_name
                if mf.exists():
                    try:
                        with open(mf, encoding="utf-8") as f:
                            mdata = json.load(f)
                        ms = mdata.get("map_size")
                        if ms and int(ms) > 0:
                            self.spn_watch_map.setValue(int(ms))
                            break
                    except Exception:
                        pass

    def _watch_selected_model(self):
        row = self.tbl_models.currentRow()
        if row < 0:
            QMessageBox.information(self, "Модели", "Выберите модель в таблице")
            return
        self._sync_watch_model()
        self.tabs.setCurrentIndex(5)
        self._toggle_watch(start=True)

    def _finetune_selected(self):
        p = self._selected_model_path()
        if not p:
            QMessageBox.information(self, "Модели", "Выберите модель")
            return
        model_file = pick_model_file(p)
        if model_file is None:
            QMessageBox.warning(self, "Модели", "Нет final/best/checkpoint .pt")
            return
        self._start_training(resume_model=model_file)

    def _open_selected_folder(self):
        p = self._selected_model_path()
        if p:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _delete_selected(self):
        p = self._selected_model_path()
        if not p:
            return
        ret = QMessageBox.question(self, "Удаление",
                                   f"Удалить модель {p.name} безвозвратно?")
        if ret == QMessageBox.Yes:
            try:
                self.registry.delete(p.name)
                self._refresh_models()
            except (OSError, FileNotFoundError) as e:
                self.log("error", str(e))

    # ─────────────────────────── watch ───────────────────────────

    def _random_watch_seed(self):
        self.spn_watch_seed.setValue(random.randint(1, 999_999_999))

    def _stop_watch_proc(self) -> None:
        """Остановить процесс наблюдения ВМЕСТЕ с raylib-окном.

        `kill()` бьёт только по `watch_champion.py`: окно игры — его ребёнок, и на
        Windows оно переживает родителя (дерево процессов там не убивается).
        Сирота продолжала читать `actions.txt`/писать `state.json`, из-за чего
        следующее наблюдение выглядело «мёртвым». Поэтому — `taskkill /T /F`.
        """
        proc = self._watch_proc
        self._watch_proc = None
        if proc is None:
            return
        if proc.poll() is None:
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=10,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except (OSError, subprocess.SubprocessError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=2.0)
                except (OSError, subprocess.SubprocessError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
        try:
            proc.wait(timeout=3)
        except subprocess.SubprocessError:
            pass

    def _toggle_watch(self, start: bool = False):
        if self._watch_proc and self._watch_proc.poll() is None:
            self._stop_watch_proc()
            self._watch_timer.stop()
            self.btn_watch.setText("👁 Наблюдать")
            self.log("info", "Наблюдение остановлено")
            return
        if not start and self.btn_watch.text().startswith("■"):
            return
        model_dir = self.cmb_watch_model.currentData()
        if not model_dir:
            # Раньше это выглядело как «кнопка ничего не делает»: список пуст,
            # потому что каталог прогона без final_model.pt (прерванное обучение)
            # моделью не считался. Сообщаем, где ищем и что считаем весами.
            from train_ui2.models import MODEL_WEIGHT_NAMES
            msg = (f"Нет моделей для наблюдения.\n\n"
                   f"Каталог: {self.registry.root}\n"
                   f"Моделью считается папка прогона с {' / '.join(MODEL_WEIGHT_NAMES)} "
                   f"или checkpoint_*.pt внутри.\n\n"
                   f"Обучите модель (вкладка «Обучение») или нажмите «⟳ Обновить» "
                   f"на вкладке «Модели», если прогон уже есть.")
            self.log("warn", msg.replace("\n", " "))
            QMessageBox.information(self, "Наблюдение", msg)
            return
        seed = self.spn_watch_seed.value()
        speed_txt = self.cmb_watch_speed.currentText()
        checked_buildings = [bid for bid, chk in self._building_checks.items() if chk.isChecked()]
        if len(checked_buildings) == len(ALL_BUILD_IDS):
            unlock_ids = ""
        else:
            unlock_ids = ",".join(checked_buildings)

        args = [sys.executable, "-u", str(_PROJECT / "watch_champion.py"),
                "--model-dir", str(model_dir),
                "--episodes", "1000000", "--max-steps", "1000000",
                "--device", "cpu",
                "--map-size", str(self.spn_watch_map.value()),
                "--speed", "0" if speed_txt == "max" else speed_txt,
                "--curriculum-stage", str(self.cmb_stage.currentIndex()),
                "--unlock-ids", unlock_ids]
        if seed > 0:
            args += ["--seed", str(seed)]
        log_file = os.path.join(tempfile.gettempdir(),
                                f"colony_watch2_{int(time.time()*1000)}.log")
        args += ["--log-file", log_file]
        if self.chk_watch_visual.isChecked():
            args.append("--visual")
        if self.chk_watch_sample.isChecked():
            args.append("--sample")
        popen_kwargs: Dict[str, Any] = {
            "cwd": str(_PROJECT),
            "stderr": subprocess.STDOUT,
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
        if os.name != "nt":
            # Своя сессия/группа процессов: `_stop_watch_proc` бьёт по дереву
            # (os.killpg), а не только по watch_champion.py — иначе raylib-окно
            # переживает остановку наблюдения и доедает чужой actions.txt.
            popen_kwargs["start_new_session"] = True
        fh = open(log_file, "a", encoding="utf-8")
        try:
            # Ручку закрываем у себя: дочерний процесс уже унаследовал дескриптор,
            # а открытый fd в родителе — утечка на каждый запуск наблюдения.
            self._watch_proc = subprocess.Popen(args, stdout=fh, **popen_kwargs)
        except OSError as e:
            self.log("error", f"не удалось запустить наблюдение: {e}")
            return
        finally:
            fh.close()
        self._watch_log, self._watch_offset = log_file, 0
        self.chart_watch.clear()
        self._watch_timer.start(500)
        self.btn_watch.setText("■ Стоп")
        self.log("info", f"Наблюдение: {Path(model_dir).name} "
                         f"(seed={'случайная' if seed == 0 else seed}, "
                         f"карта {self.spn_watch_map.value()})")

    def _poll_watch(self):
        self._drain_watch_log()
        self._check_watch_exit()

    def _drain_watch_log(self) -> None:
        """Прочитать и показать всё, что дописалось в лог наблюдения."""
        if not self._watch_log:
            return
        try:
            size = os.path.getsize(self._watch_log)
        except OSError:
            size = self._watch_offset
        if size < self._watch_offset:  # файл переписан с нуля
            self._watch_offset = 0
            try:
                size = os.path.getsize(self._watch_log)
            except OSError:
                size = 0
        if size > self._watch_offset:
            try:
                with open(self._watch_log, "r", encoding="utf-8") as f:
                    f.seek(self._watch_offset)
                    data = f.read()
                    tell = f.tell()
            except (OSError, UnicodeDecodeError):
                data, tell = "", self._watch_offset
            nl = data.rfind("\n")
            if data and nl != len(data) - 1:
                # Последняя строка ещё дописывается драйвером: не рвём её пополам
                # (иначе в лог UI попадает обрезок JSON), дочитаем в следующий poll.
                tail = data[nl + 1:]
                data = data[:nl + 1]
                tell -= len(tail.encode("utf-8", "replace"))
            self._watch_offset = tell
            for line in data.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        self.log("info", line)
                        continue
                    if d.get("type") == "step":
                        self.w_day.set_value(str(d.get("day", "—")))
                        self.w_money.set_value(f"{d.get('money', 0):,}")
                        self.w_people.set_value(str(d.get("people", "—")))
                        self.w_bases.set_value(str(d.get("bases", "—")))
                        self.w_action.set_value(str(d.get("action", "—"))[:14])
                        self.chart_watch.push({"reward": d.get("reward", 0)})
                        continue
                    if d.get("type") in ("log", "error", "done"):
                        self.log(d.get("level", "info"), d.get("message", ""))
                        continue
                self.log("info", line)

    def _check_watch_exit(self) -> None:
        """Завершился ли процесс наблюдения; показать код возврата.

        Код возврата — единственная подсказка, когда raylib-окно так и не
        открылось: watch_champion.py пишет диагноз в этот же лог
        (run_visual_watch / describe_gui_failure) и выходит с 1.
        """
        proc = self._watch_proc
        if proc is None or proc.poll() is None:
            return
        rc = proc.returncode
        self._watch_proc = None
        self._watch_timer.stop()
        self.btn_watch.setText("👁 Наблюдать")
        # Процесс уже умер, но его последние строки (traceback!) могли попасть
        # в файл ПОСЛЕ нашего последнего чтения. Без этого добора UI печатал
        # «причина — в строках выше», а самих строк в панели не было.
        self._drain_watch_log()
        if rc not in (0, None):
            self.log("error", f"Наблюдение завершилось с кодом {rc} "
                              f"(причина — в строках выше)")


# _esc imported from train_ui2.icons
