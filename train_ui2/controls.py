"""Compact bound controls for UI 2.0.

A ParamRow binds one spinbox to a config key. Unlike the old UI it renders
label+spin in a single 22px row with no per-row ×2/÷2 buttons (those were the
"enormous buttons" complaint) — scaling moved to two buttons per group.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QWidget

from train_ui2 import theme as T
from train_ui2.parameter_widget import PARAM_SPECS, REWARD_SPECS, ParamSpec
from train_ui2.constants import HUMAN_REWARD_HELP

SPECS: Dict[str, ParamSpec] = {s.key: s for s in PARAM_SPECS + REWARD_SPECS}


class ParamRow(QWidget):
    value_changed = Signal(str, object)

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        spec = SPECS[key]
        self.key = key
        self.spec = spec
        lay = QGridLayout(self)
        lay.setSpacing(3)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setColumnMinimumWidth(0, 96)
        lay.setColumnStretch(1, 1)
        lay.addWidget(T.field_label(spec.label, spec.tooltip), 0, 0)
        decimals = spec.decimals if spec.decimals > 0 else T.decimals_for(spec.min)
        self.spin = T.spin(spec.min, spec.max, spec.default, spec.is_int,
                           step=spec.step, decimals=decimals,
                           tooltip=spec.tooltip)
        self.spin.valueChanged.connect(self._emit)
        lay.addWidget(self.spin, 0, 1, Qt.AlignLeft)
        # Пояснение простыми словами (глоссарий train_ui2/constants.py):
        # подписи вроде «Бонус закрытия потребности» звучит программистски,
        # одна строка серым под параметром снимает вопросы «что это вообще».
        human = HUMAN_REWARD_HELP.get(key)
        if human:
            hint = T.label(human, T.DIM, size=10, word_wrap=True)
            lay.addWidget(hint, 1, 0, 1, 2)

    def _emit(self, v):
        self.value_changed.emit(self.key, self.value())

    def value(self) -> Any:
        return int(self.spin.value()) if self.spec.is_int else float(self.spin.value())

    def set_value(self, v: float):
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)

    def scale(self, factor: float):
        self.set_value(_scaled(self.value(), factor, self.spec.is_int, self.spec.min))

    def set_enabled(self, on: bool):
        self.spin.setEnabled(on)


def _scaled(value: float, factor: float, is_int: bool, min_value: float) -> float:
    v = value * factor
    if is_int:
        return max(int(min_value), int(round(v)))
    return max(float(min_value), round(v, 8))


class ParamGroup(QWidget):
    """Themed group of ParamRows in a 2-column grid + ×2/÷2 for the group."""

    def __init__(self, title: str, keys: List[str], tooltip: str = "", parent=None):
        super().__init__(parent)
        self.rows: Dict[str, ParamRow] = {}
        box = T.group(title)
        grid = box.layout()
        half = (len(keys) + 1) // 2
        for i, key in enumerate(keys):
            if key not in SPECS:
                continue
            row = ParamRow(key)
            row.value_changed.connect(self._changed)
            self.rows[key] = row
            grid.addWidget(row, i % half, (i // half) * 2)
            grid.addWidget(QWidget(), i % half, (i // half) * 2 + 1)
        outer = T.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(box)
        if tooltip:
            box.setToolTip(tooltip)

    value_changed = Signal(str, object)

    def _changed(self, key, val):
        self.value_changed.emit(key, val)

    def values(self) -> Dict[str, Any]:
        return {k: r.value() for k, r in self.rows.items()}

    def set_values(self, data: Dict[str, Any]):
        for k, r in self.rows.items():
            v = data.get(k)
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            if v is None:
                continue
            try:
                r.set_value(float(v))
            except (TypeError, ValueError):
                pass

    def scale_all(self, factor: float):
        for r in self.rows.values():
            r.scale(factor)

    def set_enabled(self, on: bool):
        for r in self.rows.values():
            r.set_enabled(on)


class StatCard(QWidget):
    """One compact KPI tile: caption + big value + optional delta."""

    def __init__(self, caption: str, value: str = "—", color: str = T.TXT, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(44)
        self.setMaximumHeight(52)
        lay = T.QVBoxLayout(self)
        lay.setContentsMargins(7, 4, 7, 4)
        lay.setSpacing(0)
        self.cap = T.label(caption.upper(), T.DIM, size=9)
        self.val = T.label(value, color, bold=True, size=15)
        lay.addWidget(self.cap)
        lay.addWidget(self.val)
        self.setStyleSheet(
            f"background:{T.PANEL}; border:1px solid {T.LINE}; border-radius:4px;")

    def set_value(self, text: str, color: Optional[str] = None):
        self.val.setText(text)
        if color:
            self.val.setStyleSheet(
                f"color:{color}; font-size:15px; font-weight:bold; background:transparent;")
