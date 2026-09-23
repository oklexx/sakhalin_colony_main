"""Lightweight QPainter charts for UI 2.0 (no matplotlib dependency).

Chart keeps rolling history per series, auto-scales, draws grid + legend +
last values. Designed for ~120px tall panels stacked in the Monitoring tab.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from train_ui2.monitor import LEGAL_NARROW_PCT, REASON_RU, fmt_pct, rows_with_sticky
from train_ui2.theme import DIM, FIELD, LINE, TXT, color_for

# Русские названия действий модели (ключ — английское имя из action_names).
_ACTION_RU = {
    "DAY": "День",
    "WEEK": "Неделя",
    "IMPROVE_LAND": "Улучш. землю",
    "REPAIR": "Ремонт",
    "REPAIR_ALL": "Ремонт всех",
    "DEMOLISH": "Снос",
    "PRESERVE": "Законсерв.",
    "UNPRESERVE": "Расконсерв.",
    "SELL_SURPLUS": "Продать излишки",
    "BUY_FOOD": "Купить еду",
    "TAKE_LOAN": "Кредит",
    "REPAY_LOAN": "Погасить кредит",
    "PAY_TAX": "Налог",
    "BUILD_CITY": "Город",
    "BUILD_FARM": "Ферма",
    "BUILD_GARDEN": "Сад",
    "BUILD_WATERCHANNEL": "Водоканал",
    "BUILD_SAWMILL": "Лесопилка",
    "BUILD_COALMINE": "Шахта",
    "BUILD_IRONMINE": "Карьер",
    "BUILD_REFINERY": "Нефтедобыча",
    "BUILD_GOLDMINE": "Золотой прииск",
    "BUILD_POWERSTATION": "Электростанция",
    "BUILD_HYDROSTATION": "Гидростанция",
    "BUILD_ROAD": "Дорога",
    "BUILD_FISH": "Рыбный промысел",
    "BUILD_COALCUT": "Угольный разрез",
    "BUILD_HUNTINGLAND": "Охотничьи угодья",
    "BUILD_COWFARM": "Животноводч. ферма",
    "BUILD_MUSHROOM": "Грибная плантация",
    "BUILD_BIGHOUSE": "Жилой район",
    "BUILD_BIGFARM": "Хозяйство",
    "BUILD_APIARY": "Пасека",
    "BUILD_TORCHLIGHT": "Факел",
    "BUILD_HOthouse": "Теплица",
    "BUILD_SUPERHOUSE": "Жилой центр",
    "BUILD_BIGSAWMILL": "Лесоповал",
    "BUILD_WATERMILL": "Водокачка",
    "BUILD_BIGREFINARY": "Нефтенасос",
    "BUILD_PUERPERAL": "Дом матери и ребёнка",
    "BUILD_BIGIRONMINE": "Катакомбы",
    "BUILD_AIRSTATION": "Ветряная электростанция",
    "BUILD_SMALLATOMSTATION": "Малая АЭС",
    "BUILD_ATOMSTATION": "АЭС",
}

# Заполняется лениво из configs/bases.json (captions на русском).
_BUILD_CAPTION: Dict[str, str] = {}


def _load_build_captions() -> None:
    global _BUILD_CAPTION
    if _BUILD_CAPTION:
        return
    try:
        import json as _json
        from pathlib import Path as _Path
        root = _Path(__file__).resolve().parent.parent
        data = _json.loads((root / "configs" / "bases.json").read_text(encoding="utf-8"))
        for d in data:
            cap = d.get("caption")
            bid = d.get("id")
            if cap and bid:
                _BUILD_CAPTION[bid.upper()] = cap
    except Exception:
        pass


def action_ru(name: str) -> str:
    """Русское название действия (по англ. имени), с fallback на исходное."""
    if name in _ACTION_RU:
        return _ACTION_RU[name]
    n = name.upper()
    if n in _ACTION_RU:
        return _ACTION_RU[n]
    if n.startswith("BUILD_"):
        b = n[len("BUILD_"):]
        if not _BUILD_CAPTION:
            _load_build_captions()
        if b in _BUILD_CAPTION:
            return _BUILD_CAPTION[b]
    # Голый id (без BUILD_): так называются действия в одиночной обёртке
    # python/cpp_env.py — без этой ветки «WaterChannel» и «BUILD_WATERCHANNEL»
    # дали бы две строки на одну шкалу.
    if not _BUILD_CAPTION:
        _load_build_captions()
    if n in _BUILD_CAPTION:
        return _BUILD_CAPTION[n]
    return name


class Chart(QWidget):
    """Multi-series rolling line chart."""

    def __init__(self, title: str = "", maxlen: int = 400, height: int = 130,
                 y_zero_line: bool = False, parent=None):
        super().__init__(parent)
        self.title = title
        self.maxlen = maxlen
        self.series: Dict[str, deque] = {}
        self.y_zero_line = y_zero_line
        self.setMinimumHeight(height)
        self.setMaximumHeight(height + 40)
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())

    def add_series(self, name: str, color: Optional[QColor] = None):
        if name not in self.series:
            self.series[name] = deque(maxlen=self.maxlen)
        return color or color_for(name)

    def push(self, values: Dict[str, float]):
        """Append one sample for each named series (missing keys -> gap)."""
        if not values:
            return
        for name, dq in self.series.items():
            v = values.get(name)
            try:
                v = float(v)
                if v != v or v in (float("inf"), float("-inf")):
                    v = None
            except (TypeError, ValueError):
                v = None
            dq.append(v)
        self.update()

    def clear(self):
        for dq in self.series.values():
            dq.clear()
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(FIELD))

        pad_l, pad_r, pad_t, pad_b = 6, 6, 16 if self.title else 6, 14
        plot = QRectF(pad_l, pad_t, max(10, w - pad_l - pad_r),
                      max(10, h - pad_t - pad_b))

        # collect range
        vmin, vmax = float("inf"), float("-inf")
        n = 0
        for dq in self.series.values():
            n = max(n, len(dq))
            for v in dq:
                if v is None:
                    continue
                vmin = min(vmin, v)
                vmax = max(vmax, v)
        if n == 0 or vmin == float("inf"):
            p.setPen(QColor(DIM))
            p.drawText(plot, Qt.AlignCenter, "нет данных")
            if self.title:
                p.drawText(QRectF(4, 1, w - 8, 14), Qt.AlignLeft | Qt.AlignVCenter,
                           self.title)
            p.end()
            return
        if vmax - vmin < 1e-9:
            vmax += 1.0
            vmin -= 1.0
        span = vmax - vmin
        vmin -= span * 0.08
        vmax += span * 0.08
        if self.y_zero_line and vmin > 0 > vmin - span:
            vmin = min(vmin, 0.0)
            vmax = max(vmax, 0.0)

        # grid + y labels
        p.setPen(QPen(QColor(LINE), 1, Qt.DotLine))
        for i in range(5):
            y = plot.top() + plot.height() * i / 4
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            val = vmax - (vmax - vmin) * i / 4
            p.setPen(QColor(DIM))
            p.drawText(QRectF(2, y - 7, 44, 14), Qt.AlignLeft | Qt.AlignVCenter,
                       _fmt(val))
            p.setPen(QPen(QColor(LINE), 1, Qt.DotLine))
        if self.y_zero_line and vmin < 0 < vmax:
            y0 = plot.top() + plot.height() * (vmax - 0) / (vmax - vmin)
            p.setPen(QPen(QColor(90, 90, 90), 1, Qt.DashLine))
            p.drawLine(QPointF(plot.left(), y0), QPointF(plot.right(), y0))

        # series
        legend_x = plot.left() + 4
        for name, dq in self.series.items():
            if not dq:
                continue
            col = color_for(name)
            pen = QPen(col, 1.6)
            p.setPen(pen)
            path = QPainterPath()
            started = False
            m = len(dq)
            for i, v in enumerate(dq):
                if v is None:
                    started = False
                    continue
                x = plot.left() + plot.width() * (i / max(1, m - 1))
                y = plot.top() + plot.height() * (vmax - v) / (vmax - vmin)
                if not started:
                    path.moveTo(x, y)
                    started = True
                else:
                    path.lineTo(x, y)
            p.drawPath(path)
            # legend chip + last value
            last = next((v for v in reversed(dq) if v is not None), None)
            txt = f"{name} {_fmt(last) if last is not None else '—'}"
            p.setPen(col)
            p.drawText(QRectF(legend_x, 1, 160, 14), Qt.AlignLeft | Qt.AlignVCenter, txt)
            legend_x += 12 + p.fontMetrics().horizontalAdvance(txt)
        if self.title:
            p.setPen(QColor(DIM))
            p.drawText(QRectF(plot.right() - 220, 1, 220, 14),
                       Qt.AlignRight | Qt.AlignVCenter, self.title)
        p.end()


def _fmt(v: float) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if a >= 10_000:
        return f"{v/1000:.1f}k"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


_ROW_MIN_H = 9    # ниже этой высоты строка нечитаема — лучше обрезать список
_LABEL_W = 130    # колонка подписей, как было исторически
_TEXT_W = 112     # колонка «доля · легальность» справа


class Bars(QWidget):
    """Горизонтальные полоски долей действий (мониторинг политики).

    Серый след под полоской = доля шагов роллаута, на которых действие было
    ЛЕГАЛЬНЫМ (маска = 1). Без этой второй шкалы «строка пропала» невозможно
    прочитать: политика разлюбила действие или оно весь роллаут было закрыто
    (для водоканала — не было свободного участка с водой). Закреплённые
    действия (`keep`) не исчезают при нулевой доле, а остаются серой строкой
    с нулём — ровно тот случай, который раньше выглядел как баг мониторинга.
    """

    def __init__(self, title: str = "", height: int = 240,
                 keep: Sequence[str] = (), legend: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.legend = legend
        self.keep: List[str] = list(keep or ())
        # (имя, доля%, легальность% | None, липкая строка?)
        self.items: List[tuple] = []
        # label → русская причина закрытия маски («деньги», «нет участка», …);
        # параллельно items, чтобы не ломать контракт 4-кортежей строк.
        self._reasons: Dict[str, str] = {}
        self.setMinimumHeight(height)
        self.setMaximumHeight(height + 40)

    def _top_offset(self) -> int:
        return 18 if self.title else 2

    def set_items(self, items: Dict[str, float],
                  legality: Optional[Dict[str, float]] = None,
                  reasons: Optional[Dict[str, str]] = None) -> None:
        """Принять доли (и, если есть, легальность с причинами) и перерисовать.

        Сколько строк влезает — решает виджет, а не тренер: отсечка top-15 в
        `rl/async_trainer.py` и вытеснение по индексу действия раньше прятали
        живые строки. Обрезок здесь чисто визуальный (не влезло — не показано),
        но закреплённые `self.keep` строки попадают в список всегда: нулевая
        доля ключевого здания — это диагноз, а не пустое место.

        `reasons` — action_mask_reasons ({action: MASK_REASON_KEY}); для строк
        с узким окном легальности подписывается короткая причина закрытия
        («· деньги»), перевод ярлыка — monitor.REASON_RU.
        """
        avail_h = max(1, self.height() - self._top_offset() - 4)
        max_rows = max(1, avail_h // _ROW_MIN_H)
        rows = rows_with_sticky(items or {}, legality or {},
                                keep=self.keep, max_rows=max_rows)
        merged: Dict[str, tuple] = {}
        merged_reasons: Dict[str, str] = {}
        for name, pct, legal, sticky in rows:
            label = action_ru(name)
            prev = merged.get(label)
            # Две англ. подписи могут сойтись в одну русскую: оставляем строку
            # с большей долей, чтобы панель не «съедала» реальное значение.
            if prev is not None and prev[1] >= pct:
                continue
            merged[label] = (label, pct, legal, sticky)
            ru = REASON_RU.get((reasons or {}).get(name, ""), "")
            if ru:
                merged_reasons[label] = ru
        self.items = list(merged.values())
        self._reasons = merged_reasons
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(FIELD))

        top_offset = self._top_offset()
        if self.title:
            font = p.font()
            font.setBold(True)
            p.setFont(font)
            p.setPen(QColor(TXT))
            p.drawText(QRectF(4, 2, w - 8, 16), Qt.AlignLeft | Qt.AlignVCenter, self.title)
            font.setBold(False)
            p.setFont(font)
            if self.legend:
                p.setPen(QColor(DIM))
                p.drawText(QRectF(w * 0.42, 2, w * 0.58 - 6, 16),
                           Qt.AlignRight | Qt.AlignVCenter, self.legend)

        if not self.items:
            p.setPen(QColor(DIM))
            p.drawText(QRectF(0, top_offset, w, h - top_offset), Qt.AlignCenter, "нет данных")
            p.end()
            return

        avail_h = h - top_offset - 4
        row_h = max(_ROW_MIN_H, min(18, avail_h // max(1, len(self.items))))
        max_pct = max((r[1] for r in self.items), default=1.0) or 1.0
        has_legality = any(r[2] is not None for r in self.items)
        has_reason = bool(self._reasons)
        # Колонка цифр: без легальности второй надписи нет — не съедаем ширину;
        # причина («· нет участка») шире процентов — колонка шире на 40px
        text_w = _TEXT_W if has_legality else 56
        if has_reason:
            text_w += 40
        bar_x = _LABEL_W + 10
        bar_w_total = max(8.0, w - bar_x - text_w - 6)
        for i, (name, pct, legal, sticky) in enumerate(self.items):
            y = top_offset + i * row_h + 2
            row_color = QColor(DIM) if sticky else color_for("return")
            p.setPen(QColor(DIM) if sticky else QColor(TXT))
            p.drawText(QRectF(4, y, _LABEL_W, row_h - 3), Qt.AlignLeft | Qt.AlignVCenter,
                       name[:18])
            if has_legality and legal is not None:
                # «след» легальности: своя шкала 0..100% ширины панели
                p.fillRect(QRectF(bar_x, y + 2, bar_w_total * min(1.0, legal / 100.0),
                                  row_h - 8), QColor(LINE))
            p.fillRect(QRectF(bar_x, y + 2,
                              max(2.0, bar_w_total * (pct / max_pct)), row_h - 8), row_color)
            p.setPen(QColor(DIM))
            txt = fmt_pct(pct)
            rsn = self._reasons.get(name)
            # Причина — только когда маска реально резала (узкое окно):
            # при легальности 100% показывать «· деньги» было бы враньём.
            if rsn and (legal is None or legal < LEGAL_NARROW_PCT):
                if has_legality and legal is not None:
                    txt += f" · {legal:.0f}%"
                txt += f" · {rsn}"
            elif has_legality:
                txt += f" · {legal:.0f}% лег" if legal is not None else " · лег ?"
            p.drawText(QRectF(w - text_w, y, text_w - 6, row_h - 3),
                       Qt.AlignRight | Qt.AlignVCenter, txt)
        p.end()
