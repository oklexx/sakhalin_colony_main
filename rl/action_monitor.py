"""Счётчики легальности действий для UI-монитора (2026-09-23).

Зачем: панель «Активности / Строительство зданий» показывала только долю
нажатий. Если доля равна нулю, невозможно отличить «политика разлюбила
действие» от «действие было нелегальным» (маска = 0). Для водоканала это
принципиально: его окно легальности короткое — участок с водой появляется
только когда туда доехала дорога (tests/cpp/water_mask_check.cpp W3/W6), и
закрывается, как только тайл занят дорогой/постройкой или сгорел
(`destroyed_lots`).

Модуль намеренно без torch: он вызывается на каждом шаге сбора роллаута,
и его стоимость — одно сложение по строкам маски.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class ActionLegalityMonitor:
    """Доля env-шагов, на которых действие было легальным (маска == 1).

    Окно = один роллаут: `pop_percent` отдаёт проценты и обнуляет счётчики,
    поэтому каждое обновление монитора описывает ровно тот роллаут, который
    только что собран (а не последние 1000 действий из 32768, как было
    раньше — отсюда и «мерцание» строк).
    """

    def __init__(self, n_actions: int) -> None:
        self._n = max(0, int(n_actions))
        self._legal = np.zeros(self._n, dtype=np.float64)
        self._steps = 0

    @property
    def steps(self) -> int:
        """Сколько env-шаго́в накоплено в текущем окне (0 после pop)."""
        return int(self._steps)

    def reset(self) -> None:
        self._legal[:] = 0.0
        self._steps = 0

    def add_step(self, masks: Optional[np.ndarray]) -> None:
        """Сложить в счётчик маску [n_envs, n_actions], которую реально видел актор.

        Любое отклонение по форме (нет масок, старый .pyd с другим числом
        действий, 1D-мусор) игнорируется молча: метрика наблюдательная, и
        ронять из-за неё обучение хуже, чем не показать след.
        """
        if self._n <= 0 or masks is None:
            return
        arr = np.asarray(masks)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return
        cols = min(int(arr.shape[1]), self._n)
        if cols <= 0:
            return
        self._legal[:cols] += np.asarray(arr[:, :cols], dtype=np.float64).sum(axis=0)
        self._steps += int(arr.shape[0])

    def pop_percent(self, names: Sequence[str]) -> Dict[str, float]:
        """Проценты легальности по имени действия (0..100) и сброс окна.

        Пустой ответ — признак «окно пустое» (первый шаг или маски не_read_),
        а не «действие недоступно»: UI обязан различать эти состояния.
        """
        out: Dict[str, float] = {}
        if self._steps <= 0:
            return out
        steps = float(self._steps)
        for i, nm in enumerate(list(names)[: self._n]):
            out[str(nm)] = round(float(self._legal[i]) / steps * 100.0, 2)
        self.reset()
        return out
