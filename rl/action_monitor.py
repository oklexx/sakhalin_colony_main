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

#: Коды MaskReason, порядок ОБЯЗАН совпадать с MASK_REASON_NAMES в
#: include/colony/constants.h — регресс:
#: tests/test_monitor_action_stats.py::test_mask_reason_keys_in_sync_with_cpp
#: (first-fail W2: курикулум → деньги → участок → прочее, open всегда первый).
MASK_REASON_KEYS: Sequence[str] = (
    "open", "curriculum", "money", "no_lot", "other")


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


class MaskReasonMonitor:
    """Доминирующая причина закрытия маски по каждому действию (2026-09).

    Дополняет `ActionLegalityMonitor`: легальность отвечает «ЗАКРЫТО или
    нет», этот класс — «ПОЧЕМУ закрыто» (деньги / нет участка / курикулум,
    коды MaskReason из include/colony/constants.h). Источник —
    `CppVecEnv.mask_reasons`, который C++ пишет тем же проходом action_mask,
    что и бит (docs/MONITOR_ACTIONS_2026_09.md §4).

    Окно = один роллаут: `pop_dominant` отдаёт {имя_действия: ключ_причины}
    и обнуляет счётчики. В ответ попадают только действия, которые ХОТЯ БЫ
    РАЗ были закрыты (открытое все шаги — причины нет, не врём).
    """

    def __init__(self, n_actions: int) -> None:
        self._n = max(0, int(n_actions))
        # по строке на действие, по столбцу на MASK_REASON_KEYS
        self._hist = np.zeros((self._n, len(MASK_REASON_KEYS)), dtype=np.int64)
        self._steps = 0

    @property
    def steps(self) -> int:
        """Сколько env-шаго́в накоплено в текущем окне (0 после pop)."""
        return int(self._steps)

    def reset(self) -> None:
        self._hist[:] = 0
        self._steps = 0

    def add_step(self, reasons: Optional[np.ndarray]) -> None:
        """Сложить коды MaskReason [n_envs, n_actions] из CppVecEnv.mask_reasons.

        Как и у ActionLegalityMonitor: форма мимо / None / коды вне корзин
        игнорируются молча — метрика наблюдательная и не должна ронять шаг
        (старый .pyd под COLONY_ALLOW_STALE_PYD отдаёт None или мусор).
        """
        if self._n <= 0 or reasons is None:
            return
        arr = np.asarray(reasons)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return
        cols = min(int(arr.shape[1]), self._n)
        if cols <= 0:
            return
        codes = arr[:, :cols]
        # one-hot суммирование: codes вне [0, K) не попадают ни в одну корзину
        for r in range(len(MASK_REASON_KEYS)):
            self._hist[:cols, r] += np.count_nonzero(codes == r, axis=0)
        self._steps += int(arr.shape[0])

    def pop_dominant(self, names: Sequence[str]) -> Dict[str, str]:
        """{имя_действия: доминирующая причина закрытия} и сброс окна.

        Доминирующая = максимум счётчика среди закрытых корзин (1..K-1);
        тай-брейк — порядок MASK_REASON_KEYS[1:] (argmax детерминирован).
        Пустой dict = окно пустое или ни одно действие не закрывалось.
        """
        out: Dict[str, str] = {}
        if self._steps <= 0:
            return out
        for i, nm in enumerate(list(names)[: self._n]):
            blocked = self._hist[i, 1:]  # 0-я корзина — open, она не «причина»
            if int(blocked.sum()) <= 0:
                continue
            best = int(np.argmax(blocked))
            out[str(nm)] = MASK_REASON_KEYS[1 + best]
        self.reset()
        return out
