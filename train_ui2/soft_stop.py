"""Мягкая остановка воркера обучения (P1-1 ревью 2026-09-24).

Раньше «Стоп» в UI писал команду `stop_training` и в ту же миллисекунду делал
`terminate()`: трейнер не успевал сохранить `final_model.pt`/`.norm.json`,
воркер — `meta.json`, а `episode_diagnostics.jsonl` оставался без `run_end`.

Теперь порядок такой:
1. UI отправляет `stop_training` и ждёт выхода процесса (индикатор
   «останавливаем…»); трейнер видит команду на ближайшем шаге среды,
   сохраняет модель и завершается, пропуская долгий финальный турнир.
2. Если за `grace_s` процесс не вышел, UI применяет `terminate()` (fallback);
   на POSIX воркер превращает SIGTERM в SystemExit и всё равно пишет `run_end`
   со `status="killed"`.
3. Повторное нажатие «Стоп» во время ожидания — немедленный `terminate()`.

Модуль без Qt: состояние и решения тестируются в песочнице без libGL,
а `main_window` только вызывает `request()`/`should_force()` по таймеру.
"""
from __future__ import annotations

import time
from collections.abc import Callable

#: Сколько ждать мягкой остановки. Хватает на дособранный шаг среды,
#: PPO-апдейт текущего роллаута и сохранение модели; турнир при остановке
#: пользователем не запускается.
STOP_GRACE_S: float = 60.0


class SoftStop:
    """Состояние одного запроса мягкой остановки."""

    def __init__(self, grace_s: float = STOP_GRACE_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.grace_s = float(grace_s)
        self._clock = clock
        self._requested_at: float | None = None
        self.forced = False

    @property
    def pending(self) -> bool:
        """Команда отправлена, процесс ещё не вышел."""
        return self._requested_at is not None

    def request(self) -> bool:
        """Отметить запрос. True — первый (слать команду), False — повторный.

        Повторный запрос означает «не жду, убей сейчас»: вызывающий делает
        `terminate()` и вызывает `mark_forced()`.
        """
        if self._requested_at is None:
            self._requested_at = self._clock()
            return True
        return False

    def elapsed(self) -> float:
        if self._requested_at is None:
            return 0.0
        return max(0.0, self._clock() - self._requested_at)

    def should_force(self) -> bool:
        """Таймаут мягкой остановки истёк, а fallback ещё не применён."""
        return self.pending and not self.forced and self.elapsed() >= self.grace_s

    def mark_forced(self) -> None:
        self.forced = True

    def reset(self) -> None:
        """Процесс завершился — готовы к следующему запуску."""
        self._requested_at = None
        self.forced = False
