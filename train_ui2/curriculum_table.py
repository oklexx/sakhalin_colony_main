"""Разбор таблицы расписания курикулума (вкладка «Курикулум», P2-1 ревью 2026-09-24).

Раньше `_sort_curriculum_table`/`_collect_curriculum_schedule` разбирали строки
под голым `except: pass`: строка с опечаткой («200к», пустая ячейка, этап 7)
молча исчезала из расписания, и пользователь видел «курикулум как будто не
применился» (тот же класс, что в docs/CURRICULUM_FIX_2026_09.md). Кроме того,
голый `except` глотал KeyboardInterrupt/SystemExit.

Здесь — чистая функция без Qt: разбор + список понятных причин пропуска.
UI показывает причины в статус-баре и логе, тесты гоняют её в песочнице.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

Cell = Optional[str]


@dataclass
class ScheduleParse:
    """Результат разбора: валидные пары и предупреждения (пропуски, повторы)."""

    rows: List[List[int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _parse_int(text: Cell, what: str) -> Tuple[Optional[int], Optional[str]]:
    if text is None or not str(text).strip():
        return None, f"пустое поле «{what}»"
    raw = str(text).strip().replace("_", "").replace(" ", "").replace("\u00a0", "")
    try:
        return int(raw), None
    except ValueError:
        return None, f"«{what}» = {str(text).strip()!r} — не целое число"


def parse_schedule_rows(cells: Iterable[Sequence[Cell]], max_stage: int) -> ScheduleParse:
    """Разобрать строки таблицы `(шаг, этап)` → отсортированное расписание.

    * номера строк в сообщениях — с 1, как их видит пользователь;
    * шаг ≥ 0, этап в 0..max_stage (max_stage — из `rl.curriculum.STAGE_MAP`);
    * повтор шага: побеждает последняя строка (как при правке ячейки), о чём
      тоже сообщается — иначе «исчезнувшая» строка снова выглядела бы багом.
    """
    out = ScheduleParse()
    by_step: dict = {}
    for n, row in enumerate(cells, start=1):
        step_txt = row[0] if len(row) > 0 else None
        stage_txt = row[1] if len(row) > 1 else None
        step, err_step = _parse_int(step_txt, "шаг")
        stage, err_stage = _parse_int(stage_txt, "этап")
        errors = [e for e in (err_step, err_stage) if e]
        if step is not None and step < 0:
            errors.append(f"шаг {step} < 0")
        if stage is not None and not 0 <= stage <= max_stage:
            errors.append(f"этап {stage} вне 0–{max_stage}")
        if errors:
            out.warnings.append(f"строка {n}: " + "; ".join(errors))
            continue
        assert step is not None and stage is not None
        if step in by_step:
            out.warnings.append(
                f"строка {n}: шаг {step} повторяется — взят этап {stage} "
                f"(вместо {by_step[step]})")
        by_step[step] = stage
    out.rows = [[s, by_step[s]] for s in sorted(by_step)]
    return out
