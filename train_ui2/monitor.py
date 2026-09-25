"""Чистая (без Qt) логика панели действий во вкладке «Мониторинг».

Вынесено из `main_window._handle_msg` и `charts.Bars` по двум причинам:

1. Тестируемость без PySide6/Qt — CI и `make test` прогоняют это headless.
2. Правило «одного источника»: как именно отличать «действие нельзя» от
   «политика его не выбирает» должно жить в одном месте, а не в двух
   (тренер резал top-15, а UI ещё раз резал и переводил имена).

Данные приходят из `rl/async_trainer.py`: `top_actions` (доли шагов роллаута),
`action_counts` (сырые счётчики) и `action_legality` (доля шагов, на которых
действие было легально, т.е. маска = 1). Именно нехватка последнего поля и
делала поведение «водоканал появился и пропал» нечитаемым.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypedDict

# Ключи действий-построек, как их отдаёт среда (`python/cpp_vecenv.py`:
# "BUILD_" + id.upper()). Строгое сравнение с ALL_BUILD_IDS раньше было
# регистрозависимым, и «голые» id (WATERCHANNEL) уезжали в панель «Активности».
_BUILD_MARKERS: tuple[str, ...] = ("BUILD:", "A_BUILD", "BUILD_")

# Пороги вердикта (в процентах шагов роллаута). Это эвристика подписи, а не
# параметр обучения: менять их в Config нельзя, там живут только режимы сбора.
_LEGAL_NONE_PCT: float = 1.0   # ниже — действие практически всегда закрыто маской
LEGAL_NARROW_PCT: float = 25.0  # ниже — окно доступности узкое, доля закономерно скачет
# (публичный — charts.py показывает колонку причин только при узком окне)

# Русские ярлыки причин закрытия маски. Ключи — из rl/action_monitor.
# MASK_REASON_KEYS (= MASK_REASON_NAMES в include/colony/constants.h, сверка
# test_mask_reason_keys_in_sync_with_cpp): деньги / нет участка / курикулум.
REASON_RU: dict[str, str] = {
    "curriculum": "курикулум",
    "money": "деньги",
    "no_lot": "нет участка",
    "other": "иное",
    "open": "открыто",
}


def is_build_action(name: str, build_ids: Iterable[str] | None = None) -> bool:
    """True, если действие — постройка (по имени действия или по id здания)."""
    u = str(name).upper()
    if any(marker in u for marker in _BUILD_MARKERS):
        return True
    if build_ids:
        return u in {str(b).upper() for b in build_ids}
    return False


def split_by_panel(top_actions: dict[str, float],
                   build_ids: Iterable[str] | None = None,
                   ) -> tuple[dict[str, float], dict[str, float]]:
    """Разделить `top_actions` на («Активности», «Строительство зданий»)."""
    ids_upper = {str(b).upper() for b in (build_ids or [])}
    actions: dict[str, float] = {}
    builds: dict[str, float] = {}
    for key, value in (top_actions or {}).items():
        try:
            pct = float(value)
        except (TypeError, ValueError):
            # Мусор в сообщении не должен ронять обработку метрик всей
            # панели: строку просто не показываем (кривой ключ виден в логе).
            continue
        if is_build_action(key, ids_upper):
            builds[key] = pct
        else:
            actions[key] = pct
    return actions, builds


def action_verdict(pct: float, legal_pct: float | None,
                   reason: str | None = None) -> str:
    """Одна фраза диагностики: действие закрыто маской или его не выбирают.

    `legal_pct` — доля шагов, на которых действие было ДОСТУПНО (маска = 1),
    а не доля легальных выборов: выбрать закрытое действие политика не может
    вовсе (маска зануляет логит). Отсюда подписи «доступно …» (P3-7 ревью
    2026-09-24); имя поля протокола `action_legality` оставлено ради
    совместимости.

    `legal_pct is None` — легальность не измерялась (нет масок/флаг выключен):
    выдумывать «заблокировано» нельзя, поэтому вердикт пустой либо честное
    «нет данных о доступности».

    `reason` — доминирующая причина закрытия из action_mask_reasons
    (MASK_REASON_KEYS): «заблокировано маской» уточняется до
    «заблокировано: деньги / нет участка / курикулум» (docs/MONITOR_ACTIONS_2026_09.md §4).
    Без причины (None/неизвестный ключ) — прежние строки без изменений.
    """
    if legal_pct is None:
        return "" if pct > 0.0 else "нет данных о доступности"
    if pct > 0.0:
        return "выбирает"
    ru = REASON_RU.get(reason or "", "")
    if legal_pct < _LEGAL_NONE_PCT:
        return f"заблокировано: {ru}" if ru else "заблокировано маской"
    if legal_pct < LEGAL_NARROW_PCT:
        return f"окно доступности узкое: {ru}" if ru else "окно доступности узкое"
    return "доступно, но не выбирает"


def rows_with_sticky(items: dict[str, float], legality: dict[str, float],
                     keep: Sequence[str] = (), max_rows: int = 0,
                     ) -> list[tuple[str, float, float | None, bool]]:
    """Строки панели: (имя, доля%, легальность% | None, липкая?).

    `keep` — закрепленные действия (ключевые здания этапа). Они попадают в
    список ВСЕГДА: иначе низкодольное здание снова «исчезает», стоит набраться
    более частым действиям — ровно та жалоба, из-за которой всё затевалось.
    Остальные места добиваются по убыванию доли, хвост отрезается.

    `sticky=True` у строки значит «доли нет, но она закреплена» — панель рисует
    такие серыми. `max_rows <= 0` — не обрезать (сколько влезает, решает виджет).
    """
    items = items or {}
    legality = legality or {}
    pinned: list[str] = []
    for name in keep or ():
        if name not in pinned:
            pinned.append(name)
    pinned_set = set(pinned)

    rows: list[tuple[str, float, float | None, bool]] = []
    for name in pinned:
        pct = float(items.get(name, 0.0) or 0.0)
        rows.append((name, pct, legality.get(name), pct == 0.0))
    rest = sorted(((n, float(p)) for n, p in items.items() if n not in pinned_set),
                  key=lambda kv: -kv[1])
    limit = len(items) + len(pinned) if max_rows <= 0 else max_rows
    for name, pct in rest:
        if len(rows) >= limit:
            break
        rows.append((name, pct, legality.get(name), False))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows[:limit]


def fmt_pct(v: float) -> str:
    """Доля в процентах без «0.0%» на живом действии.

    При окне в целый роллаут (32768 шагов) единичные постройки = 0.003–0.05%,
    и фиксированный один знак после запятой показывал бы «0.0%» там, где
    действие есть. Пороги — зеркало `charts._fmt`, но без Qt-зависимости.
    """
    v = float(v)
    if v <= 0.0:
        return "0%"
    if v >= 10.0:
        return f"{v:.1f}%"
    if v >= 1.0:
        return f"{v:.2f}%"
    return f"{v:.3f}%"


class WatchRow(TypedDict):
    """Строка `watch_report`: одно закреплённое действие."""

    name: str
    count: int
    pct: float
    legal: float | None
    reason: str | None
    verdict: str


def watch_report(counts: dict[str, int], legality: dict[str, float],
                 names: Sequence[str], total_actions: int = 0,
                 shares: dict[str, float] | None = None,
                 reasons: dict[str, str] | None = None,
                 ) -> list[WatchRow]:
    """Отчёт по закреплённым действиям: {name, count, pct, legal, reason, verdict}.

    `shares` (готовые доли из `top_actions`) имеют приоритет над пересчётом из
    счётчиков: цифра в подписи обязана совпадать с цифрой на шкале, иначе
    панель начнёт «врать двумя способами сразу». Сырой `count` рядом нужен
    потому, что на длинном роллауте доля редкого действия мельчает до 0.01%,
    а «4 шага» читается сразу.

    `reasons` — action_mask_reasons из протокола ({action: MASK_REASON_KEY});
    уходит в поле `reason` строки и в вердикт (см. action_verdict).
    """
    total = max(int(total_actions or 0), 0)
    out: list[WatchRow] = []
    for name in names or ():
        cnt = int((counts or {}).get(name, 0) or 0)
        pct: float | None = None
        if shares and name in shares:
            try:
                pct = float(shares[name])
            except (TypeError, ValueError):
                pct = None
        if pct is None:
            pct = round(cnt / total * 100.0, 3) if total > 0 else 0.0
        legal = (legality or {}).get(name)
        reason = (reasons or {}).get(name)
        out.append({"name": name, "count": cnt, "pct": pct, "legal": legal,
                    "reason": reason,
                    "verdict": action_verdict(pct, legal, reason)})
    return out
