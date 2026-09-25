# Один экран в графическом виде: светлая карта с иконками слева, панель
# «Книга/Карта/Долг» всегда справа, меню сверху, легенда/палитра внизу.
# Карта: колесо — зум, перетаскивание мышью — сдвиг, ПКМ — строить.

from __future__ import annotations

import os
import string
import sys

import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C
from core.bases import STATE_NEED_SUNDUK, STATE_NEED_WORKERS
from core.game import Game, new_game
from core.resources import Sunduk
from core.save import load_game, save_game
from ui.render import (
    BG,
    FG,
    FG_ACCENT,
    FG_DIM,
    FG_MENU,
    FG_TITLE,
    ICON_BUILD,
    ICON_DEAD,
    ICON_NEED_SUNDUK,
    ICON_NEED_WORKERS,
    ICON_NO_SEASON,
    ICON_PRESERVE,
    LEGEND,
    LEGEND_BG,
    PANEL_BG,
    PANEL_FG,
    PANEL_HEAD,
    RES_SHORT,
    TERRAIN_BG,
    _asset,
    _scaled,
    base_resource_caption,
    earth_img,
    icon,
    lot_caption,
)

TITLE = "# Сахалинская колония 3.47"
AUTHOR = "Жмулевский Григорий"

WINDOW_W, WINDOW_H = 1160, 720
SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "saves", "slot.json")

# шрифт / текстовая сетка (для панелей и диалогов)
CH = 22
CW = 11
TOP_H = 2 * CH + 2           # заголовок + меню
BSTEP = 32                   # шаг кнопок построек (оригинал: l += 32)
PAL_ROWS = 4                 # ряды палитры построек (33 постройки: 4 ряда по 9)
PAL_COLS = 9
MAP_Y = TOP_H + PAL_ROWS * BSTEP   # карта под палитрой (оригинал: pb1.Top=96)
BOTTOM_H = 2 * CH + 2        # статус + легенда
MAP_W = 880                  # ширина поля карты
PANEL_X = MAP_W              # x правой панели
MAP_H = WINDOW_H - MAP_Y - BOTTOM_H
MAP_X = 0

ZOOMS = (16, 20, 25, 32, 40, 50, 64)   # размер ячейки (в оригинале CellW=25)

# кнопки-иконки imlTools справа от палитры построек
# (оригинал unMain.dfm, координаты внутри pnClient: Left/Top/ImageIndex)
# ряд 0: Земля(352,9) Поиск(384,15) Ремонт(416,16) Ремонт всех(448,24)
#        Снос(480,13) Отмена(512,26)
# ряд 1: Банк(352,25) Консервация(384,23)
# ряд 2: Купить(352,12) Продать(384,11) День(480,10) Неделя(512,14)
TOOL_X0 = 352
TOOL_BUTTONS = [
    ("Земля", 9, 0, 0), ("Поиск", 15, 0, 1), ("Ремонт", 16, 0, 2),
    ("Ремонт всех", 24, 0, 3), ("Снос", 13, 0, 4), ("Отмена", 26, 0, 5),
    ("Банк", 25, 1, 0), ("Консервация", 23, 1, 1),
    ("Купить", 12, 2, 0), ("Продать", 11, 2, 1),
    ("День", 10, 2, 4), ("Неделя", 14, 2, 5),
]

# полоса даты: 4 сезона (оригинал PaintDateProgress, цвета BGR -> RGB)
SEASON_COLORS = [
    (186, 216, 148),   # весна 0x0094D89C
    (233, 235, 158),   # лето  0x009EEBE9
    (255, 177, 140),   # осень 0x008CB1FF
    (186, 224, 226),   # зима  0x00E2E0BA
]

MENU_ITEMS = {
    "Игра": [("Новая игра...", "Новая игра"), ("Загрузить...", "Загрузить"),
             ("Сохранить как...", "Сохранить как"), ("Выход", "Выход")],
    "Сервис": [("Звук", "Звук"), ("Музыка", "Музыка"),
               ("Полный экран", "Полный экран"), ("Параметры...", "Параметры")],
    "?": [("Помощь", "Помощь"), ("О программе...", "О программе")],
}
MENU_X = {"Игра": 2, "Сервис": 12, "?": 25}

HELP_TEXT = [
    "Сахалинская колония — экономическая стратегия.",
    "Ваша колония начинает жизнь 1 марта 1890 года.",
    "Цель — выжить как можно дольше.",
    "",
    "ОСНОВНАЯ МЕХАНИКА",
    "  Каждый день рабочие добывают ресурсы, постройки изнашиваются",
    "  и требуют ремонта, люди рождаются и умирают.",
    "  Каждый год (1 марта) уплачивается ежегодный налог, раз в 10 лет",
    "  — главный налог. Пока существует Город, деньги и ресурсы",
    "  в безопасности. Потеря Города = крах колонии.",
    "",
    "ЗЕМЛЯ",
    "  Иконки местности: вода, лес, уголь, железо, нефть, золото.",
    "  На каждой земле можно строить только подходящие постройки.",
    "  Улучшенная земля (иконка с плюсом) даёт больше прибыли.",
    "",
    "ПОСТРОЙКИ",
    "  Стройте дома для людей, фермы и промыслы для еды,",
    "  добывайте уголь, железо, нефть, золото.",
    "  Значки у постройки: строящаяся, законсервированная,",
    "  нужны ресурсы, нужны рабочие, не сезон, изношена.",
    "  Изношенные постройки требуют ремонта, иначе разрушатся.",
    "",
    "ГОРЯЧИЕ КЛАВИШИ",
    "  Стрелки — перемещение по карте",
    "  Ctrl+буква — построить (A..Z по списку построек)",
    "  N — следующий день, W — следующая неделя",
    "  Кнопки под мини-картой: День, Неделя, Месяц — перемотка времени",
    "  B — рынок (покупка), S — рынок (продажа)",
    "  K — банк, G — улучшить землю",
    "  R — ремонт, A — ремонт всех, P — консервация",
    "  D — снос, U — отмена, F — найти изношенную постройку",
    "  колесо мыши — масштаб; ЛКМ по мини-карте — переход/перетаскивание",
    "  F1 — помощь, F2 — сохранить, F4 — новая игра,",
    "  F5 — загрузить, F9 — полный экран",
]

ABOUT_TEXT = [
    "Сахалинская колония 3.47",
    "Автор оригинала: Жмулевский Григорий (zgsprojects.narod.ru)",
    "Воссоздание интерфейса и механики по исходникам 3.08.",
    "",
    "Игра воспроизведена 1-в-1: параметры, цены, налоги,",
    "механика построек и карта — по исходному коду автора.",
]

_font = None
_font_b = None
_font_title = None


def _fonts():
    global _font, _font_b, _font_title
    if _font is None:
        _font = pygame.font.SysFont("Consolas", 16)
        _font_b = pygame.font.SysFont("Consolas", 16, bold=True)
        _font_title = pygame.font.SysFont("Consolas", 19, bold=True)
    return _font, _font_b, _font_title


class Dialog:
    def __init__(self, title: str, lines: list[str]) -> None:
        self.title, self.lines = title, lines

    def draw(self, scr) -> None:
        w = 640
        n = 2 + len(self.lines) + 2
        h = n * CH + 10
        x = (WINDOW_W - w) // 2
        y = (WINDOW_H - h) // 2
        box = pygame.Surface((w, h))
        box.fill((30, 36, 32))
        pygame.draw.rect(box, (140, 170, 150), box.get_rect(), 2)
        scr.screen.blit(box, (x, y))
        scr.text(y + 4, x + 12, self.title, FG_ACCENT, bold=True)
        for i, line in enumerate(self.lines):
            scr.text(y + 4 + (2 + i) * CH, x + 12, line, FG)
        scr.text(y + h - 22, x + 12, "[Enter/Пробел] — закрыть", FG_DIM)


class TextDialog(Dialog):
    def __init__(self, title: str, lines: list[str], choices: list[str]) -> None:
        super().__init__(title, lines)
        self.choices = choices

    def draw(self, scr) -> None:
        super().draw(scr)
        w = 640
        x = (WINDOW_W - w) // 2
        h = (2 + len(self.lines) + 2) * CH + 10
        y = (WINDOW_H - h) // 2
        for i, c in enumerate(self.choices):
            scr.text(y + 4 + (2 + len(self.lines) + i) * CH, x + 12,
                     f"{i + 1}. {c}", FG_MENU)

    def choose(self, key: str) -> int | None:
        if key.isdigit() and int(key) in range(1, len(self.choices) + 1):
            return int(key) - 1
        return None

    def click(self, row: int) -> int | None:
        h = (2 + len(self.lines) + 2) * CH + 10
        y = (WINDOW_H - h) // 2
        gy = (row - (y + 4) - (2 + len(self.lines)) * CH) // CH
        if 0 <= gy < len(self.choices):
            return gy
        return None


class HelpDialog:
    """Помощь: длинный текст с прокруткой (F1)."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.offset = 0

    def draw(self, scr) -> None:
        w, h = 760, 560
        x = (WINDOW_W - w) // 2
        y = (WINDOW_H - h) // 2
        box = pygame.Surface((w, h))
        box.fill((30, 36, 32))
        pygame.draw.rect(box, (140, 170, 150), box.get_rect(), 2)
        scr.screen.blit(box, (x, y))
        scr.text(y + 4, x + 12, "Помощь", FG_ACCENT, bold=True)
        vis = (h - 56) // CH
        for i in range(vis):
            li = self.offset + i
            if li >= len(self.lines):
                break
            scr.text(y + 4 + (1 + i) * CH, x + 12, self.lines[li], FG)
        scr.text(y + h - 22, x + 12,
                 f"↑/↓, колесо — листать   [Enter/Пробел] — закрыть"
                 f"   ({self.offset + 1}..{min(self.offset + vis, len(self.lines))} из {len(self.lines)})",
                 FG_DIM)

    def key(self, key: int, unicode: str) -> None:
        if key == pygame.K_UP:
            self.offset = max(0, self.offset - 1)
        elif key == pygame.K_DOWN:
            self.offset = min(len(self.lines) - 1, self.offset + 1)

    def wheel(self, dy: int) -> None:
        self.offset = max(0, min(len(self.lines) - 1, self.offset - dy))


class MarketDialog:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.buying = True
        self.msg = ""

    def lines(self) -> list[str]:
        head = (f"Купля (пробел/клик — продажа).   Деньги: "
                f"{C.thousands(self.game.money)}")
        rows = [head, ""]
        for i in range(9):
            rows.append(f"{i + 1}. {RES_SHORT[i]:<16} сундук: "
                        f"{self.game.sunduk.items[i]:<6} куп "
                        f"{C.BUY_SUNDUK[i]:>5}  прод {C.SALE_SUNDUK[i]:>5}")
        rows.append("")
        rows.append(f"Цифра ресурса — сделка на 1 ед.   Esc — закрыть.  {self.msg}")
        return rows

    def draw(self, scr) -> None:
        Dialog("Рынок", self.lines()).draw(scr)

    def key(self, scr, key: str) -> None:
        if key == " ":
            self.buying = not self.buying
            self.msg = ""
            return
        if key.isdigit() and int(key) in range(1, 10):
            self.trade(int(key) - 1)

    def click(self, row: int) -> None:
        h = (2 + len(self.lines()) + 2) * CH + 10
        y = (WINDOW_H - h) // 2
        if y + 4 + 3 * CH <= row < y + 4 + 4 * CH:
            self.buying = not self.buying
            self.msg = ""
        elif y + 4 + 5 * CH <= row < y + 4 + 14 * CH:
            self.trade((row - (y + 4 + 5 * CH)) // CH)

    def trade(self, idx: int) -> None:
        items = [0] * 9
        items[idx] = 1
        if self.buying:
            ok, msg, _ = self.game.market_buy(Sunduk(items))
        else:
            ok, msg, _ = self.game.market_sell(Sunduk(items))
        self.msg = msg or ("сделано" if ok else "")


class BankDialog:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.msg = ""

    def lines(self) -> list[str]:
        if self.game.credit:
            head = (f"Кредит: {C.thousands(self.game.credit)}   "
                    f"(долг растёт +0.2%/день)")
        else:
            head = "Вы ещё не брали кредит."
        rows = [head, ""]
        rows.append("1. Взять 1000         4. Вернуть 1000")
        rows.append("2. Взять 10000        5. Вернуть 10000")
        rows.append("3. Взять 100000       6. Вернуть 100000")
        rows.append(f"Цифры 1..6 или клик.  Esc — закрыть.  {self.msg}")
        return rows

    def draw(self, scr) -> None:
        Dialog("Банк", self.lines()).draw(scr)

    def key(self, scr, key: str) -> None:
        act = {"1": ("take", 1000), "2": ("take", 10000), "3": ("take", 100000),
               "4": ("give", 1000), "5": ("give", 10000), "6": ("give", 100000)}
        if key in act:
            kind, amount = act[key]
            self.do(kind, amount)

    def click(self, row: int, col: int) -> None:
        w = 640
        h = (2 + len(self.lines()) + 2) * CH + 10
        y = (WINDOW_H - h) // 2
        x = (WINDOW_W - w) // 2
        r = (row - (y + 4 + 3 * CH)) // CH
        if 0 <= r <= 2:
            amounts = (1000, 10000, 100000)
            kind = "take" if col < x + w // 2 else "give"
            self.do(kind, amounts[r])

    def do(self, kind: str, amount: int) -> None:
        self.msg = (self.game.bank_take(amount) if kind == "take"
                    else self.game.bank_give(amount)) or "сделано"


class NalogDialog:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.msg = ""

    def lines(self) -> list[str]:
        g = self.game
        rows = [f"Ежегодный (1 марта): {C.thousands(g.annual_tax_amount())}"
                f"   — {'к уплате' if g.annual_tax_due() else 'уплачен'}",
                f"Главный (1 ноября):  {C.thousands(g.main_tax_amount())}"
                f"   — {'к уплате' if g.main_tax_due() else 'уплачен'}",
                ""]
        rows.append("1. Уплатить ежегодный")
        rows.append("2. Уплатить главный")
        rows.append(f"Цифры 1,2 или клик.  Esc — закрыть.  {self.msg}")
        return rows

    def draw(self, scr) -> None:
        Dialog("Налоги", self.lines()).draw(scr)

    def key(self, scr, key: str) -> None:
        if key == "1":
            self.msg = self.game.pay_annual_tax() or "сделано"
        elif key == "2":
            self.msg = self.game.pay_main_tax() or "сделано"

    def click(self, row: int) -> None:
        h = (2 + len(self.lines()) + 2) * CH + 10
        y = (WINDOW_H - h) // 2
        if y + 4 + 6 * CH <= row < y + 4 + 7 * CH:
            self.msg = self.game.pay_annual_tax() or "сделано"
        elif y + 4 + 7 * CH <= row < y + 4 + 8 * CH:
            self.msg = self.game.pay_main_tax() or "сделано"


class GameWindow:
    def __init__(self, game: Game) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Сахалинская колония")
        try:
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "extracted", "icon_000.ico"
            )
            if os.path.exists(icon_path):
                pygame.display.set_icon(pygame.image.load(icon_path))
        except Exception:
            pass
        _fonts()
        self.game = game
        self.cx, self.cy = game.init_sel_x, game.init_sel_y
        self.selected = "Farm"
        self.dialog = None
        self.messages: list[Dialog] = []
        self.menu_open = None
        self.map_dirty = True
        self.zoom_i = 2                 # индекс в ZOOMS (ячейка 25, как в оригинале)
        self.sel_frame = 0              # кадр анимации рамки выделения (0..11)
        self.blink_on = True            # мигание иконки критического износа
        self._alarm_notified: set[tuple[int, int]] = set()
        self.map_off_x = self.map_off_y = 0
        self.grab = None        # (px, py, cx, cy, moved, remx, remy) — ЛКМ на карте
        self.plan_drag = False          # перетаскивание по мини-карте
        self._time_buttons: list[pygame.Rect] = []   # День/Неделя/Месяц
        self.running = True
        self.game_over_handled = False
        self._plan_cache = None
        self._plan_rect = pygame.Rect(0, 0, 0, 0)   # мини-карта (x, y, w, h)
        self._date_rect = pygame.Rect(0, 0, 0, 0)   # полоса даты (клик = время)
        self._recenter()

    # ================================================================ helpers
    @property
    def tile(self) -> int:
        return ZOOMS[self.zoom_i]

    @property
    def cols(self) -> int:
        return max(1, MAP_W // self.tile)

    @property
    def rows(self) -> int:
        return max(1, MAP_H // self.tile)

    @property
    def base_ids(self) -> list[str]:
        return [d.id for d in self.game.base_data]

    def text(self, y: int, x: int, s: str, fg=FG, bold=False, bg=None) -> None:
        f, fb, ft = _fonts()
        img = (fb if bold else f).render(s, True, fg, bg)
        self.screen.blit(img, (x, y))

    def textg(self, row: int, col: int, s: str, fg=FG, bold=False,
              bg=None) -> None:
        self.text(row * CH + 2, col * CW, s, fg, bold, bg)

    def _log(self, title: str, lines: list[str]) -> None:
        if lines:
            self.messages.append(Dialog(title, lines))

    def _apply_result(self, res) -> None:
        if isinstance(res, str):
            self.messages.append(Dialog("Внимание", [res]))
            self._check_alarms()
            return
        if isinstance(res, list):
            for r in res:
                self._apply_result(r)
            return
        lines = []
        if res.season_changed:
            lines.append(f"Сезон сменился: {res.season_new}")
        if res.people_arrived:
            lines.append(C.SM_ARRIVAL % res.people_arrived)
        if res.reminder_may:
            lines.append(C.SM_REMINDER)
        if res.home_overflow:
            lines.append(C.SM_DIED_OVERFLOW % res.died)
        elif res.died:
            lines.append(C.SM_DIED % res.died)
        if res.born:
            lines.append(C.SM_BORN % res.born)
        if res.base_lost:
            lines.append(C.SM_DESTROYED)
        for msg, cons in res.events:
            lines.append(msg)
            if cons:
                lines.append(cons)
        self._log("Отчёт за день", lines)
        self._check_alarms()

    def _check_alarms(self) -> None:
        """Однократное оповещение о постройке, вошедшей в критический износ."""
        g = self.game
        alive: set[tuple[int, int]] = set()
        for b in g.bases:
            if not b.is_alarm():
                continue
            alive.add((b.x, b.y))
            if (b.x, b.y) in self._alarm_notified:
                continue
            self._alarm_notified.add((b.x, b.y))
            self._log("Внимание",
                      [f"{b.data.caption} в критическом износе — "
                       f"осталось ~{b.live_time} дней, отремонтируйте!"])
        # отремонтированные (вышедшие из износа) снова смогут оповестить
        self._alarm_notified = {p for p in self._alarm_notified if p in alive}

    # ================================================================ actions
    def do_action(self, label: str) -> None:
        g = self.game
        if label == "День":
            self._apply_result(g.advance_day())
            self.map_dirty = True
        elif label == "Неделя":
            self._apply_result(g.advance_week())
            self.map_dirty = True
        elif label == "Месяц":
            days = C.DAYS_IN_MONTH[g.month - 1] - g.day + 1
            for _ in range(min(days, 35)):
                self._apply_result(g.advance_day())
            self.map_dirty = True
        elif label in ("Рынок", "Купить", "Продать"):
            self.dialog = MarketDialog(g)
            self.dialog.buying = label != "Продать"
        elif label == "Банк":
            self.dialog = BankDialog(g)
        elif label == "Налог":
            self.dialog = NalogDialog(g)
        elif label == "Отмена":
            g.undo()
            self.map_dirty = True
        elif label == "Земля":
            ok, msg = g.good_earth(self.cx, self.cy)
            self._log("Улучшение земли", [msg] if not ok and msg else [])
            self.map_dirty = True
        elif label == "Снос":
            ok, msg = g.destroy(self.cx, self.cy)
            self._log("Снос", [msg] if not ok and msg else [])
            self.map_dirty = True
        elif label == "Ремонт":
            ok, msg, _, _ = g.restore(self.cx, self.cy)
            self._log("Ремонт", [msg] if not ok and msg else [])
            self.map_dirty = True
        elif label == "Ремонт всех":
            ok, msg, total, n = g.restore_all()
            self._log("Ремонт всех", [msg] if not ok and msg else
                      [f"Отремонтировано построек: {n} на {C.thousands(total)}"])
            self.map_dirty = True
        elif label == "Поиск":
            b = g.find_slowest_base()
            if b is None:
                self._log("Поиск", [C.SM_NOFINDSLOW])
            else:
                self.cx, self.cy = b.x, b.y
                self._ensure_visible()
                self.map_dirty = True
        elif label == "Консервация":
            ok, msg = g.preserve(self.cx, self.cy)
            self._log("Консервация", [msg] if not ok and msg else [])
            self.map_dirty = True
        elif label == "Строить":
            ok, msg = g.build(self.selected, self.cx, self.cy)
            self._log("Строительство", [msg] if not ok and msg else [])
            self.map_dirty = True
        elif label == "Сохранить":
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            try:
                save_game(g, SAVE_PATH)
                self._log("Сохранение", ["Игра сохранена в saves/slot.json"])
            except Exception as e:  # noqa: BLE001
                self._log("Ошибка", [str(e)])
        elif label == "Сохранить как":
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            try:
                save_game(g, SAVE_PATH)
                self._log("Сохранение", ["Игра сохранена в saves/slot.json"])
            except Exception as e:  # noqa: BLE001
                self._log("Ошибка", [str(e)])
        elif label == "Звук":
            self._log("Звук", ["Звуковые эффекты в текущей сборке не озвучены"])
        elif label == "Музыка":
            self._log("Музыка", ["Музыка в текущей сборке не воспроизводится"])
        elif label == "Полный экран":
            pygame.display.toggle_fullscreen()
        elif label == "Параметры":
            self._log("Параметры",
                      ["Разрешение: фиксированное (1160x720)",
                       "Масштаб карты: колесо мыши",
                       "Остальные параметры в исходной игре не изменялись"])
        elif label == "Помощь":
            self.dialog = HelpDialog(HELP_TEXT)
        elif label == "О программе":
            self.dialog = Dialog("О программе", ABOUT_TEXT)
        elif label == "Загрузить":
            self._load()
        elif label == "Новая игра":
            self.game = new_game(g.map_seed + 1, map_size=g.map_size)
            self.cx, self.cy = self.game.init_sel_x, self.game.init_sel_y
            self.map_dirty = True
            self.game_over_handled = False
            self._recenter()
        elif label == "Выход":
            self.running = False
        self._check_over()

    def _load(self) -> None:
        if not os.path.exists(SAVE_PATH):
            self._log("Загрузка", ["Сохранение не найдено"])
            return
        try:
            self.game = load_game(SAVE_PATH)
            self.cx, self.cy = self.game.init_sel_x, self.game.init_sel_y
            self.map_dirty = True
            self._recenter()
            self._log("Загрузка", ["Игра загружена"])
        except Exception as e:  # noqa: BLE001
            self._log("Ошибка", [str(e)])

    def _check_over(self) -> None:
        if self.game_over_handled:
            return
        go = self.game.game_over()
        if go is not None:
            self.game_over_handled = True
            self.dialog = TextDialog("Игра окончена",
                                     [f"Причина: {go.reason}",
                                      f"Дата: {self.game.date_text()}",
                                      f"Людей: {self.game.people}, "
                                      f"денег: {C.thousands(self.game.money)}"],
                                     ["Новая игра", "Выход"])

    # ================================================================ map
    def _recenter(self) -> None:
        self.map_off_x = min(max(self.cx - self.cols // 2, 0),
                             self.game.map_size - self.cols)
        self.map_off_y = min(max(self.cy - self.rows // 2, 0),
                             self.game.map_size - self.rows)

    def _ensure_visible(self) -> None:
        """Прокрутка, только если выделение вышло за пределы экрана
        (аналог pb1->ShowRect в оригинале: выделение не всегда в центре)."""
        max_off_x = max(0, self.game.map_size - self.cols)
        max_off_y = max(0, self.game.map_size - self.rows)
        if self.cx < self.map_off_x:
            self.map_off_x = max(0, self.cx)
        elif self.cx >= self.map_off_x + self.cols:
            self.map_off_x = min(max_off_x, self.cx - self.cols + 1)
        if self.cy < self.map_off_y:
            self.map_off_y = max(0, self.cy)
        elif self.cy >= self.map_off_y + self.rows:
            self.map_off_y = min(max_off_y, self.cy - self.rows + 1)

    def _set_zoom(self, idx: int) -> None:
        idx = max(0, min(len(ZOOMS) - 1, idx))
        if idx == self.zoom_i:
            return
        # сохраняем мировую позицию центра видимой области, чтобы выделение
        # не «прыгало» при зуме (выделение может быть не в центре)
        cx_world = self.map_off_x + self.cols // 2
        cy_world = self.map_off_y + self.rows // 2
        self.zoom_i = idx
        self.map_off_x = max(0, min(cx_world - self.cols // 2,
                                    self.game.map_size - self.cols))
        self.map_off_y = max(0, min(cy_world - self.rows // 2,
                                    self.game.map_size - self.rows))
        self._ensure_visible()
        self.map_dirty = True

    def _cell_at(self, px: int, py: int):
        if MAP_X <= px < MAP_X + MAP_W and MAP_Y <= py < MAP_Y + MAP_H:
            col = (px - MAP_X) // self.tile
            row = (py - MAP_Y) // self.tile
            if 0 <= col < self.cols and 0 <= row < self.rows:
                return self.map_off_x + col, self.map_off_y + row
        return None

    # ================================================================ drawing
    def _draw_top(self) -> None:
        self.textg(0, 0, TITLE, FG_TITLE, bold=True)
        for name, x in MENU_X.items():
            self.textg(1, x, name, FG_MENU, bold=True)
        # палитра построек сверху (оригинал: InitBaseButton, 4 ряда по 9, шаг 32)
        bs = 25
        y0 = 2 * CH + 3
        for i, bid in enumerate(self.base_ids):
            col = i % PAL_COLS
            row = i // PAL_COLS
            x = col * BSTEP
            y = y0 + row * BSTEP
            sel = bid == self.selected
            pygame.draw.rect(self.screen, (36, 46, 40) if sel else (20, 28, 23),
                             (x, y, bs, bs))
            if sel:
                pygame.draw.rect(self.screen, FG_ACCENT, (x, y, bs, bs), 1)
            self.screen.blit(_scaled(_asset(f"imlBases_{self._img(bid):02d}"),
                                     bs - 6), (x + 3, y + 3))
            letter = string.ascii_uppercase[i] if i < 26 else ""
            # буква в углу кнопки (оригинал: lb 15x15 на (l+14, t+14))
            self.text(y + 13, x + 16, letter, FG_DIM)
        # кнопки действий справа от палитры построек (оригинал unMain.dfm)
        for _, img_idx, row, col in TOOL_BUTTONS:
            x = TOOL_X0 + col * BSTEP
            y = y0 + row * BSTEP
            pygame.draw.rect(self.screen, (30, 40, 34), (x, y, bs, bs),
                             border_radius=3)
            self.screen.blit(_scaled(_asset(f"imlTools_{img_idx:02d}"), bs - 6),
                             (x + 3, y + 3))
        pygame.draw.line(self.screen, (70, 90, 80),
                         (0, MAP_Y - 6), (WINDOW_W, MAP_Y - 6), 1)
        if self.menu_open:
            f, fb, ft = _fonts()
            items = MENU_ITEMS[self.menu_open]
            x = MENU_X[self.menu_open] * CW
            for i, (label, _) in enumerate(items):
                w = fb.size(label)[0] + 12
                rect = pygame.Rect(x, TOP_H + i * CH, w, CH - 2)
                pygame.draw.rect(self.screen, (40, 46, 44), rect)
                self.text(rect.y + 2, rect.x + 6, label, FG_MENU)

    def _draw_map(self) -> None:
        g = self.game
        t = self.tile
        for row in range(self.rows):
            for col in range(self.cols):
                gx = self.map_off_x + col
                gy = self.map_off_y + row
                lot = int(g.earth.lots[gy, gx])
                kind = {C.LT_NONE: "none", C.LT_WATER: "water",
                        C.LT_WOOD: "wood", C.LT_COAL: "coal",
                        C.LT_IRON: "iron", C.LT_OIL: "oil",
                        C.LT_GOLD: "gold"}.get(lot, "land")
                px = MAP_X + col * t
                py = MAP_Y + row * t
                # фон ячейки, поверх — прозрачная иконка местности
                pygame.draw.rect(self.screen, TERRAIN_BG[lot], (px, py, t, t))
                if kind != "land" and kind != "none":
                    img = earth_img(lot, int(g.earth.subtype[gy, gx]))
                    self.screen.blit(_scaled(_asset(f"imlEarth_{img:02d}"), t),
                                     (px, py))
                b = g.base_in_box(gx, gy)
                if b is not None:
                    # иконка постройки — на весь размер клетки
                    bs = t
                    self.screen.blit(_scaled(_asset(
                        f"imlBases_{b.data.image_index:02d}"), bs), (px, py))
                    # статусные значки imlIcons внизу ячейки (оригинал PaintBase)
                    if b.build_days:
                        self._blit_icons(ICON_BUILD, px, py, t)
                    if b.preserved:
                        self._blit_icons(ICON_PRESERVE, px, py, t)
                    if STATE_NEED_SUNDUK in b.state:
                        self._blit_icons(ICON_NEED_SUNDUK, px, py, t)
                    if STATE_NEED_WORKERS in b.state:
                        self._blit_icons(ICON_NEED_WORKERS, px, py, t)
                    if b.data.work_seasons and g.season not in b.data.work_seasons:
                        self._blit_icons(ICON_NO_SEASON, px, py, t)
                    if b.is_alarm() and self.blink_on:
                        self._blit_icons(ICON_DEAD, px, py, t)
                elif g.destroyed_lots[gy, gx]:
                    self.screen.blit(icon("destroyed", t), (px, py))
                elif g.good_lots[gy, gx]:
                    self.screen.blit(icon("good", t), (px, py))
        # рамка выделения (оригинал: imlSelectEarth / imlSelectNone, кадры 0..11)
        if 0 <= self.cx - self.map_off_x < self.cols and \
                0 <= self.cy - self.map_off_y < self.rows:
            x = MAP_X + (self.cx - self.map_off_x) * t
            y = MAP_Y + (self.cy - self.map_off_y) * t
            lot = int(g.earth.lots[self.cy, self.cx])
            if lot == C.LT_NONE:
                self.screen.blit(_scaled(_asset("imlSelectNone_"
                                                f"{self.sel_frame % 12:02d}"), t),
                                 (x, y))
            else:
                self.screen.blit(_scaled(_asset("imlSelectEarth_"
                                                f"{self.sel_frame % 12:02d}"), t),
                                 (x, y))
        # рамка карты
        pygame.draw.rect(self.screen, (60, 80, 70),
                         (MAP_X, MAP_Y, MAP_W, MAP_H), 1)

    def _render_plan(self, plan_w: int, plan_h: int) -> pygame.Surface:
        """Мини-карта целиком (как оригинальный PlanPaint, но фоном — карта)."""
        g = self.game
        surf = pygame.Surface((plan_w, plan_h))
        gs = g.map_size
        scale = plan_w / gs
        for gy in range(gs):
            for gx in range(gs):
                pygame.draw.rect(surf, TERRAIN_BG[int(g.earth.lots[gy, gx])],
                                 (gx * scale, gy * scale,
                                  scale + 0.5, scale + 0.5))
        for b in g.bases:
            bx = b.x * scale
            by = b.y * scale
            if b.data.plan:  # отображается на мини-карте
                pygame.draw.rect(surf, (200, 40, 30), (bx - 2, by - 2, 4, 4))
            if b.is_alarm() and self.blink_on:
                surf.blit(_asset("imlIcons_00"), (bx - 7, by - 7))
        return surf

    def _plan_set(self, px: int, py: int) -> None:
        """Клик/перетаскивание по мини-карте: видимая область большой карты
        = эта же точка в миниатюре (вид центрируется на клике)."""
        g = self.game
        if not self._plan_rect.w:
            return
        scale = self._plan_rect.w / g.map_size
        ncx = min(max(int((px - self._plan_rect.x) / scale), 0),
                  g.map_size - 1)
        ncy = min(max(int((py - self._plan_rect.y) / scale), 0),
                  g.map_size - 1)
        if (ncx, ncy) != (self.cx, self.cy):
            self.cx, self.cy = ncx, ncy
            self.map_dirty = True
        self.map_off_x = min(max(ncx - self.cols // 2, 0),
                             g.map_size - self.cols)
        self.map_off_y = min(max(ncy - self.rows // 2, 0),
                             g.map_size - self.rows)
        self.map_dirty = True

    def _jump_date(self, px: int) -> None:
        """Клик по полосе даты: продвигаем время до дня года в этой точке."""
        g = self.game
        r = self._date_rect
        if not r.w:
            return
        leap = C.is_leap(g.year)
        days_year = 365 if leap else 364
        abs_day = g.day
        for i in range(g.month - 1):
            abs_day += C.DAYS_IN_MONTH[i]
        if leap and g.month > 2:
            abs_day += 1
        target = max(0, min(days_year, (px - r.x) * days_year // r.w))
        steps = 0
        while abs_day != target and steps < 1000:
            if g.month == 12 and g.day == 31:
                break
            abs_day += 1
            if abs_day > days_year:
                break
            res = g.advance_day()
            self._apply_result(res)
            steps += 1
        self.map_dirty = True

    def _blit_icons(self, idx: int, px: int, py: int, t: int) -> None:
        """Значок imlIcons в нижнем левом углу ячейки (оригинал: y+CellH-14)."""
        s = _scaled(_asset(f"imlIcons_{idx:02d}"), max(8, t - 9))
        self.screen.blit(s, (px, py + t - s.get_height()))

    def _draw_panel(self) -> None:
        # панель (в оригинале pnRight, clNavy)
        pygame.draw.rect(self.screen, PANEL_BG,
                         (PANEL_X, TOP_H, WINDOW_W - PANEL_X,
                          WINDOW_H - BOTTOM_H - TOP_H))
        g = self.game
        x = PANEL_X + 10

        def head(name: str, y: int) -> None:
            pygame.draw.rect(self.screen, PANEL_HEAD,
                             (PANEL_X, y, WINDOW_W - PANEL_X, CH))
            self.text(y + 2, x, name, (10, 15, 12), bold=True)

        # ---- мини-карта (оригинал Plan: 210x210, постройки красными) ----
        plan_w = WINDOW_W - PANEL_X - 20
        plan_h = min(210, WINDOW_H - BOTTOM_H - TOP_H - 12 * CH - 20)
        plan_x, plan_y = PANEL_X + 10, TOP_H + 4
        self._plan_rect = pygame.Rect(plan_x, plan_y, plan_w, plan_h)
        pygame.draw.rect(self.screen, (20, 30, 24), (plan_x - 3, plan_y - 3,
                                                     plan_w + 6, plan_h + 6))
        if self._plan_cache is None:
            self._plan_cache = self._render_plan(plan_w, plan_h)
        self.screen.blit(self._plan_cache, (plan_x, plan_y))
        # рамка видимого участка большой карты (те же клетки, что видны на экране)
        scale = plan_w / g.map_size
        vx0 = plan_x + self.map_off_x * scale
        vy0 = plan_y + self.map_off_y * scale
        vw = self.cols * scale
        vh = self.rows * scale
        pygame.draw.rect(self.screen, (250, 250, 235),
                         (vx0, vy0, vw, vh), 1)
        pygame.draw.rect(self.screen, (250, 250, 235),
                         (vx0 - 1, vy0 - 1, vw + 2, vh + 2), 1)
        y = plan_y + plan_h + 6
        # ---- полоса даты (оригинал PaintDateProgress) ----
        self.text(y, x, "Дата:", PANEL_FG)
        y += CH - 2
        pb = pygame.Rect(x, y, plan_w, 6)
        self._date_rect = pygame.Rect(x, y, plan_w, 6)
        pygame.draw.rect(self.screen, (0, 0, 0), pb)
        # границы сезонов: зима(беж) | весна(зел) | лето(голуб) | осень(оранж)
        leap = C.is_leap(g.year)
        days_year = 365 if leap else 364
        d1 = 31 + C.DAYS_IN_MONTH[1] + (1 if leap else 0)
        d2 = d1 + 31 + 30 + 31
        d3 = d2 + 30 + 31 + 31
        d4 = d3 + 30 + 31 + 30
        for (a, b, col) in ((0, d1, SEASON_COLORS[3]),
                    (d1, d2, SEASON_COLORS[0]),
                    (d2, d3, SEASON_COLORS[1]),
                    (d3, d4, SEASON_COLORS[2]),
                    (d4, days_year, SEASON_COLORS[3])):
            x0 = pb.x + pb.w * a // days_year
            x1 = pb.x + pb.w * b // days_year
            if x1 > x0:
                pygame.draw.rect(self.screen, col, (x0, pb.y, x1 - x0, 6))
        # красный маркер текущего дня (оригинал: центр полосы)
        abs_day = g.day
        for i in range(g.month - 1):
            abs_day += C.DAYS_IN_MONTH[i]
        if leap and g.month > 2:
            abs_day += 1
        mx = pb.x + pb.w * abs_day // days_year
        pygame.draw.rect(self.screen, (200, 30, 30),
                         (mx - 2, pb.y - 2, 4, 4))
        y += 12
        # кнопки перемотки времени: день / неделя / месяц
        self._time_buttons = []
        bw, bh = 52, 20
        for i, (lab, _act) in enumerate((("День", "День"), ("Неделя", "Неделя"),
                                        ("Месяц", "Месяц"))):
            bx = x + i * (bw + 4)
            pygame.draw.rect(self.screen, PANEL_HEAD, (bx, y, bw, bh),
                             border_radius=3)
            self.text(y + 2, bx + 6, lab, (10, 15, 12), bold=True)
            self._time_buttons.append(pygame.Rect(bx, y, bw, bh))
        y += bh + 4
        # ---- ресурсы с иконками (оригинал: pc01..pc09, imlTools 0..8) ----
        for i in range(C.SUNDUK_SIZE):
            self.screen.blit(_scaled(_asset(f"imlTools_{i:02d}"), CH - 6),
                             (x, y))
            self.text(y + 1, x + CH, f"{RES_SHORT[i]}: {g.sunduk.items[i]}",
                      PANEL_FG)
            y += CH - 4
        y += 2
        # деньги и люди (оригинал: pc1=17, pc2=18)
        for idx, label, val in ((17, "Деньги", C.thousands(g.money)),
                                (18, "Люди", g.people)):
            self.screen.blit(_scaled(_asset(f"imlTools_{idx:02d}"), CH - 6),
                             (x, y))
            self.text(y + 1, x + CH, f"{label}: {val}", PANEL_FG)
            y += CH - 4
        y += 2
        for label, val in (("Жильё", g.now_home_places()),
                           ("Рабочие места", g.now_need_workers()),
                           ("Не работают", max(0, g.free_people())),
                           ("Кредит", C.thousands(g.credit))):
            self.text(y + 2, x, f"{label}: {val}", PANEL_FG)
            y += CH - 4

    def _draw_bottom(self) -> None:
        g = self.game
        y = WINDOW_H - BOTTOM_H
        pygame.draw.rect(self.screen, (8, 14, 10), (0, y, WINDOW_W, BOTTOM_H))
        # статус текущей клетки (в левом нижнем углу под картой)
        lot = int(g.earth.lots[self.cy, self.cx])
        b = g.base_in_box(self.cx, self.cy)
        status = f"Клетка ({self.cx},{self.cy}): {lot_caption(lot)}"
        if g.good_lots[self.cy, self.cx]:
            status += ", улучшена"
        if b is not None:
            res = base_resource_caption(b)
            suffix = f" ({res})" if res else ""
            if b.build_days:
                status += f" | {b.data.caption}{suffix}, строится ({b.build_days})"
            else:
                status += f" | {b.data.caption}{suffix}, прослужит {b.live_time}"
            if b.preserved:
                status += ", законсервирована"
            if STATE_NEED_SUNDUK in b.state:
                status += ", нужны ресурсы"
            if STATE_NEED_WORKERS in b.state:
                status += ", нужны рабочие"
            if b.is_alarm():
                status += ", ИЗНОШЕНА!"
        elif g.destroyed_lots[self.cy, self.cx]:
            status += ", сгоревший участок"
        self.text(y + 4, 6, status, FG_ACCENT, bold=True)
        # легенда
        yy = y + CH + 2
        # жёлтая полоса легенды (как в оригинале)
        pygame.draw.rect(self.screen, LEGEND_BG, (0, yy, WINDOW_W, CH))
        self.text(yy, 6, "Карта: ", (40, 50, 30))
        cx = 6 + 7 * CW
        for lot, name in LEGEND:
            t = 16
            self.screen.blit(icon(lot_kind(lot), t), (cx, yy))
            self.text(yy + 2, cx + t + 4, name, (40, 50, 30))
            cx += t + 6 + len(name) * CW
        self.text(yy, cx + 6,
                  "ЛКМ — выбрать, ПКМ — строить, колесо — зум", (70, 70, 40))

    def _caption(self, bid: str) -> str:
        for d in self.game.base_data:
            if d.id == bid:
                return d.caption
        return bid

    def _img(self, bid: str) -> int:
        for d in self.game.base_data:
            if d.id == bid:
                return d.image_index
        return 0

    def _draw_dialog(self, d) -> None:
        if isinstance(d, (MarketDialog, BankDialog, NalogDialog)):
            d.draw(self)
        else:
            d.draw(self)

    # ================================================================ input
    def _click_menu(self, px: int, py: int) -> None:
        if self.menu_open:
            items = MENU_ITEMS[self.menu_open]
            x = MENU_X[self.menu_open]
            f, fb, ft = _fonts()
            for i, (label, action) in enumerate(items):
                w = fb.size(label)[0] + 12
                if (TOP_H + i * CH <= py < TOP_H + (i + 1) * CH
                        and x * CW <= px < x * CW + w):
                    self.menu_open = None
                    self.do_action(action)
                    return
            self.menu_open = None
            return
        for name, x in MENU_X.items():
            if x * CW <= px < (x + len(name)) * CW and py < TOP_H:
                self.menu_open = name
                return

    def _tool_index(self, px: int, py: int) -> int | None:
        bs = 25
        y0 = 2 * CH + 3
        for i, (_, _, row, col) in enumerate(TOOL_BUTTONS):
            x = TOOL_X0 + col * BSTEP
            y = y0 + row * BSTEP
            if x <= px < x + bs and y <= py < y + bs:
                return i
        return None

    def _palette_index(self, px: int, py: int) -> int | None:
        y0 = 2 * CH + 3
        col = (px - MAP_X) // BSTEP
        row = (py - y0) // BSTEP
        if not (0 <= col < PAL_COLS and 0 <= row < PAL_ROWS):
            return None
        i = row * PAL_COLS + col
        if i < len(self.base_ids):
            return i
        return None

    def handle_event(self, ev) -> None:
        if ev.type == pygame.QUIT:
            self.running = False
            return

        if self.dialog is not None:
            if isinstance(self.dialog, HelpDialog):
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_RETURN, pygame.K_ESCAPE,
                                  pygame.K_SPACE):
                        self.dialog = None
                    else:
                        self.dialog.key(ev.key, ev.unicode or "")
                elif ev.type == pygame.MOUSEWHEEL:
                    self.dialog.wheel(ev.y)
                return
            if ev.type == pygame.KEYDOWN:
                if isinstance(self.dialog, TextDialog):
                    i = self.dialog.choose(ev.unicode or "")
                    if i == 0:
                        self.dialog = None
                        self.do_action("Новая игра")
                    elif i == 1:
                        self.running = False
                    return
                if isinstance(self.dialog, (MarketDialog, BankDialog, NalogDialog)):
                    if ev.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                        self.dialog = None
                    else:
                        self.dialog.key(self, ev.unicode or "")
                    return
                if ev.key in (pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_SPACE):
                    self.dialog = None
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                px, py = ev.pos
                if isinstance(self.dialog, MarketDialog):
                    self.dialog.click(py)
                elif isinstance(self.dialog, BankDialog):
                    self.dialog.click(py, px)
                elif isinstance(self.dialog, NalogDialog):
                    self.dialog.click(py)
                elif isinstance(self.dialog, TextDialog):
                    i = self.dialog.click(py)
                    if i == 0:
                        self.dialog = None
                        self.do_action("Новая игра")
                    elif i == 1:
                        self.running = False
                    elif i is not None:
                        self.dialog = None
                else:
                    self.dialog = None
            return

        if self.messages:
            if (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN,
                                                         pygame.K_SPACE,
                                                         pygame.K_ESCAPE)) \
                    or (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1):
                self.messages.pop(0)
            return

        g = self.game
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                px, py = ev.pos
                dropdown_h = 0
                if self.menu_open:
                    dropdown_h = len(MENU_ITEMS[self.menu_open]) * CH + 4
                if self.menu_open and py < TOP_H + dropdown_h:
                    self._click_menu(px, py)
                    return
                idx = self._tool_index(px, py)
                if idx is not None:
                    self.do_action(TOOL_BUTTONS[idx][0])
                    return
                if py < TOP_H:
                    self._click_menu(px, py)
                    return
                # клик по мини-карте — быстрое перемещение (удержание = перетаскивание)
                if self._plan_rect.w and self._plan_rect.collidepoint(px, py):
                    self.plan_drag = True
                    self._plan_set(px, py)
                    return
                # кнопки перемотки времени: День/Неделя/Месяц
                for i, r in enumerate(self._time_buttons):
                    if r.collidepoint(px, py):
                        self.do_action(("День", "Неделя", "Месяц")[i])
                        return
                # клик по полосе даты — перемещение во времени
                if self._date_rect.w and self._date_rect.collidepoint(px, py):
                    self._jump_date(px)
                    return
                idx = self._palette_index(px, py)
                if idx is not None and idx < len(self.base_ids):
                    self.selected = self.base_ids[idx]
                    return
                if self._cell_at(px, py) is not None:
                    cell = self._cell_at(px, py)
                    self.grab = (px, py, cell[0], cell[1], False, 0, 0)
                else:
                    self.menu_open = None
            elif ev.button == 3:
                self.do_action("Строить")
            return
        if ev.type == pygame.MOUSEMOTION:
            if self.grab is not None:
                # плавное перетаскивание: карта едет за мышью в 2 раза
                # медленнее (накапливаем пиксели, без прыжков-центрирования)
                px0, py0, cx0, cy0, moved, remx, remy = self.grab
                rel = getattr(ev, "rel", None)
                if rel is not None:
                    rx, ry = rel[0], rel[1]
                else:
                    rx, ry = ev.pos[0] - px0, ev.pos[1] - py0
                if rx or ry:
                    moved = True
                remx += rx
                remy += ry
                # int() — округление к нулю: корректно и для отрицательных
                dx = int(remx / (2 * self.tile))
                dy = int(remy / (2 * self.tile))
                remx -= dx * 2 * self.tile
                remy -= dy * 2 * self.tile
                ncx = min(max(self.cx - dx, 0), g.map_size - 1)
                ncy = min(max(self.cy - dy, 0), g.map_size - 1)
                if (ncx, ncy) != (self.cx, self.cy):
                    self.cx, self.cy = ncx, ncy
                    self.map_dirty = True
                self.grab = (px0, py0, cx0, cy0, moved, remx, remy)
                return
            if self.plan_drag:
                self._plan_set(ev.pos[0], ev.pos[1])
                return
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if self.grab is not None:
                px0, py0, _, _, moved, _, _ = self.grab
                self.grab = None
                if not moved:
                    cell = self._cell_at(px0, py0)
                    if cell is not None:
                        self.cx, self.cy = cell
                        self.map_dirty = True
                        self._ensure_visible()
            self.plan_drag = False
            return
        if ev.type == pygame.MOUSEWHEEL:
            self._set_zoom(self.zoom_i + (1 if ev.y > 0 else -1))
            return

        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        # Ctrl+буква — выбрать постройку и строить (как в исходниках)
        if getattr(ev, "mod", 0) & pygame.KMOD_CTRL:
            if pygame.K_a <= k <= pygame.K_z:
                i = k - pygame.K_a
                if i < len(self.base_ids):
                    self.selected = self.base_ids[i]
                    self.do_action("Строить")
                return
        if k in (pygame.K_LEFT,) and self.cx > 0:
            self.cx -= 1
            self.map_dirty = True
            self._ensure_visible()
        elif k in (pygame.K_RIGHT,) and self.cx < g.map_size - 1:
            self.cx += 1
            self.map_dirty = True
            self._ensure_visible()
        elif k in (pygame.K_UP,) and self.cy > 0:
            self.cy -= 1
            self.map_dirty = True
            self._ensure_visible()
        elif k in (pygame.K_DOWN,) and self.cy < g.map_size - 1:
            self.cy += 1
            self.map_dirty = True
            self._ensure_visible()
        elif k == pygame.K_SPACE:
            self.do_action("День")
        elif k == pygame.K_m:
            self.do_action("Налог")
        elif k == pygame.K_b:
            self.do_action("Рынок")
        elif k == pygame.K_s:
            self.do_action("Продать")
        elif k == pygame.K_k:
            self.do_action("Банк")
        elif k == pygame.K_n:
            self.do_action("День")
        elif k == pygame.K_w:
            self.do_action("Неделя")
        elif k == pygame.K_u:
            self.do_action("Отмена")
        elif k == pygame.K_g:
            self.do_action("Земля")
        elif k == pygame.K_d:
            self.do_action("Снос")
        elif k == pygame.K_r:
            self.do_action("Ремонт")
        elif k == pygame.K_a:
            self.do_action("Ремонт всех")
        elif k == pygame.K_f:
            self.do_action("Поиск")
        elif k == pygame.K_p:
            self.do_action("Консервация")
        elif k in (pygame.K_2, pygame.K_3):
            # (оставлено для совместимости) сохранение/загрузка через F2/F5
            self.do_action("Сохранить" if k == pygame.K_2 else "Загрузить")
        elif k == pygame.K_ESCAPE:
            self.menu_open = None if self.menu_open else "Игра"
        elif k == pygame.K_F1:
            self.do_action("Помощь")
        elif k == pygame.K_F2:
            self.do_action("Сохранить как")
        elif k == pygame.K_F4:
            self.do_action("Новая игра")
        elif k == pygame.K_F5:
            self.do_action("Загрузить")
        elif k == pygame.K_F7:
            self.do_action("Звук")
        elif k == pygame.K_F8:
            self.do_action("Музыка")
        elif k == pygame.K_F9:
            self.do_action("Полный экран")
        elif k == pygame.K_F12:
            self.do_action("Параметры")
        elif k in (pygame.K_PLUS, pygame.K_EQUALS):
            self._set_zoom(self.zoom_i + 1)
        elif k in (pygame.K_MINUS,):
            self._set_zoom(self.zoom_i - 1)

    # ================================================================ loop
    def run(self) -> None:
        frame = 0
        while self.running:
            for ev in pygame.event.get():
                self.handle_event(ev)
            frame += 1
            if frame % 4 == 0:          # ~60 мс: анимация рамки (оригинал 55 мс)
                self.sel_frame = (self.sel_frame + 1) % 12
            if frame % 8 == 0:          # ~128 мс: мигание иконки износа
                self.blink_on = not self.blink_on
            if self.map_dirty:
                self.map_dirty = False
                self._plan_cache = None
            self.screen.fill(BG)
            self._draw_top()
            self._draw_map()
            self._draw_panel()
            self._draw_bottom()
            if self.dialog is not None:
                self._draw_dialog(self.dialog)
            if self.messages:
                self.messages[0].draw(self)
            pygame.display.flip()
            pygame.time.wait(16)
        pygame.event.set_grab(False)
        pygame.quit()


def lot_kind(lot: int) -> str:
    return {C.LT_NONE: "none", C.LT_WATER: "water", C.LT_WOOD: "wood",
            C.LT_COAL: "coal", C.LT_IRON: "iron", C.LT_OIL: "oil",
            C.LT_GOLD: "gold"}.get(lot, "land")


def run_game(game: Game) -> None:
    GameWindow(game).run()
