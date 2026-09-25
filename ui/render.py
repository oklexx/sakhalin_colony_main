# Графический интерфейс: иконки ресурсов и построек, извлечённые из
# оригинальных исходников (unData.dfm, TImageList) в ui/assets/*.png.

from __future__ import annotations

import os

import pygame

from core import constants as C

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# ---- палитра (стиль оригинала) ------------------------------------------
BG = (0, 7, 0)
FG = (215, 215, 205)
FG_DIM = (125, 135, 125)
FG_TITLE = (120, 200, 120)
FG_ACCENT = (255, 220, 120)
FG_ERROR = (255, 110, 110)
FG_MENU = (225, 225, 235)

PANEL_HEAD = (132, 207, 228)   # светло-голубая шапка панели
PANEL_BG = (222, 228, 220)     # светлый фон панели
PANEL_FG = (20, 25, 20)        # тёмный текст на панели
LEGEND_BG = (224, 224, 144)    # желтоватая полоса легенды
CURSOR_COLOR = (255, 80, 80)   # курсор

# ---- типы местности: фон ячейки -----------------------------------------
TERRAIN_BG: dict[int, tuple] = {
    C.LT_NONE: (186, 211, 178),         # чистое поле — обычная земля
    C.LT_NORMAL: (186, 211, 178),   # ровная земля — светло-зелёная
    C.LT_WATER: (131, 208, 227),    # океан — голубой
    C.LT_WOOD: (186, 211, 178),
    C.LT_COAL: (186, 211, 178),
    C.LT_IRON: (186, 211, 178),
    C.LT_OIL: (186, 211, 178),
    C.LT_GOLD: (186, 211, 178),
}

GOOD_BG = (170, 230, 170)
DESTROYED_BG = (200, 120, 100)

# ---- русские названия ----------------------------------------------------
LOT_NAMES: dict[int, str] = {
    C.LT_NONE: "ничего",
    C.LT_NORMAL: "ровная земля",
    C.LT_WATER: "вода",
    C.LT_WOOD: "лес",
    C.LT_COAL: "уголь",
    C.LT_IRON: "железо",
    C.LT_OIL: "нефть",
    C.LT_GOLD: "золото",
}

RES_SHORT = ["Золото", "Продовольствие", "Уголь", "Железо", "Нефть",
             "Камень", "Вода", "Дерево", "Энергия"]

# --------------------------------------------------------------------------
# иконки извлечены из оригинала (unData.dfm, TImageList) в ui/assets/*.png.
# Маппинг imlEarth: (LotType - ltWater) * 10 + SubType  (unMain.cpp:195).
# imlBases: ImageIndex из BASES.INI.  imlIcons: служебные значки 14x14.
# --------------------------------------------------------------------------

IMG_WATER = (C.LT_WATER - 2) * 10     # 0..9  вода
IMG_WOOD = (C.LT_WOOD - 2) * 10       # 10..19 лес
IMG_COAL = (C.LT_COAL - 2) * 10       # 20..29 уголь
IMG_IRON = (C.LT_IRON - 2) * 10       # 30..39 железо
IMG_OIL = (C.LT_OIL - 2) * 10         # 40..49 нефть
IMG_GOLD = (C.LT_GOLD - 2) * 10       # 50..59 золото
IMG_DESTROYED = 64                    # сгоревший лот (unMain.cpp:197)

# imlIcons: 0=умирает, 1=строится, 2=хорошая земля, 3=законсервирована,
#           4=нужны ресурсы, 5=не сезон, 6=нужны рабочие (unMainUtils.cpp)
ICON_DEAD, ICON_BUILD, ICON_GOOD, ICON_PRESERVE, ICON_NEED_SUNDUK, \
    ICON_NO_SEASON, ICON_NEED_WORKERS = range(7)

# imlTools (панель): 0..8 ресурсы, 9..16 действия, 17 деньги, 18 люди
TOOL_GOLD, TOOL_FOOD, TOOL_COAL, TOOL_IRON, TOOL_OIL, TOOL_STONE, \
    TOOL_WATER, TOOL_WOOD, TOOL_ENERGY, TOOL_GOOD_EARTH, TOOL_NEW_DAY, \
    TOOL_SALE, TOOL_BUY, TOOL_DESTROY, TOOL_NEW_WEEK, TOOL_FIND_SLOW, \
    TOOL_RESTORE, TOOL_MONEY, TOOL_PEOPLE = range(19)

_asset_cache: dict[str, pygame.Surface] = {}


def _asset(name: str) -> pygame.Surface:
    s = _asset_cache.get(name)
    if s is None:
        s = pygame.image.load(os.path.join(ASSETS, name + ".png")).convert_alpha()
        _asset_cache[name] = s
    return s


def _scaled(surf: pygame.Surface, size: int) -> pygame.Surface:
    if surf.get_width() == size:
        return surf
    return pygame.transform.smoothscale(surf, (size, size))


def earth_img(lot_type: int, sub: int = 0) -> int:
    base = {C.LT_WATER: IMG_WATER, C.LT_WOOD: IMG_WOOD, C.LT_COAL: IMG_COAL,
            C.LT_IRON: IMG_IRON, C.LT_OIL: IMG_OIL,
            C.LT_GOLD: IMG_GOLD}.get(lot_type)
    if base is None:
        return 0
    return base + sub


def icon(kind: str, size: int) -> pygame.Surface:
    if kind == "good":
        name = f"imlIcons_{ICON_GOOD:02d}"
    elif kind == "destroyed":
        name = f"imlEarth_{IMG_DESTROYED:02d}"
    elif kind in BASE_ICON:
        name = f"imlBases_{BASE_ICON[kind][0]:02d}"
    elif kind in ("water", "wood", "coal", "iron", "oil", "gold"):
        lt = {"water": C.LT_WATER, "wood": C.LT_WOOD, "coal": C.LT_COAL,
              "iron": C.LT_IRON, "oil": C.LT_OIL,
              "gold": C.LT_GOLD}[kind]
        name = f"imlEarth_{earth_img(lt):02d}"
    else:
        name = ""
    if not name:
        s = pygame.Surface((size, size))
        s.fill(TERRAIN_BG.get(C.LT_NORMAL, (0, 0, 0)))
        return s
    return _scaled(_asset(name), size)


def base_icon_by_index(image_index: int, size: int) -> pygame.Surface:
    return _scaled(_asset(f"imlBases_{image_index:02d}"), size)


def tool_icon(idx: int, size: int) -> pygame.Surface:
    return _scaled(_asset(f"imlTools_{idx:02d}"), size)


def icons_icon(idx: int, size: int) -> pygame.Surface:
    return _scaled(_asset(f"imlIcons_{idx:02d}"), size)


# ---- легенда -------------------------------------------------------------
LEGEND = [
    (C.LT_WATER, "вода"),
    (C.LT_NORMAL, "земля"),
    (C.LT_WOOD, "лес"),
    (C.LT_COAL, "уголь"),
    (C.LT_IRON, "железо"),
    (C.LT_OIL, "нефть"),
    (C.LT_GOLD, "золото"),
]


def lot_caption(lot: int) -> str:
    return LOT_NAMES.get(lot, "?")


def base_resource_caption(base) -> str:
    prof = base.data.profit
    cons = base.data.consume
    for s in (prof, cons):
        for i in range(C.SUNDUK_SIZE):
            if s.items[i] != 0:
                return RES_SHORT[i]
    return ""


# ---- соответствие постройки -> иконка imlBases (ImageIndex из BASES.INI 3.47) --
BASE_ICON: dict[str, tuple[int, str]] = {
    "City": (10, "Город"),
    "Farm": (0, "Ферма"),
    "Garden": (1, "Сад"),
    "WaterChannel": (2, "Водоканал"),
    "Sawmill": (3, "Лесопилка"),
    "Coalmine": (4, "Шахта"),
    "Ironmine": (5, "Карьер"),
    "Refinery": (6, "Нефтедобыча"),
    "Goldmine": (7, "Золотой прииск"),
    "PowerStation": (8, "Электростанция"),
    "HydroStation": (9, "Гидростанция"),
    "Road": (12, "Дорога"),
    "House": (13, "Жилой дом"),
    "SmallHouse": (14, "Хижина"),
    "Fish": (15, "Рыбный промысел"),
    "CoalCut": (11, "Угольный разрез"),
    "HuntingLand": (16, "Охотничьи угодья"),
    "CowFarm": (17, "Животноводческая ферма"),
    "Mushroom": (18, "Грибная плантация"),
    "BigHouse": (19, "Жилой район"),
    "BigFarm": (20, "Хозяйство"),
    "Apiary": (21, "Пасека"),
    "Torchlight": (22, "Факел"),
    "Hothouse": (23, "Теплица"),
    "SuperHouse": (24, "Жилой центр"),
    "BigSawmill": (25, "Лесоповал"),
    "WaterMill": (26, "Водокачка"),
    "BigRefinary": (27, "Нефтенасос"),
    "Puerperal": (28, "Дом матери и ребенка"),
    "BigIronmine": (29, "Катакомбы"),
    "AirStation": (30, "Ветряная электростанция"),
    "SmallAtomStation": (31, "Малая АЭС"),
    "AtomStation": (32, "АЭС"),
}
