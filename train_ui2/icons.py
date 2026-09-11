"""Icon loading helpers — extracted from main_window for testability."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from train_ui2.constants import BUILD_IMAGE_INDEX

_PROJECT = Path(__file__).resolve().parent.parent


def load_icon_pixmap(name: str, size: int = 24) -> Optional[QPixmap]:
    """Try to load an icon from assets; return None if not found."""
    candidates = [
        _PROJECT / "assets" / name,
        _PROJECT / "ui" / "assets" / name,
        _PROJECT / "train_ui2" / name,
    ]
    for path in candidates:
        if path.exists():
            pm = QPixmap(str(path))
            if not pm.isNull():
                return pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return None


def building_icon(bid: str) -> Optional[QPixmap]:
    idx = BUILD_IMAGE_INDEX.get(bid)
    if idx is None:
        return None
    return load_icon_pixmap(f"imlBases_{idx:02d}.png", 24)


def resource_icon(rid: str) -> Optional[QPixmap]:
    mapping = {"gold": 0, "food": 1, "coal": 2, "iron": 3, "oil": 4, "stone": 5, "water": 6, "wood": 7, "energy": 8}
    idx = mapping.get(rid, 0)
    pm = load_icon_pixmap(f"imlMarketItem_{idx:02d}.png", 20)
    if pm is None:
        pm = load_icon_pixmap(f"imlIcons_0{idx % 6}.png", 20)
    return pm


def format_steps(n: float) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1000:.0f}k"
    return str(n)


def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
