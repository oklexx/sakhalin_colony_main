"""Общие пути для тестов: корень репозитория и каталог python/.

Большинство тестов импортируют `colony_cpp` и модули `rl.*` «как есть». Раньше
каждый файл сам делал sys.path.insert(...), поэтому работоспособность зависела от
порядка сбора: `tests/test_minimap_radius.py` импортирует `colony_cpp` на уровне
модуля и падает, если до него не успел импортироваться файл, добавивший python/ в
sys.path. Здесь пути выставляются один раз до сбора тестов.

Ничего не импортируем на уровне модуля, кроме stdlib: этот файл грузится всегда,
даже когда colony_cpp не собран.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "python"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
