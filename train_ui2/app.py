"""Application entry for Training UI 2.0.

Runs side by side with the old UI: separate state file
(~/colony_runs/sakhalin_colony_ui2/config.json) and separate temp
message/command files (colony_ui2_*), so both can drive the same worker
script concurrently without clashing.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from PySide6.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Sakhalin Colony UI 2.0")
    from train_ui2 import theme as T
    from train_ui2.main_window import MainWindow2
    app.setStyleSheet(T.QSS)
    win = MainWindow2()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
