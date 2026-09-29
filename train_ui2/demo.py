"""Human-game demonstration discovery and validation for Training UI 2.0.

The GUI records JSONL transitions through ``--record-demo``.  This module is
intentionally dependency-free so the UI can inspect/import games before torch
or the C++ extension is loaded.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DemoStats:
    path: Path
    transitions: int = 0
    episodes: int = 0
    obs_size: int | None = None
    n_actions: int | None = None
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.transitions > 0 and not self.errors

    @property
    def status(self) -> str:
        if self.valid:
            return "OK"
        if self.transitions == 0 and not self.errors:
            return "пусто"
        return "ошибка"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def inspect_demo_file(path: str | Path) -> DemoStats:
    """Validate one JSONL demo and return human-readable aggregate stats.

    The checks mirror the invariants consumed by ``rl.bc_dataset``: every
    transition must have an observation, action mask, integer action, reward
    and terminal flag; dimensions must stay constant within a file.
    """
    p = Path(path)
    errors: list[str] = []
    transitions = 0
    episodes = 0
    obs_size: int | None = None
    n_actions: int | None = None

    try:
        fh = p.open(encoding="utf-8")
    except OSError as exc:
        return DemoStats(p, errors=(f"не удалось открыть: {exc}",))

    with fh:
        for line_no, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"строка {line_no}: JSON: {exc.msg}")
                continue
            if not isinstance(row, dict):
                errors.append(f"строка {line_no}: нужен JSON-объект")
                continue

            missing = [key for key in ("obs", "action", "action_mask", "reward", "terminated")
                       if key not in row]
            if missing:
                errors.append(f"строка {line_no}: нет {', '.join(missing)}")
                continue
            obs = row["obs"]
            mask = row["action_mask"]
            action = row["action"]
            if not isinstance(obs, list) or not obs:
                errors.append(f"строка {line_no}: obs не является непустым списком")
                continue
            if not all(_is_number(x) for x in obs):
                errors.append(f"строка {line_no}: obs содержит нечисловое значение")
                continue
            if not isinstance(mask, list) or not mask:
                errors.append(f"строка {line_no}: action_mask не является непустым списком")
                continue
            if not all(_is_number(x) for x in mask):
                errors.append(f"строка {line_no}: action_mask содержит нечисловое значение")
                continue
            if not isinstance(action, int) or isinstance(action, bool):
                errors.append(f"строка {line_no}: action должен быть целым числом")
                continue
            if action < 0 or action >= len(mask):
                errors.append(f"строка {line_no}: action={action} вне mask[0:{len(mask)})")
            if not _is_number(row["reward"]):
                errors.append(f"строка {line_no}: reward должен быть числом")
            if not isinstance(row["terminated"], bool):
                errors.append(f"строка {line_no}: terminated должен быть bool")

            transitions += 1
            episodes += int(row.get("terminated") is True)
            if obs_size is None:
                obs_size = len(obs)
            elif obs_size != len(obs):
                errors.append(f"строка {line_no}: obs имеет размер {len(obs)}, ожидался {obs_size}")
            if n_actions is None:
                n_actions = len(mask)
            elif n_actions != len(mask):
                errors.append(f"строка {line_no}: mask имеет размер {len(mask)}, ожидался {n_actions}")

    # Do not flood the UI when a damaged file has thousands of bad rows.
    if len(errors) > 8:
        errors = errors[:8] + [f"… и ещё {len(errors) - 8} ошибок"]
    return DemoStats(p, transitions, episodes, obs_size, n_actions, tuple(errors))


def scan_demo_dir(root: str | Path) -> list[DemoStats]:
    """Return all JSONL games in a directory, sorted by filename."""
    directory = Path(root)
    if not directory.exists() or not directory.is_dir():
        return []
    return [inspect_demo_file(path) for path in sorted(directory.glob("*.jsonl"))]


def copy_demo_files(files: Iterable[str | Path], destination: str | Path) -> list[Path]:
    """Copy selected JSONL games into the UI's demo directory."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source_raw in files:
        source = Path(source_raw)
        if source.suffix.lower() != ".jsonl" or not source.is_file():
            continue
        out = target / source.name
        if out.resolve() != source.resolve():
            shutil.copy2(source, out)
        copied.append(out)
    return copied


def aggregate_demo_stats(stats: Iterable[DemoStats]) -> dict[str, int | None | bool]:
    """Summarize a collection for the BC preflight check."""
    rows = list(stats)
    valid = [row for row in rows if row.valid]
    obs_sizes = {row.obs_size for row in valid if row.obs_size is not None}
    action_sizes = {row.n_actions for row in valid if row.n_actions is not None}
    return {
        "files": len(rows),
        "valid_files": len(valid),
        "transitions": sum(row.transitions for row in valid),
        "episodes": sum(row.episodes for row in valid),
        "obs_size": next(iter(obs_sizes)) if len(obs_sizes) == 1 else None,
        "n_actions": next(iter(action_sizes)) if len(action_sizes) == 1 else None,
        # BC receives the whole directory, so one malformed file must block the
        # launch instead of failing several minutes later inside the trainer.
        "consistent": bool(rows) and len(valid) == len(rows)
        and len(obs_sizes) == 1 and len(action_sizes) == 1,
    }
