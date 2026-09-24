"""Append-only JSONL records for terminal training episodes.

The file is deliberately independent of TensorBoard and the UI message stream:
`episode_diagnostics.jsonl` stays beside a run's checkpoints and can be copied
for offline analysis even if the GUI/worker exits unexpectedly.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import uuid
from typing import Any


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Convert common Python/numpy scalars and containers to JSON values.

    NaN/±Inf → None (JSON null): запись идёт с `allow_nan=False`, и один
    нефинитный reward (расходящееся обучение) давал ValueError внутри
    `append_episode`; трейнер ловит его как «warning once», так что
    диагностика молча прекращалась до конца прогона (P3-5 ревью 2026-09-24).
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


class EpisodeDiagnosticsWriter:
    """Thread-safe, fsynced JSONL writer with a small, versioned schema."""

    def __init__(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None,
                 run_id: str | None = None, fsync: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex
        self._fsync = bool(fsync)
        self._lock = threading.RLock()
        self._episode_index = 0
        self._append({
            "record_type": "run_start",
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp_utc": _utc_now(),
            "metadata": _json_safe(metadata or {}),
        })

    def _append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
                stream.flush()
                if self._fsync:
                    os.fsync(stream.fileno())

    def append_episode(self, info: Mapping[str, Any], *, total_timesteps: int,
                       env_index: int, curriculum_stage: int) -> bool:
        """Append one done-info object; return False only if it is not an episode."""
        episode = info.get("episode")
        if not isinstance(episode, Mapping):
            return False
        metrics = episode.get("metrics", info.get("episode_metrics"))
        metrics_available = isinstance(metrics, Mapping)
        with self._lock:
            self._episode_index += 1
            episode_index = self._episode_index
            final_state = {
                key: episode.get(key)
                for key in ("days", "people", "money", "bases")
                if key in episode
            }
            self._append({
                "record_type": "episode",
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "timestamp_utc": _utc_now(),
                "episode_index": episode_index,
                "total_timesteps": int(total_timesteps),
                "env_index": int(env_index),
                "curriculum_stage": int(curriculum_stage),
                "seed": episode.get("seed"),
                "episode_return": episode.get("r"),
                "episode_length": episode.get("l"),
                "final_state": final_state,
                "metrics_available": metrics_available,
                "episode_metrics": _json_safe(metrics) if metrics_available else None,
            })
        return True

    def close(self, *, total_timesteps: int, status: str = "completed") -> None:
        self._append({
            "record_type": "run_end",
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp_utc": _utc_now(),
            "total_timesteps": int(total_timesteps),
            "status": str(status),
            "episodes_logged": self._episode_index,
        })
