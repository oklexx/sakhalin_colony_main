from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MsgType(str, Enum):
    READY = "ready"
    LOG = "log"
    PROGRESS = "progress"
    SAVED = "saved"
    DONE = "done"
    ERROR = "error"
    COMMAND = "command"


@dataclass
class ReadyMsg:
    type: MsgType = field(default=MsgType.READY, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "ready"}


@dataclass
class LogMsg:
    level: str
    message: str
    type: MsgType = field(default=MsgType.LOG, init=False)

    def __post_init__(self):
        if self.level not in ("info", "warn", "error"):
            raise ValueError(f"invalid log level: {self.level!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "log", "level": self.level, "message": self.message}


def _safe_float(x: float, default: float = 0.0) -> float:
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return float(x)


def _safe_int(x: int, default: int = 0) -> int:
    """Целое из «сырого» числа метрик: NaN/inf/мусор не должны валить JSONL."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return int(f)


@dataclass
class ProgressMsg:
    done: int
    total: int
    fps: float = 0.0
    best_reward: float = 0.0
    episodes: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl: float = 0.0
    ent_coef: float = 0.0  # display-only, always provided by caller
    top_actions: Dict[str, float] = field(default_factory=dict)
    # Мониторинг действий (2026-09-23): сырые счётчики за роллаут и доля
    # шагов, в которых действие было легальным (маска = 1). Без них «строка
    # пропала» не читается: политика разлюбила или действие закрыто маской.
    action_counts: Dict[str, int] = field(default_factory=dict)
    action_legality: Dict[str, float] = field(default_factory=dict)
    # Доминирующая причина закрытия маски (MASK_REASON_KEYS): {action: reason}.
    # Пусто = не измерялось; UI не гадает и показывает прежний вердикт.
    action_mask_reasons: Dict[str, str] = field(default_factory=dict)
    loop_detected: bool = False
    loop_action_name: Optional[str] = None
    envs_with_loops: int = 0
    curriculum_stage_active: int = 0
    curriculum_next_at_step: Optional[int] = None
    # Curriculum extended fields
    curriculum_stage: int = 0
    curriculum_progress_percent: float = 0.0
    # False = прогресс не измерялся (нет расписания этапов или среда не
    # отдаёт его): UI показывает «н/д», а не вечные 0%.
    curriculum_progress_valid: bool = True
    # Знаменатель долей действий = шагов в собранном роллауте (n_steps*n_envs).
    action_total_steps: int = 0
    curriculum_available_actions: str = ""
    curriculum_upcoming_stages: List[Dict[str, int]] = field(default_factory=list)
    # Return statistics
    avg_return: float = 0.0
    median_return: float = 0.0
    max_return: float = 0.0
    min_return: float = 0.0
    n_episodes_for_stats: int = 0
    type: MsgType = field(default=MsgType.PROGRESS, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "progress",
            "done": int(self.done),
            "total": int(self.total),
            "fps": _safe_float(self.fps),
            "best_reward": _safe_float(self.best_reward),
            "episodes": int(self.episodes),
            "policy_loss": _safe_float(self.policy_loss),
            "value_loss": _safe_float(self.value_loss),
            "entropy": _safe_float(self.entropy),
            "kl": _safe_float(self.kl),
            "ent_coef": _safe_float(self.ent_coef),
            "top_actions": self.top_actions,
            "action_counts": {k: _safe_int(v) for k, v in self.action_counts.items()},
            "action_legality": {k: _safe_float(v) for k, v in self.action_legality.items()},
            # JSON-safe: None выкидываем, прочее → строка (значения — ключи MASK_REASON_KEYS)
            "action_mask_reasons": {
                str(k): str(v) for k, v in self.action_mask_reasons.items()
                if v is not None
            },
            "action_total_steps": int(self.action_total_steps),
            "loop_detected": self.loop_detected,
            "loop_action_name": self.loop_action_name,
            "envs_with_loops": int(self.envs_with_loops),
            "curriculum_stage_active": int(self.curriculum_stage_active),
            "curriculum_next_at_step": (
                int(self.curriculum_next_at_step)
                if self.curriculum_next_at_step is not None
                else None
            ),
            "curriculum_stage": int(self.curriculum_stage),
            "curriculum_progress_percent": _safe_float(self.curriculum_progress_percent),
            "curriculum_progress_valid": bool(self.curriculum_progress_valid),
            "curriculum_available_actions": self.curriculum_available_actions,
            "curriculum_upcoming_stages": self.curriculum_upcoming_stages,
            "avg_return": _safe_float(self.avg_return),
            "median_return": _safe_float(self.median_return),
            "max_return": _safe_float(self.max_return),
            "min_return": _safe_float(self.min_return),
            "n_episodes_for_stats": int(self.n_episodes_for_stats),
        }


@dataclass
class SavedMsg:
    path: str
    type: MsgType = field(default=MsgType.SAVED, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "saved", "path": self.path}


@dataclass
class DoneMsg:
    total: int
    time_s: float
    best_reward: float
    episodes: int
    type: MsgType = field(default=MsgType.DONE, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "done",
            "total": int(self.total),
            "time_s": _safe_float(self.time_s),
            "best_reward": _safe_float(self.best_reward),
            "episodes": int(self.episodes),
        }


@dataclass
class ErrorMsg:
    message: str
    type: MsgType = field(default=MsgType.ERROR, init=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "error", "message": self.message}


# ── Команды UI → worker → trainer ─────────────────────────────────────────
# Единственный источник строк команд (P1-4 ревью 2026-09-24). Раньше контракт
# жил в трёх несогласованных видах: UI писал `stop_training`/`pause_training`,
# stdin-читатель воркера понимал `stop`/`pause`, а трейнер — только
# `*_training`; `pause` из stdin уходил в трейнер и молча игнорировался.
# Теперь UI, воркер и `AsyncTrainer._process_commands` берут строки отсюда,
# а старые короткие имена приводятся к каноническим через `normalize_command`.
CMD_STOP = "stop_training"
CMD_PAUSE = "pause_training"
CMD_RESUME = "resume_training"
CMD_BOOST_ENTROPY = "boost_entropy"
CMD_RESET_CURRICULUM = "reset_curriculum"


class CommandType(str, Enum):
    """Канонические команды; значения совпадают с константами `CMD_*`."""

    STOP_TRAINING = CMD_STOP
    PAUSE_TRAINING = CMD_PAUSE
    RESUME_TRAINING = CMD_RESUME
    BOOST_ENTROPY = CMD_BOOST_ENTROPY
    RESET_CURRICULUM = CMD_RESET_CURRICULUM


#: Всё, что трейнер обязан уметь обработать (тест паритета UI → trainer).
KNOWN_COMMANDS: frozenset = frozenset(c.value for c in CommandType)

#: Устаревшие короткие имена (stdin-протокол, `encode_stop` до 2026-09-24).
LEGACY_COMMAND_ALIASES: Dict[str, str] = {
    "stop": CMD_STOP,
    "pause": CMD_PAUSE,
    "resume": CMD_RESUME,
}


def normalize_command(cmd: Any) -> Optional[str]:
    """Каноническое имя команды или None, если команда неизвестна.

    None — сигнал вызывающему громко предупредить (опечатка, чужой формат),
    а не молча выбросить команду, как это было до 2026-09-24.
    """
    if not isinstance(cmd, str):
        return None
    name = LEGACY_COMMAND_ALIASES.get(cmd, cmd)
    return name if name in KNOWN_COMMANDS else None


@dataclass
class CommandMsg:
    cmd: str
    payload: Optional[Dict[str, Any]] = None
    type: MsgType = field(default=MsgType.COMMAND, init=False)

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "command",
            "cmd": str(self.cmd),
            "payload": self.payload or {},
        }


def encode_command(cmd: str, payload: Optional[Dict[str, Any]] = None) -> str:
    """Encode a command to JSON."""
    msg = CommandMsg(cmd=cmd, payload=payload or {})
    return json.dumps(msg.to_dict(), ensure_ascii=False, allow_nan=False)


def decode_command(line: str) -> Dict[str, Any]:
    """Parse a command from JSON line.
    
    Returns dict with 'cmd' and 'payload' keys.
    """
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    
    if not isinstance(d, dict):
        raise ValueError("command must be a JSON object")
    
    t = d.get("type")
    if t != "command":
        raise ValueError(f"expected command type, got: {t}")
    
    cmd = d.get("cmd")
    if cmd is None:
        raise ValueError("missing 'cmd' field")
    
    payload = d.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("'payload' must be an object")
    
    return {"cmd": cmd, "payload": payload}


Msg = ReadyMsg | LogMsg | ProgressMsg | SavedMsg | DoneMsg | ErrorMsg | CommandMsg

_REQUIRED: Dict[MsgType, tuple] = {
    MsgType.READY: (),
    MsgType.LOG: ("level", "message"),
    MsgType.PROGRESS: ("done", "total"),
    MsgType.SAVED: ("path",),
    MsgType.DONE: ("total", "time_s", "best_reward", "episodes"),
    MsgType.ERROR: ("message",),
    MsgType.COMMAND: ("cmd",),
}


def encode(msg: Msg) -> str:
    """Serialize a message to a single JSON line (no trailing newline)."""
    d = msg.to_dict()
    return json.dumps(d, ensure_ascii=False, allow_nan=False)


def decode(line: str) -> Msg:
    """Parse one JSON line into a typed message.

    Raises ValueError on malformed input or unknown type.
    """
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(d, dict):
        raise ValueError("message must be a JSON object")
    t = d.get("type")
    try:
        mt = MsgType(t)
    except ValueError as e:
        raise ValueError(f"unknown message type: {t!r}") from e
    
    # Handle command messages
    if mt is MsgType.COMMAND:
        return CommandMsg(
            cmd=d["cmd"],
            payload=d.get("payload", {}),
        )
    
    missing = [k for k in _REQUIRED[mt] if k not in d]
    if missing:
        raise ValueError(f"missing fields for {mt.value}: {missing}")
    if mt is MsgType.READY:
        return ReadyMsg()
    if mt is MsgType.LOG:
        return LogMsg(level=d["level"], message=d["message"])
    if mt is MsgType.PROGRESS:
        return ProgressMsg(
            done=d["done"],
            total=d["total"],
            fps=d.get("fps", 0.0),
            best_reward=d.get("best_reward", float("-inf")),
            episodes=d.get("episodes", 0),
            policy_loss=d.get("policy_loss", 0.0),
            value_loss=d.get("value_loss", 0.0),
            entropy=d.get("entropy", 0.0),
            kl=d.get("kl", 0.0),
            ent_coef=d.get("ent_coef", 0.005),
            top_actions={k: _safe_float(v) for k, v in d.get("top_actions", {}).items()},
            loop_detected=d.get("loop_detected", False),
            loop_action_name=d.get("loop_action_name"),
            envs_with_loops=d.get("envs_with_loops", 0),
            curriculum_stage_active=d.get("curriculum_stage_active", 0),
            curriculum_next_at_step=d.get("curriculum_next_at_step"),
            curriculum_stage=d.get("curriculum_stage", 0),
            curriculum_progress_percent=d.get("curriculum_progress_percent", 0.0),
            curriculum_progress_valid=bool(d.get("curriculum_progress_valid", True)),
            action_counts={k: _safe_int(v) for k, v in d.get("action_counts", {}).items()},
            action_legality={k: _safe_float(v) for k, v in d.get("action_legality", {}).items()},
            action_mask_reasons=(
                {str(k): str(v) for k, v in d.get("action_mask_reasons").items()}
                if isinstance(d.get("action_mask_reasons"), dict) else {}
            ),
            action_total_steps=_safe_int(d.get("action_total_steps", 0)),
            curriculum_available_actions=d.get("curriculum_available_actions", ""),
            curriculum_upcoming_stages=d.get("curriculum_upcoming_stages", []),
            avg_return=d.get("avg_return", 0.0),
            median_return=d.get("median_return", 0.0),
            max_return=d.get("max_return", 0.0),
            min_return=d.get("min_return", 0.0),
            n_episodes_for_stats=d.get("n_episodes_for_stats", 0),
        )
    if mt is MsgType.SAVED:
        return SavedMsg(path=d["path"])
    if mt is MsgType.DONE:
        return DoneMsg(
            total=d["total"],
            time_s=d["time_s"],
            best_reward=d["best_reward"],
            episodes=d["episodes"],
        )
    return ErrorMsg(message=d["message"])


def encode_stop() -> str:
    """Команда мягкой остановки: трейнер сохраняет final_model/мету и выходит."""
    payload = {"final_save": True}
    msg = CommandMsg(cmd=CMD_STOP, payload=payload)
    return json.dumps(msg.to_dict(), ensure_ascii=False, allow_nan=False)
