from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvState:
    """State tracking for a single environment."""
    env_idx: int
    last_action: str = ""
    consecutive_count: int = 0
    action_sequence: deque = field(default_factory=lambda: deque(maxlen=100))


class LoopDetector:
    """Detects action loops in training environments.

    Tracks consecutive same-action sequences per environment and alerts
    when an action repeats beyond a configurable threshold.
    """

    def __init__(self, threshold_config: dict[str, Any] | None = None):
        self._consecutive_threshold = 3
        if threshold_config:
            self._consecutive_threshold = threshold_config.get(
                "consecutive_threshold", 3
            )
        self.history_size = (
            threshold_config.get("history_size", 1000) if threshold_config else 1000
        )

        self._state: dict[int, EnvState] = {}
        self._history: list[dict[str, Any]] = []

        self._total_loops_detected = 0
        self._total_actions_sampled = 0
        self._loop_start_times: dict[int, float] = {}

    def _get_or_create_env_state(self, env_idx: int) -> EnvState:
        if env_idx not in self._state:
            self._state[env_idx] = EnvState(env_idx=env_idx)
        return self._state[env_idx]

    def update_batch(self, action_data: list[dict[str, Any]]) -> dict[int, str | None]:
        """Process multiple environments at once.

        Args:
            action_data: List of dicts with keys: env_idx, action, step (optional).

        Returns:
            Dict mapping env_idx -> action_name if loop detected, else None.
        """
        alerts: dict[int, str | None] = {}

        for data in action_data:
            env_idx = data["env_idx"]
            action = data.get("action", "")

            state = self._get_or_create_env_state(env_idx)

            # Update consecutive count
            if action == state.last_action:
                state.consecutive_count += 1
            else:
                state.consecutive_count = 1
            state.last_action = action

            state.action_sequence.append(action)
            self._total_actions_sampled += 1

            # Check for loop
            if state.consecutive_count >= self._consecutive_threshold:
                if env_idx not in alerts:
                    alerts[env_idx] = action
                    self._total_loops_detected += 1
                    self._loop_start_times[env_idx] = time.time()

            # Add to global history (sample every 10 updates)
            if self._total_actions_sampled % 10 == 0:
                self._history.append({
                    "timestamp": time.time(),
                    "envs_with_loops": len(alerts),
                    "alerts": alerts.copy(),
                })
                while len(self._history) > self.history_size:
                    self._history.pop(0)

        return alerts

    def get_stats(self) -> dict[str, Any]:
        if not self._state:
            return {
                "total_envs": 0,
                "envs_with_loops": 0,
                "max_consecutive": 0,
                "avg_consecutive": 0.0,
                "loop_rate_percent": 0.0,
                "consecutive_threshold": self._consecutive_threshold,
                "total_loops_detected": 0,
            }

        max_consec = 0
        total_consec = 0
        envs_with_loops = 0

        for state in self._state.values():
            max_consec = max(max_consec, state.consecutive_count)
            total_consec += state.consecutive_count
            if state.consecutive_count >= self._consecutive_threshold:
                envs_with_loops += 1

        avg_consec = total_consec / len(self._state) if self._state else 0.0

        total_samples = sum(len(d.get("alerts", {})) for d in self._history)
        loop_rate = (
            (total_samples / (len(self._history) * max(1, self._consecutive_threshold))) * 100
            if self._history
            else 0.0
        )

        return {
            "total_envs": len(self._state),
            "envs_with_loops": envs_with_loops,
            "max_consecutive": max_consec,
            "avg_consecutive": avg_consec,
            "loop_rate_percent": loop_rate,
            "consecutive_threshold": self._consecutive_threshold,
            "total_loops_detected": self._total_loops_detected,
        }

    def clear(self) -> None:
        self._state.clear()
        self._history.clear()
        self._loop_start_times.clear()
        self._total_loops_detected = 0
        self._total_actions_sampled = 0

    @property
    def consecutive_threshold(self) -> int:
        return self._consecutive_threshold

    @consecutive_threshold.setter
    def consecutive_threshold(self, value: int) -> None:
        self._consecutive_threshold = max(1, value)
        if hasattr(self, '_state') and self._state:
            for state in self._state.values():
                if state.consecutive_count >= value:
                    state.consecutive_count = value
