from __future__ import annotations

import json
import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.config import Config
from rl.env_manager import EnvManager
from rl.episode_diagnostics import EpisodeDiagnosticsWriter
from rl.loop_detector import LoopDetector
from train_ui2 import protocol as P


@dataclass
class TrainMetrics:
    total_timesteps: int = 0
    fps: float = 0.0
    episode_return: float = 0.0
    episode_length: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    learning_rate: float = 0.0
    gpu_mem_mb: float = 0.0
    wall_time_s: float = 0.0
    n_episodes: int = 0
    best_reward: float = float("-inf")
    eval_days: float = 0.0
    eval_people: float = 0.0
    eval_bases: float = 0.0
    best_eval_days: float = 0.0
    eval_score: float = 0.0
    best_score: float = 0.0
    ent_coef: float = 0.005
    avg_return: float = 0.0
    median_return: float = 0.0
    max_return: float = 0.0
    min_return: float = 0.0
    n_episodes_for_stats: int = 0
    top_actions: dict[str, float] = field(default_factory=dict)
    # Ненулевые счётчики действий за роллаут: по ним UI восстанавливает числа
    # для закрепленных строк, даже если действие не попало в `top_actions`.
    action_counts: dict[str, int] = field(default_factory=dict)
    # Доля шагов роллаута, на которых действие было легальным (маска == 1),
    # 0..100. Без неё «строка пропала» неоднозначна: политика разлюбила или
    # действие закрыто маской (для водоканала — не было свободного LT_WATER).
    action_legality: dict[str, float] = field(default_factory=dict)
    # Доминирующая причина закрытия маски за роллаут ("money"/"no_lot"/
    # "curriculum"/"other") по действию; пусто = не измерялось или маска не
    # закрывалась (rl/action_monitor.MaskReasonMonitor, §4 панели).
    action_mask_reasons: dict[str, str] = field(default_factory=dict)
    # Знаменатель долей = шагов в собранном роллауте (n_steps*n_envs).
    action_total_steps: int = 0
    loop_detected: bool = False
    loop_action_name: str | None = None
    envs_with_loops: int = 0
    curriculum_stage: int = 0
    # Доля 0..1 (не проценты!): UI умножает на 100. Название оставлено ради
    # совместимости протокола — см. curriculum_progress_valid.
    curriculum_progress_percent: float = 0.0
    # False = прогресс не измерялся (среда/воркер его не предоставили или
    # расписания этапов нет). Отличает честное «н/д» от вечно нулевых 0%.
    curriculum_progress_valid: bool = True
    curriculum_available_actions: str = ""
    curriculum_next_at_step: int | None = None
    curriculum_upcoming_stages: list = field(default_factory=list)
    # Итог прогона (P1-1 ревью 2026-09-24): воркер пишет их в meta.json, чтобы
    # остановку пользователем можно было отличить от завершения по шагам.
    # stop_reason: "" (дошли до total) | "user" | "early_stop".
    stop_reason: str = ""
    # tournament: "done" | "skipped_user_stop" | "interrupted_user_stop" |
    # "error" | "" (не дошли).
    tournament: str = ""


class AsyncTrainer:
    """Async PPO trainer: producer (env steps) + consumer (GPU PPO update)."""

    def __init__(
        self,
        cfg: Config,
        env_manager: EnvManager,
        logger: Any | None = None,
        progress_callback: Callable[[TrainMetrics], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ):
        self.cfg = cfg
        self.em = env_manager
        self.device = env_manager.device
        self.logger = logger
        self.progress_callback = progress_callback
        self.stop_check = stop_check

        self._stop = False
        # Почему выставлен _stop: "user" (команда/stop_check) | "early_stop".
        self._stop_reason: str = ""
        # Очередь команд текущего train(): опрашивается и внутри роллаута.
        self._command_queue: queue.Queue | None = None
        self._paused = False
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=cfg.queue_size)

        self.metrics = TrainMetrics()
        self.best_reward = float("-inf")
        self.best_eval_days: float | None = None
        self.best_score: float | None = None
        self._es_patience = 0
        self._es_best_score: float | None = None
        self._ep_returns: list[float] = []
        self._ep_returns_maxlen = 10000
        self._ep_lengths: list[int] = []
        self._episode_diagnostics: EpisodeDiagnosticsWriter | None = None
        self._diagnostic_total_timesteps = 0
        self._episode_diagnostics_warned = False
        self._curriculum_stage = getattr(cfg, "curriculum_stage", 0)
        self._curriculum_progress_step = 0

        # Loop detection
        self.loop_detector = (
            LoopDetector({"consecutive_threshold": cfg.loop_consecutive_threshold})
            if getattr(cfg, "loop_detection_enabled", False) else None
        )

        # Action names from C++ env
        self._action_names: list[str] = getattr(env_manager, 'action_names', [])
        if not self._action_names:
            # Fallback if env doesn't provide action_names. Index-aligned
            # with the standard layout (DAY, WEEK, 32 builds, 11 managers,
            # 4 road directions); build slots stay generic — only the env knows
            # the real order (a hardcoded guess here once mislabeled Road as
            # GOLDMINE).
            self._action_names = (
                ["DAY", "WEEK"]
                + [f"BUILD_{i}" for i in range(32)]
                + ["IMPROVE_LAND", "REPAIR", "REPAIR_ALL", "DEMOLISH",
                   "PRESERVE", "UNPRESERVE", "SELL_SURPLUS", "BUY_FOOD",
                   "TAKE_LOAN", "REPAY_LOAN", "PAY_TAX"]
                + ["ROAD_E", "ROAD_W", "ROAD_S", "ROAD_N"]
            )

        # Action history for loop detection (safe tracking). Окно намеренно
        # маленькое: детектору циклов достаточно недавней истории, а мониторинг
        # долей считает отдельным счётчиком на весь роллаут (см. ниже) — иначе
        # «циклы» начали бы срабатывать на всём роллауте сразу.
        self._action_history: deque = deque(maxlen=1000)

        # ── мониторинг долей действий (вкладка «Мониторинг») ──
        # Счётчик на весь роллаут: исторически доли считались по последним 1000
        # действиям, что при n_steps*n_envs = 32768 (дефолт) показывало ~3%
        # роллаута. Отсюда и жалоба «водоканал появился, подрос и пропал»:
        # строка исчезала, когда в ХВОСТЕ роллаута не оставалось нажатий, а не
        # когда политика перестала строить.
        n_act = max(1, len(self._action_names))
        self._rollout_action_counts: np.ndarray = np.zeros(n_act, dtype=np.int64)
        self._monitor_warned: bool = False

    def _request_stop(self, reason: str = "user") -> None:
        if not self._stop:
            self._stop_reason = reason
        self._stop = True

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def _check_stop(self) -> bool:
        if self._stop:
            return True
        if self.stop_check and self.stop_check():
            self._request_stop("user")
            return True
        return False

    def _record_episode_diagnostic(self, info: dict[str, Any], env_index: int) -> None:
        writer = self._episode_diagnostics
        if writer is None:
            return
        try:
            writer.append_episode(
                info,
                total_timesteps=self._diagnostic_total_timesteps,
                env_index=env_index,
                curriculum_stage=self._curriculum_stage,
            )
        except Exception as exc:  # diagnostics must never take down PPO training
            if not self._episode_diagnostics_warned:
                self._episode_diagnostics_warned = True
                self._log(f"[EpisodeDiagnostics] WARNING: не удалось записать JSONL: {exc}")

    def _collect_rollout(self, obs: torch.Tensor) -> dict[str, Any]:
        """Collect n_steps of transitions."""
        n_envs = self.em.n_envs
        action_names = self._action_names
        # Окно монитора = один роллаут: обнуляем счётчик ДО сбора, иначе
        # доли «наследовались» бы от предыдущего роллаута.
        self._reset_rollout_actions()

        for _ in range(self.cfg.n_steps):
            # Команды опрашиваются на каждом шаге среды, а не раз в роллаут:
            # при n_steps*n_envs = 32768 «Стоп» иначе ждал бы минуты. Пауза,
            # пришедшая посреди роллаута, применяется на его границе.
            if self._command_queue is not None and not self._command_queue.empty():
                self._process_commands(self._command_queue)
            if self._check_stop():
                break

            new_obs, infos = self.em.collect_step(obs)
            self._diagnostic_total_timesteps += n_envs

            # Действия шага: счётчик долей (весь роллаут) + история для
            # детектора циклов (недавний хвост).
            self._track_step_actions(n_envs, action_names)

            # Terminal infos carry complete C++ EpisodeMetrics. Persist them
            # independently from the transient progress/UI stream.
            for env_index, info in enumerate(infos):
                ep = info.get("episode")
                if ep is not None:
                    self._record_episode_diagnostic(info, env_index)
                    r = ep.get("r")
                    ep_len = ep.get("l")
                    if r is not None:
                        self._ep_returns.append(r)
                        if len(self._ep_returns) > self._ep_returns_maxlen:
                            self._ep_returns.pop(0)
                        self._ep_lengths.append(ep_len or 0)
                        if r > self.best_reward:
                            self.best_reward = r
                ep_ret = info.get("ep_return")
                if ep_ret is not None and info.get("steps", 0) > 0:
                    self._ep_returns.append(float(ep_ret))
                    if len(self._ep_returns) > self._ep_returns_maxlen:
                        self._ep_returns.pop(0)
                    self._ep_lengths.append(int(info.get("steps", 0)))
                    if float(ep_ret) > self.best_reward:
                        self.best_reward = float(ep_ret)

            obs = new_obs

        if self._stop:
            # Роллаут неполный и в PPO не пойдёт (train() выходит сразу) —
            # бутстрап по недособранному буферу не считаем.
            return {"stopped": True, "final_obs": obs}

        # Single terminated read after loop (not per step)
        last_pos = self.em.buffer.pos - 1
        terminated = (self.em.buffer.terminated[last_pos * n_envs:(last_pos + 1) * n_envs]
                      .cpu().numpy())

        with torch.no_grad():
            # Bootstrap the critic with the same action-availability context
            # used during rollout/update. This is especially important at a
            # mechanic unlock: the fixed action head stays 49-wide, but the
            # value function must see that the manager branch changed.
            _vec = getattr(self.em, "vec_env", None)
            _mask_np = getattr(_vec, "action_masks", None) if _vec is not None else None
            last_masks = (torch.as_tensor(
                np.asarray(_mask_np, dtype=np.float32),
                device=self.device, dtype=torch.float32
            ) if _mask_np is not None else None)
            last_value = self._bootstrap_value(obs, last_masks)
            last_done = torch.tensor(terminated, dtype=torch.bool, device=self.device)

        return {
            "last_value": last_value.cpu().numpy(),
            "last_done": last_done,
            "final_obs": obs,
        }

    # ── obs-mode aware critic bootstrap ──

    def _obs_mode(self) -> str:
        """Observation mode of this run: 'flat' | 'minimap' | 'hybrid'.

        Read from the trainer config first, then from the env manager. The old
        `getattr(self.em, "obs_mode", "flat")` ALWAYS answered "flat": EnvManager
        kept the mode only inside `cfg`, so a hybrid run bootstrapped the critic
        with the raw `(flat, minimap)` tuple and the first Linear died with
        "TypeError: linear(): argument 'input' (position 1) must be Tensor, not
        tuple" at the very first rollout.
        """
        for src in (self.cfg, self.em, getattr(self.em, "cfg", None)):
            mode = getattr(src, "obs_mode", None)
            if isinstance(mode, str) and mode:
                return mode
        return "flat"

    @staticmethod
    def _split_hybrid_obs(obs: Any) -> tuple[Any, Any]:
        """Unpack a hybrid observation into (flat, minimap).

        EnvManager._policy_obs() yields a tuple, but a dict form also exists
        (both are accepted by ActorCriticHybrid) — accept either here so the
        bootstrap can never feed a container into a Linear.
        """
        if isinstance(obs, dict):
            return obs.get("flat"), obs.get("minimap")
        if isinstance(obs, (tuple, list)):
            if len(obs) != 2:
                raise TypeError(
                    f"hybrid obs must be a (flat, minimap) pair, got {len(obs)} items")
            return obs[0], obs[1]
        return obs, None

    def _bootstrap_value(self, obs: Any, last_masks: torch.Tensor | None) -> torch.Tensor:
        """V(s_last) for GAE — obs-mode aware, masks passed by keyword.

        Positional masks are exactly how the hybrid crash happened (the second
        positional of ActorCriticHybrid.get_value is `minimap`, not masks), so
        every branch passes `action_masks=` explicitly.
        """
        model = self.em.ppo.model
        if self._obs_mode() == "hybrid":
            flat, minimap = self._split_hybrid_obs(obs)
            return model.get_value(flat, minimap, action_masks=last_masks)
        if isinstance(obs, (tuple, list, dict)):
            # Defensive: a container obs on a non-hybrid mode means the env and
            # the policy disagree — fail with a message, not inside a Linear.
            flat, minimap = self._split_hybrid_obs(obs)
            if minimap is not None:
                raise TypeError(
                    f"obs_mode={self._obs_mode()!r} but the env returned a "
                    f"(flat, minimap) pair — set obs_mode='hybrid' (or 'minimap') "
                    f"to match the observation the env emits")
            obs = flat
        return model.get_value(obs, action_masks=last_masks)

    def _reset_rollout_actions(self) -> None:
        """Новый роллаут — новый отсчёт долей (окно = ровно один роллаут)."""
        self._rollout_action_counts[:] = 0

    def _record_rollout_actions(self, actions: Any) -> None:
        """Сложить действия одного шага всех env в счётчик роллаута."""
        arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return
        n = int(self._rollout_action_counts.size)
        valid = (arr >= 0) & (arr < n)
        if not bool(np.any(valid)):
            return
        self._rollout_action_counts += np.bincount(arr[valid], minlength=n)[:n]

    def _calculate_action_distribution(self, window: int = 0) -> list[int]:
        """Счётчики действий для монитора.

        ``window <= 0`` (норма) — всё, что накоплено за текущий роллаут: доля
        тогда означает «сколько шагов роллаута ушло на действие», и строка не
        гаснет из-за случайного пустого хвоста лога.

        ``window > 0`` — legacy-режим по последним N записям `_action_history`
        (исторически это был единственный способ считать; оставлен для отладки
        и тестов, которым нужна именно скользящая window).
        """
        if window > 0:
            counts = [0] * len(self._action_names)
            recent = list(self._action_history)[-window:]
            for _, action_name in recent:
                try:
                    idx = self._action_names.index(action_name)
                except ValueError:
                    # Имя из fallback-списка может отсутствовать в списке среды:
                    # считаем по реальным именам, молча пропуская чужие.
                    continue
                if 0 <= idx < len(counts):
                    counts[idx] += 1
            return counts
        n = min(len(self._rollout_action_counts), len(self._action_names))
        return [int(c) for c in self._rollout_action_counts[:n]]

    def _top_actions_dict(self, action_counts: list[int], total_actions: int,
                          limit: int | None = None) -> dict[str, float]:
        """Доли действий для монитора; нулевые сюда не попадают (их рисует UI).

        `limit` по умолчанию = `Config.monitor_action_top` (0 = не обрезать).
        Раньше тут был жёсткий `[:15]`, и при равных count стабильная
        сортировка отдавала предпочтение меньшему индексу действия: здание с
        count=1 вылетало из списка, как только набиралось 15 действий с
        count>=2, — хотя его собственная доля не падала. Это и было второй
        половиной жалобы «водоканал рос и пропал».
        """
        if limit is None:
            limit = int(getattr(self.cfg, "monitor_action_top", 0) or 0)
        order = sorted(range(len(action_counts)),
                       key=lambda i: action_counts[i], reverse=True)
        if limit > 0:
            order = order[:limit]
        total = max(int(total_actions), 1)
        names = self._action_names
        return {
            names[i]: round(action_counts[i] / total * 100, 2)
            for i in order
            if i < len(names) and action_counts[i] > 0
        }

    def _track_step_actions(self, n_envs: int, action_names: list[str]) -> None:
        """Записать действия последнего шага: счётчик роллаута + историю циклов.

        Одно чтение буфера на двух потребителей: раньше оба жили в одном
        `try/except: pass`, и молча обнулённый мониторинг было не отличить от
        «политика ничего не строит».
        """
        try:
            pos = self.em.buffer.pos - 1
            step_actions = (self.em.buffer.actions[pos * n_envs:(pos + 1) * n_envs]
                            .cpu().numpy())
        except Exception as e:  # noqa: BLE001 - наблюдательный счётчик не валит обучение
            self._monitor_warn_once(
                f"не удалось прочитать actions шага: {type(e).__name__}: {e}")
            return
        self._record_rollout_actions(step_actions)
        for env_idx in range(n_envs):
            action_idx = int(step_actions[env_idx])
            action_name = (action_names[action_idx]
                           if action_idx < len(action_names) else f"ACTION_{action_idx}")
            self._action_history.append((env_idx, action_name))

    def _monitor_warn_once(self, message: str) -> None:
        """Предупреждение о недоступной метрике — ровно один раз за прогон.

        Тихий `except Exception` под RULES.md запрещён: отсутствие счётчика
        должно быть видно, но не должно спамить на каждый роллаут.
        """
        if self._monitor_warned:
            return
        self._monitor_warned = True
        self._log(f"[Monitor] {message}")

    def _pop_action_legality(self) -> dict[str, float]:
        """Доля легальных шагов по каждому действию за собранный роллаут."""
        pop = getattr(self.em, "pop_action_legality", None)
        if pop is None:
            self._monitor_warn_once(
                "env_manager не предоставляет pop_action_legality — доля "
                "легальных шагов не измеряется (монитор не сможет отличить "
                "«маска закрыла» от «политика не хочет»)"
            )
            return {}
        try:
            return dict(pop() or {})
        except Exception as e:  # noqa: BLE001 - наблюдательная метрика не валит обучение
            self._monitor_warn_once(f"pop_action_legality упал: {type(e).__name__}: {e}")
            return {}

    def _pop_action_mask_reasons(self) -> dict[str, str]:
        """Доминирующая причина закрытой маски (деньги / нет участка / курикулум)."""
        pop = getattr(self.em, "pop_action_mask_reasons", None)
        if pop is None:
            self._monitor_warn_once(
                "env_manager не предоставляет pop_action_mask_reasons — "
                "атрибуция причины маски не измеряется (в колонке панели "
                "останется обобщённое «заблокировано маской»)"
            )
            return {}
        try:
            return dict(pop() or {})
        except Exception as e:  # noqa: BLE001 - наблюдательная метрика не валит обучение
            self._monitor_warn_once(
                f"pop_action_mask_reasons упал: {type(e).__name__}: {e}")
            return {}

    def _curriculum_progress_view(self, step: int) -> tuple[float, bool]:
        """(доля 0..1, измерена ли) — прогресс текущего этапа курикулума.

        Вторым значением возвращаем честный признак «нет данных»: до 2026-09-23
        несуществующий `EnvManager.get_curriculum_progress` глушился
        `except Exception` и UI вечно рисовал «этап N (0%)».
        """
        get = getattr(self.em, "get_curriculum_progress", None)
        if get is None:
            self._monitor_warn_once(
                "env_manager.get_curriculum_progress отсутствует — прогресс "
                "этапа скрыт (в UI: «н/д» вместо вечных 0%)"
            )
            return 0.0, False
        try:
            prog = get(int(step)) or {}
        except Exception as e:  # noqa: BLE001 - наблюдательная метрика не валит обучение
            self._monitor_warn_once(
                f"get_curriculum_progress упал: {type(e).__name__}: {e}")
            return 0.0, False
        raw = prog.get("progress")
        if raw is None:  # этап без расписания: доли нет, и 0% показывать нельзя
            return 0.0, False
        try:
            return float(raw), True
        except (TypeError, ValueError):
            self._monitor_warn_once(
                f"get_curriculum_progress вернул progress={raw!r} (не число)")
            return 0.0, False

    def _process_commands(self, command_queue: queue.Queue | None) -> None:
        """Выполнить команды UI из очереди (строки — `train_ui2.protocol.CMD_*`)."""
        if command_queue is None:
            return
        while True:
            try:
                _, cmd_data = command_queue.get_nowait()
            except queue.Empty:
                break
            raw = cmd_data.get("cmd") if isinstance(cmd_data, dict) else None
            payload = (cmd_data.get("payload") or {}) if isinstance(cmd_data, dict) else {}
            cmd = P.normalize_command(raw)
            if cmd == P.CMD_BOOST_ENTROPY:
                factor = payload.get("factor", 2.0)
                old = self.em.ppo.ent_coef
                new = min(old * factor, 0.05)
                self.em.ppo.ent_coef = new
                self.metrics.ent_coef = new
                self._log(f"[Command] Boost ent_coef: {old:.5f} -> {new:.5f}")
            elif cmd == P.CMD_RESET_CURRICULUM:
                stage = payload.get("stage", 0)
                self.em.set_curriculum_stage(stage)
                self._curriculum_stage = stage
                self._log(f"[Command] Reset curriculum to stage {stage}")
            elif cmd == P.CMD_PAUSE:
                self._paused = True
                self._log("[Command] Training paused")
            elif cmd == P.CMD_RESUME:
                self._paused = False
                self._log("[Command] Training resumed")
            elif cmd == P.CMD_STOP:
                self._request_stop("user")
                self._log("[Command] Training stop requested — сохраняю final_model и выхожу")
                return
            else:
                # Раньше незнакомая команда исчезала без следа (P1-4).
                self._log(f"[Command] WARNING: unknown command {raw!r} — ignored "
                          f"(known: {sorted(P.KNOWN_COMMANDS)})")

    def _update_ppo(self, rollout: dict[str, Any]) -> dict[str, float]:
        """Run PPO update on GPU."""
        last_value = torch.tensor(rollout["last_value"], dtype=torch.float32, device=self.device)
        # rollout["last_done"] is already a torch tensor — torch.tensor() on a
        # tensor copies via a UserWarning path; move it instead.
        last_done = rollout["last_done"].to(self.device)

        t0 = time.perf_counter()
        stats = self.em.ppo.update(last_value=last_value, last_done=last_done)
        update_time = time.perf_counter() - t0

        stats["update_time_s"] = update_time
        if torch.cuda.is_available() and self.device.type == "cuda":
            stats["gpu_mem_mb"] = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            stats["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024 * 1024)

        return stats

    def _update_loop_detector(self, window: int, total_done: int) -> tuple[int, str | None]:
        """Прогнать детектор циклов по хвосту истории действий после роллаута.

        Возвращает (число сред с циклом, имя зацикленного действия или None).
        `_action_history` — deque(maxlen=1000): срез deque бросает TypeError,
        из-за чего включённый `loop_detection_enabled` ронял обучение на первом
        же роллауте (ревью 2026-09-24, P3-3/P3-8). Окно фактически ≤ 1000
        записей — семантика «цикл только что был» (STATE.md).
        """
        if self.loop_detector is None or not self._action_history or window <= 0:
            return 0, None
        recent = list(self._action_history)[-window:]
        action_data = [
            {"env_idx": env_idx, "action": action_name, "step": total_done + i}
            for i, (env_idx, action_name) in enumerate(recent)
        ]
        alerts = self.loop_detector.update_batch(action_data)
        envs_with_loops = len(alerts)
        loop_action_name = self._get_current_loop_action({}) if envs_with_loops > 0 else None
        return envs_with_loops, loop_action_name

    def _get_current_loop_action(self, stats: dict[str, Any]) -> str | None:
        """Get the action name currently in loop if any."""
        if self.loop_detector is None:
            return None
        if not hasattr(self.loop_detector, '_state') or not self.loop_detector._state:
            return None
        # Find env with highest consecutive count
        best_env = None
        best_count = 0
        for _env_idx, state in self.loop_detector._state.items():
            if state.consecutive_count > best_count:
                best_count = state.consecutive_count
                best_env = state
        if best_env and best_env.consecutive_count >= self.loop_detector.consecutive_threshold:
            return best_env.last_action
        return None

    def _curriculum_meta(self) -> dict[str, Any]:
        """Curriculum fields stored in meta.json next to a checkpoint.

        Persisting them is what lets eval/watch restore the exact scenario the
        model was trained in (stage + manual set from the «Курикулум» tab).
        """
        return {
            "curriculum_stage_at_best": int(self._curriculum_stage),
            "unlock_ids": self.cfg.effective_unlock_ids(),
            "use_curriculum_tab": bool(getattr(self.cfg, "use_curriculum_tab", False)),
            "curriculum_resources": str(getattr(self.cfg, "curriculum_resources", "") or ""),
            "tax_to_debt": bool(getattr(self.cfg, "tax_to_debt", True)),
            # PR 5: without this the eval-dir meta reads as legacy v0 and the
            # stored-vs-env check fails mid-training on a v1 run.
            "obs_version": int(getattr(self.cfg, "obs_version", 2)),
            "disabled_mechanics": list(getattr(self.cfg, "disabled_mechanics", [])),
            "mechanics_unlock_schedule": getattr(self.cfg, "mechanics_unlock_schedule", []),
            "enabled_mechanics": list(self.cfg.enabled_mechanics_at(self._curriculum_progress_step)),
        }

    def _curriculum_kwargs(self) -> dict[str, Any]:
        """Curriculum kwargs for `run_eval` (keep eval == training scenario)."""
        return {
            "curriculum_stage": int(self._curriculum_stage),
            "unlock_ids": self.cfg.effective_unlock_ids(),
            "use_curriculum_tab": bool(getattr(self.cfg, "use_curriculum_tab", False)),
            # PR 5: run_eval's version is explicit-only, so a --obs-version 0
            # training run must thread it here or in-training eval errors out.
            "obs_version": int(getattr(self.cfg, "obs_version", 2)),
            "tax_to_debt": bool(getattr(self.cfg, "tax_to_debt", True)),
            "enabled_mechanics": list(self.cfg.enabled_mechanics_at(self._curriculum_progress_step)),
        }

    def _update_best_meta_curriculum(self, save_dir: Path) -> None:
        """Merge current curriculum fields into an existing best_model.meta.json.

        The tournament may overwrite best_model.pt after the last eval, leaving a
        meta file that describes a different curriculum; watch/eval read it, so
        keep it in sync.
        """
        meta_path = save_dir / "best_model.meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                return
            meta.update(self._curriculum_meta())
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except (OSError, ValueError) as ex:
            self._log(f"[Meta] curriculum update failed: {ex}")

    def _eval(self, total_done: int) -> dict[str, float]:
        """Run evaluation episodes with the current policy.

        Returns dict with days, people, bases, avg_return, score.
        Saves best_model.pt if composite score improves AND thresholds are met.
        """
        from train_ui2.evaluator import run_eval

        save_dir = Path(self.cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate from a DEDICATED subdir with a FRESH meta.json.
        # run_eval reads reward/difficulty from meta files next to the model;
        # when the temp model sat in model_dir it picked up the PREVIOUS best
        # model's reward config, so eval returns were computed with stale
        # rewards (and a stale difficulty).
        eval_dir = save_dir / "_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_model_path = eval_dir / "_eval_temp.pt"
        self.em.ppo.save(str(eval_model_path))
        with open(eval_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "difficulty": getattr(self.cfg, "difficulty", "normal"),
                "config": {"reward": self.cfg.reward.to_dict()},
                # eval must play under the SAME curriculum as training
                **self._curriculum_meta(),
            }, f)

        if self.loop_detector:
            self.loop_detector.clear()

        norm_path = save_dir / "normalization.json"
        norm_str = str(norm_path) if norm_path.exists() else None

        if norm_str is None:
            self._log("[Eval] WARNING: normalization.json не найден — eval БЕЗ нормализации obs!")

        seeds = getattr(self.cfg, "eval_seeds", [42]) or [42]
        all_days: list[float] = []
        all_bases: list[float] = []
        all_people: list[float] = []
        all_returns: list[float] = []
        all_water: list[float] = []
        all_water_ready: list[float] = []

        try:
            for seed in seeds:
                result = run_eval(
                    model_path=str(eval_model_path),
                    episodes=self.cfg.eval_episodes,
                    max_days=10000,
                    seed=seed,
                    device=str(self.device),
                    normalization_path=norm_str,
                    map_size=self.em.cfg.map_size,
                    mode=getattr(self.cfg, "obs_mode", "flat"),
                    minimap_radius=getattr(self.cfg, "minimap_radius", 14),
                    difficulty=getattr(self.cfg, "difficulty", "normal"),
                    **self._curriculum_kwargs(),
                )
                all_days.extend(result.get("episode_days", [result["days"]]))
                all_bases.extend(result.get("episode_bases", [result["bases"]]))
                all_people.extend(result.get("episode_people", [result["people"]]))
                all_returns.extend(result.get("episode_returns", [result["avg_return"]]))
                all_water.extend(result.get("episode_water", [result.get("water_channels", 0.0)]))
                all_water_ready.extend(result.get("episode_water_ready", [result.get("water_ready", 0.0)]))
        finally:
            if eval_model_path.exists():
                eval_model_path.unlink()

        use_median = getattr(self.cfg, "eval_use_median", True)
        if use_median:
            days_agg = float(np.median(all_days))
            bases_agg = float(np.median(all_bases))
            people_agg = float(np.median(all_people))
            return_agg = float(np.median(all_returns))
            water_agg = float(np.median(all_water)) if all_water else 0.0
            water_ready_agg = float(np.median(all_water_ready)) if all_water_ready else 0.0
        else:
            days_agg = float(np.mean(all_days))
            bases_agg = float(np.mean(all_bases))
            people_agg = float(np.mean(all_people))
            return_agg = float(np.mean(all_returns))
            water_agg = float(np.mean(all_water)) if all_water else 0.0
            water_ready_agg = float(np.mean(all_water_ready)) if all_water_ready else 0.0

        bases_std = float(np.std(all_bases)) if all_bases else 0.0
        bases_p25 = float(np.percentile(all_bases, 25)) if all_bases else 0.0
        days_std = float(np.std(all_days)) if all_days else 0.0

        w1, w2, w3, w4 = getattr(self.cfg, "eval_score_weights", (0.10, 1.0, 0.10, 0.0001))
        variance_penalty = 0.2 * bases_std + 0.001 * days_std

        # Water chain bonus in eval score:
        # Starting WaterChannel construction gives +20, operational WaterChannel gives +50.
        water_score = 20.0 * min(water_agg, 2.0) + 50.0 * min(water_ready_agg, 2.0)

        score = (days_agg * w1 + bases_agg * w2
                 + people_agg * w3 + max(0.0, return_agg) * w4
                 + water_score
                 - variance_penalty)

        min_bases = getattr(self.cfg, "eval_min_bases", 5)
        min_days = getattr(self.cfg, "eval_min_days", 730.0)
        # Standard threshold OR water breakthrough threshold (built water channel + 2 bases + survived 60 days)
        water_qualified = (water_agg >= 1.0 and bases_agg >= 2 and days_agg >= 60.0)
        thresholds_met = ((bases_agg >= min_bases) and (bases_p25 >= min_bases * 0.7) and (days_agg >= min_days)) or water_qualified

        ci95 = {}
        water_str = f" water={water_agg:.1f} (ready={water_ready_agg:.1f})" if (water_agg > 0 or water_ready_agg > 0) else ""
        if len(all_days) >= 4:
            ci95["days"] = [float(np.percentile(all_days, 2.5)), float(np.percentile(all_days, 97.5))]
            ci95["bases"] = [float(np.percentile(all_bases, 2.5)), float(np.percentile(all_bases, 97.5))]
            ci95["people"] = [float(np.percentile(all_people, 2.5)), float(np.percentile(all_people, 97.5))]
            self._log(
                f"[Eval @ {total_done:,}] days={days_agg:.1f} "
                f"people={people_agg:.1f} bases={bases_agg:.1f}{water_str} "
                f"return={return_agg:.1f} score={score:.2f} "
                f"CI95 days={ci95['days'][0]:.1f}-{ci95['days'][1]:.1f} "
                f"bases={ci95['bases'][0]:.1f}-{ci95['bases'][1]:.1f} "
                f"thresholds={'PASS' if thresholds_met else 'FAIL'}"
            )
        else:
            self._log(
                f"[Eval @ {total_done:,}] days={days_agg:.1f} "
                f"people={people_agg:.1f} bases={bases_agg:.1f}{water_str} "
                f"return={return_agg:.1f} score={score:.2f} "
                f"thresholds={'PASS' if thresholds_met else 'FAIL'}"
            )

        if thresholds_met and (self.best_score is None or score > self.best_score):
            self.best_score = score
            self.best_eval_days = days_agg
            best_path = save_dir / "best_model.pt"
            self.em.ppo.save(str(best_path))

            # Save CURRENT normalization stats (not the stale file from training start)
            norm_path = save_dir / "best_model.norm.json"
            self.em.vec_env.venv.save_normalization(str(norm_path))

            meta = {
                "best_score": score,
                "best_days": days_agg,
                "best_bases": bases_agg,
                "best_people": people_agg,
                "best_return": return_agg,
                "best_water": water_agg,
                "best_water_ready": water_ready_agg,
                "water_score": water_score,
                "score_weights": list(getattr(self.cfg, "eval_score_weights", (0.10, 1.0, 0.10, 0.0001))),
                "min_bases": min_bases,
                "min_days": min_days,
                "map_size": int(getattr(self.cfg, "map_size", 280)),
                "total_timesteps": total_done,
                "episodes": len(all_days),
                # stage + manual set → watch/eval can rebuild the same scenario
                **self._curriculum_meta(),
                "ci95": ci95,
                "difficulty": getattr(self.cfg, "difficulty", "normal"),
                "config": {"reward": self.cfg.reward.to_dict()},
            }
            meta_path = save_dir / "best_model.meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            self._log(f"[Best] Saved best_model.pt (score={score:.2f}, days={days_agg:.1f}, bases={bases_agg:.1f}{water_str})")

        return {
            "days": days_agg,
            "bases": bases_agg,
            "people": people_agg,
            "water": water_agg,
            "water_ready": water_ready_agg,
            "avg_return": return_agg,
            "score": score,
            "saved": thresholds_met and (self.best_score == score),
        }

    def train(self, total_timesteps: int | None = None,
              command_queue: queue.Queue | None = None) -> TrainMetrics:
        total = total_timesteps or self.cfg.total_timesteps
        n_envs = self.em.n_envs

        self._log(f"[Trainer] Starting: total={total:,} steps, n_envs={n_envs}, "
                  f"n_steps={self.cfg.n_steps}, batch={self.cfg.batch_size}, "
                  f"epochs={self.cfg.n_epochs}, device={self.device}")
        self._log(f"[Trainer] AMP={self.cfg.use_amp} ({self.cfg.amp_dtype}), "
                  f"compile={self.cfg.torch_compile}")
        self._log(f"[Trainer] curriculum_stage={self._curriculum_stage}, "
                  f"schedule={self.cfg.curriculum_schedule}")

        save_dir = Path(self.cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._diagnostic_total_timesteps = 0
        self._episode_diagnostics_warned = False
        try:
            self._episode_diagnostics = EpisodeDiagnosticsWriter(
                save_dir / "episode_diagnostics.jsonl",
                metadata={
                    "run_name": save_dir.name,
                    "seed": getattr(self.cfg, "seed", None),
                    "n_envs": n_envs,
                    "map_size": getattr(self.cfg, "map_size", None),
                    "difficulty": getattr(self.cfg, "difficulty", None),
                    "obs_mode": getattr(self.cfg, "obs_mode", "flat"),
                    "obs_version": getattr(self.cfg, "obs_version", None),
                    "curriculum_stage": self._curriculum_stage,
                    "learning_rate": getattr(self.cfg, "learning_rate", None),
                },
            )
        except Exception as exc:  # keep diagnostics optional if disk is read-only/full
            self._episode_diagnostics = None
            self._episode_diagnostics_warned = True
            self._log(f"[EpisodeDiagnostics] WARNING: JSONL отключён: {exc}")

        self._command_queue = command_queue
        try:
            self._train_body(total, save_dir, command_queue)
        except BaseException as exc:
            # Любой аварийный выход (исключение, SIGTERM → SystemExit из
            # воркера, Ctrl+C) всё равно закрывает JSONL записью run_end —
            # иначе файл неотличим от «процесс исчез» (P1-1).
            if isinstance(exc, SystemExit):
                status = "killed"
            elif isinstance(exc, KeyboardInterrupt):
                status = "interrupted"
            else:
                status = "error"
            self._close_episode_diagnostics(status)
            raise
        finally:
            self._command_queue = None
        self._close_episode_diagnostics(self._run_status())
        return self.metrics

    def _run_status(self) -> str:
        """Статус для run_end: completed | stopped (пользователь) | early_stopped."""
        if not self._stop:
            return "completed"
        return "early_stopped" if self._stop_reason == "early_stop" else "stopped"

    def _close_episode_diagnostics(self, status: str) -> None:
        """Записать run_end ровно один раз (повторный вызов — no-op)."""
        writer = self._episode_diagnostics
        if writer is None:
            return
        self._episode_diagnostics = None
        try:
            writer.close(total_timesteps=self._diagnostic_total_timesteps, status=status)
            self._log(f"[EpisodeDiagnostics] run_end status={status}: {writer.path}")
        except Exception as exc:  # диагностика не должна маскировать исходную ошибку
            if not self._episode_diagnostics_warned:
                self._episode_diagnostics_warned = True
                self._log(f"[EpisodeDiagnostics] WARNING: закрытие JSONL не удалось: {exc}")

    def _train_body(self, total: int, save_dir: Path,
                    command_queue: queue.Queue | None) -> None:
        """Цикл обучения + финальное сохранение и турнир (без закрытия JSONL)."""
        n_envs = self.em.n_envs
        steps_per_rollout = self.cfg.n_steps * n_envs
        obs = self.em.reset()
        self.em.vec_env.venv.save_normalization(str(save_dir / "normalization.json"))
        t_start = time.perf_counter()
        total_done = 0
        rollout_idx = 0
        self._paused = False

        save_every = (
            max(1, int(round(self.cfg.save_freq / steps_per_rollout)))
            if self.cfg.save_freq > 0
            else 0
        )
        eval_every = (
            max(1, int(round(self.cfg.eval_freq / steps_per_rollout)))
            if self.cfg.eval_freq > 0
            else 0
        )

        while total_done < total and not self._stop:
            # Process commands from UI
            self._process_commands(command_queue)

            # Wait if paused. _check_stop(), а не голый self._stop: «Стоп» из UI
            # приходит через stop_check (stop_event воркера), и раньше пауза
            # его не видела — мягкая остановка зависала до жёсткого kill.
            while self._paused and not self._check_stop():
                time.sleep(0.5)
                self._process_commands(command_queue)

            if self._stop:
                break

            rollout = self._collect_rollout(obs)
            # Следующий роллаут продолжает с текущего состояния среды. Раньше
            # здесь оставался obs первого reset(): collect_step(obs) затирал им
            # em._obs, и первый шаг КАЖДОГО роллаута выбирал действие по
            # наблюдению со старта обучения (ревью 2026-09-24, P1-5).
            obs = rollout["final_obs"]

            envs_with_loops, loop_action_name = self._update_loop_detector(
                self.cfg.n_steps * n_envs, total_done)

            if self._stop:
                break

            stats = self._update_ppo(rollout)

            total_done += steps_per_rollout
            rollout_idx += 1
            elapsed = time.perf_counter() - t_start
            fps = total_done / max(elapsed, 1e-10)

            self.metrics.total_timesteps = total_done
            self.metrics.fps = fps
            self.metrics.policy_loss = stats.get("policy_loss", 0.0)
            self.metrics.value_loss = stats.get("value_loss", 0.0)
            self.metrics.entropy = stats.get("entropy", 0.0)
            self.metrics.approx_kl = stats.get("approx_kl", 0.0)
            self.metrics.learning_rate = stats.get("learning_rate", 0.0)
            self.metrics.gpu_mem_mb = stats.get("gpu_mem_mb", 0.0)
            self.metrics.wall_time_s = elapsed
            self.metrics.n_episodes = len(self._ep_returns)
            self.metrics.best_reward = self.best_reward
            self.metrics.ent_coef = self.em.ppo.ent_coef

            # Доли действий — по всему роллауту (не по хвосту в 1000 шагов),
            # плюс сырые счётчики и легальность по маскам: UI по ним держит
            # «липкие» строки и различает «нельзя» / «не хочет».
            action_counts = self._calculate_action_distribution()
            total_actions = sum(action_counts)
            top_actions = self._top_actions_dict(action_counts, total_actions)
            counts_by_name = {
                name: int(cnt)
                # strict=False осознанно: fallback-имена могут не совпасть по длине
                # с n_actions — счётчики для UI не должны ронять обучение.
                for name, cnt in zip(self._action_names, action_counts, strict=False)
                if cnt > 0
            }
            action_legality = self._pop_action_legality()
            action_mask_reasons = self._pop_action_mask_reasons()

            # Return statistics
            avg_return = 0.0
            median_return = 0.0
            max_return = 0.0
            min_return = 0.0
            n_episodes_for_stats = len(self._ep_returns)
            if self._ep_returns:
                recent_returns = self._ep_returns[-50:]
                avg_return = float(np.mean(recent_returns))
                median_return = float(np.median(recent_returns))
                max_return = float(np.max(recent_returns))
                min_return = float(np.min(recent_returns))

            # Curriculum data
            curriculum_next_at_step = None
            schedule = getattr(self.cfg, "curriculum_schedule", None)
            if schedule:
                for threshold, stage in schedule:
                    if total_done >= threshold and stage > self._curriculum_stage:
                        curriculum_next_at_step = threshold
                        break

            # Always show the effective allowed set (stage preset UNION
            # manual set — get_allowed_buildings_for_stage is manual-aware),
            # not just under a schedule: with a fixed manual set the widget
            # would otherwise stay blank.
            available_actions = ""
            try:
                get_allowed = getattr(self.em, "get_allowed_buildings_for_stage", None)
                if get_allowed is not None:
                    allowed = list(get_allowed(self._curriculum_stage) or [])
                    if allowed:
                        available_actions = " | ".join(allowed[:5])
                        if len(allowed) > 5:
                            available_actions += f" (+{len(allowed) - 5} more)"
            except Exception:
                pass

            # Реальный прогресс текущего этапа (с проверкой, что он вообще
            # измеряется: вечно нулевой ноль хуже, чем «н/д»).
            curriculum_progress, curriculum_progress_valid = \
                self._curriculum_progress_view(total_done)

            upcoming_stages = []
            if schedule:
                for threshold, stage in schedule:
                    if stage > self._curriculum_stage:
                        upcoming_stages.append({"stage": stage, "at_step": threshold})
                upcoming_stages = upcoming_stages[:3]

            # Populate metrics with all computed data
            self.metrics.avg_return = avg_return
            self.metrics.median_return = median_return
            self.metrics.max_return = max_return
            self.metrics.min_return = min_return
            self.metrics.n_episodes_for_stats = n_episodes_for_stats
            self.metrics.top_actions = top_actions
            self.metrics.action_counts = counts_by_name
            self.metrics.action_legality = action_legality
            self.metrics.action_mask_reasons = action_mask_reasons
            self.metrics.action_total_steps = int(total_actions)
            self.metrics.loop_detected = envs_with_loops > 0
            self.metrics.loop_action_name = loop_action_name
            self.metrics.envs_with_loops = envs_with_loops
            self.metrics.curriculum_stage = self._curriculum_stage
            self.metrics.curriculum_progress_percent = curriculum_progress
            self.metrics.curriculum_progress_valid = curriculum_progress_valid
            self.metrics.curriculum_available_actions = available_actions
            self.metrics.curriculum_next_at_step = curriculum_next_at_step
            self.metrics.curriculum_upcoming_stages = upcoming_stages

            self._log(
                f"[Step {total_done:,}/{total:,}] "
                f"FPS={fps:,.0f} | "
                f"episodes={self.metrics.n_episodes} "
                f"best_score={self.best_score or 0:.1f} "
                f"best_reward={self.best_reward:.2f} | "
                f"p_loss={stats.get('policy_loss', 0):.4f} "
                f"v_loss={stats.get('value_loss', 0):.4f} "
                f"ent={stats.get('entropy', 0):.4f} "
                f"KL={stats.get('approx_kl', 0):.5f} | "
                f"GPU_mem={stats.get('gpu_mem_mb', 0):.0f}MB "
                f"loops={envs_with_loops}"
            )

            if self.progress_callback:
                try:
                    self.progress_callback(self.metrics)
                except Exception:
                    pass

            save_dir = Path(self.cfg.model_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            if save_every > 0 and rollout_idx % save_every == 0:
                ckpt_path = save_dir / f"checkpoint_{total_done}_steps.pt"
                self.em.ppo.save(str(ckpt_path))
                norm_path = str(ckpt_path).replace(".pt", ".norm.json")
                self.em.vec_env.venv.save_normalization(norm_path)
                self.em.vec_env.venv.save_normalization(str(save_dir / "normalization.json"))
                # Записать метаданные чекпоинта (стадия курикулума, шаги, obs_version)
                meta_file = Path(str(ckpt_path).replace(".pt", ".meta.json"))
                try:
                    with open(meta_file, "w", encoding="utf-8") as mf:
                        json.dump({
                            "curriculum_stage": int(self._curriculum_stage),
                            "total_timesteps": total_done,
                            "unlock_ids": self.cfg.effective_unlock_ids(),
                            "use_curriculum_tab": bool(getattr(self.cfg, "use_curriculum_tab", False)),
                            "curriculum_resources": str(getattr(self.cfg, "curriculum_resources", "") or ""),
                            "obs_version": int(getattr(self.cfg, "obs_version", 2)),
                            "tax_to_debt": bool(getattr(self.cfg, "tax_to_debt", True)),
                            "enabled_mechanics": list(self.cfg.enabled_mechanics_at(self._curriculum_progress_step)),
                            "disabled_mechanics": list(getattr(self.cfg, "disabled_mechanics", [])),
                            "mechanics_unlock_schedule": getattr(self.cfg, "mechanics_unlock_schedule", []),
                        }, mf)
                except Exception:
                    pass
                self._log(f"[Save] {ckpt_path}")

            # Curriculum stage switching
            if schedule:
                for step_threshold, stage in schedule:
                    if total_done >= step_threshold and self._curriculum_stage < stage:
                        self.em.set_curriculum_stage(stage)
                        self._curriculum_stage = stage
                        self._log(f"[Curriculum] Stage -> {stage} at step {total_done:,}")
                        break

            # Mechanic unlocks are an additive allow-list over the fixed action
            # head. Apply only when the schedule actually changes it; unlike a
            # stage replacement this can never re-lock an action mid-episode.
            target_mechanics = self.cfg.enabled_mechanics_at(total_done)
            current_mechanics = self.cfg.enabled_mechanics_at(self._curriculum_progress_step)
            if target_mechanics != current_mechanics:
                self.em.set_curriculum_progress(total_done)
                self._log(f"[Curriculum] Mechanics -> {list(target_mechanics)} at step {total_done:,}")
            self._curriculum_progress_step = total_done

            # Run eval (не при остановке: eval идёт минутами, а мягкий «Стоп»
            # должен уложиться в таймаут UI; stop_check опрашиваем здесь, т.к.
            # во время PPO-апдейта stop_event никто не читал).
            if eval_every > 0 and rollout_idx % eval_every == 0 and not self._check_stop():
                eval_result = self._eval(total_done)

                es_patience = getattr(self.cfg, "early_stopping_patience", 0)
                if es_patience > 0:
                    if self.best_score is not None and (self._es_best_score is None
                                                        or self.best_score > self._es_best_score):
                        self._es_best_score = self.best_score
                        self._es_patience = 0
                    else:
                        self._es_patience += 1
                        self._log(f"[EarlyStop] {self._es_patience}/{es_patience} evals without improvement")
                        if self._es_patience >= es_patience:
                            self._log(f"[EarlyStop] Stopping at step {total_done:,} (best_score={self.best_score:.2f})")
                            self._request_stop("early_stop")

                if self.progress_callback:
                    try:
                        self.metrics.eval_days = eval_result["days"]
                        self.metrics.eval_people = eval_result["people"]
                        self.metrics.eval_bases = eval_result["bases"]
                        self.metrics.eval_score = eval_result.get("score", 0.0)
                        self.metrics.best_eval_days = self.best_eval_days or 0.0
                        self.metrics.best_score = self.best_score or 0.0
                        self.progress_callback(self.metrics)
                    except Exception:
                        pass

        elapsed = time.perf_counter() - t_start
        final_fps = total_done / max(elapsed, 1e-10)

        save_dir = Path(self.cfg.model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "final_model.pt"
        self.em.ppo.save(str(final_path))
        norm_path = str(final_path).replace(".pt", ".norm.json")
        self.em.vec_env.venv.save_normalization(norm_path)
        self._log(f"[Save] Final model: {final_path}")

        # End-of-Training Tournament: evaluate all candidates and ensure best_model.pt is the true champion
        # Остановка пользователем («Стоп» в UI) турнир пропускает: он занимает
        # минуты (кандидаты × 5 сидов × eval_episodes), а мягкий стоп должен
        # укладываться в таймаут UI. best_model.pt остаётся от последнего eval;
        # в meta.json это видно по полю tournament="skipped_user_stop".
        if self._stop and self._stop_reason == "user":
            self.metrics.tournament = "skipped_user_stop"
            self._log("[Tournament] Skipped: training stopped by user "
                      "(best_model.pt — по последнему eval)")
        else:
            self.metrics.tournament = "done"
            try:
                from train_ui2.evaluator import run_eval
                self._log("[Tournament] Running end-of-training model tournament across checkpoints & final...")
                best_cand_path = None
                best_cand_score = -1e9
                cand_paths = sorted(save_dir.glob("checkpoint_*_steps.pt")) + [final_path]
                if (save_dir / "best_model.pt").exists():
                    cand_paths.append(save_dir / "best_model.pt")
                # Keep the progress denominator stable and do not evaluate the same
                # path twice if best_model.pt happens to be a listed candidate.
                cand_paths = sorted(set(cand_paths), key=lambda p: str(p).lower())
                tournament_total = len(cand_paths)
                tournament_done = 0
                tournament_started = time.perf_counter()
                self._log(f"[Tournament] Progress: 0.0% (0/{tournament_total})")

                import random
                seeds = [random.randint(1, 999999) for _ in range(5)]
                use_median = getattr(self.cfg, "eval_use_median", True)
                w1, w2, w3, w4 = getattr(self.cfg, "eval_score_weights", (0.10, 1.0, 0.10, 0.0001))
                min_b = getattr(self.cfg, "eval_min_bases", 5)
                min_d = getattr(self.cfg, "eval_min_days", 730.0)

                for candidate_index, cp in enumerate(cand_paths, start=1):
                    if self._check_stop():
                        # «Стоп» во время турнира: final_model уже сохранён,
                        # победителя по неполной выборке не объявляем.
                        self.metrics.tournament = "interrupted_user_stop"
                        best_cand_path = None
                        self._log(f"[Tournament] Interrupted by user after "
                                  f"{tournament_done}/{tournament_total} candidates "
                                  f"— best_model.pt не меняется")
                        break
                    if not cp.exists():
                        tournament_done += 1
                        percent = 100.0 * tournament_done / max(1, tournament_total)
                        self._log(f"[Tournament] Progress: {percent:.1f}% "
                                  f"({tournament_done}/{tournament_total}), "
                                  f"missing={cp.name}")
                        continue
                    self._log(f"[Tournament] Candidate {candidate_index}/{tournament_total}: "
                              f"{cp.name} (progress "
                              f"{100.0 * (candidate_index - 1) / max(1, tournament_total):.1f}%)")
                    try:
                        cp_norm = Path(str(cp).replace(".pt", ".norm.json"))
                        cp_norm_str = str(cp_norm) if cp_norm.exists() else norm_path

                        # Оценивать чекпоинт под той стадией курикулума, под которой он учился
                        eval_curriculum = dict(self._curriculum_kwargs())
                        cp_meta = Path(str(cp).replace(".pt", ".meta.json"))
                        if cp_meta.exists():
                            try:
                                with open(cp_meta, encoding="utf-8") as cmf:
                                    cdata = json.load(cmf)
                                    if "curriculum_stage" in cdata:
                                        eval_curriculum["curriculum_stage"] = int(cdata["curriculum_stage"])
                                    if "enabled_mechanics" in cdata:
                                        eval_curriculum["enabled_mechanics"] = list(cdata["enabled_mechanics"])
                                    if "tax_to_debt" in cdata:
                                        eval_curriculum["tax_to_debt"] = bool(cdata["tax_to_debt"])
                            except Exception:
                                pass

                        all_days = []
                        all_bases = []
                        all_people = []
                        all_returns = []
                        all_water = []
                        all_water_ready = []
                        for seed in seeds:
                            res = run_eval(
                                model_path=str(cp),
                                episodes=max(10, getattr(self.cfg, "eval_episodes", 20)),
                                max_days=10000,
                                seed=seed,
                                device=str(self.device),
                                normalization_path=cp_norm_str,
                                map_size=self.em.cfg.map_size,
                                mode=getattr(self.cfg, "obs_mode", "flat"),
                                minimap_radius=getattr(self.cfg, "minimap_radius", 14),
                                difficulty=getattr(self.cfg, "difficulty", "normal"),
                                **eval_curriculum,
                            )
                            all_days.extend(res.get("episode_days", [res["days"]]))
                            all_bases.extend(res.get("episode_bases", [res["bases"]]))
                            all_people.extend(res.get("episode_people", [res["people"]]))
                            all_returns.extend(res.get("episode_returns", [res["avg_return"]]))
                            all_water.extend(res.get("episode_water", [res.get("water_channels", 0.0)]))
                            all_water_ready.extend(res.get("episode_water_ready", [res.get("water_ready", 0.0)]))

                        if use_median:
                            days_agg = float(np.median(all_days))
                            bases_agg = float(np.median(all_bases))
                            people_agg = float(np.median(all_people))
                            return_agg = float(np.median(all_returns))
                            water_agg = float(np.median(all_water)) if all_water else 0.0
                            water_ready_agg = float(np.median(all_water_ready)) if all_water_ready else 0.0
                        else:
                            days_agg = float(np.mean(all_days))
                            bases_agg = float(np.mean(all_bases))
                            people_agg = float(np.mean(all_people))
                            return_agg = float(np.mean(all_returns))
                            water_agg = float(np.mean(all_water)) if all_water else 0.0
                            water_ready_agg = float(np.mean(all_water_ready)) if all_water_ready else 0.0

                        bases_std = float(np.std(all_bases)) if all_bases else 0.0
                        days_std = float(np.std(all_days)) if all_days else 0.0
                        variance_penalty = 0.2 * bases_std + 0.001 * days_std

                        # Балансировка очков турнира:
                        # 1) Ограничиваем вклад числа баз (до 15), чтобы спам сараев в долг не давал победу
                        effective_bases = min(bases_agg, 15.0)
                        # 2) Реальный вес возврата (0.005 вместо 0.0001): здоровая экономика +2000 даёт +10 очков
                        return_score = return_agg * 0.005
                        # 3) Бонус за создание водной инфраструктуры (водоканал строящийся +20, работающий +50)
                        water_score = 20.0 * min(water_agg, 2.0) + 50.0 * min(water_ready_agg, 2.0)
                        sc = (days_agg * w1 + effective_bases * w2
                              + people_agg * w3 + return_score
                              + water_score
                              - variance_penalty)

                        # Модель побеждает, если:
                        #   - выполнила стандартные требования (базы >= min_b, дни >= min_d)
                        #   ИЛИ
                        #   - совершила прорыв по воде (water >= 1.0, базы >= 2, дни >= 60)
                        # И нет катастрофического банкротства (return > -5000)
                        water_qualified = (water_agg >= 1.0 and bases_agg >= 2 and days_agg >= 60.0)
                        thresholds_ok = (bases_agg >= min_b and days_agg >= min_d) or water_qualified
                        if thresholds_ok and return_agg > -5000.0 and sc > best_cand_score:
                            best_cand_score = sc
                            best_cand_path = cp
                    except Exception as ex:
                        self._log(f"[Tournament] Candidate {cp.name} evaluation error: {ex}")
                    finally:
                        tournament_done += 1
                        percent = 100.0 * tournament_done / max(1, tournament_total)
                        elapsed_t = time.perf_counter() - tournament_started
                        eta_s = (elapsed_t / tournament_done) * (tournament_total - tournament_done)
                        eta_min = eta_s / 60.0
                        self._log(f"[Tournament] Progress: {percent:.1f}% "
                                  f"({tournament_done}/{tournament_total}), "
                                  f"remaining≈{eta_min:.1f} min")

                if best_cand_path is not None:
                    self._log(f"[Tournament] Winner: {best_cand_path.name} (score={best_cand_score:.1f})")
                    import shutil
                    shutil.copy(str(best_cand_path), str(save_dir / "best_model.pt"))
                    cp_norm = Path(str(best_cand_path).replace(".pt", ".norm.json"))
                    if cp_norm.exists():
                        shutil.copy(str(cp_norm), str(save_dir / "best_model.norm.json"))
                    self._update_best_meta_curriculum(save_dir)
            except Exception as te:
                self.metrics.tournament = "error"
                self._log(f"[Tournament] Error during tournament: {te}")

        self.metrics.total_timesteps = total_done
        self.metrics.fps = final_fps
        self.metrics.wall_time_s = elapsed
        self.metrics.best_reward = self.best_reward

        self._log(f"[Done] {total_done:,} steps in {elapsed:.1f}s "
                  f"({final_fps:,.0f} FPS), best_reward={self.best_reward:.2f}, "
                  f"episodes={self.metrics.n_episodes}")
        self.metrics.stop_reason = self._stop_reason if self._stop else ""

    def close(self) -> None:
        self.em.close()
