from __future__ import annotations

import time
import threading
import queue
from collections import deque
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

from rl.config import Config
from rl.env_manager import EnvManager
from rl.loop_detector import LoopDetector


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
    top_actions: Dict[str, float] = field(default_factory=dict)
    loop_detected: bool = False
    loop_action_name: Optional[str] = None
    envs_with_loops: int = 0
    curriculum_stage: int = 0
    curriculum_progress_percent: float = 0.0
    curriculum_available_actions: str = ""
    curriculum_next_at_step: Optional[int] = None
    curriculum_upcoming_stages: List = field(default_factory=list)


class AsyncTrainer:
    """Async PPO trainer: producer (env steps) + consumer (GPU PPO update)."""

    def __init__(
        self,
        cfg: Config,
        env_manager: EnvManager,
        logger: Optional[Any] = None,
        progress_callback: Optional[Callable[[TrainMetrics], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self.cfg = cfg
        self.em = env_manager
        self.device = env_manager.device
        self.logger = logger
        self.progress_callback = progress_callback
        self.stop_check = stop_check

        self._stop = False
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=cfg.queue_size)

        self.metrics = TrainMetrics()
        self.best_reward = float("-inf")
        self.best_eval_days: Optional[float] = None
        self.best_score: Optional[float] = None
        self._es_patience = 0
        self._es_best_score: Optional[float] = None
        self._ep_returns: list[float] = []
        self._ep_returns_maxlen = 10000
        self._ep_lengths: list[int] = []
        self._curriculum_stage = getattr(cfg, "curriculum_stage", 0)

        # Loop detection
        self.loop_detector = (
            LoopDetector({"consecutive_threshold": cfg.loop_consecutive_threshold})
            if getattr(cfg, "loop_detection_enabled", False) else None
        )

        # Action names from C++ env
        self._action_names: List[str] = getattr(env_manager, 'action_names', [])
        if not self._action_names:
            # Fallback if env doesn't provide action_names
            self._action_names = [
                "DAY", "WEEK",
                "BUILD_HOUSE", "BUILD_FARM", "BUILD_ROAD", "BUILD_GARDEN",
                "BUILD_SMALL_HOUSE", "BUILD_SAWMILL", "BUILD_WATER_CHANNEL",
                "BUILD_COALMINE", "BUILD_IRONMINE", "BUILD_REFINERY",
                "BUILD_GOLDMINE", "BUILD_POWER_STATION", "BUILD_HYDRO_STATION",
                "IMPROVE_LAND", "REPAIR", "REPAIR_ALL", "DEMOLISH",
                "PRESERVE", "UNPRESERVE", "SELL_SURPLUS", "BUY_FOOD",
                "TAKE_LOAN", "REPAY_LOAN", "PAY_TAX",
            ]

        # Action history for loop detection (safe tracking)
        self._action_history: deque = deque(maxlen=1000)

    def _request_stop(self):
        self._stop = True

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def _check_stop(self) -> bool:
        if self._stop:
            return True
        if self.stop_check and self.stop_check():
            self._stop = True
            return True
        return False

    def _collect_rollout(self, obs: torch.Tensor) -> Dict[str, Any]:
        """Collect n_steps of transitions."""
        n_envs = self.em.n_envs
        action_names = self._action_names

        for _ in range(self.cfg.n_steps):
            if self._check_stop():
                break

            new_obs, infos = self.em.collect_step(obs)

            # Track actions for loop detection (read only current step slice)
            try:
                pos = self.em.buffer.pos - 1
                step_actions = (self.em.buffer.actions[pos * n_envs:(pos + 1) * n_envs]
                                .cpu().numpy())
                for env_idx in range(n_envs):
                    action_idx = int(step_actions[env_idx])
                    action_name = (action_names[action_idx]
                                   if action_idx < len(action_names) else f"ACTION_{action_idx}")
                    self._action_history.append((env_idx, action_name))
            except Exception:
                pass

            # Track per-episode returns
            for info in infos:
                ep = info.get("episode")
                if ep is not None:
                    r = ep.get("r")
                    l = ep.get("l")
                    if r is not None:
                        self._ep_returns.append(r)
                        if len(self._ep_returns) > self._ep_returns_maxlen:
                            self._ep_returns.pop(0)
                        self._ep_lengths.append(l or 0)
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

        # Single terminated read after loop (not per step)
        last_pos = self.em.buffer.pos - 1
        terminated = (self.em.buffer.terminated[last_pos * n_envs:(last_pos + 1) * n_envs]
                      .cpu().numpy())

        with torch.no_grad():
            if getattr(self.em, "obs_mode", "flat") == "hybrid":
                last_flat, last_minimap = obs
                last_value = self.em.ppo.model.get_value(last_flat, last_minimap)
            else:
                last_value = self.em.ppo.model.get_value(obs)
            last_done = torch.tensor(terminated, dtype=torch.bool, device=self.device)

        return {
            "last_value": last_value.cpu().numpy(),
            "last_done": last_done,
            "final_obs": obs,
        }

    def _calculate_action_distribution(self) -> List[int]:
        """Calculate action frequency counts from action history."""
        action_counts = [0] * len(self._action_names)
        recent = list(self._action_history)[-1000:]
        for _, action_name in recent:
            try:
                idx = self._action_names.index(action_name)
                if 0 <= idx < len(action_counts):
                    action_counts[idx] += 1
            except ValueError:
                pass
        return action_counts

    def _process_commands(self, command_queue: Optional[queue.Queue]):
        """Process queued commands from the UI."""
        if command_queue is None:
            return
        while not command_queue.empty():
            try:
                _, cmd_data = command_queue.get_nowait()
                cmd = cmd_data["cmd"]
                payload = cmd_data.get("payload", {})
                if cmd == "boost_entropy":
                    factor = payload.get("factor", 2.0)
                    old = self.em.ppo.ent_coef
                    new = min(old * factor, 0.05)
                    self.em.ppo.ent_coef = new
                    self.metrics.ent_coef = new
                    self._log(f"[Command] Boost ent_coef: {old:.5f} -> {new:.5f}")
                elif cmd == "reset_curriculum":
                    stage = payload.get("stage", 0)
                    self.em.set_curriculum_stage(stage)
                    self._curriculum_stage = stage
                    self._log(f"[Command] Reset curriculum to stage {stage}")
                elif cmd == "pause_training":
                    self._paused = True
                    self._log("[Command] Training paused")
                elif cmd == "resume_training":
                    self._paused = False
                    self._log("[Command] Training resumed")
                elif cmd == "stop_training":
                    self._stop = True
                    self._log("[Command] Training stop requested")
                    return
            except queue.Empty:
                break

    def _update_ppo(self, rollout: Dict[str, Any]) -> Dict[str, float]:
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

    def _get_current_loop_action(self, stats: Dict[str, Any]) -> Optional[str]:
        """Get the action name currently in loop if any."""
        if self.loop_detector is None:
            return None
        if not hasattr(self.loop_detector, '_state') or not self.loop_detector._state:
            return None
        # Find env with highest consecutive count
        best_env = None
        best_count = 0
        for env_idx, state in self.loop_detector._state.items():
            if state.consecutive_count > best_count:
                best_count = state.consecutive_count
                best_env = state
        if best_env and best_env.consecutive_count >= self.loop_detector.consecutive_threshold:
            return best_env.last_action
        return None

    def _eval(self, total_done: int) -> Dict[str, float]:
        """Run evaluation episodes with the current policy.

        Returns dict with days, people, bases, avg_return, score.
        Saves best_model.pt if composite score improves AND thresholds are met.
        """
        import json
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
                )
                all_days.extend(result.get("episode_days", [result["days"]]))
                all_bases.extend(result.get("episode_bases", [result["bases"]]))
                all_people.extend(result.get("episode_people", [result["people"]]))
                all_returns.extend(result.get("episode_returns", [result["avg_return"]]))
        finally:
            if eval_model_path.exists():
                eval_model_path.unlink()

        use_median = getattr(self.cfg, "eval_use_median", True)
        if use_median:
            days_agg = float(np.median(all_days))
            bases_agg = float(np.median(all_bases))
            people_agg = float(np.median(all_people))
            return_agg = float(np.median(all_returns))
        else:
            days_agg = float(np.mean(all_days))
            bases_agg = float(np.mean(all_bases))
            people_agg = float(np.mean(all_people))
            return_agg = float(np.mean(all_returns))

        bases_std = float(np.std(all_bases)) if all_bases else 0.0
        bases_p25 = float(np.percentile(all_bases, 25)) if all_bases else 0.0
        days_std = float(np.std(all_days)) if all_days else 0.0

        w1, w2, w3, w4 = getattr(self.cfg, "eval_score_weights", (0.10, 1.0, 0.10, 0.0001))
        variance_penalty = 0.2 * bases_std + 0.001 * days_std
        score = (days_agg * w1 + bases_agg * w2
                 + people_agg * w3 + max(0.0, return_agg) * w4
                 - variance_penalty)

        min_bases = getattr(self.cfg, "eval_min_bases", 5)
        min_days = getattr(self.cfg, "eval_min_days", 730.0)
        thresholds_met = (bases_agg >= min_bases) and (bases_p25 >= min_bases * 0.7) and (days_agg >= min_days)

        ci95 = {}
        if len(all_days) >= 4:
            ci95["days"] = [float(np.percentile(all_days, 2.5)), float(np.percentile(all_days, 97.5))]
            ci95["bases"] = [float(np.percentile(all_bases, 2.5)), float(np.percentile(all_bases, 97.5))]
            ci95["people"] = [float(np.percentile(all_people, 2.5)), float(np.percentile(all_people, 97.5))]
            self._log(
                f"[Eval @ {total_done:,}] days={days_agg:.1f} "
                f"people={people_agg:.1f} bases={bases_agg:.1f} "
                f"return={return_agg:.1f} score={score:.2f} "
                f"CI95 days={ci95['days'][0]:.1f}-{ci95['days'][1]:.1f} "
                f"bases={ci95['bases'][0]:.1f}-{ci95['bases'][1]:.1f} "
                f"thresholds={'PASS' if thresholds_met else 'FAIL'}"
            )
        else:
            self._log(
                f"[Eval @ {total_done:,}] days={days_agg:.1f} "
                f"people={people_agg:.1f} bases={bases_agg:.1f} "
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
            (getattr(self.em, "vec_env", None) or self.em.env).venv.save_normalization(str(norm_path))

            meta = {
                "best_score": score,
                "best_days": days_agg,
                "best_bases": bases_agg,
                "best_people": people_agg,
                "best_return": return_agg,
                "score_weights": list(getattr(self.cfg, "eval_score_weights", (0.10, 1.0, 0.10, 0.0001))),
                "min_bases": min_bases,
                "min_days": min_days,
                "total_timesteps": total_done,
                "episodes": len(all_days),
                "curriculum_stage_at_best": self._curriculum_stage,
                "ci95": ci95,
                "difficulty": getattr(self.cfg, "difficulty", "normal"),
                "config": {"reward": self.cfg.reward.to_dict()},
            }
            meta_path = save_dir / "best_model.meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            self._log(f"[Best] Saved best_model.pt (score={score:.2f}, days={days_agg:.1f}, bases={bases_agg:.1f})")

        return {
            "days": days_agg,
            "bases": bases_agg,
            "people": people_agg,
            "avg_return": return_agg,
            "score": score,
            "saved": thresholds_met and (self.best_score == score),
        }

    def train(self, total_timesteps: Optional[int] = None,
              command_queue: Optional[queue.Queue] = None) -> TrainMetrics:
        total = total_timesteps or self.cfg.total_timesteps
        n_envs = self.em.n_envs
        steps_per_rollout = self.cfg.n_steps * n_envs

        self._log(f"[Trainer] Starting: total={total:,} steps, n_envs={n_envs}, "
                  f"n_steps={self.cfg.n_steps}, batch={self.cfg.batch_size}, "
                  f"epochs={self.cfg.n_epochs}, device={self.device}")
        self._log(f"[Trainer] AMP={self.cfg.use_amp} ({self.cfg.amp_dtype}), "
                  f"compile={self.cfg.torch_compile}")
        self._log(f"[Trainer] curriculum_stage={self._curriculum_stage}, "
                  f"schedule={self.cfg.curriculum_schedule}")

        obs = self.em.reset()
        Path(self.cfg.model_dir).mkdir(parents=True, exist_ok=True)
        (getattr(self.em, "vec_env", None) or self.em.env).venv.save_normalization(str(Path(self.cfg.model_dir) / "normalization.json"))
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

            # Wait if paused
            while self._paused and not self._stop:
                time.sleep(0.5)
                self._process_commands(command_queue)

            if self._stop:
                break

            t_rollout_start = time.perf_counter()
            rollout = self._collect_rollout(obs)
            rollout_time = time.perf_counter() - t_rollout_start

            # Update loop detector after each rollout
            loop_detected = False
            loop_action_name = None
            envs_with_loops = 0

            if self.loop_detector and self._action_history:
                recent = self._action_history[-(self.cfg.n_steps * n_envs):]
                action_data = [
                    {"env_idx": env_idx, "action": action_name, "step": total_done + i}
                    for i, (env_idx, action_name) in enumerate(recent)
                ]
                alerts = self.loop_detector.update_batch(action_data)
                envs_with_loops = len(alerts)
                if envs_with_loops > 0:
                    loop_detected = True
                    loop_action_name = self._get_current_loop_action({})

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

            # Calculate top actions from history
            action_counts = self._calculate_action_distribution()
            total_actions = sum(action_counts)
            order = sorted(range(len(action_counts)), key=lambda i: action_counts[i], reverse=True)[:15]
            top_actions = {self._action_names[i]: round(action_counts[i] / max(total_actions, 1) * 100, 2)
                           for i in order if i < len(self._action_names)}

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

            available_actions = ""
            if schedule:
                allowed = self.em.get_allowed_buildings_for_stage(self._curriculum_stage)
                available_actions = " | ".join(allowed[:5])

            # Real progress of the current stage for the UI widget (was
            # hardcoded 0.0 before).
            try:
                prog = self.em.get_curriculum_progress(total_done)
                curriculum_progress = float(prog.get("progress_percent", 0.0))
            except Exception:
                curriculum_progress = 0.0

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
            self.metrics.loop_detected = envs_with_loops > 0
            self.metrics.loop_action_name = loop_action_name
            self.metrics.envs_with_loops = envs_with_loops
            self.metrics.curriculum_stage = self._curriculum_stage
            self.metrics.curriculum_progress_percent = curriculum_progress
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
                (getattr(self.em, "vec_env", None) or self.em.env).venv.save_normalization(norm_path)
                (getattr(self.em, "vec_env", None) or self.em.env).venv.save_normalization(str(save_dir / "normalization.json"))
                self._log(f"[Save] {ckpt_path}")

            # Curriculum stage switching
            if schedule:
                for step_threshold, stage in schedule:
                    if total_done >= step_threshold and self._curriculum_stage < stage:
                        self.em.set_curriculum_stage(stage)
                        self._curriculum_stage = stage
                        self._log(f"[Curriculum] Stage -> {stage} at step {total_done:,}")
                        break

            # Run eval
            if eval_every > 0 and rollout_idx % eval_every == 0:
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
                            self._stop = True

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
        (getattr(self.em, "vec_env", None) or self.em.env).venv.save_normalization(norm_path)
        self._log(f"[Save] Final model: {final_path}")

        # End-of-Training Tournament: evaluate all candidates and ensure best_model.pt is the true champion
        try:
            from train_ui2.evaluator import run_eval
            self._log("[Tournament] Running end-of-training model tournament across checkpoints & final...")
            best_cand_path = None
            best_cand_score = -1e9
            cand_paths = sorted(save_dir.glob("checkpoint_*_steps.pt")) + [final_path]
            if (save_dir / "best_model.pt").exists():
                cand_paths.append(save_dir / "best_model.pt")

            import random
            seeds = [random.randint(1, 999999) for _ in range(5)]
            use_median = getattr(self.cfg, "eval_use_median", True)
            w1, w2, w3, w4 = getattr(self.cfg, "eval_score_weights", (0.10, 1.0, 0.10, 0.0001))
            min_b = getattr(self.cfg, "eval_min_bases", 5)
            min_d = getattr(self.cfg, "eval_min_days", 730.0)

            for cp in set(cand_paths):
                if not cp.exists():
                    continue
                try:
                    cp_norm = Path(str(cp).replace(".pt", ".norm.json"))
                    cp_norm_str = str(cp_norm) if cp_norm.exists() else norm_str

                    all_days = []
                    all_bases = []
                    all_people = []
                    all_returns = []
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
                        )
                        all_days.extend(res.get("episode_days", [res["days"]]))
                        all_bases.extend(res.get("episode_bases", [res["bases"]]))
                        all_people.extend(res.get("episode_people", [res["people"]]))
                        all_returns.extend(res.get("episode_returns", [res["avg_return"]]))

                    if use_median:
                        days_agg = float(np.median(all_days))
                        bases_agg = float(np.median(all_bases))
                        people_agg = float(np.median(all_people))
                        return_agg = float(np.median(all_returns))
                    else:
                        days_agg = float(np.mean(all_days))
                        bases_agg = float(np.mean(all_bases))
                        people_agg = float(np.mean(all_people))
                        return_agg = float(np.mean(all_returns))

                    bases_std = float(np.std(all_bases)) if all_bases else 0.0
                    days_std = float(np.std(all_days)) if all_days else 0.0
                    variance_penalty = 0.2 * bases_std + 0.001 * days_std
                    sc = (days_agg * w1 + bases_agg * w2
                             + people_agg * w3 + max(0.0, return_agg) * w4
                             - variance_penalty)

                    if bases_agg >= min_b and days_agg >= min_d and sc > best_cand_score:
                        best_cand_score = sc
                        best_cand_path = cp
                except Exception as ex:
                    self._log(f"[Tournament] Candidate {cp.name} evaluation error: {ex}")

            if best_cand_path is not None:
                self._log(f"[Tournament] Winner: {best_cand_path.name} (score={best_cand_score:.1f})")
                import shutil
                shutil.copy(str(best_cand_path), str(save_dir / "best_model.pt"))
                cp_norm = Path(str(best_cand_path).replace(".pt", ".norm.json"))
                if cp_norm.exists():
                    shutil.copy(str(cp_norm), str(save_dir / "best_model.norm.json"))
        except Exception as te:
            self._log(f"[Tournament] Error during tournament: {te}")

        self.metrics.total_timesteps = total_done
        self.metrics.fps = final_fps
        self.metrics.wall_time_s = elapsed
        self.metrics.best_reward = self.best_reward

        self._log(f"[Done] {total_done:,} steps in {elapsed:.1f}s "
                  f"({final_fps:,.0f} FPS), best_reward={self.best_reward:.2f}, "
                  f"episodes={self.metrics.n_episodes}")

        return self.metrics

    def _get_steps_in_curriculum_stage(self, current_step: int, next_threshold: int) -> int:
        if next_threshold <= current_step:
            return 0
        schedule = getattr(self.cfg, "curriculum_schedule", None)
        if not schedule or len(schedule) < 2:
            return max(next_threshold - current_step, 10000)
        prev_threshold = 0
        for threshold, _ in schedule:
            if threshold > current_step:
                break
            prev_threshold = threshold
        stage_steps = next_threshold - prev_threshold
        return min(stage_steps, 100000)

    def close(self):
        self.em.close()
