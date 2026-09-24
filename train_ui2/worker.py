from __future__ import annotations
import builtins
import queue as _queue
#!/usr/bin/env python3
"""Worker process for train_ui: runs training or eval in a separate process.

Protocol: JSONL messages written to a file (--output). Commands on stdin.

Usage:
  python train_ui/worker.py --config <config.json> --name <run_name> --output <msg.jsonl>
  python train_ui/worker.py --eval-model <model.pt> --output <msg.jsonl>
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

from train_ui2 import protocol as P
from training_lr import apply_configured_learning_rate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sakhalin Colony train/eval worker")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=str, help="path to config JSON (train mode)")
    mode.add_argument("--eval-model", type=str, dest="eval_model", help="model path (eval mode)")
    p.add_argument("--name", type=str, default="", help="run name (train mode)")
    p.add_argument("--output", type=str, default="", help="path to JSONL message file")
    p.add_argument("--command-file", type=str, default="", dest="command_file",
                   help="path to JSONL command file (optional)")
    p.add_argument("--resume-model", type=str, default="", help="path to model .pt to load weights from (fine-tuning)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-days", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--normalization", type=str, default="", dest="normalization",
                   help="path to normalization.json (eval mode)")
    p.add_argument("--command-queue-size", type=int, default=100, help="max size of command queue")
    return p


def _read_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("config must be a JSON object")
    return d


class MsgFile:
    """Thread-safe writer that appends JSONL messages to a file."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._fh: Optional[TextIO] = None

    def open(self) -> None:
        self._fh = open(self._path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, msg: P.Msg) -> None:
        if self._fh is None:
            return
        line = P.encode(msg) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def write_raw(self, text: str) -> None:
        if self._fh is None:
            return
        with self._lock:
            self._fh.write(text)
            self._fh.flush()
            os.fsync(self._fh.fileno())


def _default_warn(message: str) -> None:
    import logging
    logging.getLogger("Worker").warning(message)


WarnFn = Callable[[str], None]


def dispatch_command(d: Any, stop_event: threading.Event,
                     command_queue: Optional[_queue.Queue] = None,
                     warn: WarnFn = _default_warn, source: str = "command") -> bool:
    """Разобрать одну команду UI и передать её трейнеру. True = пришёл «Стоп».

    Единая точка для stdin- и файлового каналов (P1-2/P1-4 ревью 2026-09-24):
    * имя приводится к каноническому (`P.normalize_command`), неизвестное —
      предупреждение, а не молчаливый no-op;
    * «Стоп» ставит `stop_event`: трейнер опрашивает его на КАЖДОМ шаге среды
      (`AsyncTrainer._check_stop`), поэтому остановка не ждёт конца роллаута,
      а штатный путь сохранения (final_model, meta.json, run_end) успевает
      отработать до того, как UI применит жёсткий fallback;
    * переполненная очередь — громкое предупреждение с именем команды.
    """
    if not isinstance(d, dict):
        warn(f"[Worker] {source}: команда должна быть JSON-объектом, получено {type(d).__name__}")
        return False
    raw = d.get("cmd")
    payload = d.get("payload") or {}
    if not isinstance(payload, dict):
        warn(f"[Worker] {source}: payload команды {raw!r} не объект — игнорирую payload")
        payload = {}
    cmd = P.normalize_command(raw)
    if cmd is None:
        warn(f"[Worker] {source}: неизвестная команда {raw!r} — пропущена "
             f"(известные: {sorted(P.KNOWN_COMMANDS)})")
        return False
    if cmd == P.CMD_STOP:
        stop_event.set()
        # Дублируем в очередь, чтобы трейнер залогировал причину остановки;
        # сама остановка от очереди не зависит (её несёт stop_event).
        if command_queue is not None:
            try:
                command_queue.put_nowait(("command", {"cmd": cmd, "payload": payload}))
            except _queue.Full:
                pass  # stop_event уже выставлен — потеря дубля безвредна
        return True
    if command_queue is None:
        warn(f"[Worker] {source}: нет очереди команд — {cmd!r} не доставлена")
        return False
    try:
        command_queue.put(("command", {"cmd": cmd, "payload": payload}), timeout=1)
    except _queue.Full:
        warn(f"[Worker] {source}: очередь команд переполнена "
             f"({command_queue.maxsize}) — команда {cmd!r} ПОТЕРЯНА")
    return False


def _parse_command_line(line: str, warn: WarnFn, source: str) -> Optional[Any]:
    """JSON одной строки команды; битая строка — предупреждение и None."""
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        warn(f"[Worker] {source}: битая строка команды пропущена ({e}): {line[:120]!r}")
        return None


def _watch_stdin(stop_event: threading.Event, command_queue: Optional[_queue.Queue] = None,
                 warn: WarnFn = _default_warn) -> None:
    """Команды из stdin (резервный канал; UI пишет в файл команд)."""
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            d = _parse_command_line(line, warn, "stdin")
            if d is None:
                continue
            if dispatch_command(d, stop_event, command_queue, warn, "stdin"):
                return
    except (ValueError, OSError) as e:
        # stdin закрыт/не читается (запуск без консоли) — канал просто недоступен
        warn(f"[Worker] stdin недоступен: {e}")


def process_command_lines(data: str, stop_event: threading.Event,
                          command_queue: Optional[_queue.Queue] = None,
                          warn: WarnFn = _default_warn) -> bool:
    """Обработать порцию строк файла команд. True = получен «Стоп»."""
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        d = _parse_command_line(line, warn, "command-file")
        if d is None:
            continue
        if dispatch_command(d, stop_event, command_queue, warn, "command-file"):
            return True
    return False


def _watch_commands(command_file: str, stop_event: threading.Event,
                    command_queue: Optional[_queue.Queue] = None,
                    warn: WarnFn = _default_warn) -> None:
    """Файл команд JSONL от UI (основной канал)."""
    offset = 0
    while not stop_event.is_set():
        try:
            size = os.path.getsize(command_file)
        except OSError:
            time.sleep(0.5)  # файла ещё нет: UI создаёт его первой командой
            continue
        if size <= offset:
            time.sleep(0.5)
            continue
        try:
            with open(command_file, "rb") as f:
                f.seek(offset)
                chunk = f.read()
        except OSError as e:
            warn(f"[Worker] файл команд не читается ({e}), повтор")
            time.sleep(0.5)
            continue
        # Недописанную последнюю строку оставляем до следующего опроса, иначе
        # она разобьётся на две «битые» половины.
        cut = chunk.rfind(b"\n")
        if cut < 0:
            time.sleep(0.2)
            continue
        chunk = chunk[:cut + 1]
        offset += len(chunk)
        data = chunk.decode("utf-8", errors="replace")
        if process_command_lines(data, stop_event, command_queue, warn):
            return
        time.sleep(0.5)


def run_train(cfg_dict: Dict[str, Any], run_name: str, mf: MsgFile, stop_event: threading.Event,
              resume_model: str = "", command_queue: Optional[_queue.Queue] = None) -> int:
    # import queue already at top
    _orig_print = builtins.print

    def _print(*a, **kw):
        sep = kw.get("sep", " ")
        end = kw.get("end", "\n")
        text = sep.join(str(x) for x in a) + end
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                mf.write(P.LogMsg(level="info", message=line))
            except Exception:
                pass

    builtins.print = _print  # type: ignore[assignment]

    try:
        return _run_train_inner(cfg_dict, run_name, mf, stop_event, resume_model, command_queue)
    finally:
        builtins.print = _orig_print  # type: ignore[assignment]


def _run_train_inner(cfg_dict: Dict[str, Any], run_name: str, mf: MsgFile, stop_event: threading.Event,
                      resume_model: str = "", command_queue: Optional[_queue.Queue] = None) -> int:
    import torch
    from rl.config import Config, RewardConfig
    from rl.env_manager import EnvManager
    from rl.async_trainer import AsyncTrainer

    home = Path.home()
    model_dir = str(home / "colony_runs" / "models" / run_name)
    log_dir = str(home / "colony_runs" / "logs" / run_name)

    # Build the FULL config via Config.from_dict so that every supported field
    # (eval_seeds, eval_score_weights, early_stopping_patience, unlock_ids,
    # loop_detection_*, target_kl, difficulty, ...) is honoured. The previous
    # hand-written field list silently dropped ~10 fields ??? anything the UI or
    # a JSON profile set for them was ignored ("params don't stick" bug).
    work = dict(cfg_dict)
    work.pop("name", None)
    work.pop("n_layers", None)
    cfg = Config.from_dict(work)

    # Reward: UI writes reward keys flat at top level; JSON profiles may nest
    # them under "reward". RewardConfig.from_dict handles both.
    cfg.reward = RewardConfig.from_dict(cfg_dict)

    cfg.log_dir = log_dir
    cfg.model_dir = model_dir

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    def log(level: str, message: str) -> None:
        mf.write(P.LogMsg(level=level, message=message))

    def progress_cb(metrics, extra_data=None) -> None:
        try:
            mf.write(P.ProgressMsg(
                done=metrics.total_timesteps,
                total=cfg.total_timesteps,
                fps=float(getattr(metrics, 'fps', 0.0)),
                best_reward=float(getattr(metrics, 'best_reward', 0.0)) if getattr(metrics, 'best_reward', float("-inf")) != float("-inf") else 0.0,
                episodes=int(getattr(metrics, 'n_episodes', 0)),
                policy_loss=float(getattr(metrics, 'policy_loss', 0.0)),
                value_loss=float(getattr(metrics, 'value_loss', 0.0)),
                entropy=float(getattr(metrics, 'entropy', 0.0)),
                kl=float(getattr(metrics, 'approx_kl', 0.0)),
                ent_coef=float(getattr(metrics, 'ent_coef', cfg.ent_coef)),
                top_actions=dict(getattr(metrics, 'top_actions', {})),
                action_counts=dict(getattr(metrics, 'action_counts', {})),
                action_legality=dict(getattr(metrics, 'action_legality', {})),
                action_mask_reasons=dict(getattr(metrics, 'action_mask_reasons', {})),
                action_total_steps=int(getattr(metrics, 'action_total_steps', 0) or 0),
                loop_detected=bool(getattr(metrics, 'loop_detected', False)),
                loop_action_name=getattr(metrics, 'loop_action_name', None),
                envs_with_loops=int(getattr(metrics, 'envs_with_loops', 0)),
                curriculum_stage_active=int(getattr(metrics, 'curriculum_stage', 0)),
                curriculum_next_at_step=getattr(metrics, 'curriculum_next_at_step', None),
                curriculum_stage=int(getattr(metrics, 'curriculum_stage', 0)),
                curriculum_progress_percent=float(getattr(metrics, 'curriculum_progress_percent', 0.0)),
                curriculum_progress_valid=bool(getattr(metrics, 'curriculum_progress_valid', True)),
                curriculum_available_actions=str(getattr(metrics, 'curriculum_available_actions', '')),
                curriculum_upcoming_stages=list(getattr(metrics, 'curriculum_upcoming_stages', [])),
                avg_return=float(getattr(metrics, 'avg_return', 0.0)),
                median_return=float(getattr(metrics, 'median_return', 0.0)),
                max_return=float(getattr(metrics, 'max_return', 0.0)),
                min_return=float(getattr(metrics, 'min_return', 0.0)),
                n_episodes_for_stats=int(getattr(metrics, 'n_episodes_for_stats', 0)),
            ))
        except Exception as e:
            try:
                mf.write(P.LogMsg(level="warn", message=f"[Worker] progress_cb error: {type(e).__name__}: {e}"))
            except Exception:
                pass

    log("info", f"[Worker] run={run_name}")
    log("info", f"[Worker] === TRAINING PARAMETERS ===")
    log("info", f"[Worker] steps={cfg.total_timesteps:,} envs={cfg.n_envs} n_steps={cfg.n_steps} "
                f"batch_size={cfg.batch_size} n_epochs={cfg.n_epochs}")
    log("info", f"[Worker] lr={cfg.learning_rate} gamma={cfg.gamma} gae_lambda={cfg.gae_lambda} "
                f"clip_range={cfg.clip_range}")
    log("info", f"[Worker] ent_coef={cfg.ent_coef} vf_coef={cfg.vf_coef} "
                f"max_grad_norm={cfg.max_grad_norm}")
    log("info", f"[Worker] net_arch={cfg.net_arch} obs_mode={cfg.obs_mode} "
                f"minimap_radius={cfg.minimap_radius}")
    log("info", f"[Worker] device={device} amp={cfg.use_amp}({cfg.amp_dtype}) "
                f"compile={cfg.torch_compile}")
    log("info", f"[Worker] seed={cfg.seed} map_size={cfg.map_size}")
    log("info", f"[Worker] reward: build={cfg.reward.build_bonus} chain={cfg.reward.chain_bonus} "
                f"novelty={cfg.reward.novelty} income={cfg.reward.daily_income} "
                f"error={cfg.reward.error_penalty} game_over={cfg.reward.game_over_penalty}")
    log("info", f"[Worker] curriculum_stage={cfg.curriculum_stage} "
                f"schedule={cfg.curriculum_schedule}")
    # Явный лог разрешённых зданий: так видно, что ручной набор из вкладки
    # «Курикулум» действительно дошёл до среды (иначе сценарий молча теряется).
    try:
        from rl.curriculum import RESOURCE_NAMES as _RES_NAMES
        from rl.curriculum import allowed_ids as _allowed_ids
        _manual = cfg.effective_unlock_ids()
        _allowed = _allowed_ids(cfg.curriculum_stage, _manual, True)
        _st = cfg.curriculum_state()
        if _st.all_resources:
            _prio = "all"
        else:
            _picked = [r for r, w in zip(_RES_NAMES, _st.resource_weights, strict=False) if w > 0]
            _w = "[" + ",".join(str(int(w)) for w in _st.resource_weights) + "]"
            _prio = f"{','.join(_picked)} (weights={_w})"
        log("info", f"[Worker] curriculum: stage={cfg.curriculum_stage} "
                    f"manual={_manual or '—'} checkbox={bool(cfg.use_curriculum_tab)} "
                    f"→ {len(_allowed)} buildings allowed, priority={_prio}; "
                    f"mechanics={list(_st.enabled_mechanics)} tax_to_debt={cfg.tax_to_debt}")
    except Exception as _ex:  # noqa: BLE001 — не роняем обучение из-за лога
        log("warn", f"[Worker] curriculum log failed: {type(_ex).__name__}: {_ex}")
    log("info", f"[Worker] model_dir={cfg.model_dir}")

    t0 = time.perf_counter()
    em = EnvManager(cfg, device)
    log("info", f"[Worker] env ready: obs={em.obs_size} actions={em.n_actions}")

    if resume_model and resume_model.strip():
        rm_path = Path(resume_model)
        ckpt = torch.load(rm_path, map_location=device, weights_only=False)
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
        from rl._nn_common import clean_policy_state, load_policy_state
        clean = clean_policy_state(state)
        # PR 5: a v0 checkpoint on a v1 env (or vice versa) must fail here
        # with a clear message, not in load_state_dict with a shape error.
        from rl.curriculum import (
            check_obs_version_compat,
            check_policy_obs_compat,
            ckpt_flat_width,
        )
        check_obs_version_compat(
            ckpt.get("obs_version"), cfg.obs_version, ckpt_path=resume_model)
        _flat_w = ckpt_flat_width(clean)
        if _flat_w is not None:
            check_policy_obs_compat(
                _flat_w, int(em.obs_size), ckpt_path=resume_model)
        load_policy_state(em.model, clean, ckpt_path=resume_model,
                          log=lambda m: log("info", f"[Worker] {m}"))
        log("info", f"[Worker] loaded weights from {resume_model}")

        optimizer_loaded = False
        if "optimizer_state" in ckpt:
            try:
                em.ppo.optimizer.load_state_dict(ckpt["optimizer_state"])
                optimizer_loaded = True
                log("info", "[Worker] loaded optimizer moments")
            except Exception as e:
                log("warn", f"[Worker] optimizer state not loaded: {e}")
        previous_lrs = apply_configured_learning_rate(
            em.ppo.optimizer, cfg.learning_rate, scheduler=em.ppo.scheduler)
        em.ppo.lr = float(cfg.learning_rate)
        if optimizer_loaded:
            log("info", "[Worker] fine-tune LR: checkpoint "
                f"{previous_lrs} -> configured {cfg.learning_rate}")
        else:
            log("info", f"[Worker] fine-tune LR from config: {cfg.learning_rate}")

        norm_candidates = [
            rm_path.with_name(rm_path.name.replace(".pt", ".norm.json")),
            rm_path.parent / "normalization.json",
            rm_path.parent / "best_model.norm.json",
        ]
        norm_loaded = False
        for nc in norm_candidates:
            if nc.exists():
                try:
                    (getattr(em, "vec_env", None) or em.env).venv.load_normalization(str(nc))
                    log("info", f"[Worker] loaded normalization stats from {nc}")
                    norm_loaded = True
                    break
                except Exception as ex:
                    log("warn", f"[Worker] failed to load normalization from {nc}: {ex}")
        if not norm_loaded:
            log("warn", f"[Worker] WARNING: no normalization file found for resume_model {resume_model}!")

    trainer = AsyncTrainer(
        cfg=cfg,
        env_manager=em,
        logger=None,
        progress_callback=progress_cb,
        stop_check=lambda: stop_event.is_set(),
    )

    metrics = trainer.train(command_queue=command_queue)
    em.close()

    elapsed = time.perf_counter() - t0
    run_dir = Path(cfg.model_dir)
    final_path = run_dir / "final_model.pt"
    episode_diagnostics_path = run_dir / "episode_diagnostics.jsonl"
    if episode_diagnostics_path.is_file():
        log("info", f"[Worker] episode diagnostics: {episode_diagnostics_path}")

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": int(metrics.total_timesteps),
        "best_reward": float(metrics.best_reward) if metrics.best_reward != float("-inf") else 0.0,
        "episodes": int(metrics.n_episodes),
        "train_time_sec": float(elapsed),
        # «completed» | «stopped» (кнопка «Стоп») | «early_stopped». При
        # остановке пользователем финальный турнир не запускается, и
        # best_model.pt — от последнего eval (tournament="skipped_user_stop").
        "status": (
            "completed" if not metrics.stop_reason
            else "early_stopped" if metrics.stop_reason == "early_stop"
            else "stopped"
        ),
        "stop_reason": metrics.stop_reason or None,
        "tournament": metrics.tournament or None,
        "final_model": final_path.name if final_path.is_file() else None,
        "episode_diagnostics_file": (
            episode_diagnostics_path.name if episode_diagnostics_path.is_file() else None
        ),
        "config": cfg.to_dict(),
    }
    meta_path = run_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("info", f"[Worker] meta saved: {meta_path}")

    mf.write(P.SavedMsg(path=str(final_path)))
    mf.write(P.DoneMsg(
        total=int(metrics.total_timesteps),
        time_s=float(elapsed),
        best_reward=float(metrics.best_reward) if metrics.best_reward != float("-inf") else 0.0,
        episodes=int(metrics.n_episodes),
    ))
    return 0


def run_eval(model_path: str, episodes: int, max_days: int, seed: int,
             device: str, mf: MsgFile, normalization_path: str = "") -> int:
    from train_ui2.evaluator import run_eval as _run_eval

    def log(level: str, message: str) -> None:
        mf.write(P.LogMsg(level=level, message=message))

    norm = Path(normalization_path) if normalization_path else None
    log("info", f"[Eval] model={model_path} episodes={episodes} max_days={max_days}")

    log_path = Path("logs") / f"eval_{Path(model_path).stem}_{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    result = _run_eval(model_path, episodes=episodes, max_days=max_days,
                       seed=seed, device=device, normalization_path=norm,
                       log_path=log_path)
    log("info", f"[Eval] days={result['days']:.1f} people={result['people']:.1f} "
                f"bases={result['bases']:.1f} avg_return={result['avg_return']:.2f}")
    log("info", f"[Eval] step log written to: {log_path}")
    mf.write_raw(json.dumps({"type": "eval_result", **result, "log_path": str(log_path)}, ensure_ascii=False) + "\n")
    return 0


def _install_termination_handler() -> None:
    """SIGTERM → SystemExit, чтобы `AsyncTrainer.train` дописал `run_end`.

    Жёсткое завершение — это fallback UI после таймаута мягкой остановки.
    Без обработчика SIGTERM убивает процесс мгновенно, и
    `episode_diagnostics.jsonl` остаётся без `run_end` — неотличим от падения.
    С обработчиком исключение раскручивает стек, и трейнер закрывает JSONL
    со `status="killed"`. На Windows `TerminateProcess` перехватить нельзя —
    там спасает только мягкий путь (команда stop_training).
    """
    import signal

    def _on_sigterm(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    if hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _on_sigterm)


def main() -> int:
    args = _build_parser().parse_args()

    mf = MsgFile(args.output) if args.output else None
    if mf:
        mf.open()

    stop_event = threading.Event()

    command_queue: _queue.Queue = _queue.Queue(maxsize=max(1, int(args.command_queue_size)))

    def warn(message: str) -> None:
        # Проблемы с командами видны в логе UI, а не только в stderr воркера
        # (UI запускает его с stderr=DEVNULL).
        if mf:
            mf.write(P.LogMsg(level="warn", message=message))
        else:
            _default_warn(message)

    _install_termination_handler()

    stdin_thread = threading.Thread(target=_watch_stdin, args=(stop_event, command_queue, warn),
                                    daemon=True)
    stdin_thread.start()

    # Start file-based command watcher if command file is provided
    cmd_file = getattr(args, 'command_file', '')
    if cmd_file:
        cmd_thread = threading.Thread(
            target=_watch_commands, args=(cmd_file, stop_event, command_queue, warn), daemon=True)
        cmd_thread.start()

    if mf:
        mf.write(P.ReadyMsg())

    rc = 1
    try:
        if args.config:
            cfg_dict = _read_config(Path(args.config))
            run_name = args.name or cfg_dict.get("name", "") or f"run_{int(time.time())}"
            rc = run_train(cfg_dict, run_name, mf, stop_event,
                           resume_model=args.resume_model,
                           command_queue=command_queue)
        else:
            rc = run_eval(args.eval_model, args.episodes, args.max_days,
                          args.seed, args.device, mf,
                          normalization_path=args.normalization)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if mf:
            try:
                mf.write(P.ErrorMsg(message=f"{type(e).__name__}: {e}\n{tb}"))
            except Exception:
                pass
        rc = 1
    finally:
        if mf:
            mf.close()

    return rc


if __name__ == "__main__":
    sys.exit(main())

