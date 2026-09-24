"""Этап 1 ревью 2026-09-24: мягкий «Стоп», протокол команд, obs между роллаутами.

P1-1  «Стоп» из UI → final_model.pt + meta.json + run_end(status="stopped"),
      турнир пропущен; пауза не блокирует остановку; аварийный выход тоже
      закрывает JSONL; SoftStop (логика UI без Qt).
P1-2  файловый канал команд: битая строка/переполнение очереди/незнакомая
      команда — предупреждение, а не молчаливый `except Exception: pass`.
P1-4  один словарь команд (`train_ui2.protocol`), паритет UI → trainer,
      `else`-ветка трейнера для неизвестных команд.
P1-5  следующий роллаут продолжает с текущего obs, а не с obs первого reset().
"""
from __future__ import annotations

import ast
import json
import queue
import threading
from pathlib import Path
from typing import Any, List

import pytest
import torch

from rl.async_trainer import AsyncTrainer
from rl.config import Config
from tests.test_async_trainer import FakeEnvManager
from train_ui2 import protocol as P
from train_ui2 import worker as W
from train_ui2.soft_stop import SoftStop

REPO = Path(__file__).resolve().parent.parent


def _cfg(tmp_path: Path, **kw: Any) -> Config:
    base = dict(n_envs=2, n_steps=3, total_timesteps=60, save_freq=0, eval_freq=0,
                use_amp=False, model_dir=str(tmp_path))
    base.update(kw)
    return Config(**base)


def _records(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _Logs:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, msg: str) -> None:
        self.lines.append(msg)

    def any(self, needle: str) -> bool:
        return any(needle in line for line in self.lines)


# ── P1-4: словарь команд ────────────────────────────────────────────────────

def test_command_vocabulary_is_single_and_canonical():
    assert P.KNOWN_COMMANDS == {
        "stop_training", "pause_training", "resume_training",
        "boost_entropy", "reset_curriculum",
    }
    assert {c.value for c in P.CommandType} == P.KNOWN_COMMANDS
    # старые короткие имена stdin-протокола приводятся к каноническим
    assert P.normalize_command("pause") == P.CMD_PAUSE
    assert P.normalize_command("resume") == P.CMD_RESUME
    assert P.normalize_command("stop") == P.CMD_STOP
    for bad in ("start", "stop_trainig", "", None, 3):
        assert P.normalize_command(bad) is None


def _ui_sent_commands() -> set:
    """Команды, которые шлёт main_window: только через P.CMD_*, без литералов."""
    tree = ast.parse((REPO / "train_ui2" / "main_window.py").read_text(encoding="utf-8"))
    sent: set = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("_send_command", "encode_command") and node.args):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.IfExp):
            options = [arg.body, arg.orelse]
        else:
            options = [arg]
        for opt in options:
            if isinstance(opt, ast.Constant):
                pytest.fail(f"main_window.py:{node.lineno}: команда строкой {opt.value!r} — "
                            f"используйте train_ui2.protocol.CMD_*")
            if isinstance(opt, ast.Attribute) and opt.attr.startswith("CMD_"):
                sent.add(getattr(P, opt.attr))
            elif isinstance(opt, ast.Name) and opt.id == "cmd":
                continue  # переменная; её значения — P.CMD_* (собраны ниже)
            else:
                pytest.fail(f"main_window.py:{node.lineno}: не могу проверить команду {ast.dump(opt)}")
    # Команда может идти через переменную (`cmd = P.CMD_RESUME if … else …`):
    # считаем отправляемыми все упоминания P.CMD_* в модуле UI.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr.startswith("CMD_")
                and isinstance(node.value, ast.Name) and node.value.id == "P"):
            sent.add(getattr(P, node.attr))
    return sent


def test_ui_commands_subset_of_trainer_commands(tmp_path):
    sent = _ui_sent_commands()
    assert {P.CMD_STOP, P.CMD_PAUSE, P.CMD_RESUME, P.CMD_BOOST_ENTROPY} <= sent
    assert sent <= P.KNOWN_COMMANDS

    # Трейнер реально обрабатывает каждую известную команду (нет warning)
    logs = _Logs()
    trainer = AsyncTrainer(cfg=_cfg(tmp_path), env_manager=FakeEnvManager())
    trainer._log = logs  # type: ignore[method-assign]
    q: queue.Queue = queue.Queue()
    for cmd in sorted(P.KNOWN_COMMANDS - {P.CMD_STOP}) + [P.CMD_STOP]:
        q.put(("command", {"cmd": cmd, "payload": {}}))
    trainer._process_commands(q)
    assert not logs.any("unknown command"), logs.lines
    assert trainer._stop and trainer._stop_reason == "user"


def test_trainer_warns_on_unknown_command(tmp_path):
    logs = _Logs()
    trainer = AsyncTrainer(cfg=_cfg(tmp_path), env_manager=FakeEnvManager())
    trainer._log = logs  # type: ignore[method-assign]
    q: queue.Queue = queue.Queue()
    q.put(("command", {"cmd": "start", "payload": {}}))
    q.put(("command", {"cmd": "pause", "payload": {}}))  # legacy alias — работает
    trainer._process_commands(q)
    assert logs.any("unknown command 'start'")
    assert trainer._paused is True


def test_encode_stop_uses_canonical_name():
    d = json.loads(P.encode_stop())
    assert d["cmd"] == P.CMD_STOP


# ── P1-2: канал команд воркера ──────────────────────────────────────────────

def test_broken_line_warns_and_next_command_still_processed():
    stop, q, logs = threading.Event(), queue.Queue(), _Logs()
    data = "{broken\n" + P.encode_command(P.CMD_BOOST_ENTROPY, {"factor": 2.0}) + "\n"
    assert W.process_command_lines(data, stop, q, logs) is False
    assert logs.any("битая строка")
    assert q.get_nowait() == ("command", {"cmd": P.CMD_BOOST_ENTROPY, "payload": {"factor": 2.0}})


def test_unknown_command_warns_and_is_not_queued():
    stop, q, logs = threading.Event(), queue.Queue(), _Logs()
    W.process_command_lines('{"cmd": "stop_trainig"}\n', stop, q, logs)
    assert logs.any("неизвестная команда 'stop_trainig'")
    assert q.empty() and not stop.is_set()


def test_full_queue_is_reported_with_command_name(monkeypatch):
    stop, logs = threading.Event(), _Logs()
    q: queue.Queue = queue.Queue(maxsize=1)
    q.put(("command", {"cmd": P.CMD_PAUSE, "payload": {}}))
    monkeypatch.setattr(q, "put", lambda *a, **k: (_ for _ in ()).throw(queue.Full()))
    W.dispatch_command({"cmd": P.CMD_BOOST_ENTROPY}, stop, q, logs)
    assert logs.any("ПОТЕРЯНА") and logs.any("boost_entropy")


def test_stop_sets_event_immediately_and_legacy_names_normalized():
    stop, q, logs = threading.Event(), queue.Queue(), _Logs()
    assert W.dispatch_command({"cmd": "pause"}, stop, q, logs) is False
    assert q.get_nowait()[1]["cmd"] == P.CMD_PAUSE
    assert W.dispatch_command({"cmd": P.CMD_STOP}, stop, q, logs) is True
    assert stop.is_set()
    assert not logs.lines


def test_watch_commands_waits_for_complete_line(tmp_path):
    """Недописанная строка не режется пополам в «битые» половины."""
    cmd_file = tmp_path / "cmd.jsonl"
    full = P.encode_command(P.CMD_STOP) + "\n"
    cmd_file.write_text(full[:10], encoding="utf-8")
    stop, q, logs = threading.Event(), queue.Queue(), _Logs()
    t = threading.Thread(target=W._watch_commands, args=(str(cmd_file), stop, q, logs), daemon=True)
    t.start()
    t.join(0.8)
    assert not stop.is_set() and not logs.lines
    with open(cmd_file, "a", encoding="utf-8") as f:
        f.write(full[10:])
    t.join(3.0)
    assert stop.is_set() and not t.is_alive()
    assert not logs.lines


def test_worker_has_no_catch_all_in_command_path():
    src = (REPO / "train_ui2" / "worker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in (
                "dispatch_command", "process_command_lines", "_watch_commands", "_watch_stdin"):
            for h in ast.walk(fn):
                if isinstance(h, ast.ExceptHandler):
                    names = ast.unparse(h.type) if h.type is not None else "<bare>"
                    assert "Exception" not in names and names != "<bare>", (fn.name, names)


# ── P1-1: мягкая остановка в трейнере ───────────────────────────────────────

class _StopAtStepEM(FakeEnvManager):
    """Посылает «Стоп» посреди роллаута на заданном шаге среды."""

    def __init__(self, on_step, stop_at: int, **kw: Any) -> None:
        super().__init__(**kw)
        self._on_step = on_step
        self._stop_at = stop_at

    def collect_step(self, obs):
        out = super().collect_step(obs)
        if self._step_count == self._stop_at:
            self._on_step()
        return out


def test_stop_command_mid_rollout_saves_final_and_skips_tournament(tmp_path, monkeypatch):
    import train_ui2.evaluator as ev

    def _no_tournament(*a, **k):
        raise AssertionError("турнир не должен запускаться при остановке пользователем")

    monkeypatch.setattr(ev, "run_eval", _no_tournament)
    q: queue.Queue = queue.Queue()
    em = _StopAtStepEM(lambda: q.put(("command", {"cmd": P.CMD_STOP, "payload": {}})),
                       stop_at=5, n_envs=2, n_steps=3)
    trainer = AsyncTrainer(cfg=_cfg(tmp_path), env_manager=em)
    metrics = trainer.train(command_queue=q)

    # остановились на ближайшем шаге (6-й шаг не собирался), не дожидаясь total=60
    assert em._step_count == 5
    assert metrics.stop_reason == "user"
    assert metrics.tournament == "skipped_user_stop"
    assert (tmp_path / "final_model.pt").is_file()
    rec = _records(tmp_path / "episode_diagnostics.jsonl")
    assert rec[-1]["record_type"] == "run_end" and rec[-1]["status"] == "stopped"


def test_stop_check_is_honoured_while_paused(tmp_path):
    """Пауза + «Стоп» через stop_event: раньше цикл паузы его не видел."""
    stop = threading.Event()
    q: queue.Queue = queue.Queue()
    q.put(("command", {"cmd": P.CMD_PAUSE, "payload": {}}))
    trainer = AsyncTrainer(cfg=_cfg(tmp_path), env_manager=FakeEnvManager(),
                           stop_check=stop.is_set)
    threading.Timer(0.3, stop.set).start()
    done: List[Any] = []
    t = threading.Thread(target=lambda: done.append(trainer.train(command_queue=q)), daemon=True)
    t.start()
    t.join(10.0)
    assert not t.is_alive(), "мягкая остановка на паузе зависла"
    assert done[0].stop_reason == "user"
    assert (tmp_path / "final_model.pt").is_file()


def test_crash_still_writes_run_end(tmp_path):
    class _Boom(FakeEnvManager):
        def collect_step(self, obs):
            if self._step_count >= 2:
                raise RuntimeError("env exploded")
            return super().collect_step(obs)

    trainer = AsyncTrainer(cfg=_cfg(tmp_path), env_manager=_Boom())
    with pytest.raises(RuntimeError, match="env exploded"):
        trainer.train()
    rec = _records(tmp_path / "episode_diagnostics.jsonl")
    assert rec[-1]["record_type"] == "run_end" and rec[-1]["status"] == "error"


def test_sigterm_like_exit_writes_killed(tmp_path):
    class _Killed(FakeEnvManager):
        def collect_step(self, obs):
            if self._step_count >= 1:
                raise SystemExit(143)  # так worker превращает SIGTERM
            return super().collect_step(obs)

    trainer = AsyncTrainer(cfg=_cfg(tmp_path), env_manager=_Killed())
    with pytest.raises(SystemExit):
        trainer.train()
    rec = _records(tmp_path / "episode_diagnostics.jsonl")
    assert rec[-1]["status"] == "killed"
    assert sum(r["record_type"] == "run_end" for r in rec) == 1


def test_worker_stop_end_to_end_writes_final_meta_and_run_end(tmp_path, monkeypatch):
    """DoD P1-1: команда stop_training через канал воркера → final_model.pt,
    meta.json (status=stopped, tournament skipped) и run_end в JSONL."""
    import rl.env_manager as envm

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    stop = threading.Event()
    cq: queue.Queue = queue.Queue()

    def send_stop() -> None:
        W.process_command_lines(P.encode_stop() + "\n", stop, cq)

    class _EM(_StopAtStepEM):
        def __init__(self, cfg, device):
            super().__init__(send_stop, stop_at=4, n_envs=cfg.n_envs, n_steps=cfg.n_steps)

    monkeypatch.setattr(envm, "EnvManager", _EM)
    out = tmp_path / "msg.jsonl"
    mf = W.MsgFile(str(out))
    mf.open()
    try:
        rc = W.run_train({"n_envs": 2, "n_steps": 3, "total_timesteps": 600,
                          "eval_freq": 0, "save_freq": 0, "use_amp": False,
                          "device": "cpu", "torch_compile": False},
                         "stop_run", mf, stop, command_queue=cq)
    finally:
        mf.close()
    assert rc == 0
    run_dir = tmp_path / "colony_runs" / "models" / "stop_run"
    assert (run_dir / "final_model.pt").is_file()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "stopped"
    assert meta["tournament"] == "skipped_user_stop"
    assert meta["final_model"] == "final_model.pt"
    rec = _records(run_dir / "episode_diagnostics.jsonl")
    assert rec[-1]["record_type"] == "run_end" and rec[-1]["status"] == "stopped"
    msgs = [json.loads(line)["type"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert "done" in msgs and "saved" in msgs


# ── P1-1: логика UI (без Qt) ────────────────────────────────────────────────

def test_soft_stop_timeline():
    now = [100.0]
    s = SoftStop(grace_s=60.0, clock=lambda: now[0])
    assert not s.pending and not s.should_force()
    assert s.request() is True           # первое нажатие: слать команду
    now[0] += 59.0
    assert s.pending and not s.should_force()
    now[0] += 1.0
    assert s.should_force()              # таймаут → fallback terminate()
    s.mark_forced()
    assert not s.should_force()          # второй раз не убиваем
    assert s.request() is False          # повторное нажатие = «сразу»
    s.reset()
    assert not s.pending and not s.forced


def test_ui_stop_no_longer_terminates_immediately():
    """_stop_training не зовёт terminate() сам — только через _force_stop."""
    tree = ast.parse((REPO / "train_ui2" / "main_window.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_stop_training")
    calls = {n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "terminate" not in calls and "kill" not in calls
    assert "_send_command" in calls and "_force_stop" in calls


# ── P1-5: obs между роллаутами ──────────────────────────────────────────────

def test_next_rollout_starts_from_current_obs_not_initial_reset(tmp_path):
    class _Track(FakeEnvManager):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.passed: List[torch.Tensor] = []
            self.returned: List[torch.Tensor] = []

        def collect_step(self, obs):
            self.passed.append(obs)
            new_obs, infos = super().collect_step(obs)
            self.returned.append(new_obs)
            return new_obs, infos

    em = _Track(n_envs=2, n_steps=3)
    trainer = AsyncTrainer(cfg=_cfg(tmp_path, total_timesteps=12), env_manager=em)
    trainer.train()
    assert len(em.passed) == 6  # 2 роллаута × 3 шага
    # первый шаг второго роллаута получает obs последнего шага первого
    assert em.passed[3] is em.returned[2]


def test_stop_during_tournament_interrupts_it(tmp_path, monkeypatch):
    """«Стоп» после обычного завершения, пока идёт турнир, не ждёт всех кандидатов."""
    import train_ui2.evaluator as ev

    stop = threading.Event()
    calls: List[str] = []

    def _eval(model_path: str, **kw: Any) -> dict:
        calls.append(model_path)
        stop.set()  # пользователь нажал «Стоп» во время первого кандидата
        return {"days": 1.0, "people": 1.0, "bases": 1.0, "avg_return": 0.0}

    monkeypatch.setattr(ev, "run_eval", _eval)
    for i in (6, 12):  # два лишних кандидата помимо final_model.pt
        (tmp_path / f"checkpoint_{i}_steps.pt").write_bytes(b"x")
    trainer = AsyncTrainer(cfg=_cfg(tmp_path, total_timesteps=6), env_manager=FakeEnvManager(),
                           stop_check=stop.is_set)
    metrics = trainer.train()
    assert metrics.tournament == "interrupted_user_stop"
    assert len({c for c in calls}) == 1
    assert not (tmp_path / "best_model.pt").exists()
