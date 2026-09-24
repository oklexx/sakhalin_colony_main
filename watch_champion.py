#!/usr/bin/env python3
"""Watch a trained champion model play the colony game in real time.

Usage:
  python watch_champion.py --model-dir ~/colony_runs/models/run_003
  python watch_champion.py --model-dir ~/colony_runs/models/run_003 --speed 5
  python watch_champion.py --model-dir ~/colony_runs/models/run_003 --episodes 3
"""
import argparse
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from colony_cpp_api import require_colony
from cpp_env import CppColonyEnv
from train_ui2.evaluator import _load_policy
from train_ui2.models import latest_checkpoint


def _same_file(stream, path: Path) -> bool:
    """Is `stream` already pointing at `path`?

    The GUI launches us with stdout AND stderr redirected into --log-file
    (`subprocess.Popen(stdout=fh, stderr=STDOUT)`). Opening the same file a
    second time here gives a SECOND file object with its OWN offset: the two
    writers then overwrite each other's bytes, and whatever lands last wins.
    That is why a crash showed up in the UI as «завершилось с кодом 1» with the
    traceback nowhere to be seen — it was written at an offset that the
    duplicated `print` output had already clobbered.
    """
    try:
        s1 = os.fstat(stream.fileno())
    except (OSError, ValueError, AttributeError):
        return False
    try:
        s2 = os.stat(path)
    except OSError:
        return False
    if os.name == "nt":
        # st_ino is meaningful on Windows for Python 3.8+ (via GetFileInformation).
        return (s1.st_dev, s1.st_ino) == (s2.st_dev, s2.st_ino) and s1.st_ino != 0
    return (s1.st_dev, s1.st_ino) == (s2.st_dev, s2.st_ino)


class TeeWriter:
    """Write to both stdout and a file (for UI log capture).

    When stdout IS the file (UI launch), the duplicate write is suppressed —
    see `_same_file`.
    """

    def __init__(self, stdout, file, duplicate: bool = True):
        self.stdout = stdout
        self.file = file
        self.duplicate = duplicate

    def write(self, data):
        if self.duplicate:
            self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        if self.duplicate:
            try:
                self.stdout.flush()
            except (OSError, ValueError):
                pass
        self.file.flush()


def read_curriculum_from_meta(model_dir):
    """Read stage + manual set + checkbox flag from the model meta files."""
    from rl.curriculum import read_curriculum_meta

    return read_curriculum_meta(Path(model_dir))


def read_stage_from_meta(model_dir):
    """Read curriculum_stage_at_best from best_model.meta.json. Returns None if not found."""
    return read_curriculum_from_meta(model_dir)["curriculum_stage"]


def curriculum_action_mask(allowed, action_names):
    """Action mask (1.0/0.0) that locks BUILD_* actions outside `allowed`.

    Safety net for the GUI watch: the C++ exe only restricts the env if it was
    rebuilt with --unlock-ids support. Intersecting the mask here guarantees the
    watched policy can never pick a building the curriculum forbids. Returns
    None when the curriculum does not restrict anything.
    """
    if allowed is None:
        return None
    allowed_norm = {str(b).replace("_", "").upper() for b in allowed}
    mask = []
    restricted = False
    for name in action_names:
        if name.startswith("BUILD_"):
            bid = name[len("BUILD_"):].replace("_", "").upper()
            ok = bid in allowed_norm
            restricted = restricted or not ok
            mask.append(1.0 if ok else 0.0)
        else:
            mask.append(1.0)
    return mask if restricted else None


def write_action(action_file: Path, action: int, timeout: float = 2.0) -> bool:
    """Write action int to the IPC file (atomic tmp + rename).

    Waits for C++ to delete the file (consumed previous action) before writing.
    This prevents overwriting an unread action.

    Returns True when the previous action was consumed in time. False means the
    GUI is not reading the file at all (мертвое окно, чужой exe, зависший
    headless-ai) — caller turns that into a diagnosis instead of waiting forever.

    The write is atomic (`os.replace`): the GUI used to be able to open a
    truncated actions.txt, fail the `f >> action` parse and still delete the
    file — the action was lost and BOTH sides waited for each other forever
    (window open, game frozen, log silent).
    """
    deadline = time.monotonic() + timeout
    consumed = True
    while action_file.exists():
        if time.monotonic() >= deadline:
            consumed = False
            break
        time.sleep(0.005)
    tmp = action_file.with_name(action_file.name + ".tmp")
    tmp.write_text(str(action), encoding="utf-8")
    os.replace(tmp, action_file)
    return consumed


GUI_EXE_NAMES = (
    "sakhalin_colony_gui.exe",
    "SkhClny3.exe",
    "colony_gui.exe",
)

GUI_EXE_DIRS = (
    ".",
    "Release",
    "build",
    "build/Release",
    "x64/Release",
    "out/build/x64-Release",
    "python",
)


def resolve_model_file(model_dir: Path, requested: str = "best_model.pt") -> Path:
    """Какой файл весов грузить: запрошенный → final → best → свежий чекпойнт.

    `final_model.pt` пишется только в конце обучения, поэтому у остановленного
    вручную или упавшего прогона его нет. Раньше цепочка запасных вариантов его
    не знала: `--model-file final_model.pt` на прерванном прогоне перескакивал
    сразу на чекпойнт, хотя рядом лежал `best_model.pt` — чемпион турнира
    (обычно сильнее произвольного чекпойнта). Порядок совпадает с
    `train_ui2.models.pick_model_file`, чтобы UI и CLI выбирали одно и то же.

    Если нет ничего, возвращается запрошенный путь: ошибка ниже назовёт именно
    тот файл, который просил пользователь.
    """
    wanted = model_dir / requested
    if wanted.exists():
        return wanted
    for name in ("final_model.pt", "best_model.pt"):
        cand = model_dir / name
        if cand.exists():
            return cand
    latest = latest_checkpoint(model_dir)
    return latest if latest is not None else wanted


def find_gui_exe() -> "Path | None":
    """Locate the raylib GUI executable.

    build_gui.bat emits it to the project root, but CMake/MSBuild layouts put
    it under Release/ or build/Release — the old single hardcoded path made
    the visual watch fail with "GUI exe not found" even when it was built.
    An explicit COLONY_GUI_EXE env var wins over the search.
    """
    import os
    env_path = os.environ.get("COLONY_GUI_EXE", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
    for d in GUI_EXE_DIRS:
        for name in GUI_EXE_NAMES:
            p = (PROJECT_ROOT / d / name)
            if p.exists():
                return p
    return None


def read_state(state_file: Path) -> dict | None:
    """Read state JSON from the IPC file. Returns None if not found."""
    import json
    if not state_file.exists():
        return None
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ── visual watch: IPC hygiene and diagnostics ────────────────────────────────
#
# Окно наблюдения «не запускалось» молча: драйвер не проверял ни старт exe, ни
# появление state.json, и в любой непонятной ситуации уходил в бесконечный
# `time.sleep(0.05)` (или в бесконечный рестарт exe). Ниже — явные таймауты,
# heartbeat в лог UI, ограниченные рестарты и текст диагноза.

GUI_START_TIMEOUT = 25.0    # сколько ждём первый state.json от окна
GUI_STEP_TIMEOUT = 30.0     # сколько ждём ответ на одно действие
GUI_RESET_TIMEOUT = 20.0    # C++ сам перезапускает карту через 5 с (gui.cpp)
GUI_HEARTBEAT_EVERY = 5.0   # раз в столько секунд пишем «ещё жду» в лог UI
MAX_GUI_RESTARTS = 3        # exe, который не стартует, не должен крутиться вечно


class GuiStartupError(RuntimeError):
    """The raylib GUI never came up (exited early or never answered over IPC)."""


#: Счётчик вызовов `make_ipc_dir` в этом процессе: pid+мс уникальны МЕЖДУ
#: процессами, но два запуска в одном процессе (тесты, повторное наблюдение без
#: перезапуска воркера) могут попасть в одну миллисекунду — и, если предыдущий
#: каталог уже убран за собой, получить тот же путь.
_IPC_SEQ = itertools.count()


def make_ipc_dir() -> Path:
    """Unique per-run IPC directory for actions.txt / state.json.

    A fixed `%TEMP%/colony_watch` was shared by every run: the UI kills only
    `watch_champion.py` (Windows does not kill the process tree), so a raylib
    window that outlived its driver kept eating actions.txt and writing
    state.json — the NEXT watch then talked to the orphan, and its own window
    looked dead. Per-run dir + cleanup makes that impossible.
    """
    root = Path(tempfile.gettempdir()) / "colony_watch"
    seq = next(_IPC_SEQ)
    stamp = f"{os.getpid()}_{int(time.time() * 1000)}" + (f"_s{seq}" if seq else "")
    for attempt in range(100):
        d = root / (stamp if attempt == 0 else f"{stamp}_{attempt}")
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue
    # крайне маловероятно: 100 каталогов за один запуск уже существуют
    d = root / f"{stamp}_fallback"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gui_exe_stale_sources(exe_path) -> "list[str]":
    """Sources newer than the built exe — a stale exe ignores new CLI flags.

    Наблюдение передаёт `--tax-to-debt`, `--curriculum` JSON и `--minimap-radius`;
    exe, собранный до этих правок, молча играет в другую игру (или не понимает
    протокол). Предупреждение печатается в лог UI до запуска окна.
    """
    exe = Path(exe_path)
    try:
        exe_mtime = exe.stat().st_mtime
    except OSError:
        return []
    newer: "list[str]" = []
    for pattern in ("src/*.cpp", "include/colony/*.h", "include/*.h",
                    "configs/*.json", "build_gui.bat"):
        for f in PROJECT_ROOT.glob(pattern):
            try:
                if f.stat().st_mtime > exe_mtime:
                    newer.append(f.relative_to(PROJECT_ROOT).as_posix())
            except OSError:
                continue
    return sorted(newer)


def state_stamp(state_file: Path):
    """(mtime_ns, size) of state.json or None — «свежесть» ответа окна."""
    try:
        st = Path(state_file).stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def wait_for_state(
    state_file: Path,
    since=None,
    timeout: "float | None" = None,
    heartbeat: "float | None" = None,
    on_wait=None,
    proc=None,
) -> "dict | None":
    """Wait for a state.json NEWER than `since` (None = any state at all).

    Lock-step: an action must be computed from the state that resulted from the
    PREVIOUS action. Without the freshness check the driver could answer with a
    stale state and send two actions per GUI step.

    `on_wait(seconds)` is called every `heartbeat` seconds — the UI log gets a
    visible «жду окно» instead of total silence. Returns None on timeout (or if
    `proc` died while waiting — the caller then reports the exit code).
    """
    # Константы читаем в теле (не в дефолтах): так их можно подкрутить и в
    # тестах, и при отладке медленного старта окна.
    timeout = GUI_STEP_TIMEOUT if timeout is None else float(timeout)
    heartbeat = GUI_HEARTBEAT_EVERY if heartbeat is None else float(heartbeat)
    deadline = time.monotonic() + timeout
    next_beat = time.monotonic() + heartbeat
    while True:
        stamp = state_stamp(state_file)
        if stamp is not None and (since is None or stamp != since):
            state = read_state(state_file)
            if state is not None:
                return state
        if proc is not None and proc.poll() is not None:
            return None
        now = time.monotonic()
        if now >= deadline:
            return None
        if on_wait is not None and now >= next_beat:
            next_beat = now + heartbeat
            on_wait(timeout - (deadline - now))
        time.sleep(0.02)


def describe_gui_failure(proc, exe_path, gui_log: "Path | None", ipc_dir: Path,
                         waited_for: str) -> str:
    """Human-readable diagnosis for «окно не поднялось»."""
    lines = [f"ERROR: GUI-окно не запустилось ({waited_for})."]
    rc = proc.poll() if proc is not None else None
    if rc is not None:
        hint = ""
        if rc < 0:
            hint = f" (signal {-rc})"
        elif rc >= 0x80000000 or rc in (0xC0000135, -1073741515):
            hint = " — 0xC0000135 = не найден DLL (raylib.dll рядом с exe?)"
        lines.append(f"  процесс завершился сразу: exit code {rc}{hint}")
    else:
        lines.append("  процесс жив, но не пишет state.json — окно могло не открыться "
                     "(проверьте ai_debug_gui.log в корне проекта)")
    lines.append(f"  exe: {exe_path}")
    lines.append(f"  IPC: {ipc_dir}")
    if gui_log is not None and Path(gui_log).exists():
        try:
            tail = Path(gui_log).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            tail = ""
        if tail:
            lines.append("  вывод GUI (последние строки):")
            lines.extend(f"    | {ln}" for ln in tail.splitlines()[-15:])
        else:
            lines.append("  вывод GUI пуст (exe не успел ничего напечатать)")
    stale = gui_exe_stale_sources(exe_path)
    if stale:
        lines.append(f"  ВНИМАНИЕ: exe старше исходников {stale[:6]}"
                     f"{' …' if len(stale) > 6 else ''} — пересоберите build_gui.bat")
    return "\n".join(lines)


def _terminate_process_tree(proc, *, windows: bool | None = None,
                            grace_seconds: float = 2.0) -> None:
    """Stop the GUI process and every child before removing its IPC directory.

    On Windows ``Popen.kill()`` only terminates the batch/cmd wrapper (the
    common ``.cmd -> python fake/launcher -> GUI`` case), leaving descendants to
    hold actions.txt/state.json open. ``taskkill /T`` follows the process tree.
    On POSIX the GUI is started in a fresh session, so signalling its process
    group also catches grandchildren.
    """
    if proc is None:
        return
    if windows is None:
        windows = os.name == "nt"

    pid = getattr(proc, "pid", None)
    if windows:
        if pid is not None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=max(3.0, grace_seconds + 1.0),
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError):
                # Fallback below still stops the wrapper if taskkill is missing
                # or the system refuses the tree-level request.
                pass
        if proc.poll() is None:
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
    else:
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                try:
                    proc.terminate()
                except (OSError, ProcessLookupError):
                    pass
        elif proc.poll() is None:
            try:
                proc.terminate()
            except (OSError, ProcessLookupError):
                pass

        try:
            proc.wait(timeout=max(0.0, grace_seconds))
        except subprocess.TimeoutExpired:
            pass
        # The group may outlive its leader (or ignore SIGTERM). Always send the
        # final group signal, even if proc.wait() already observed the leader.
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except (OSError, ProcessLookupError):
                        pass

    try:
        proc.wait(timeout=max(0.1, grace_seconds))
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=1.0)
        except (subprocess.SubprocessError, OSError):
            pass
    except (subprocess.SubprocessError, OSError):
        pass


def _configure_utf8_stdio(streams=None) -> None:
    """Prefer UTF-8 for console output (notably the Windows cp1251 console)."""
    if streams is None:
        streams = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    seen = set()
    for stream in streams:
        if stream is None or id(stream) in seen:
            continue
        seen.add(id(stream))
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            # pytest/IDE capture streams and some embedded consoles cannot be
            # reconfigured; file logs are explicitly opened as UTF-8 anyway.
            continue


def launch_visual_watch(
    model_dir: Path,
    exe_path: str,
    actions_file: Path,
    state_file: Path,
    seed: int,
    map_size: int,
    curriculum=None,  # CurriculumState | dict | None (None = explicitly unrestricted)
    reward_config_path: str | None = None,
    minimap_radius: int | None = None,
    tax_to_debt: bool = True,
    gui_log: "Path | None" = None,
) -> "subprocess.Popen":
    """Launch the GUI exe in headless-ai mode.

    `tax_to_debt=True` (по умолчанию) повторяет налоговую политику обучения:
    неоплаченный налог уходит в долг, календарь идёт. У GUI-окна для людей
    политика обратная (диалог налогов + «Нет» = конец игры), поэтому режим
    наблюдения сообщает его явно — иначе watched-модель играет в другую игру
    и «умирает» на 365-м дне там, где обучение уже не умирает
    (см. tests/cpp/gui_watch_check.cpp).

    `gui_log` перехватывает stdout/stderr окна: без него причина «не запустилось»
    (битый --curriculum JSON, отсутствующий DLL, ошибка в конфиге) уходила в
    никуда — консоль у watch-процесса скрыта (CREATE_NO_WINDOW из UI).
    """
    import json as _json
    import subprocess
    args = [
        exe_path,
        "--headless-ai",
        "--actions-file", str(actions_file),
        "--state-file", str(state_file),
        "--seed", str(seed),
        "--map-size", str(map_size),
    ]
    args.append("--tax-to-debt" if tax_to_debt else "--tax-dialog")
    # PR 1: the GUI env takes the same computed state over --curriculum JSON —
    # the flag is never skipped, so its action_mask can't silently re-enable
    # buildings (the watched policy would pick e.g. прииск again).
    if curriculum is None:
        args.extend(["--curriculum-all"])
    else:
        payload = curriculum.to_dict() if hasattr(curriculum, "to_dict") else curriculum
        args.extend(["--curriculum", _json.dumps(payload)])
    if reward_config_path:
        args.extend(["--reward-config", reward_config_path])
    if minimap_radius is not None:
        args.extend(["--minimap-radius", str(minimap_radius)])
    workdir = str(PROJECT_ROOT)
    kwargs: dict = {"cwd": workdir}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    if gui_log is not None:
        # line-buffered text file: строки ошибки видны в логе UI сразу. Ручку
        # закрываем у себя — дочерний процесс уже унаследовал дескриптор.
        fh = open(gui_log, "a", encoding="utf-8", errors="replace")
        try:
            kwargs["stdout"] = fh
            kwargs["stderr"] = subprocess.STDOUT
            return subprocess.Popen(args, **kwargs)
        finally:
            fh.close()
    return subprocess.Popen(args, **kwargs)



def run_visual_watch(
    *,
    exe_path: str,
    ipc_dir: Path,
    model_dir: Path,
    seed: int,
    map_size: int,
    curriculum=None,
    reward_config_path: "str | None" = None,
    minimap_radius: "int | None" = None,
    tax_to_debt: bool = True,
    normalizer=None,
    policy=None,
    is_hybrid: bool = False,
    is_cnn: bool = False,
    device=None,
    cur_mask=None,
    action_names: "list[str] | None" = None,
    speed: float = 1.0,
    episodes: int = 1,
    temperature: float = 0.0,
    emit_step=None,
    emit_log=None,
) -> int:
    """Drive the raylib GUI over the file IPC and feed it policy actions.

    Протокол (оба файла — в `ipc_dir`):
      ``actions.txt`` : драйвер пишет int-действие, окно читает и удаляет файл;
      ``state.json``  : окно пишет day/money/people/bases/reward/obs/
                        action_mask/minimap после каждого шага.

    Всё, что раньше выглядело как «галочка GUI-окно → ничего не происходит,
    в логе тишина», теперь имеет таймаут и диагноз: старт exe, первый
    state.json, ответ на каждое действие, авто-рестарт карты. Рестарты упавшего
    exe ограничены (``MAX_GUI_RESTARTS``), вывод окна пишется в
    ``ipc_dir/gui_output.log`` и показывается в логе UI вместе с кодом возврата.

    Returns 0 on a clean stop, 1 when the GUI never came up.
    """
    actions_file = ipc_dir / "actions.txt"
    state_file = ipc_dir / "state.json"
    gui_log = ipc_dir / "gui_output.log"
    names = list(action_names or [])

    def _log(msg: str, level: str = "info") -> None:
        # либо JSON для UI, либо текст в консоль — но не то и другое сразу
        # (иначе каждая строка наблюдения дублируется в панели лога UI).
        if emit_log is not None:
            emit_log(msg, level=level)
        else:
            print(msg, flush=True)

    stale = gui_exe_stale_sources(exe_path)
    if stale:
        _log(f"WARNING: GUI exe собран ДО правок в {', '.join(stale[:4])}"
             f"{' …' if len(stale) > 4 else ''} — пересоберите build_gui.bat, "
             f"иначе окно играет по старым правилам/протоколу.", level="warning")

    def _launch():
        proc = launch_visual_watch(
            model_dir=model_dir,
            exe_path=exe_path,
            actions_file=actions_file,
            state_file=state_file,
            seed=seed,
            map_size=map_size,
            curriculum=curriculum,
            reward_config_path=reward_config_path,
            minimap_radius=minimap_radius,
            tax_to_debt=tax_to_debt,
            gui_log=gui_log,
        )
        _log(f"GUI-окно запущено: pid={proc.pid}, exe={exe_path}")
        _log(f"IPC: {ipc_dir} ({actions_file.name} / {state_file.name}); "
             f"вывод окна → {gui_log}")
        return proc

    def _handshake(proc) -> dict:
        """Первый state.json от окна; старый exe толкаем действием 0 (DAY)."""
        state = wait_for_state(state_file, since=None, timeout=3.0, proc=proc)
        if state is not None:
            return state
        if proc.poll() is not None:
            raise GuiStartupError(describe_gui_failure(
                proc, exe_path, gui_log, ipc_dir, "exe завершился при старте"))
        write_action(actions_file, 0)  # 0 = DAY
        state = wait_for_state(
            state_file, since=None, timeout=GUI_START_TIMEOUT, proc=proc,
            on_wait=lambda s: _log(f"жду первое состояние от GUI-окна… {s:.0f} с"))
        if state is None:
            raise GuiStartupError(describe_gui_failure(
                proc, exe_path, gui_log, ipc_dir,
                f"нет state.json за {GUI_START_TIMEOUT:.0f} с"))
        return state

    def _minimap_tensor(minimap_data):
        """8×G×G тензор из плоского списка; None, если окно миникарту не прислало."""
        if not minimap_data:
            return None
        n_mm = len(minimap_data)
        n_ch = 8
        grid = int(round((n_mm / n_ch) ** 0.5))
        if grid * grid * n_ch != n_mm:
            _log(f"WARNING: миникарта из окна не раскладывается в 8×G×G "
                 f"({n_mm} чисел) — ветвь CNN останется без входа", level="warning")
            return None
        return torch.from_numpy(
            np.array(minimap_data, dtype=np.float32).reshape(1, n_ch, grid, grid)
        ).to(device)

    def _choose_action(state: dict) -> "tuple[int, str]":
        obs = state.get("obs") or []
        if not obs:
            return 0, "DAY"          # окно ещё не прислало наблюдение
        obs_np = np.array(obs, dtype=np.float32)
        if normalizer is not None:
            obs_np = normalizer.normalize(obs_np)
        mm_t = _minimap_tensor(state.get("minimap") or [])
        if (is_hybrid or is_cnn) and mm_t is None:
            grid = getattr(policy, "grid_size", None)
            _log(f"ERROR: модель {'hybrid' if is_hybrid else 'minimap'}, а окно не "
                 f"прислало миникарту (нужно 8×{grid}×{grid} в state.json). "
                 f"Пересоберите GUI (build_gui.bat): старый exe не пишет поле "
                 f"minimap и наблюдать такую модель нельзя.", level="error")
            return 0, "DAY"
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).to(device).reshape(1, -1)
            if is_hybrid:
                logits, _ = policy(obs_t, mm_t)
            elif is_cnn:
                logits, _ = policy(mm_t)
            else:
                logits, _ = policy(obs_t)
            # Маски — как в обучении (rl/ppo.py): -1e9, а не -inf (NaN-safe).
            mask = state.get("action_mask")
            if mask is not None:
                mask_t = torch.tensor(mask, dtype=torch.float32,
                                      device=device).reshape(1, -1)
                logits = logits.masked_fill(mask_t == 0, -1e9)
            # ...и курикулум-маска (exe мог собраться до --unlock-ids).
            if cur_mask is not None:
                cm_t = torch.tensor(cur_mask, dtype=torch.float32,
                                    device=device).reshape(1, -1)
                logits = logits.masked_fill(cm_t == 0, -1e9)

            probs = torch.softmax(logits, dim=-1).squeeze(0)
            if temperature > 0.0:
                sample_probs = torch.softmax(logits / max(1e-4, temperature), dim=-1).squeeze(0)
                action = int(torch.multinomial(sample_probs, 1).item())
            else:
                action = int(logits.argmax(dim=-1).item())
        name = names[action] if action < len(names) else str(action)
        top3_idx = torch.topk(probs, min(3, probs.numel())).indices.tolist()
        top3_info = ", ".join(f"{names[i] if i < len(names) else i}: {probs[i].item():.1%}" for i in top3_idx)
        return action, name, top3_info

    restarts = 0
    step_count = 0
    episode = 0
    total_reward = 0.0
    infer_errors = 0
    total_episodes = max(1, episodes)
    proc = None
    try:
        proc = _launch()
        state = _handshake(proc)
        _log(f"Visual watch running. Speed: {speed if speed > 0 else 1.0} steps/s. "
             f"Закройте окно, чтобы остановить.")

        while episode < total_episodes:
            if state.get("terminated"):
                episode += 1
                total_reward = 0.0
                _log(f"[Game Over at day {state.get('day')}] "
                     f"episode {episode}/{total_episodes}")
                if episode >= total_episodes:
                    break
                # C++ сам сбрасывает карту через 5 с (gui.cpp) и пишет свежий
                # state — раньше драйвер спал вслепую 6 с и чистил IPC-файлы.
                stamp = state_stamp(state_file)
                state = wait_for_state(
                    state_file, since=stamp, timeout=GUI_RESET_TIMEOUT, proc=proc,
                    on_wait=lambda s: _log(f"жду авто-рестарт карты в окне… {s:.0f} с"))
                if state is None:
                    if proc.poll() is None:
                        raise GuiStartupError(describe_gui_failure(
                            proc, exe_path, gui_log, ipc_dir,
                            f"окно не перезапустило карту за {GUI_RESET_TIMEOUT:.0f} с"))
                    if restarts >= MAX_GUI_RESTARTS:
                        raise GuiStartupError(describe_gui_failure(
                            proc, exe_path, gui_log, ipc_dir,
                            f"лимит рестартов ({MAX_GUI_RESTARTS}) исчерпан"))
                    restarts += 1
                    _log(f"[GUI process exited, restarting… {restarts}/"
                         f"{MAX_GUI_RESTARTS}]", level="warning")
                    _terminate_process_tree(proc)
                    proc = _launch()
                    state = _handshake(proc)
                continue

            step_count += 1
            day = state.get("day", "?")
            money = state.get("money", "?")
            bases = state.get("bases", "?")
            top3_info = ""
            try:
                action, action_name, top3_info = _choose_action(state)
            except Exception as e:
                infer_errors += 1
                action, action_name = 0, "DAY"
                if infer_errors <= 3 or infer_errors % 50 == 0:
                    _log(f"[WARN] ошибка инференса на шаге {step_count}: "
                         f"{type(e).__name__}: {e} → действие 0 (DAY)",
                         level="warning")
                if infer_errors == 3:
                    _log("ERROR: инференс падает на каждом шаге — наблюдение "
                         "бессмысленно (проверьте obs_version/нормализатор "
                         "модели и что окно присылает obs той же размерности).",
                         level="error")
            else:
                if step_count <= 10 or step_count % 50 == 0:
                    top_suffix = f" [{top3_info}]" if top3_info else ""
                    print(f"  step {step_count}  day {day}  money={money} "
                          f"bases={bases}  -> action {action} ({action_name}){top_suffix}",
                          flush=True)

            stamp = state_stamp(state_file)
            if not write_action(actions_file, action):
                _log("WARNING: окно не забирает actions.txt 2 с — оно зависло "
                     "или не в режиме --headless-ai", level="warning")
            state = wait_for_state(
                state_file, since=stamp, timeout=GUI_STEP_TIMEOUT, proc=proc,
                on_wait=lambda s: _log(f"жду ответ от GUI-окна… {s:.0f} с"))
            if state is None:
                if proc.poll() is None:
                    raise GuiStartupError(describe_gui_failure(
                        proc, exe_path, gui_log, ipc_dir,
                        f"окно не ответило на действие за {GUI_STEP_TIMEOUT:.0f} с"))
                if restarts >= MAX_GUI_RESTARTS:
                    raise GuiStartupError(describe_gui_failure(
                        proc, exe_path, gui_log, ipc_dir,
                        f"лимит рестартов ({MAX_GUI_RESTARTS}) исчерпан"))
                restarts += 1
                _log(f"[GUI process exited, restarting… {restarts}/"
                     f"{MAX_GUI_RESTARTS}]", level="warning")
                _terminate_process_tree(proc)
                proc = _launch()
                state = _handshake(proc)
                continue

            reward = float(state.get("reward", 0.0) or 0.0)
            total_reward += reward
            # Тот же JSONL-протокол, что в текстовом режиме: без него карточки
            # «День/Касса/Люди/Базы/Действие» и график вкладки Наблюдение пусты.
            if emit_step:
                emit_step(
                    step=step_count, day=state.get("day", "?"),
                    action=action_name, reward=reward,
                    total_reward=round(total_reward, 2),
                    people=state.get("people", 0), bases=state.get("bases", 0),
                    money=state.get("money", 0),
                )
            if speed > 0:
                time.sleep(1.0 / speed)

    except GuiStartupError as e:
        _log(str(e), level="error")
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        _terminate_process_tree(proc)
        # IPC-каталог свой у каждого запуска — удаляем целиком (actions.txt,
        # state.json, reward_config.json, gui_output.log).
        shutil.rmtree(ipc_dir, ignore_errors=True)
        _log("Visual watch stopped.")
    return 0


def main():
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Watch champion model play")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model directory (contains best_model.pt)")
    parser.add_argument("--model-file", type=str, default="best_model.pt",
                        help="Model filename (default: best_model.pt)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Steps per second (0=no delay, 1=1 step/sec)")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Map seed. Default -1 = random map on every "
                             "launch (the seed is printed so an interesting "
                             "map can be re-watched with --seed N)")
    parser.add_argument("--map-size", type=int, default=280,
                        help="Map size; MUST match the value the model was "
                             "trained with (training default is 280)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--minimap-radius", type=int, default=None,
                        help="Minimap radius for minimap/hybrid models. Default: "
                             "taken from the checkpoint's grid_size")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Also write output to this file (for UI capture)")
    parser.add_argument("--step-log", type=str, default=None,
                        help="C++ step-log file: per-step reward breakdown + state (err/tax/build/"
                             "div/prox/nov/mile/surv/idle/...) for debugging model behavior")
    parser.add_argument("--curriculum-stage", type=int, default=None,
                        help="Override curriculum stage (0=all buildings, 1-3). "
                             "If not set, reads from best_model.meta.json")
    parser.add_argument("--unlock-ids", type=str, default=None,
                        help="Comma-separated building ids from the «Курикулум» tab "
                             "(manual set). If not set, reads unlock_ids from the "
                             "model meta. Pass \"\" to explicitly allow all buildings.")
    parser.add_argument("--curriculum-resources", type=str, default=None,
                        help="Resource priority set (CSV) from the «Курикулум» tab. "
                             "If not set, reads curriculum_resources from the model "
                             "meta. Pass \"\" to explicitly use all resources.")
    parser.add_argument("--obs-version", type=int, default=None, choices=[0, 1, 2],
                        help="Obs layout: 0 = 248-dim, 1 = 289-dim frame, "
                             "2 = 299-dim + resource directions. Default: from the "
                             "checkpoint meta (watch in the layout it was trained "
                             "with), else the current default (2). An explicit value "
                             "wins; a mismatch with the checkpoint still errors out.")
    parser.add_argument("--visual", action="store_true",
                        help="Open visual GUI window (requires sakhalin_colony_gui.exe)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=deterministic argmax, >0=stochastic sampling, e.g. 0.8)")
    parser.add_argument("--sample", action="store_true",
                        help="Enable stochastic action sampling (default temperature 0.8)")
    parser.add_argument("--allow-stale-pyd", action="store_true",
                        help="Debugging only: run even if the colony_cpp binary is stale "
                             "(same as COLONY_ALLOW_STALE_PYD=1). Expect wrong behaviour.")
    args = parser.parse_args()

    # PR 3: fail fast on a stale colony_cpp binary, before loading the model.
    require_colony(allow_stale=args.allow_stale_pyd)

    if args.log_file:
        # "a", not "w": the GUI already opened this very file and handed it to
        # us as stdout. Truncating it here threw away lines and left the two
        # writers fighting over offsets, so the crash reason never survived to
        # the UI panel («причина — в строках выше» pointed at nothing).
        log_f = open(args.log_file, "a", encoding="utf-8")
        _dup = not _same_file(sys.__stdout__, Path(args.log_file))
        sys.stdout = TeeWriter(sys.__stdout__, log_f, duplicate=_dup)
        import json as _json
        def emit_step(**kw):
            print(_json.dumps({"type": "step", **kw}, ensure_ascii=False))
        def emit_log(msg, level="info"):
            print(_json.dumps({"type": "log", "level": level, "message": msg}, ensure_ascii=False))
        def emit_done(msg):
            print(_json.dumps({"type": "done", "message": msg}, ensure_ascii=False))
    else:
        emit_step = None
        emit_log = None
        emit_done = None

    def say(msg: str, level: str = "info") -> None:
        """Одна строка — ровно один раз.

        С `--log-file` лог читает UI (`_poll_watch`): он парсит JSON-строки и
        отдельно показывает обычный текст, поэтому `print` + `emit_log` одной и
        той же фразы давали в панели лога ДВА одинаковых сообщения.
        """
        if emit_log is not None:
            emit_log(msg, level=level)
        else:
            print(msg, flush=True)

    # Randomize the map unless an explicit seed was requested. Same seed =
    # same island (Earth(seed) generates terrain/lakes/deposits), and with a
    # fixed default seed the colony always replayed ONE map — sometimes an
    # unwinnable one (no lake reachable from the center => no water => no food).
    if args.seed < 0:
        import random as _rnd
        args.seed = _rnd.randint(1, 999_999_999)
        _msg = (f"Random map seed: {args.seed} "
                f"(re-watch this exact map with --seed {args.seed})")
        say(_msg)

    model_dir = Path(args.model_dir).expanduser()
    model_path = resolve_model_file(model_dir, args.model_file)
    norm_path = model_dir / "normalization.json"
    if not norm_path.exists():
        norm_path = model_path.with_suffix(".norm.json")
    if not norm_path.exists():
        # Look up the matching checkpoint norm by total_timesteps from meta.json
        for meta_name in ("best_model.meta.json", "meta.json"):
            meta_path = model_dir / meta_name
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ts = meta.get("total_timesteps") or meta.get("steps")
                    if ts:
                        candidate = model_dir / f"checkpoint_{ts}_steps.norm.json"
                        if candidate.exists():
                            norm_path = candidate
                            break
                except (json.JSONDecodeError, OSError):
                    pass
    # Fallback: try final_model.norm.json or latest checkpoint norm
    if not norm_path.exists():
        final_norm = model_dir / "final_model.norm.json"
        if final_norm.exists():
            norm_path = final_norm
        else:
            # числовой порядок: лексикографически 999999 «новее» 1000000
            ckpt_norm = latest_checkpoint(model_dir, "checkpoint_*_steps.norm.json",
                                          loose=False)
            if ckpt_norm is not None:
                norm_path = ckpt_norm

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        print(f"  Looked in: {model_dir}")
        print(f"  Files: {[f.name for f in model_dir.iterdir() if f.is_file()]}")
        sys.exit(1)

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Loading policy from {model_path} on {dev}")
    policy = _load_policy(model_path, dev)

    is_cnn = hasattr(policy, "cnn")
    is_hybrid = hasattr(policy, "flat_proj") and hasattr(policy, "cnn")
    if is_hybrid:
        print(f"Model: hybrid (flat + minimap CNN)")
    elif is_cnn:
        print(f"Model: CNN minimap")

    print(f"Creating env (map_size={args.map_size})")
    # Read reward config from model's meta.json to match training parameters
    reward_cfg = None
    tax_to_debt = True
    meta = {}
    meta_paths = [model_path.with_suffix(".meta.json"),
                  model_dir / "best_model.meta.json", model_dir / "meta.json"]
    for meta_path in meta_paths:
        if not meta_path.exists():
            continue
        try:
            import json as _json
            loaded = _json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                continue
            for key, value in loaded.items():
                if key not in meta:
                    meta[key] = value
            cfg_meta = loaded.get("config", {})
            if reward_cfg is None and isinstance(cfg_meta, dict):
                reward_cfg = cfg_meta.get("reward")
            if "tax_to_debt" in loaded:
                tax_to_debt = bool(loaded["tax_to_debt"])
            elif isinstance(cfg_meta, dict) and "tax_to_debt" in cfg_meta:
                tax_to_debt = bool(cfg_meta["tax_to_debt"])
            if reward_cfg:
                print(f"Loaded reward config from {meta_path.name}")
        except (Exception,):
            pass

    # Curriculum: the watched env MUST have the same allowed-buildings set as
    # training. Previously the manual set from the «Курикулум» tab was lost here
    # and stage 0 unlocked every building — the champion then built Goldmine
    # (прииск) even though only WaterChannel was allowed during training.
    from rl.curriculum import (
        CURRENT_OBS_VERSION,
        obs_size_for_version,
        resolve_curriculum,
        resolve_obs_version,
        resolve_state,
        stored_obs_version,
    )

    # Версия obs: явный --obs-version → раскладка чекпойнта → текущий дефолт.
    # Без этого старые модели (289) не смотрелись бы на новом дефолте (299):
    # UI не передаёт --obs-version, значит версия обязана следовать за моделью.
    # `meta` was merged with the checkpoint sidecar taking priority above.
    stored_v = stored_obs_version(None, meta)
    obs_version = resolve_obs_version(args.obs_version, None, meta)
    _vsrc = ("--obs-version" if args.obs_version is not None
             else "meta модели" if stored_v is not None
             else f"дефолт {CURRENT_OBS_VERSION}")
    _vmsg = (f"Obs layout: v{obs_version} "
             f"({obs_size_for_version(obs_version)} dims, из {_vsrc})")
    say(_vmsg)

    resolved = resolve_curriculum(
        None, meta=meta,
        curriculum_stage=args.curriculum_stage,
        unlock_ids=args.unlock_ids,
        resources=args.curriculum_resources,
    )
    # PR 1: the env takes ONE computed state; the resolved dict stays for
    # logging and the GUI safety-net mask.
    st = resolve_state(
        None, meta=meta,
        curriculum_stage=args.curriculum_stage,
        unlock_ids=args.unlock_ids,
        resources=args.curriculum_resources,
        obs_version=obs_version,
    )
    stage = int(resolved["curriculum_stage"])
    manual_csv = str(resolved["unlock_ids"])
    n_allowed = len(resolved["allowed"])

    env = CppColonyEnv(
        map_size=args.map_size,
        reward_config=reward_cfg,
        curriculum=st,
        tax_to_debt=tax_to_debt,
    )
    # PR 5: obs-layout compatibility (same rule as eval: the version is
    # explicit, a stored mismatch errors out instead of misaligning).
    from rl.curriculum import (
        check_obs_version_compat,
        check_policy_obs_compat,
        stored_obs_version,
    )
    check_obs_version_compat(
        stored_v, st.obs_version, ckpt_path=str(model_path),
    )
    _flat_dim = getattr(policy, "obs_size", None)
    if _flat_dim is not None:  # CNN-only policies have no flat input
        check_policy_obs_compat(
            int(_flat_dim), int(env.observation_space.shape[0]),
            ckpt_path=str(model_path),
        )
    # Match the env's minimap grid to the policy's (see train_ui/evaluator.py).
    # Without this, watching a hybrid/minimap model trained with
    # minimap_radius != 14 dies with a shape mismatch on the first step.
    resolved_minimap_radius = args.minimap_radius
    if resolved_minimap_radius is None and hasattr(policy, "grid_size"):
        resolved_minimap_radius = int(policy.grid_size) // 2

    if args.minimap_radius is not None:
        env.cpp_env.set_minimap_radius(int(args.minimap_radius))
        print(f"Minimap radius: {args.minimap_radius} (grid {2*args.minimap_radius+1})")
    elif hasattr(policy, "grid_size"):
        r = int(policy.grid_size) // 2
        env.cpp_env.set_minimap_radius(r)
        print(f"Minimap radius: {r} from model grid_size={policy.grid_size}")
    if norm_path.exists():
        env.normalizer.load(str(norm_path))
        env.normalizer.set_update(False)
        print(f"Loaded normalization from {norm_path.name} (obs_size={env.normalizer._obs_size})")
        if env.normalizer._obs_size != env.observation_space.shape[0]:
            print(f"WARNING: normalizer obs_size={env.normalizer._obs_size} "
                  f"!= env obs_size={env.observation_space.shape[0]}, normalization may be wrong")
    else:
        print(f"WARNING: no normalization found in {model_dir}, using raw observations")
        print(f"  Available files: {[f.name for f in model_dir.iterdir() if f.suffix in ('.json', '.pt')]}")

    # (the curriculum state was already applied at construction above)
    stored = read_curriculum_from_meta(model_dir)
    if stored["curriculum_stage"] is not None or stored["unlock_ids"] is not None:
        print(f"Curriculum (from model meta): stage={stage}"
              f"{f', manual={manual_csv}' if manual_csv else ''}"
              f" → {n_allowed} buildings allowed")
    else:
        print("Curriculum: not set (all buildings)"
              if not (stage or manual_csv) else
              f"Curriculum (CLI): stage={stage}"
              f"{f', manual={manual_csv}' if manual_csv else ''}"
              f" → {n_allowed} buildings allowed")

    _tax_policy = "долг (как при обучении)" if env.cpp_env.tax_to_debt() else "диалог"
    _taxmsg = f"Tax policy: {_tax_policy}"
    say(_taxmsg)

    action_names = env._action_names
    cur_mask = curriculum_action_mask(resolved["allowed"], action_names)
    if cur_mask is not None:
        locked = sum(1 for v in cur_mask[2:] if v == 0.0)
        print(f"Action mask: {locked} locked build action(s) forced off")

    if args.step_log:
        env.set_step_log(args.step_log)
        print(f"Step log (per-step reward breakdown) -> {args.step_log}")

    # Sampling temperature — ОБЯЗАН быть вычислен ДО headless-цикла ниже.
    # Здесь он стоял после цикла (перед `if args.visual:`), а Python считает имя
    # локальным на всю функцию: любой прогон без `--visual` (режим по умолчанию,
    # команда из README) падал с UnboundLocalError на первом шаге выбора
    # действия, а `--sample`/`--temperature` были недостижимы.
    watch_temp = args.temperature if args.temperature > 0.0 else (0.8 if args.sample else 0.0)
    # То же правило для предупреждения о карте: печатать его после прогона, как
    # раньше, — значит сообщать о несовпадении, когда эпизод уже отыгран.
    stored_map_size = meta.get("map_size")
    if stored_map_size and int(stored_map_size) != args.map_size:
        say(f"WARNING: наблюдение запущено с картой {args.map_size}, а модель обучалась на карте {stored_map_size}! Рекомендуется запускать с --map-size {stored_map_size}.", level="warning")

    if not args.visual:  # headless text mode (visual mode returns below)
        for ep in range(args.episodes):
            print(f"\n{'=' * 70}")
            print(f"EPISODE {ep + 1}/{args.episodes}")
            print(f"{'=' * 70}")

            obs, _info = env.reset(seed=args.seed + ep)
            total_reward = 0.0
            step = 0

            for step in range(1, args.max_steps + 1):
                with torch.no_grad():
                    if is_hybrid:
                        # flat obs is already normalized by env.normalizer
                        flat_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev)
                        flat_t = flat_t.reshape(1, -1)
                        mm = env.cpp_env.minimap()
                        mm_t = torch.as_tensor(
                            np.ascontiguousarray(mm, dtype=np.float32)
                        ).to(dev).reshape(1, *np.asarray(mm).shape)
                        logits, _ = policy(flat_t, mm_t)
                    elif is_cnn:
                        mm = env.cpp_env.minimap()
                        mm_t = torch.as_tensor(
                            np.ascontiguousarray(mm, dtype=np.float32)
                        ).to(dev).reshape(1, *np.asarray(mm).shape)
                        logits, _ = policy(mm_t)
                    else:
                        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev)
                        obs_t = obs_t.reshape(1, -1)
                        logits, _ = policy(obs_t)

                    # Action masking — same as training/eval. Without it the
                    # watched policy may pick a building the curriculum locked
                    # (it would only bounce off the env with an error penalty).
                    if hasattr(env, "action_mask"):
                        try:
                            mask = np.asarray(env.action_mask(), dtype=np.float32)
                            mask_t = torch.as_tensor(mask, device=dev).reshape(1, -1)
                            # -1e9 like training (rl/ppo.py), not -inf: NaN-safe.
                            logits = logits.masked_fill(mask_t == 0, -1e9)
                        except Exception:
                            pass
                    if watch_temp > 0.0:
                        sample_probs = torch.softmax(logits / max(1e-4, watch_temp), dim=-1).squeeze(0)
                        action = int(torch.multinomial(sample_probs, 1).item())
                    else:
                        action = int(logits.argmax(dim=-1).item())

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                action_name = info.get("action_name", str(action))
                days = info.get("days", 0)
                people = info.get("people", 0)
                bases = info.get("bases", 0)
                money = info.get("money", 0)

                if emit_step:
                    emit_step(
                        step=step, day=days, action=action_name,
                        reward=round(reward, 4), total_reward=round(total_reward, 2),
                        people=people, bases=bases, money=money,
                    )
                else:
                    print(
                        f"Step {step:5d} | Day {days:5d} | "
                        f"{action_name:<16s} | "
                        f"R={reward:+8.2f} | "
                        f"TotR={total_reward:+10.1f} | "
                        f"Pop={people:3d} | Bases={bases:2d} | "
                        f"Money={money:10d}"
                    )

                if terminated or truncated:
                    reason = "TERMINATED" if terminated else "TRUNCATED"
                    msg = f"Episode {reason} at step {step}, day {days}, reward={total_reward:.1f}"
                    if emit_log:
                        emit_log(msg)
                    else:
                        print(f"\n>>> {msg}")
                    break

                if args.speed > 0:
                    time.sleep(1.0 / args.speed)

            else:
                msg = f"Max steps ({args.max_steps}) reached"
                if emit_log:
                    emit_log(msg)
                else:
                    print(f"\n>>> {msg}")

    if args.visual:
        exe = find_gui_exe()
        if exe is None:
            msg = (f"ERROR: GUI exe not found. Searched: "
                   f"{', '.join(str(PROJECT_ROOT / d / n) for d in GUI_EXE_DIRS[:3] for n in GUI_EXE_NAMES[:2])} ... "
                   f"Build it with build_gui.bat (requires raylib in raylib/), "
                   f"or set COLONY_GUI_EXE=/path/to/gui.exe")
            say(msg, level="error")
            env.close()
            sys.exit(1)
        exe_path = str(exe)
        print(f"GUI exe: {exe_path}")

        # Свой IPC-каталог на каждый запуск: общий %TEMP%/colony_watch делили
        # все прогоны, а окно переживает своего драйвера (UI убивает только
        # watch_champion.py) — сирота съедала действия следующего наблюдения.
        ipc_dir = make_ipc_dir()
        reward_cfg_path = None
        if reward_cfg:
            import json as _json
            rc_file = ipc_dir / "reward_config.json"
            rc_file.write_text(_json.dumps(reward_cfg), encoding="utf-8")
            reward_cfg_path = str(rc_file)

        print(f"Launching visual watch: {exe_path}")
        rc = run_visual_watch(
            exe_path=exe_path,
            ipc_dir=ipc_dir,
            model_dir=model_dir,
            seed=args.seed,
            map_size=args.map_size,
            curriculum=st,
            reward_config_path=reward_cfg_path,
            minimap_radius=resolved_minimap_radius,
            tax_to_debt=tax_to_debt,
            normalizer=env.normalizer,
            policy=policy,
            is_hybrid=is_hybrid,
            is_cnn=is_cnn,
            device=dev,
            cur_mask=cur_mask,
            action_names=action_names,
            speed=args.speed,
            episodes=args.episodes,
            temperature=watch_temp,
            emit_step=emit_step,
            emit_log=emit_log,
        )
        env.close()
        if rc != 0:
            sys.exit(rc)
        return

    env.close()
    final_msg = f"Done. {args.episodes} episode(s) completed."
    if emit_done:
        emit_done(final_msg)
    else:
        print(f"\n{final_msg}")


def _main_guarded() -> int:
    """Run main() and make sure the failure REASON reaches the log file.

    Без этого любое исключение после «Loading policy …» (несовпадение раскладки
    obs, битый чекпойнт, отсутствующий normalization.json, ошибка формы в
    policy) улетало в stderr интерпретатора и терялось, а UI показывал только
    «Наблюдение завершилось с кодом 1 (причина — в строках выше)» — при том что
    выше ничего не было. Теперь traceback пишется в тот же лог, и последняя
    строка исключения дублируется как JSON-сообщение уровня error, чтобы UI
    показал её в панели красным.
    """
    import traceback

    try:
        main()
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    except BaseException as e:  # noqa: BLE001 — диагностика важнее типа
        tb = traceback.format_exc()
        try:
            print(tb, flush=True)
            import json as _json
            msg = f"{type(e).__name__}: {e}"
            print(_json.dumps({"type": "error", "level": "error",
                               "message": f"Наблюдение упало — {msg}"},
                              ensure_ascii=False), flush=True)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(_main_guarded())
