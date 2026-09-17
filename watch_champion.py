#!/usr/bin/env python3
"""Watch a trained champion model play the colony game in real time.

Usage:
  python watch_champion.py --model-dir ~/colony_runs/models/run_003
  python watch_champion.py --model-dir ~/colony_runs/models/run_003 --speed 5
  python watch_champion.py --model-dir ~/colony_runs/models/run_003 --episodes 3
"""
import argparse
import sys
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


class TeeWriter:
    """Write to both stdout and a file (for UI log capture)."""

    def __init__(self, stdout, file):
        self.stdout = stdout
        self.file = file

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
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


def write_action(action_file: Path, action: int) -> None:
    """Write action int to the IPC file.

    Waits for C++ to delete the file (consumed previous action) before writing.
    This prevents overwriting an unread action.
    """
    import time as _time
    deadline = _time.monotonic() + 2.0
    while action_file.exists() and _time.monotonic() < deadline:
        _time.sleep(0.005)
    action_file.write_text(str(action), encoding="utf-8")


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
) -> "subprocess.Popen":
    """Launch the GUI exe in headless-ai mode."""
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
    return subprocess.Popen(args, cwd=workdir)


def main():
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
    parser.add_argument("--obs-version", type=int, default=2, choices=[0, 1, 2],
                        help="Obs layout: 0 = 248-dim (old checkpoints), 1 = 289-dim "
                             "frame, 2 = 299-dim + resource directions (default).")
    parser.add_argument("--visual", action="store_true",
                        help="Open visual GUI window (requires sakhalin_colony_gui.exe)")
    parser.add_argument("--allow-stale-pyd", action="store_true",
                        help="Debugging only: run even if the colony_cpp binary is stale "
                             "(same as COLONY_ALLOW_STALE_PYD=1). Expect wrong behaviour.")
    args = parser.parse_args()

    # PR 3: fail fast on a stale colony_cpp binary, before loading the model.
    require_colony(allow_stale=args.allow_stale_pyd)

    if args.log_file:
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = TeeWriter(sys.__stdout__, log_f)
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

    # Randomize the map unless an explicit seed was requested. Same seed =
    # same island (Earth(seed) generates terrain/lakes/deposits), and with a
    # fixed default seed the colony always replayed ONE map — sometimes an
    # unwinnable one (no lake reachable from the center => no water => no food).
    if args.seed < 0:
        import random as _rnd
        args.seed = _rnd.randint(1, 999_999_999)
        _msg = (f"Random map seed: {args.seed} "
                f"(re-watch this exact map with --seed {args.seed})")
        print(_msg)
        if emit_log:
            emit_log(_msg)

    model_dir = Path(args.model_dir).expanduser()
    model_path = model_dir / args.model_file
    # Fallback: if requested model doesn't exist, try final_model.pt then checkpoint_*.pt
    if not model_path.exists():
        fallback = model_dir / "final_model.pt"
        if fallback.exists():
            model_path = fallback
        else:
            checkpoints = sorted(model_dir.glob("checkpoint_*_steps.pt"))
            if checkpoints:
                model_path = checkpoints[-1]
    norm_path = model_dir / "normalization.json"
    if not norm_path.exists():
        norm_path = model_path.with_suffix(".norm.json")
    if not norm_path.exists():
        # Look up the matching checkpoint norm by total_timesteps from meta.json
        for meta_name in ("best_model.meta.json", "meta.json"):
            meta_path = model_dir / meta_name
            if meta_path.exists():
                import json as _json
                try:
                    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
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
            ckpt_norms = sorted(model_dir.glob("checkpoint_*_steps.norm.json"))
            if ckpt_norms:
                norm_path = ckpt_norms[-1]

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
    for meta_name in ("best_model.meta.json", "meta.json"):
        meta_path = model_dir / meta_name
        if meta_path.exists():
            try:
                import json as _json
                meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                reward_cfg = meta.get("config", {}).get("reward")
                if reward_cfg:
                    print(f"Loaded reward config from {meta_name}")
                    break
            except (Exception,):
                pass

    # Curriculum: the watched env MUST have the same allowed-buildings set as
    # training. Previously the manual set from the «Курикулум» tab was lost here
    # and stage 0 unlocked every building — the champion then built Goldmine
    # (прииск) even though only WaterChannel was allowed during training.
    from rl.curriculum import resolve_curriculum, resolve_state

    resolved = resolve_curriculum(
        model_dir,
        curriculum_stage=args.curriculum_stage,
        unlock_ids=args.unlock_ids,
        resources=args.curriculum_resources,
    )
    # PR 1: the env takes ONE computed state; the resolved dict stays for
    # logging and the GUI safety-net mask.
    st = resolve_state(
        model_dir,
        curriculum_stage=args.curriculum_stage,
        unlock_ids=args.unlock_ids,
        resources=args.curriculum_resources,
        obs_version=args.obs_version,
    )
    stage = int(resolved["curriculum_stage"])
    manual_csv = str(resolved["unlock_ids"])
    n_allowed = len(resolved["allowed"])

    env = CppColonyEnv(
        map_size=args.map_size,
        reward_config=reward_cfg,
        curriculum=st,
    )
    # PR 5: obs-layout compatibility (same rule as eval: the version is
    # explicit, a stored mismatch errors out instead of misaligning).
    from rl.curriculum import (
        check_obs_version_compat,
        check_policy_obs_compat,
        stored_obs_version,
    )
    check_obs_version_compat(
        stored_obs_version(model_dir), st.obs_version,
        ckpt_path=str(model_path),
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

    action_names = env._action_names
    cur_mask = curriculum_action_mask(resolved["allowed"], action_names)
    if cur_mask is not None:
        locked = sum(1 for v in cur_mask[2:] if v == 0.0)
        print(f"Action mask: {locked} locked build action(s) forced off")

    if args.step_log:
        env.set_step_log(args.step_log)
        print(f"Step log (per-step reward breakdown) -> {args.step_log}")

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
            print(msg)
            if emit_log:
                emit_log(msg, level="error")
            sys.exit(1)
        exe_path = str(exe)
        print(f"GUI exe: {exe_path}")

        import tempfile
        tmp_dir = Path(tempfile.gettempdir()) / "colony_watch"
        tmp_dir.mkdir(exist_ok=True)
        actions_file = tmp_dir / "actions.txt"
        state_file = tmp_dir / "state.json"

        # Write reward config to temp file for GUI
        reward_cfg_path = None
        if reward_cfg:
            import json as _json
            rc_file = tmp_dir / "reward_config.json"
            rc_file.write_text(_json.dumps(reward_cfg), encoding="utf-8")
            reward_cfg_path = str(rc_file)

        print(f"Launching visual watch: {exe_path}")
        proc = launch_visual_watch(
            model_dir=model_dir,
            exe_path=exe_path,
            actions_file=actions_file,
            state_file=state_file,
            seed=args.seed,
            map_size=args.map_size,
            curriculum=st,
            reward_config_path=reward_cfg_path,
            minimap_radius=resolved_minimap_radius,
        )

        for f in (actions_file, state_file):
            if f.exists():
                f.unlink()

        write_action(actions_file, 0)  # 0 = DAY

        speed = args.speed if args.speed > 0 else 1.0
        print(f"Visual watch running. Speed: {speed} steps/s. Close GUI window to stop.")

        def _restart_gui():
            """Kill old process, clean IPC, launch fresh, seed action."""
            if proc.poll() is None:
                proc.kill()
            for f in (actions_file, state_file):
                if f.exists():
                    f.unlink()
            p = launch_visual_watch(
                model_dir=model_dir,
                exe_path=exe_path,
                actions_file=actions_file,
                state_file=state_file,
                seed=args.seed,
                map_size=args.map_size,
                curriculum=st,
                reward_config_path=reward_cfg_path,
                minimap_radius=resolved_minimap_radius,
            )
            write_action(actions_file, 0)
            return p

        try:
            step_count = 0
            episode = 0
            total_reward = 0.0
            total_episodes = max(1, args.episodes)
            while episode < total_episodes:
                # Restart if process died
                if proc.poll() is not None:
                    print("  [GUI process exited, restarting...]")
                    proc = _restart_gui()
                    time.sleep(0.5)
                    continue

                state = read_state(state_file)
                if state is None:
                    time.sleep(0.05)
                    continue

                if state.get("terminated"):
                    episode += 1
                    total_reward = 0.0
                    print(f"  [Game Over at day {state.get('day')}] "
                          f"episode {episode}/{total_episodes}")
                    if episode >= total_episodes:
                        break
                    # Wait for C++ auto-reset (5s in gui.cpp) + margin
                    print("  [Waiting for C++ auto-reset...]")
                    time.sleep(6.0)
                    if proc.poll() is not None:
                        # Process died — restart it
                        proc = _restart_gui()
                        time.sleep(0.5)
                    else:
                        # Process alive — GUI auto-reset, clear IPC and send fresh action
                        for f in (actions_file, state_file):
                            if f.exists():
                                f.unlink()
                        write_action(actions_file, 0)  # 0 = DAY
                        time.sleep(0.1)
                    continue

                step_count += 1
                day = state.get("day", "?")
                money = state.get("money", "?")
                bases = state.get("bases", "?")
                if step_count <= 10 or step_count % 50 == 0:
                    print(f"  step {step_count}  day {day}  "
                          f"money={money} bases={bases} "
                          f"action_history={state.get('action')}")

                obs = state.get("obs", [])
                minimap_data = state.get("minimap", [])
                action_name = ""
                if obs:
                    try:
                        obs_arr = np.array(obs, dtype=np.float32)
                        obs_arr = env.normalizer.normalize(obs_arr)
                        # Minimap grid size inferred from data length instead
                        # of hardcoded 29x29 (radius may differ from 14).
                        mm_t = None
                        if minimap_data:
                            n_mm = len(minimap_data)
                            n_ch = 8
                            grid = int(round((n_mm / n_ch) ** 0.5))
                            if grid * grid * n_ch == n_mm:
                                mm_t = torch.from_numpy(
                                    np.array(minimap_data, dtype=np.float32)
                                    .reshape(1, n_ch, grid, grid)).to(dev)
                        with torch.no_grad():
                            if is_hybrid and mm_t is not None:
                                obs_t = torch.from_numpy(obs_arr).to(dev).reshape(1, -1)
                                logits, _ = policy(obs_t, mm_t)
                            elif is_cnn and mm_t is not None:
                                logits, _ = policy(mm_t)
                            else:
                                obs_t = torch.from_numpy(obs_arr).to(dev).reshape(1, -1)
                                logits, _ = policy(obs_t)
                            # Apply action masking from GUI env (match training behavior)
                            mask = state.get("action_mask", None)
                            if mask is not None:
                                mask_t = torch.tensor(mask, dtype=torch.float32, device=dev).reshape(1, -1)
                                # -1e9 like training (rl/ppo.py), not -inf: NaN-safe.
                                logits = logits.masked_fill(mask_t == 0, -1e9)
                            # ...and the curriculum mask (GUI exe may predate
                            # --unlock-ids support and unlock everything)
                            if cur_mask is not None:
                                cm_t = torch.tensor(cur_mask, dtype=torch.float32, device=dev).reshape(1, -1)
                                # -1e9 like training (rl/ppo.py), not -inf: NaN-safe.
                                logits = logits.masked_fill(cm_t == 0, -1e9)
                            action = int(logits.argmax(dim=-1).item())
                        action_name = action_names[action] if action < len(action_names) else str(action)
                        if step_count <= 10 or step_count % 50 == 0:
                            print(f"  -> sending action {action} ({action_name})")
                        write_action(actions_file, action)
                    except Exception as e:
                        print(f"  [WARN] obs/inference error on step {step_count}: {e}")
                        if step_count <= 10 or step_count % 50 == 0:
                            print(f"  -> fallback action 0 (DAY)")
                        write_action(actions_file, 0)
                else:
                    write_action(actions_file, 0)

                reward = state.get("reward", 0.0)
                total_reward += reward
                # Feed the UI state panel (same JSONL protocol as text mode);
                # without this the "Текущее состояние среды" panel stays empty
                # in visual mode.
                if emit_step:
                    emit_step(
                        step=step_count, day=day, action=action_name,
                        reward=reward, total_reward=round(total_reward, 2),
                        people=state.get("people", 0), bases=bases, money=money,
                    )

                if args.speed > 0:
                    time.sleep(1.0 / args.speed)

        except KeyboardInterrupt:
            pass
        finally:
            if proc.poll() is None:
                proc.kill()
            for f in (actions_file, state_file):
                if f.exists():
                    f.unlink()
            if reward_cfg_path and Path(reward_cfg_path).exists():
                Path(reward_cfg_path).unlink()
            env.close()
            print("Visual watch stopped.")
        return

    env.close()
    final_msg = f"Done. {args.episodes} episode(s) completed."
    if emit_done:
        emit_done(final_msg)
    else:
        print(f"\n{final_msg}")


if __name__ == "__main__":
    main()
