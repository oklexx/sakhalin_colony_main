from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))


def _probe_env_dims(map_size: int = 280):
    """Get (obs_size, n_actions) from a fresh colony env."""
    from cpp_env import CppColonyEnv

    env = CppColonyEnv(map_size=map_size)
    try:
        return int(env.observation_space.shape[0]), int(env.action_space.n)
    finally:
        env.close()


def _load_policy(model_path: Path, device, mode: str = "auto", minimap_radius: int = 14):
    """Load a trained policy from a checkpoint.

    mode: "auto" (detect MLP vs CNN from the state_dict), "flat", or "minimap".
    """
    import torch
    from rl.actor_critic import ActorCritic
    from rl.actor_critic_cnn import ActorCriticCNN

    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dict")
    model_state = ckpt.get("model_state")
    if model_state is None:
        raise ValueError("checkpoint missing 'model_state'")

    clean = {}
    for k, v in model_state.items():
        ck = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        clean[ck] = v
    model_state = clean

    has_cnn = any(k.startswith("cnn.") or k.startswith("conv.") for k in model_state)

    if mode == "auto":
        mode = "minimap" if has_cnn else "flat"
    if mode == "minimap" and not has_cnn:
        raise ValueError("checkpoint is not a CNN (minimap) policy; use mode='flat'")
    if mode == "flat" and has_cnn:
        raise ValueError("checkpoint is a CNN (minimap) policy; use mode='minimap'")

    hidden = ckpt.get("hidden_sizes") or None
    if hidden is None:
        hidden_keys = sorted(
            (k for k in model_state if k.startswith("trunk.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[1]),
        )
        if not hidden_keys:
            # hybrid: hidden layers live in flat_trunk + joint
            flat_keys = sorted(
                (k for k in model_state if k.startswith("flat_trunk.") and k.endswith(".weight")),
                key=lambda k: int(k.split(".")[1]),
            )
            joint_keys = sorted(
                (k for k in model_state if k.startswith("joint.") and k.endswith(".weight")),
                key=lambda k: int(k.split(".")[1]),
            )
            hidden_keys = flat_keys + joint_keys
        if not hidden_keys:
            raise ValueError("cannot infer hidden sizes from checkpoint")
        hidden = [model_state[k].shape[0] for k in hidden_keys]

    has_flat_net = any(k.startswith("flat_proj") for k in model_state)
    has_flat_trunk = any(k.startswith("flat_trunk.") for k in model_state)
    if has_flat_net and "actor.weight" in model_state:
        n_actions = model_state["actor.weight"].shape[0]
    elif "actor_head.weight" in model_state:
        n_actions = model_state["actor_head.weight"].shape[0]
    else:
        n_actions = int(ckpt.get("n_actions", 0))

    if has_flat_net or has_flat_trunk:
        from rl.actor_critic_hybrid import ActorCriticHybrid
        if has_flat_net and "flat_proj.weight" in model_state:
            obs_size = int(ckpt.get("obs_size", 0)) or int(model_state["flat_proj.weight"].shape[1])
        else:
            obs_size = int(ckpt.get("obs_size", 0)) or int(model_state["flat_trunk.0.weight"].shape[1])
        n_channels = int(ckpt.get("n_channels", 8))
        grid = int(ckpt.get("grid_size", 2 * minimap_radius + 1))
        model = ActorCriticHybrid(
            obs_size=obs_size,
            n_channels=n_channels,
            grid_size=grid,
            n_actions=int(n_actions),
            hidden_sizes=[int(h) for h in hidden],
            device=device,
        )
    elif mode == "minimap" or has_cnn:
        first_conv = None
        for k in sorted(model_state):
            if (k.startswith("cnn.") or k.startswith("conv.")) and k.endswith(".weight") and model_state[k].dim() == 4:
                first_conv = model_state[k]
                break
        if first_conv is None:
            raise ValueError("cannot infer CNN input channels from checkpoint")
        n_channels = int(first_conv.shape[1])
        grid = int(ckpt.get("grid_size", 2 * minimap_radius + 1))
        model = ActorCriticCNN(
            n_channels=n_channels,
            grid_size=grid,
            n_actions=int(n_actions),
            hidden_sizes=[int(h) for h in hidden],
            device=device,
        )
    else:
        obs_size = int(ckpt.get("obs_size", 0)) or int(model_state["trunk.0.weight"].shape[1])
        model = ActorCritic(
            obs_size=obs_size,
            n_actions=int(n_actions),
            hidden_sizes=[int(h) for h in hidden],
            device=device,
        )

    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        raise ValueError(
            "checkpoint state_dict does not match the inferred architecture"
        )
    model.eval()
    return model


def run_eval(
    model_path: str | Path,
    episodes: int = 5,
    max_days: int = 10000,
    seed: int = 7,
    device: str = "cpu",
    normalization_path: str | Path | None = None,
    map_size: int = 280,
    mode: str = "auto",
    minimap_radius: int = 14,
    log_path: str | Path | None = None,
    difficulty: str = "normal",
    curriculum_stage: int | None = None,
    unlock_ids: str | None = None,
    use_curriculum_tab: bool | None = None,
    allow_stale_pyd: bool | None = None,
) -> Dict[str, float]:
    """Run the trained policy in the colony env and return mean stats.

    mode: "auto" (detect from checkpoint), "flat" (209-dim MLP), "minimap" (CNN).
    log_path: if set, writes a per-step observation log to this file.

    Curriculum: explicit `curriculum_stage` / `unlock_ids` / `use_curriculum_tab`
    win; otherwise values stored next to the checkpoint (best_model.meta.json,
    meta.json) are restored. The eval env MUST use the same allowed-buildings
    set as training — otherwise a WaterChannel-only champion plays with every
    building unlocked and starts erecting Goldmine (прииск) instead.
    Returns dict: days, people, bases, episodes, avg_return (all non-negative).
    """
    import torch
    from cpp_env import CppColonyEnv
    from minimap import MinimapSingleEnvWrapper
    from rl.curriculum import resolve_curriculum, resolve_state

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    # PR 3: fail fast on a stale colony_cpp binary (python/ is on sys.path via
    # the module header above). None = honour COLONY_ALLOW_STALE_PYD env var.
    from colony_cpp_api import require_colony
    require_colony(allow_stale=bool(allow_stale_pyd))

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    policy = _load_policy(model_path, dev, mode=mode, minimap_radius=minimap_radius)
    from rl.actor_critic_cnn import ActorCriticCNN
    from rl.actor_critic_hybrid import ActorCriticHybrid
    use_minimap = isinstance(policy, ActorCriticCNN)
    is_hybrid = isinstance(policy, ActorCriticHybrid)

    # Read reward/curriculum config from the model's meta.json to match training
    reward_cfg = None
    model_dir = model_path.parent
    meta: Dict[str, Any] = {}
    for meta_name in ("best_model.meta.json", "meta.json"):
        meta_path = model_dir / meta_name
        if meta_path.exists():
            try:
                import json as _json
                loaded = _json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    continue
                meta = loaded
                reward_cfg = meta.get("config", {}).get("reward")
                difficulty = meta.get("difficulty", difficulty)
                if reward_cfg:
                    break
            except (Exception,):
                pass

    # Explicit args win; otherwise restore the scenario stored next to the model.
    # PR 1: the env takes ONE computed state (resolve_state); the resolved dict
    # stays for logging (stage/manual/count as before).
    resolved = resolve_curriculum(
        model_dir,
        meta=meta,
        curriculum_stage=curriculum_stage,
        unlock_ids=unlock_ids,
        use_curriculum_tab=use_curriculum_tab,
    )
    st = resolve_state(
        model_dir,
        meta=meta,
        curriculum_stage=curriculum_stage,
        unlock_ids=unlock_ids,
        use_curriculum_tab=use_curriculum_tab,
    )
    cur_stage = int(resolved["curriculum_stage"])  # type: ignore[arg-type]
    manual_csv = str(resolved["unlock_ids"])
    eval_allowed = list(st.allowed_builds)

    env = CppColonyEnv(
        map_size=map_size,
        reward_config=reward_cfg,
        difficulty=difficulty,
        curriculum=st,
    )
    print(f"[Eval] curriculum: stage={cur_stage}, "
          f"manual={manual_csv or '—'}, allowed={len(eval_allowed)} buildings",
          flush=True)
    # The policy was built for a specific minimap grid (ppo.py saves grid_size
    # in the checkpoint). A default env reports radius 14 -> grid 29, so any
    # non-default training radius must be pushed into the env here, otherwise
    # cnn_proj gets the wrong number of features:
    #   "mat1 and mat2 shapes cannot be multiplied (1x3136 and 12544x256)"
    if use_minimap or is_hybrid:
        env.cpp_env.set_minimap_radius(int(policy.grid_size) // 2)
    if normalization_path is not None and not use_minimap and not is_hybrid:
        norm_path = Path(normalization_path)
        if not norm_path.exists():
            raise FileNotFoundError(
                f"normalization file not found: {norm_path}. "
                "Refusing to run eval on un-normalized observations."
            )
        env.normalizer.load(str(norm_path))
        env.normalizer.set_update(False)
    if normalization_path is not None and is_hybrid:
        norm_path = Path(normalization_path)
        if not norm_path.exists():
            raise FileNotFoundError(
                f"normalization file not found: {norm_path} (hybrid mode requires it)")
        env.normalizer.load(str(norm_path))
        env.normalizer.set_update(False)
    mm_wrap = MinimapSingleEnvWrapper(env) if (use_minimap or is_hybrid) else None

    action_names = env._action_names if hasattr(env, "_action_names") else [str(i) for i in range(env.action_space.n)]

    days_list: list[int] = []
    people_list: list[int] = []
    bases_list: list[int] = []
    returns: list[float] = []

    log_file = None
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"EVAL LOG | model={model_path} | episodes={episodes} | seed={seed}\n")
        log_file.write(f"reward_config={reward_cfg}\n")
        log_file.write(f"curriculum: stage={cur_stage} manual={manual_csv or '—'} "
                       f"allowed={len(eval_allowed)} buildings\n")
        log_file.write(f"action_names={action_names}\n")
        log_file.write("=" * 120 + "\n")

    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=seed + 500_000 + ep)
            total_reward = 0.0
            if log_file:
                log_file.write(f"\nEPISODE {ep} | seed={seed + ep}\n")
                log_file.write("-" * 120 + "\n")
            for step in range(max_days):
                with torch.no_grad():
                    if is_hybrid:
                        flat_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev).reshape(1, -1)
                        mm = mm_wrap.minimap_obs()
                        mm_t = torch.from_numpy(np.ascontiguousarray(mm, dtype=np.float32)).to(dev).reshape(1, *mm.shape)
                        logits, _values = policy(flat_t, mm_t)
                    elif use_minimap:
                        mm = mm_wrap.minimap_obs()
                        obs_t = torch.from_numpy(np.ascontiguousarray(mm, dtype=np.float32)).to(dev).reshape(1, *mm.shape)
                        logits, _values = policy(obs_t)
                    else:
                        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(dev)
                        obs_t = obs_t.reshape(1, -1)
                        logits, _values = policy(obs_t)

                    # Apply action masking (match training behavior)
                    if hasattr(env, "action_mask"):
                        mask = env.action_mask()
                        mask_t = torch.from_numpy(mask).to(dev).reshape(1, -1)
                        logits = logits.masked_fill(mask_t == 0, float("-inf"))

                    action = int(logits.argmax(dim=-1).item())

                if log_file:
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    top3_idx = torch.topk(probs, min(3, probs.numel())).indices.tolist()
                    top3 = ", ".join(f"{action_names[i]}={probs[i].item():.3f}" for i in top3_idx)
                    action_name = action_names[action] if action < len(action_names) else str(action)
                    log_file.write(f"  step={step:4d} | action={action} ({action_name}) | top3=[{top3}]\n")

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                if log_file:
                    log_file.write(
                        f"           | reward={reward:+.4f} | days={info.get('days',0)} | "
                        f"people={info.get('people',0)} | money={info.get('money',0)} | "
                        f"bases={info.get('bases',0)} | ep_return={info.get('ep_return',0):.1f}\n"
                    )

                if terminated or truncated:
                    if log_file:
                        log_file.write(f"  *** EPISODE END | terminated={terminated} truncated={truncated} | total_reward={total_reward:.1f} ***\n")
                    break
            days_list.append(int(info.get("days", 0)))
            people_list.append(int(info.get("people", 0)))
            bases_list.append(int(info.get("bases", 0)))
            returns.append(total_reward)
    finally:
        env.close()
        if log_file:
            log_file.write("\n" + "=" * 120 + "\n")
            log_file.write(f"SUMMARY | days={sum(days_list)/max(len(days_list),1):.0f} | "
                          f"people={sum(people_list)/max(len(people_list),1):.0f} | "
                          f"bases={sum(bases_list)/max(len(bases_list),1):.0f} | "
                          f"avg_return={sum(returns)/max(len(returns),1):.1f}\n")
            log_file.close()

    n = max(len(days_list), 1)
    return {
        "days": float(sum(days_list) / n),
        "days_std": float(np.std(days_list)),
        "people": float(sum(people_list) / n),
        "bases": float(sum(bases_list) / n),
        "episodes": float(len(days_list)),
        "avg_return": float(sum(returns) / n),
        "episode_days": days_list,
        "episode_bases": bases_list,
        "episode_people": people_list,
        "episode_returns": returns,
    }
