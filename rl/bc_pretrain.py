"""Behavioral-cloning (BC) pretraining from ``--record-demo`` JSONL logs.

Usage::

    python -m rl.bc_pretrain --demo demos/session1.jsonl demos/session2.jsonl \\
        --out models/bc_pretrained/model.pt --epochs 30

or point ``--demo`` at a directory of ``*.jsonl`` files::

    python -m rl.bc_pretrain --demo demos/ --out models/bc_pretrained/model.pt

Produces a checkpoint with the exact schema ``rl/ppo.py::PPO.save()``
writes (model_state / optimizer_state / n_actions / hidden_sizes /
obs_version / obs_size), so it can be:

  * loaded as-is by ``train_ui2/evaluator.py::_load_policy`` (watch it play);
  * warm-started into PPO via ``python train.py --resume-model <out>``.

Also writes ``normalization.json`` (and ``<out>.norm.json``) next to the
checkpoint — ``mean``/``var``/``count``/``obs_size``/``clip``, the schema
``python/cpp_env.py::Normalizer`` and ``rl/async_trainer.py``'s
``save_normalization`` use — because ``train.py --resume-model`` looks for
exactly one of those two paths and otherwise warns and fine-tunes on
*unnormalized* observations, silently ruining the run.

See docs/IMITATION_LEARNING_2026_09.md for the full design write-up,
including which GUI actions are (and are not) recordable in v1, and why the
recorded pacing (one env.step() = one day, for every action) is intentional.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rl.actor_critic import ActorCritic
from rl.bc_dataset import (
    compute_normalization,
    expand_demo_paths,
    infer_obs_version,
    load_demo_files,
    split_by_episode,
    validate_consistent_dims,
)

# Matches include/colony/running_mean_std.h::RunningMeanStd::normalize.
_NORM_EPS = 1e-8


def normalize_obs(
    obs: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, clip: float
) -> torch.Tensor:
    """Reproduce RunningMeanStd::normalize (same eps, NaN handling, clip).

    A BC-pretrained policy must see observations distributed the same way a
    PPO-rollout policy does; using a different normalization convention here
    would make the pretrained weights actively harmful once resumed into
    PPO training against the C++ env's own RunningMeanStd normalizer.
    """
    denom = torch.sqrt(var.abs() + _NORM_EPS)
    v = (obs - mean) / denom
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v.clamp(-clip, clip)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--demo", nargs="+", required=True,
        help="One or more --record-demo JSONL files, glob patterns, or directories of *.jsonl",
    )
    p.add_argument(
        "--out", required=True,
        help="Output checkpoint path (e.g. models/bc_pretrained/model.pt); "
             "normalization.json is written next to it",
    )
    p.add_argument(
        "--hidden-sizes", default="256,256",
        help="Comma-separated MLP hidden layer sizes (default matches rl/config.py's default policy)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of EPISODES (not transitions) held out for validation",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--obs-version", type=int, default=None,
        help="Override auto-detected obs_version (normally inferred from the demo's obs width)",
    )
    p.add_argument(
        "--patience", type=int, default=8,
        help="Stop early after this many epochs without val-loss improvement (0 disables early stop)",
    )
    return p.parse_args(argv)


def _to_tensors(steps, obs_size: int, n_actions: int, device: torch.device):
    if not steps:
        return (
            torch.empty(0, obs_size, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, n_actions, device=device),
        )
    obs = torch.from_numpy(np.stack([s.obs for s in steps])).to(device)
    actions = torch.tensor([s.action for s in steps], dtype=torch.long, device=device)
    masks = torch.from_numpy(np.stack([s.action_mask for s in steps])).to(device)
    return obs, actions, masks


def _masked_logits(model: ActorCritic, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Exactly mirrors rl/ppo.py's masking: additive mask-aware forward, then
    a hard -1e9 fill of blocked actions before the categorical distribution.
    """
    logits, _ = model(obs, mask)
    return logits.float().masked_fill(mask == 0, -1e9)


def evaluate(
    model: ActorCritic, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor,
    mean: torch.Tensor, var: torch.Tensor, clip: float, batch_size: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = obs.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            ob = normalize_obs(obs[i:i + batch_size], mean, var, clip)
            ac = actions[i:i + batch_size]
            mk = masks[i:i + batch_size]
            logits = _masked_logits(model, ob, mk)
            loss = nn.functional.cross_entropy(logits, ac, reduction="sum")
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=-1) == ac).sum().item())
    return total_loss / max(n, 1), total_correct / max(n, 1)


def train_bc(args: argparse.Namespace) -> dict:
    """Run BC pretraining; returns a small summary dict (also used by tests)."""
    set_seed(args.seed)
    device = torch.device(args.device)

    demo_paths = expand_demo_paths(args.demo)
    print(f"[BC] loading {len(demo_paths)} demo file(s): {demo_paths}")
    steps = load_demo_files(demo_paths)
    obs_size, n_actions = validate_consistent_dims(steps)
    n_episodes = len({s.episode for s in steps})
    print(
        f"[BC] loaded {len(steps)} transitions across {n_episodes} episode(s), "
        f"obs_size={obs_size}, n_actions={n_actions}"
    )

    obs_version = args.obs_version if args.obs_version is not None else infer_obs_version(obs_size)
    print(f"[BC] obs_version={obs_version}")

    norm = compute_normalization(steps)
    mean_t = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
    var_t = torch.tensor(norm["var"], dtype=torch.float32, device=device)
    clip = float(norm["clip"])

    train_steps, val_steps = split_by_episode(steps, args.val_frac, seed=args.seed)
    print(f"[BC] split: {len(train_steps)} train / {len(val_steps)} val transitions")

    train_obs, train_actions, train_masks = _to_tensors(train_steps, obs_size, n_actions, device)
    val_obs, val_actions, val_masks = _to_tensors(val_steps, obs_size, n_actions, device)

    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",") if x.strip()]
    model = ActorCritic(obs_size=obs_size, n_actions=n_actions, hidden_sizes=hidden_sizes, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_train = train_obs.shape[0]
    batch_size = min(args.batch_size, max(n_train, 1))
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        epoch_correct = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            ob = normalize_obs(train_obs[idx], mean_t, var_t, clip)
            ac = train_actions[idx]
            mk = train_masks[idx]
            logits = _masked_logits(model, ob, mk)
            loss = nn.functional.cross_entropy(logits, ac)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * ac.shape[0]
            epoch_correct += int((logits.argmax(dim=-1) == ac).sum().item())
        train_loss = epoch_loss / max(n_train, 1)
        train_acc = epoch_correct / max(n_train, 1)

        if val_obs.shape[0] > 0:
            val_loss, val_acc = evaluate(
                model, val_obs, val_actions, val_masks, mean_t, var_t, clip, batch_size
            )
            print(
                f"[BC] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )
            improved = val_loss < best_val_loss - 1e-6
            score = val_loss
        else:
            val_loss, val_acc = None, None
            print(
                f"[BC] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} (no val split)"
            )
            improved = train_loss < best_val_loss - 1e-6
            score = train_loss

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if improved:
            best_val_loss = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if args.patience > 0 and epochs_without_improve >= args.patience:
                print(f"[BC] early stop: no improvement for {args.patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "buffer_pos": 0,
        "n_actions": n_actions,
        "hidden_sizes": hidden_sizes,
        "obs_version": obs_version,
        "obs_size": obs_size,
    }
    torch.save(payload, out_path)

    norm_path = out_path.with_suffix(".norm.json")
    also_norm_path = out_path.parent / "normalization.json"
    with open(norm_path, "w") as f:
        json.dump(norm, f)
    with open(also_norm_path, "w") as f:
        json.dump(norm, f)

    print(f"[BC] saved checkpoint: {out_path}")
    print(f"[BC] saved normalization: {norm_path} and {also_norm_path}")
    print(f"[BC] best score={best_val_loss:.4f}")

    return {
        "out_path": str(out_path),
        "norm_path": str(norm_path),
        "obs_size": obs_size,
        "n_actions": n_actions,
        "obs_version": obs_version,
        "n_transitions": len(steps),
        "n_episodes": n_episodes,
        "best_score": best_val_loss,
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_bc(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
