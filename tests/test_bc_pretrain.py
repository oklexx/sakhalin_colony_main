"""Tests for rl/bc_pretrain.py: end-to-end BC training on synthetic demos.

Uses a tiny synthetic obs/action space (not a real colony_cpp obs) so these
tests run fast and without the compiled extension; --obs-version is passed
explicitly since these obs widths intentionally aren't one of the real
248/289/299 layouts.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from rl.bc_pretrain import evaluate, normalize_obs, parse_args, train_bc

OBS_SIZE = 5
N_ACTIONS = 4


def _make_demo_file(path, n_episodes=8, steps_per_episode=20, seed=0):
    """Synthetic but *learnable* demo: action = argmax(obs[:N_ACTIONS]),
    i.e. there is a real, simple obs->action mapping a BC policy can fit,
    so the test can assert training actually reduces loss / raises accuracy
    instead of just "runs without crashing".
    """
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        for _ep in range(n_episodes):
            for t in range(steps_per_episode):
                obs = rng.normal(size=OBS_SIZE).astype(np.float32)
                action = int(np.argmax(obs[:N_ACTIONS]))
                terminated = t == steps_per_episode - 1
                row = {
                    "action": action,
                    "reward": 1.0,
                    "terminated": terminated,
                    "obs": obs.tolist(),
                    "action_mask": [1.0] * N_ACTIONS,
                }
                f.write(json.dumps(row) + "\n")


def test_normalize_obs_matches_running_mean_std_formula():
    obs = torch.tensor([[1.0, -1.0, 100.0]])
    mean = torch.tensor([0.0, 0.0, 0.0])
    var = torch.tensor([1.0, 1.0, 1.0])
    out = normalize_obs(obs, mean, var, clip=10.0)
    # var=1 -> denom = sqrt(1 + 1e-8) ~= 1; first two pass through, third clips.
    assert torch.allclose(out[0, :2], torch.tensor([1.0, -1.0]), atol=1e-3)
    assert out[0, 2].item() == 10.0  # clipped


def test_normalize_obs_handles_nan_and_inf():
    obs = torch.tensor([[float("nan"), float("inf"), float("-inf")]])
    mean = torch.tensor([0.0, 0.0, 0.0])
    var = torch.tensor([1.0, 1.0, 1.0])
    out = normalize_obs(obs, mean, var, clip=10.0)
    assert torch.all(out == 0.0)


def test_train_bc_end_to_end_produces_loadable_checkpoint(tmp_path):
    demo_path = tmp_path / "demo.jsonl"
    _make_demo_file(demo_path, n_episodes=10, steps_per_episode=25, seed=1)

    out_path = tmp_path / "model" / "bc.pt"
    args = parse_args([
        "--demo", str(demo_path),
        "--out", str(out_path),
        "--epochs", "15",
        "--batch-size", "32",
        "--lr", "1e-2",
        "--val-frac", "0.2",
        "--seed", "0",
        "--device", "cpu",
        "--obs-version", "2",
        "--patience", "0",
    ])
    summary = train_bc(args)

    assert out_path.exists()
    assert (out_path.parent / "normalization.json").exists()
    assert out_path.with_suffix(".norm.json").exists()
    assert summary["obs_size"] == OBS_SIZE
    assert summary["n_actions"] == N_ACTIONS
    assert summary["obs_version"] == 2
    assert summary["n_episodes"] == 10

    # The synthetic task (argmax of first N_ACTIONS obs dims) is trivially
    # learnable; a real MLP trained for 15 epochs should land well above
    # random-guess accuracy (1/N_ACTIONS = 0.25).
    last = summary["history"][-1]
    assert last["train_acc"] > 0.6

    # Checkpoint round-trips through the exact schema rl/ppo.py::PPO.save()
    # writes, and train_ui2/evaluator.py::_load_policy loads unmodified.
    ckpt = torch.load(out_path, map_location="cpu", weights_only=False)
    for key in ("model_state", "optimizer_state", "n_actions", "hidden_sizes", "obs_version", "obs_size"):
        assert key in ckpt
    assert ckpt["n_actions"] == N_ACTIONS
    assert ckpt["obs_size"] == OBS_SIZE
    assert ckpt["hidden_sizes"] == [256, 256]

    norm = json.loads((out_path.parent / "normalization.json").read_text())
    # Nested ColonyVecEnvCpp::save_normalization schema (contract with the
    # C++ vec-env loader — train.py --resume-model feeds it this file
    # verbatim) + top-level obs_size for the single-env PR 5 check.
    assert norm["obs_size"] == OBS_SIZE
    for key in ("obs_rms", "rew_rms", "norm_obs", "norm_reward",
                "clip_obs", "clip_reward"):
        assert key in norm, f"normalization.json missing vec-env key {key!r}"
    assert norm["obs_rms"]["count"] == summary["n_transitions"]
    assert len(norm["obs_rms"]["mean"]) == OBS_SIZE
    assert len(norm["obs_rms"]["var"]) == OBS_SIZE


def test_train_bc_loadable_by_evaluator(tmp_path):
    """The produced checkpoint must load through the project's own generic
    checkpoint loader (train_ui2/evaluator.py::_load_policy), not just via a
    raw torch.load — that's the actual compatibility contract.
    """
    demo_path = tmp_path / "demo.jsonl"
    _make_demo_file(demo_path, n_episodes=4, steps_per_episode=10, seed=2)
    out_path = tmp_path / "bc.pt"
    args = parse_args([
        "--demo", str(demo_path), "--out", str(out_path),
        "--epochs", "2", "--val-frac", "0.0", "--device", "cpu",
        "--obs-version", "2", "--patience", "0",
    ])
    train_bc(args)

    from train_ui2.evaluator import _load_policy
    model = _load_policy(out_path, torch.device("cpu"))
    assert model.obs_size == OBS_SIZE
    assert model.n_actions == N_ACTIONS


def test_train_bc_no_val_split_still_saves(tmp_path):
    demo_path = tmp_path / "demo.jsonl"
    _make_demo_file(demo_path, n_episodes=1, steps_per_episode=10, seed=3)
    out_path = tmp_path / "bc.pt"
    args = parse_args([
        "--demo", str(demo_path), "--out", str(out_path),
        "--epochs", "3", "--val-frac", "0.5", "--device", "cpu",
        "--obs-version", "2", "--patience", "0",
    ])
    summary = train_bc(args)
    # A single episode can't be split (split_by_episode falls back to no split).
    assert summary["history"][0]["val_loss"] is None
    assert out_path.exists()


def test_evaluate_matches_manual_cross_entropy():
    torch.manual_seed(0)
    from rl.actor_critic import ActorCritic
    model = ActorCritic(obs_size=3, n_actions=2, hidden_sizes=[8], device=torch.device("cpu"))
    obs = torch.zeros(4, 3)
    actions = torch.tensor([0, 1, 0, 1])
    masks = torch.ones(4, 2)
    mean = torch.zeros(3)
    var = torch.ones(3)
    loss, acc = evaluate(model, obs, actions, masks, mean, var, clip=10.0, batch_size=2)
    assert loss > 0
    assert 0.0 <= acc <= 1.0
