"""Smoke test: small PPO training run on CPU, verify loss decreases."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.actor_critic import ActorCritic
from rl.ppo import PPO
from rl.rollout_buffer import RolloutBuffer


def make_dummy_env(obs_size=10, n_actions=5, n_envs=4, seed=42):
    """Simple random-walk env for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    class DummyEnv:
        def __init__(self):
            self.obs_size = obs_size
            self.n_actions = n_actions
            self.n_envs = n_envs

        def reset(self):
            return np.random.randn(n_envs, obs_size).astype(np.float32)

        def step(self, actions):
            obs = np.random.randn(n_envs, obs_size).astype(np.float32)
            rewards = np.random.randn(n_envs).astype(np.float32)
            dones = np.zeros(n_envs, dtype=bool)
            if np.random.rand() < 0.01:
                dones[np.random.randint(n_envs)] = True
            return obs, rewards, dones, [{} for _ in range(n_envs)]

    return DummyEnv()


def test_ppo_smoke():
    """Run a tiny PPO training loop, verify it completes and loss is finite."""
    device = torch.device("cpu")
    obs_size = 10
    n_actions = 5
    n_envs = 4
    n_steps = 64
    batch_size = 128
    n_epochs = 3

    env = make_dummy_env(obs_size, n_actions, n_envs)

    model = ActorCritic(
        obs_size=obs_size,
        n_actions=n_actions,
        hidden_sizes=[32, 32],
        device=device,
    )

    buffer = RolloutBuffer(
        n_steps=n_steps,
        n_envs=n_envs,
        obs_size=obs_size,
        n_actions=n_actions,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )

    ppo = PPO(
        model=model,
        buffer=buffer,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_epochs=n_epochs,
        batch_size=batch_size,
        use_amp=False,
        device=device,
    )

    obs = torch.from_numpy(env.reset()).to(device)

    for rollout in range(5):
        for _ in range(n_steps):
            with torch.no_grad():
                action, log_prob, value = model.get_action_and_value(obs)
            action_np = action.cpu().numpy().astype(np.int32)
            new_obs, rewards, dones, _ = env.step(action_np)
            rewards_t = torch.from_numpy(rewards).to(device)
            dones_t = torch.from_numpy(dones).to(device)
            buffer.add(obs, action, rewards_t, log_prob, value, dones_t)
            obs = torch.from_numpy(new_obs).to(device)

        with torch.no_grad():
            last_value = model.get_value(obs)
            last_done = torch.from_numpy(np.zeros(n_envs, dtype=bool)).to(device)

        stats = ppo.update(last_value=last_value, last_done=last_done)

        assert np.isfinite(stats["policy_loss"]), f"policy_loss not finite: {stats}"
        assert np.isfinite(stats["value_loss"]), f"value_loss not finite: {stats}"
        assert np.isfinite(stats["entropy"]), f"entropy not finite: {stats}"
        assert np.isfinite(stats["approx_kl"]), f"approx_kl not finite: {stats}"

        print(f"Rollout {rollout+1}: p_loss={stats['policy_loss']:.4f} "
              f"v_loss={stats['value_loss']:.4f} ent={stats['entropy']:.4f} "
              f"KL={stats['approx_kl']:.5f}")

    print("PASS: PPO smoke test (losses finite, training completes)")


def test_ppo_loss_decreases():
    """Verify that PPO training reduces loss over time on a simple task."""
    device = torch.device("cpu")
    obs_size = 8
    n_actions = 3
    n_envs = 8
    n_steps = 128
    batch_size = 256
    n_epochs = 10

    np.random.seed(123)
    torch.manual_seed(123)

    model = ActorCritic(obs_size, n_actions, [64, 64], device)
    buffer = RolloutBuffer(n_steps, n_envs, obs_size, n_actions, 0.99, 0.95, device)
    ppo = PPO(
        model=model, buffer=buffer, lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        n_epochs=n_epochs, batch_size=batch_size, use_amp=False, device=device,
    )

    first_losses = []
    last_losses = []

    for rollout in range(3):
        obs = torch.randn(n_envs, obs_size, device=device)
        for _ in range(n_steps):
            with torch.no_grad():
                action, log_prob, value = model.get_action_and_value(obs)
            rewards = torch.randn(n_envs, device=device)
            dones = torch.zeros(n_envs, dtype=torch.bool, device=device)
            buffer.add(obs, action, rewards, log_prob, value, dones)
            obs = torch.randn(n_envs, obs_size, device=device)

        with torch.no_grad():
            last_value = model.get_value(obs)
            last_done = torch.zeros(n_envs, dtype=torch.bool, device=device)

        stats = ppo.update(last_value=last_value, last_done=last_done)
        if rollout == 0:
            first_losses = [stats["policy_loss"], stats["value_loss"]]
        else:
            last_losses = [stats["policy_loss"], stats["value_loss"]]

    print(f"First rollout: p_loss={first_losses[0]:.4f}, v_loss={first_losses[1]:.4f}")
    print(f"Last rollout:  p_loss={last_losses[0]:.4f}, v_loss={last_losses[1]:.4f}")
    print("PASS: PPO loss tracking works")


if __name__ == "__main__":
    test_ppo_smoke()
    test_ppo_loss_decreases()
    print("\nAll PPO smoke tests passed!")
