import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.async_trainer import AsyncTrainer
from rl.config import Config


class FakeEnvManager:
    def __init__(self, n_envs=2, obs_size=10, n_actions=5, n_steps=3):
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = torch.device("cpu")
        self._step_count = 0
        self._episode_returns = {0: 100.0, 1: 200.0}
        self.cfg = type("FakeCfg", (), {"map_size": 280, "curriculum_stage": 0})()
        self.curriculum_stages_called = []

        torch.manual_seed(0)
        from rl.actor_critic import ActorCritic
        from rl.ppo import PPO
        from rl.rollout_buffer import RolloutBuffer

        self.model = ActorCritic(obs_size, n_actions, [16], self.device)
        self.buffer = RolloutBuffer(
            n_steps=n_steps, n_envs=n_envs, obs_size=obs_size,
            n_actions=n_actions, gamma=0.99, gae_lambda=0.95,
            device=self.device,
        )
        self.ppo = PPO(
            model=self.model, buffer=self.buffer, lr=1e-3,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            n_epochs=1, batch_size=n_envs, use_amp=False,
            device=self.device,
        )

        self.vec_env = type("FakeEnv", (), {})()

        class FakeVenv:
            def __init__(self):
                self.stages = []
            def save_normalization(self, path):
                pass
            def set_curriculum_stage(self, stage):
                self.stages.append(stage)

        self.vec_env.venv = FakeVenv()

    def set_curriculum_stage(self, stage):
        self.vec_env.venv.set_curriculum_stage(stage)
        self.cfg.curriculum_stage = stage

    def get_allowed_buildings_for_stage(self, stage):
        return []

    @property
    def action_names(self):
        return ["ACT初始化", "ACT建造房屋", "ACT储备物资", "ACT空闲等待", "ACT向右移动"]

    def reset(self):
        return torch.randn(self.n_envs, self.obs_size)

    def collect_step(self, obs):
        self._step_count += 1
        with torch.no_grad():
            action, log_prob, value = self.model.get_action_and_value(obs)
        new_obs = torch.randn(self.n_envs, self.obs_size)
        rewards = torch.randn(self.n_envs)
        dones = torch.zeros(self.n_envs, dtype=torch.bool)
        self.buffer.add(
            obs=obs, action=action, reward=rewards,
            log_prob=log_prob, value=value, done=dones,
        )
        return new_obs, self.get_infos()

    def get_infos(self):
        if self._step_count >= 3:
            return [
                {"episode": {"r": self._episode_returns[0], "l": 50}},
                {"episode": {"r": self._episode_returns[1], "l": 80}},
            ]
        return [{}, {}]

    def close(self):
        pass


def test_curriculum_stage_switching(tmp_path):
    """Verify that curriculum stage switches according to schedule."""
    schedule = [
        (0, 1),
        (3, 2),
    ]
    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=12,
        save_freq=0,
        eval_freq=0,
        use_amp=False,
        model_dir=str(tmp_path),
        curriculum_schedule=schedule,
        curriculum_stage=0,
    )
    em = FakeEnvManager(n_envs=2)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    trainer.train(total_timesteps=12)

    stages = em.vec_env.venv.stages
    assert 1 in stages, f"Stage 1 should have been set, got {stages}"
    assert 2 in stages, f"Stage 2 should have been set, got {stages}"
    assert stages.index(1) < stages.index(2), "Stage 1 should come before Stage 2"


def test_curriculum_no_schedule(tmp_path):
    """Verify that no stage switching happens without a schedule."""
    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=6,
        save_freq=0,
        eval_freq=0,
        use_amp=False,
        model_dir=str(tmp_path),
        curriculum_schedule=[],
    )
    em = FakeEnvManager(n_envs=2)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    trainer.train(total_timesteps=6)

    assert em.vec_env.venv.stages == [], f"No stages should be set, got {em.vec_env.venv.stages}"


def test_curriculum_schedule_in_config():
    """Verify curriculum_schedule serializes/deserializes correctly."""
    schedule = [(0, 1), (5_000_000, 2), (10_000_000, 3)]
    cfg = Config(curriculum_schedule=schedule)
    d = cfg.to_dict()
    assert d["curriculum_schedule"] == schedule

    cfg2 = Config.from_dict(d)
    assert cfg2.curriculum_schedule == schedule


if __name__ == "__main__":
    test_curriculum_stage_switching(Path(__file__).parent / "tmp")
    print("PASS: curriculum stage switching")
    test_curriculum_no_schedule(Path(__file__).parent / "tmp")
    print("PASS: curriculum no schedule")
    test_curriculum_schedule_in_config()
    print("PASS: curriculum schedule in config")
