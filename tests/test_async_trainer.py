import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.config import Config
from rl.actor_critic import ActorCritic
from rl.rollout_buffer import RolloutBuffer
from rl.ppo import PPO
from rl.async_trainer import AsyncTrainer


class FakeEnvManager:
    """Minimal env manager that simulates 2 envs with known episode returns."""

    def __init__(self, n_envs=2, obs_size=10, n_actions=5, n_steps=3):
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.device = torch.device("cpu")
        self._step_count = 0
        self._episode_returns = {0: 100.0, 1: 200.0}

        # Stand-in for EnvManager.cfg (real EnvManager stores cfg with map_size).
        self.cfg = type("FakeCfg", (), {"map_size": 280})()

        torch.manual_seed(0)
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

        # Minimal stand-in for CppVecEnv.venv (ColonyVecEnvCpp) used by
        # AsyncTrainer.train() to persist normalization stats.
        self.env = type("FakeEnv", (), {})()
        self.env.venv = type("FakeVenv", (), {
            "save_normalization": staticmethod(lambda path: None),
        })()

    def set_curriculum_stage(self, stage):
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
                {"episode": {
                    "r": self._episode_returns[0], "l": 50, "seed": 7,
                    "metrics": {"chains_activated": 2, "max_chain_depth": 1,
                                "total_builds": 4, "builds_by_type": {"Farm": 1},
                                "reached_resources": 1,
                                "reached_resource_ids": ["food"]},
                }},
                {"episode": {
                    "r": self._episode_returns[1], "l": 80, "seed": 8,
                    "metrics": {"chains_activated": 3, "max_chain_depth": 2,
                                "total_builds": 6, "builds_by_type": {"Road": 3},
                                "reached_resources": 2,
                                "reached_resource_ids": ["food", "water"]},
                }},
            ]
        return [{}, {}]

    def close(self):
        pass


def test_per_env_episode_tracking(tmp_path):
    """Verify that episodes from different envs are tracked separately."""
    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=6,
        save_freq=0,
        eval_freq=0,
        use_amp=False,
        model_dir=str(tmp_path),
    )
    em = FakeEnvManager(n_envs=2, n_steps=3)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    trainer.train(total_timesteps=6)

    # Both episodes should be tracked separately
    assert len(trainer._ep_returns) == 2, f"Expected 2 episodes, got {len(trainer._ep_returns)}"
    # best_reward should be the max of individual episodes, not their sum
    assert trainer.best_reward == 200.0, f"Expected best_reward=200.0, got {trainer.best_reward}"
    assert 100.0 in trainer._ep_returns
    assert 200.0 in trainer._ep_returns

    import json
    diagnostics = tmp_path / "episode_diagnostics.jsonl"
    records = [json.loads(line) for line in diagnostics.read_text(encoding="utf-8").splitlines()]
    episodes = [row for row in records if row["record_type"] == "episode"]
    assert len(episodes) == 2
    assert {row["episode_metrics"]["max_chain_depth"] for row in episodes} == {1, 2}
    assert sum(row["episode_metrics"]["chains_activated"] for row in episodes) == 5
    assert all(row["metrics_available"] for row in episodes)
    assert all(row["total_timesteps"] == 6 for row in episodes)
    assert records[-1]["record_type"] == "run_end"


def test_eval_and_best_model_saving(tmp_path):
    """Verify that eval runs and best model is saved when composite score improves."""
    import json

    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=12,
        save_freq=0,
        eval_freq=3,
        eval_episodes=2,
        eval_min_bases=2,
        eval_min_days=0.0,
        eval_use_median=True,
        model_dir=str(tmp_path),
    )
    em = FakeEnvManager(n_envs=2)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    eval_results = [
        {
            "days": 10.0, "people": 5.0, "bases": 2.0, "episodes": 2.0, "avg_return": 100.0,
            "episode_days": [8.0, 12.0],
            "episode_bases": [2, 2],
            "episode_people": [4, 6],
            "episode_returns": [80.0, 120.0],
        },
        {
            "days": 20.0, "people": 10.0, "bases": 4.0, "episodes": 2.0, "avg_return": 200.0,
            "episode_days": [15.0, 25.0],
            "episode_bases": [3, 5],
            "episode_people": [8, 12],
            "episode_returns": [150.0, 250.0],
        },
    ]
    eval_call_count = [0]

    def mock_run_eval(*args, **kwargs):
        idx = min(eval_call_count[0], len(eval_results) - 1)
        eval_call_count[0] += 1
        return eval_results[idx]

    import train_ui2.evaluator as ev
    original_run_eval = ev.run_eval
    ev.run_eval = mock_run_eval

    try:
        trainer.train(total_timesteps=12)
    finally:
        ev.run_eval = original_run_eval

    best_model = tmp_path / "best_model.pt"
    best_meta = tmp_path / "best_model.meta.json"
    assert best_model.exists(), "best_model.pt should exist"
    assert best_meta.exists(), "best_model.meta.json should exist"

    with open(best_meta) as f:
        meta = json.load(f)
    assert "best_score" in meta
    assert meta["best_bases"] == 4.0
    assert meta["best_people"] == 10.0
    assert meta["best_days"] == 20.0


def test_min_bases_threshold(tmp_path):
    """Verify that model is NOT saved when bases < min_bases."""

    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=6,
        save_freq=0,
        eval_freq=3,
        eval_episodes=1,
        eval_min_bases=5,
        eval_min_days=0.0,
        eval_use_median=True,
        model_dir=str(tmp_path),
    )
    em = FakeEnvManager(n_envs=2)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    def mock_run_eval(*args, **kwargs):
        return {
            "days": 2000.0, "people": 50.0, "bases": 1.0, "episodes": 1.0, "avg_return": -100.0,
            "episode_days": [2000.0],
            "episode_bases": [1],
            "episode_people": [50],
            "episode_returns": [-100.0],
        }

    import train_ui2.evaluator as ev
    original_run_eval = ev.run_eval
    ev.run_eval = mock_run_eval

    try:
        trainer.train(total_timesteps=6)
    finally:
        ev.run_eval = original_run_eval

    best_model = tmp_path / "best_model.pt"
    assert not best_model.exists(), "best_model.pt should NOT exist (bases=1 < min_bases=5)"


def test_composite_score_calculation():
    """Verify composite score formula: score = days*w1 + bases*w2 + people*w3 + max(0,return)*w4."""

    w1, w2, w3, w4 = 0.10, 1.0, 0.10, 0.0001
    days, bases, people, ret = 100.0, 10.0, 20.0, 5000.0
    expected = days * w1 + bases * w2 + people * w3 + max(0.0, ret) * w4
    assert abs(expected - (10.0 + 10.0 + 2.0 + 0.5)) < 1e-9

    ret_neg = -100.0
    expected_neg = days * w1 + bases * w2 + people * w3 + max(0.0, ret_neg) * w4
    assert abs(expected_neg - 22.0) < 1e-9


def test_curriculum_stage_in_meta(tmp_path):
    """best_model.meta.json must contain curriculum_stage_at_best."""
    import json
    from unittest.mock import MagicMock, patch
    from rl.async_trainer import AsyncTrainer
    from rl.config import Config

    cfg = Config(
        model_dir=str(tmp_path),
        eval_episodes=1,
        eval_freq=100,
        n_steps=128,
        batch_size=64,
        total_timesteps=200,
        eval_min_bases=1,
        eval_min_days=0.0,
        curriculum_schedule=[(0, 1), (100, 2)],
    )

    mock_em = MagicMock()
    mock_em.device = torch.device("cpu")
    mock_em.cfg = cfg

    mock_ppo = MagicMock()
    mock_ppo.save = MagicMock()

    trainer = AsyncTrainer(
        cfg=cfg,
        env_manager=mock_em,
    )
    trainer.em = mock_em
    trainer.em.ppo = mock_ppo
    trainer._curriculum_stage = 2
    trainer.best_score = None

    with patch("train_ui2.evaluator.run_eval") as mock_run_eval:
        mock_run_eval.return_value = {
            "days": 100.0, "bases": 10.0, "people": 50.0,
            "avg_return": 1000.0,
            "episode_days": [100.0], "episode_bases": [10.0],
            "episode_people": [50.0], "episode_returns": [1000.0],
        }
        trainer._eval(200)

    meta_path = tmp_path / "best_model.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["curriculum_stage_at_best"] == 2


def test_multi_seed_eval(tmp_path):
    """Verify that eval runs for each seed in eval_seeds and aggregates results."""

    cfg = Config(
        n_envs=2,
        n_steps=3,
        total_timesteps=6,
        save_freq=0,
        eval_freq=3,
        eval_episodes=2,
        eval_min_bases=1,
        eval_min_days=0.0,
        eval_use_median=True,
        eval_seeds=[1, 2, 3],
        model_dir=str(tmp_path),
    )
    em = FakeEnvManager(n_envs=2)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)

    call_count = [0]

    def mock_run_eval(*args, **kwargs):
        seed = kwargs.get("seed", 42)
        call_count[0] += 1
        return {
            "days": float(10 + seed),
            "people": 5.0,
            "bases": 2.0,
            "episodes": 2.0,
            "avg_return": 100.0,
            "episode_days": [float(10 + seed), float(12 + seed)],
            "episode_bases": [2, 2],
            "episode_people": [4, 6],
            "episode_returns": [80.0, 120.0],
        }

    import train_ui2.evaluator as ev
    original_run_eval = ev.run_eval
    ev.run_eval = mock_run_eval

    try:
        trainer.train(total_timesteps=6)
    finally:
        ev.run_eval = original_run_eval

    assert call_count[0] >= 3, f"run_eval should be called for each seed, got {call_count[0]} calls"


if __name__ == "__main__":
    test_per_env_episode_tracking()
    print("PASS: per-env episode tracking")
    test_composite_score_calculation()
    print("PASS: composite score calculation")
    test_multi_seed_eval(Path(__file__).parent / "tmp")
    print("PASS: multi-seed eval")


# ── monitoring labels: real env order, always-on available set ──────────────

def test_trainer_prefers_env_action_names():
    """getattr branch: env-provided names win over the fallback list."""
    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    em = FakeEnvManager(n_envs=2, n_steps=3)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)
    assert trainer._action_names == em.action_names
    assert len(trainer._action_names) == 5  # the env list, not the 45 fallback


def test_fallback_action_names_index_aligned():
    """No env names: honest 45-slot fallback (DAY/WEEK/builds/managers)."""
    from types import SimpleNamespace

    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    em = SimpleNamespace(device=torch.device("cpu"))
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)
    names = trainer._action_names
    assert len(names) == 49  # 2 + 32 builds + 11 managers + 4 road directions
    assert names[0] == "DAY" and names[1] == "WEEK"
    assert names[2:34] == [f"BUILD_{i}" for i in range(32)]
    assert names[34:45] == ["IMPROVE_LAND", "REPAIR", "REPAIR_ALL", "DEMOLISH",
                            "PRESERVE", "UNPRESERVE", "SELL_SURPLUS", "BUY_FOOD",
                            "TAKE_LOAN", "REPAY_LOAN", "PAY_TAX"]
    assert names[45:] == ["ROAD_E", "ROAD_W", "ROAD_S", "ROAD_N"]


def test_top_actions_dict_omits_zeros_and_maps_names():
    """Доли для монитора: нули не отправляются, имена — из списка среды.

    Обновлено 2026-09-23: `_calculate_action_distribution()` по умолчанию
    считает ПО ВСЕМУ РОЛЛАУТУ (счётчик `_rollout_action_counts`), а не по
    последним 1000 записям истории — старое окно оставляло панели ~3% шагов, и
    «Водоканал» исчезал, когда в хвосте роллаута не оставалось нажатий. Скользящее
    окно теперь запрашивается явно (`window=N`) и используется как отладочный
    режим; см. регрессии в tests/test_monitor_action_stats.py.
    """
    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, use_amp=False)
    em = FakeEnvManager(n_envs=2, n_steps=3)
    trainer = AsyncTrainer(cfg=cfg, env_manager=em)
    names = em.action_names  # 5 fake actions
    trainer._action_history.extend([(0, names[1]), (1, names[1]), (0, names[3])])

    # явное окно по истории (legacy-режим)
    counts = trainer._calculate_action_distribution(window=1000)
    assert counts == [0, 2, 0, 1, 0]
    d = trainer._top_actions_dict(counts, sum(counts))
    assert d == {names[1]: round(200 / 3, 2), names[3]: round(100 / 3, 2)}

    # счётчик роллаута пуст, пока сбор не идёт: панель обязана молчать, а не
    # показывать вчерашние доли
    assert trainer._calculate_action_distribution() == [0, 0, 0, 0, 0]
    assert trainer._top_actions_dict(trainer._calculate_action_distribution(), 0) == {}


def test_env_manager_action_names_match_env_order():
    """Real env: BUILD_GOLDMINE sits at its true index (9), not 12."""
    import pytest

    pytest.importorskip("colony_cpp")
    pytest.importorskip("stable_baselines3")
    from rl.env_manager import EnvManager

    cfg = Config(n_envs=2, map_size=64, n_steps=8)
    em = EnvManager(cfg, torch.device("cpu"))
    try:
        names = em.action_names
        # 49 = 2 (DAY/WEEK) + 32 build slots + 11 managers + 4 road directions
        # (include/colony/constants.h: N_ROAD_DIRS добавил ROAD_E/W/S/N)
        assert len(names) == em.n_actions == 49
        assert names[0] == "DAY" and names[1] == "WEEK"
        assert names.index("BUILD_GOLDMINE") == 9
        assert names.index("BUILD_ROAD") == 12
        assert names == list(em.vec_env.action_names)
    finally:
        em.close()


def test_available_actions_without_schedule(tmp_path):
    """Fixed manual set (no schedule): widget shows it, open set truncates."""

    class ManualFake(FakeEnvManager):
        def __init__(self, allowed, *a, **k):
            super().__init__(*a, **k)
            self._allowed = allowed

        def get_allowed_buildings_for_stage(self, stage):
            return list(self._allowed)

    cfg = Config(n_envs=2, n_steps=3, total_timesteps=6, save_freq=0,
                 eval_freq=0, use_amp=False, model_dir=str(tmp_path))
    trainer = AsyncTrainer(cfg=cfg, env_manager=ManualFake(["Road", "WaterChannel"]))
    trainer.train(total_timesteps=6)
    assert trainer.metrics.curriculum_available_actions == "Road | WaterChannel"

    many = [f"B{i}" for i in range(32)]
    trainer2 = AsyncTrainer(cfg=cfg, env_manager=ManualFake(many))
    trainer2.train(total_timesteps=6)
    assert trainer2.metrics.curriculum_available_actions == \
        "B0 | B1 | B2 | B3 | B4 (+27 more)"
