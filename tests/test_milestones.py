import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import pytest

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False


def test_milestone_config_defaults():
    """Verify milestone config fields have correct defaults."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert rc.milestone_base_bonus == 10.0
    assert rc.milestone_people_bonus == 2.0
    assert rc.milestone_day_bonus == 2.0
    assert rc.milestone_year_bonus == 5.0
    d = rc.to_dict()
    assert d["milestone_base_bonus"] == 10.0
    assert d["milestone_people_bonus"] == 2.0
    assert d["milestone_day_bonus"] == 2.0
    assert d["milestone_year_bonus"] == 5.0


def test_milestone_cpp_bindings():
    """Verify milestone fields work in C++ RewardConfig."""
    rc = colony_cpp.RewardConfig()
    assert rc.milestone_base_bonus == 10.0
    assert rc.milestone_people_bonus == 2.0
    assert rc.milestone_day_bonus == 2.0
    assert rc.milestone_year_bonus == 5.0
    rc.milestone_base_bonus = 200.0
    assert rc.milestone_base_bonus == 200.0


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_day_milestone_100():
    """At day 100, milestone_day_bonus should fire."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.milestone_day_bonus = 77.0
    cfg.milestone_base_bonus = 0.0
    cfg.milestone_people_bonus = 0.0
    cfg.milestone_year_bonus = 0.0
    cfg.build_bonus = 0.0
    cfg.tax_daily_bonus = 0.0
    cfg.daily_income = 0.0
    cfg.chain_bonus = 0.0
    cfg.chain_daily = 0.0
    cfg.novelty = 0.0
    cfg.survival_bonus = 0.0
    cfg.game_over_penalty = 0.0
    cfg.error_penalty = 0.0
    cfg.preserve_penalty = 0.0
    cfg.manual_tax_penalty = 0.0
    cfg.idle_build_penalty = 0.0
    cfg.idle_build_penalty = 0.0
    cfg.clip_reward_min = -1000.0
    cfg.clip_reward_max = 1000.0
    cfg.disable_net_worth = True  # isolate milestone reward

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        # Advance exactly 100 days using A_DAY (action=0)
        for _ in range(100):
            _, reward, done, _, _ = env.step(0)
            rewards.append(reward)
            if done:
                break
        # On step 100 (days_alive becomes 100), milestone fires
        # rewards[99] is the 100th step (days_alive: 1→100)
        assert len(rewards) == 100, f"Expected 100 steps, got {len(rewards)}"
        # The 100th step should include milestone_day_bonus = 77
        # Other rewards (people born, etc.) may be present but non-negative
        r100 = rewards[99]
        assert r100 >= 77.0, f"Expected at least 77.0 at day 100, got {r100}"
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_day_milestone_not_double_fired():
    """Milestone should not fire twice for the same day."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.milestone_day_bonus = 77.0
    cfg.milestone_base_bonus = 0.0
    cfg.milestone_people_bonus = 0.0
    cfg.milestone_year_bonus = 0.0
    cfg.build_bonus = 0.0
    cfg.tax_daily_bonus = 0.0
    cfg.daily_income = 0.0
    cfg.chain_bonus = 0.0
    cfg.chain_daily = 0.0
    cfg.novelty = 0.0
    cfg.survival_bonus = 0.0
    cfg.game_over_penalty = 0.0
    cfg.error_penalty = 0.0
    cfg.preserve_penalty = 0.0
    cfg.manual_tax_penalty = 0.0
    cfg.idle_build_penalty = 0.0
    cfg.clip_reward_min = -1000.0
    cfg.clip_reward_max = 1000.0
    cfg.disable_net_worth = True

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(110):
            _, reward, done, _, _ = env.step(0)
            rewards.append(reward)
            if done:
                break
        # Day 100 = rewards[99] should have milestone bonus
        # Day 101-110 = rewards[100..] should NOT have milestone bonus
        if len(rewards) > 100:
            r101 = rewards[100]
            assert r101 < 77.0, f"Day 101 should not have milestone: {r101}"
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_day_milestone_200():
    """At day 200, another milestone should fire."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.milestone_day_bonus = 55.0
    cfg.milestone_base_bonus = 0.0
    cfg.milestone_people_bonus = 0.0
    cfg.milestone_year_bonus = 0.0
    cfg.build_bonus = 0.0
    cfg.tax_daily_bonus = 0.0
    cfg.daily_income = 0.0
    cfg.chain_bonus = 0.0
    cfg.chain_daily = 0.0
    cfg.novelty = 0.0
    cfg.survival_bonus = 0.0
    cfg.game_over_penalty = 0.0
    cfg.error_penalty = 0.0
    cfg.preserve_penalty = 0.0
    cfg.manual_tax_penalty = 0.0
    cfg.idle_build_penalty = 0.0
    cfg.clip_reward_min = -1000.0
    cfg.clip_reward_max = 1000.0
    cfg.disable_net_worth = True

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(200):
            _, reward, done, _, _ = env.step(0)
            rewards.append(reward)
            if done:
                break
        assert len(rewards) == 200, f"Expected 200 steps, got {len(rewards)}"
        # Day 200 = rewards[199] should include milestone
        r200 = rewards[199]
        assert r200 >= 55.0, f"Expected at least 55.0 at day 200, got {r200}"
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_year_milestone():
    """At day 365, year bonus should fire once."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.milestone_year_bonus = 200.0
    cfg.milestone_base_bonus = 0.0
    cfg.milestone_people_bonus = 0.0
    cfg.milestone_day_bonus = 0.0
    cfg.build_bonus = 0.0
    cfg.tax_daily_bonus = 0.0
    cfg.daily_income = 0.0
    cfg.chain_bonus = 0.0
    cfg.chain_daily = 0.0
    cfg.novelty = 0.0
    cfg.survival_bonus = 0.0
    cfg.game_over_penalty = 0.0
    cfg.error_penalty = 0.0
    cfg.preserve_penalty = 0.0
    cfg.manual_tax_penalty = 0.0
    cfg.idle_build_penalty = 0.0
    cfg.clip_reward_min = -1000.0
    cfg.clip_reward_max = 1000.0
    cfg.disable_net_worth = True

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(370):
            _, reward, done, _, _ = env.step(0)
            rewards.append(reward)
            if done:
                break
        assert len(rewards) >= 365, f"Not enough steps: {len(rewards)}"
        # Day 365 = rewards[364] should include year bonus = 200
        r365 = rewards[364]
        assert r365 >= 200.0, f"Expected at least 200.0 at day 365, got {r365}"
        # Day 366 = rewards[365] should NOT have year bonus
        if len(rewards) > 365:
            r366 = rewards[365]
            assert r366 < 200.0, f"Day 366 should not have year bonus: {r366}"
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_milestones_with_zero_config():
    """With all milestones set to 0, no milestone reward at day 100/200/365."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.milestone_base_bonus = 0.0
    cfg.milestone_people_bonus = 0.0
    cfg.milestone_day_bonus = 0.0
    cfg.milestone_year_bonus = 0.0
    cfg.build_bonus = 0.0
    cfg.tax_daily_bonus = 0.0
    cfg.daily_income = 0.0
    cfg.chain_bonus = 0.0
    cfg.chain_daily = 0.0
    cfg.novelty = 0.0
    cfg.survival_bonus = 0.0
    cfg.game_over_penalty = 0.0
    cfg.error_penalty = 0.0
    cfg.preserve_penalty = 0.0
    cfg.manual_tax_penalty = 0.0
    cfg.idle_build_penalty = 0.0
    cfg.clip_reward_min = -1000.0
    cfg.clip_reward_max = 1000.0
    cfg.disable_net_worth = True

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(370):
            _, reward, done, _, _ = env.step(0)
            rewards.append(reward)
            if done:
                break
        # Day 100 (index 99) should have no milestone bonus
        # With zero milestones, reward is only from people born/arrived (non-milestone)
        # People bonus = 1.0 per person, which is small. Milestone bonuses are 2-5.
        # So at milestone days, reward should be < 5.0 (the smallest possible milestone)
        if len(rewards) >= 100:
            assert rewards[99] < 5.0, f"Day 100 should have no milestone: {rewards[99]}"
        if len(rewards) >= 200:
            assert rewards[199] < 5.0, f"Day 200 should have no milestone: {rewards[199]}"
        if len(rewards) >= 365:
            assert rewards[364] < 5.0, f"Day 365 should have no year milestone: {rewards[364]}"
    finally:
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
