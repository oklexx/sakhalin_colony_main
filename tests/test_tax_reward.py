import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import colony_cpp  # noqa: F401 — только проверка доступности расширения
    ENV_OK = True
except Exception:
    ENV_OK = False


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_no_bonus_on_tax_payment():
    """
    Verify that paying annual/main tax does NOT add +15 or +30 bonus.
    With all config rewards zeroed, max possible reward per step is:
      +0.3 (tax_daily_bonus) + 4.0 (max 2 born + 2 arrived) = 4.3
    If +15 or +30 existed, we'd see values > 4.3.
    """
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.build_bonus = 0.0
    cfg.chain_bonus = 0.0
    cfg.chain_daily = 0.0
    cfg.novelty = 0.0
    cfg.daily_income = 0.0
    cfg.sale_bonus = 0.0
    cfg.tax_daily_bonus = 0.3
    cfg.survival_bonus = 0.0
    cfg.game_over_penalty = 0.0
    cfg.idle_build_penalty = 0.0
    cfg.milestone_base_bonus = 0.0
    cfg.milestone_people_bonus = 0.0
    cfg.milestone_day_bonus = 0.0
    cfg.milestone_year_bonus = 0.0

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)

        rewards = []
        for _ in range(400):
            _, reward, done, _, _ = env.step(0)  # A_DAY
            rewards.append(reward)
            if done:
                break

        # Max possible with zeroed config: tax_daily_bonus(0.3) + born(2) + arrived(2) = 4.3
        # Legacy +15 or +30 would give 15.3 or 32.3
        MAX_EXPECTED = 5.0
        for r in rewards:
            assert r <= MAX_EXPECTED, \
                f"Reward {r} exceeds {MAX_EXPECTED} -- legacy +15/+30 bonus likely still present"

    finally:
        env.close()


def test_tax_daily_bonus_unchanged():
    """Verify tax_daily_bonus default is 0.3."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert rc.tax_daily_bonus == 0.3


def test_no_tax_bonus_in_config():
    """Verify legacy tax_bonus field does not exist."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert not hasattr(rc, "tax_bonus"), "Legacy tax_bonus should be removed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
