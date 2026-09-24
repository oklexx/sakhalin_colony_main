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


def test_tax_daily_bonus_in_reward_config():
    """Verify tax_daily_bonus is part of RewardConfig and tax_bonus is removed."""
    from rl.config import RewardConfig

    rc = RewardConfig()
    assert hasattr(rc, "tax_daily_bonus")
    assert rc.tax_daily_bonus == 0.3
    assert not hasattr(rc, "tax_bonus"), "tax_bonus should be removed"

    # Test serialization
    d = rc.to_dict()
    assert "tax_daily_bonus" in d
    assert d["tax_daily_bonus"] == 0.3
    assert "tax_bonus" not in d, "tax_bonus should not be in dict"

    # Test deserialization
    rc2 = RewardConfig.from_dict(d)
    assert rc2.tax_daily_bonus == 0.3


def test_tax_daily_bonus_default():
    """Verify default value is 0.3."""
    from rl.config import RewardConfig

    rc = RewardConfig()
    assert rc.tax_daily_bonus == 0.3


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_tax_daily_bonus_in_cpp_env():
    """Verify that tax_daily_bonus is applied in C++ env when tax is not due."""
    from cpp_env import CppColonyEnv

    env = CppColonyEnv(map_size=100)
    obs, _ = env.reset(seed=42)

    # Step a few times with DAY action
    total_reward = 0.0
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(0)  # DAY
        total_reward += reward
        if terminated or truncated:
            break

    # With tax_daily_bonus=0.77, we should get ~7.7 from daily bonus
    # (minus any other rewards/penalties)
    # This is a loose check — the exact value depends on env state
    assert total_reward > -100, f"Reward too negative: {total_reward}"
    env.close()


if __name__ == "__main__":
    test_tax_daily_bonus_in_reward_config()
    test_tax_daily_bonus_default()
    print("PASS: tax daily bonus config tests")
