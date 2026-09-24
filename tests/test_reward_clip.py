import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import colony_cpp
    ENV_OK = True
except Exception:
    ENV_OK = False


def test_clip_config_fields():
    """Verify clip_reward_min/max exist in RewardConfig."""
    from rl.config import RewardConfig

    rc = RewardConfig()
    assert hasattr(rc, "clip_reward_min")
    assert hasattr(rc, "clip_reward_max")
    assert rc.clip_reward_min == -100.0
    assert rc.clip_reward_max == 100.0

    # Serialization roundtrip
    d = rc.to_dict()
    assert d["clip_reward_min"] == -100.0
    assert d["clip_reward_max"] == 100.0
    rc2 = RewardConfig.from_dict(d)
    assert rc2.clip_reward_min == -100.0
    assert rc2.clip_reward_max == 100.0


def test_clip_cpp_reward_config():
    """Verify clip fields exist in C++ RewardConfig."""
    rc = colony_cpp.RewardConfig()
    assert rc.clip_reward_min == -100.0
    assert rc.clip_reward_max == 100.0

    # Can set
    rc.clip_reward_min = -5.0
    rc.clip_reward_max = 5.0
    assert rc.clip_reward_min == -5.0
    assert rc.clip_reward_max == 5.0


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_clip_positive_reward():
    """Verify that raw reward is clipped to clip_reward_max."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.clip_reward_min = -1000.0
    cfg.clip_reward_max = 5.0  # very tight clip

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                env.reset(seed=42)
        for r in rewards:
            assert r <= 5.0 + 1e-6, f"Reward {r} exceeds clip_reward_max=5.0"
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_clip_negative_reward():
    """Verify that raw reward is clipped to clip_reward_min."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.clip_reward_min = -3.0
    cfg.clip_reward_max = 1000.0

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        rewards = []
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                env.reset(seed=42)
        for r in rewards:
            assert r >= -3.0 - 1e-6, f"Reward {r} below clip_reward_min=-3.0"
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_clip_both_directions():
    """Verify clip works in both directions simultaneously."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    cfg.clip_reward_min = -2.0
    cfg.clip_reward_max = 2.0

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        for _ in range(30):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            assert -2.0 - 1e-6 <= reward <= 2.0 + 1e-6, \
                f"Reward {reward} outside clip range [-2, 2]"
            if terminated or truncated:
                env.reset(seed=42)
    finally:
        env.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_clip_default_no_effect():
    """With default clip (-10..10), rewards should pass through unclipped for small values."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    cfg = RewardConfig()
    # defaults (v4): clip_reward_min=-100, clip_reward_max=100

    env = CppColonyEnv(map_size=100, reward_config=cfg.to_dict())
    try:
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            # Normal rewards should be within [-100, 100]
            assert -100.0 <= reward <= 100.0, \
                f"Default clip should not affect normal reward, got {reward}"
            if terminated or truncated:
                env.reset(seed=42)
    finally:
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
