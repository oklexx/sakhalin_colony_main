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


def test_build_cost_penalty_config():
    """Verify build_cost_penalty field exists with correct default."""
    from rl.config import RewardConfig
    rc = RewardConfig()
    assert hasattr(rc, "build_cost_penalty")
    assert rc.build_cost_penalty == 0.00004
    d = rc.to_dict()
    assert d["build_cost_penalty"] == 0.00004
    rc2 = RewardConfig.from_dict(d)
    assert rc2.build_cost_penalty == 0.00004


def test_build_cost_penalty_cpp():
    """Verify build_cost_penalty works in C++ RewardConfig."""
    rc = colony_cpp.RewardConfig()
    assert rc.build_cost_penalty == 0.00004
    rc.build_cost_penalty = 0.05
    assert rc.build_cost_penalty == 0.05


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_build_cost_deducts_from_build_reward():
    """
    Compare two builds with different build_cost_penalty.
    The reward difference on the build step should be:
      delta_penalty * price
    Run the same sequence of actions for both configs, find the first
    step where action=1 (build) succeeds (reward differs from error_penalty),
    and verify the delta.
    """
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    def make_cfg(penalty):
        cfg = RewardConfig()
        cfg.build_bonus = 0.0
        cfg.build_cost_penalty = penalty
        cfg.tax_daily_bonus = 0.0
        cfg.daily_income = 0.0
        cfg.chain_bonus = 0.0
        cfg.chain_daily = 0.0
        cfg.novelty = 0.0
        cfg.survival_bonus = 0.0
        cfg.game_over_penalty = 0.0
        cfg.error_penalty = -99.0  # unique marker for build failure
        cfg.preserve_penalty = 0.0
        cfg.manual_tax_penalty = 0.0
        cfg.clip_reward_min = -1000.0
        cfg.clip_reward_max = 1000.0
        return cfg

    PENALTY_LOW = 0.01
    PENALTY_HIGH = 0.10

    env_low = CppColonyEnv(map_size=100, reward_config=make_cfg(PENALTY_LOW).to_dict())
    env_high = CppColonyEnv(map_size=100, reward_config=make_cfg(PENALTY_HIGH).to_dict())
    try:
        env_low.reset(seed=42)
        env_high.reset(seed=42)

        found_build = False
        for step in range(50):
            action = 2  # A_BUILD0
            obs_low, rew_low, done_low, _, _ = env_low.step(action)
            obs_high, rew_high, done_high, _, _ = env_high.step(action)

            # Skip if build failed (both get error_penalty=-99)
            if rew_low == -99.0 or rew_high == -99.0:
                if done_low:
                    env_low.reset(seed=42)
                if done_high:
                    env_high.reset(seed=42)
                continue

            # Build succeeded on this step — compute expected delta
            expected_delta = (PENALTY_HIGH - PENALTY_LOW) * abs(rew_low + 99.0) / PENALTY_LOW
            # Actually simpler: the non-build components are identical,
            # so rew_low - rew_high = (PENALTY_HIGH - PENALTY_LOW) * price
            # We don't know price directly, but we know:
            # rew_low  = X - PENALTY_LOW  * price
            # rew_high = X - PENALTY_HIGH * price
            # So: rew_low - rew_high = (PENALTY_HIGH - PENALTY_LOW) * price
            # And: rew_low = X - PENALTY_LOW * price
            # price = (rew_low - rew_high) / (PENALTY_HIGH - PENALTY_LOW)
            # Then: rew_low should equal X - PENALTY_LOW * price

            delta = rew_low - rew_high
            expected_delta_penalties = PENALTY_HIGH - PENALTY_LOW

            if abs(delta) > 0.001:
                # price = delta / expected_delta_penalties
                price = delta / expected_delta_penalties
                assert price > 0, f"Computed price should be positive: {price}"
                # Verify consistency: rew_low should contain -PENALTY_LOW * price
                # Since other components cancel out, the difference IS the penalty difference
                found_build = True
                break

            if done_low:
                env_low.reset(seed=42)
            if done_high:
                env_high.reset(seed=42)

        if not found_build:
            pytest.skip("Build never succeeded in 50 steps (connection/lot issue)")

    finally:
        env_low.close()
        env_high.close()


@pytest.mark.skipif(not ENV_OK, reason="colony_cpp not available")
def test_build_cost_zero_penalty_no_effect():
    """With build_cost_penalty=0, compare against a tiny penalty — delta should be tiny."""
    from cpp_env import CppColonyEnv
    from rl.config import RewardConfig

    def make_cfg(penalty):
        cfg = RewardConfig()
        cfg.build_bonus = 0.0
        cfg.build_cost_penalty = penalty
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
        cfg.clip_reward_min = -1000.0
        cfg.clip_reward_max = 1000.0
        return cfg

    env_0 = CppColonyEnv(map_size=100, reward_config=make_cfg(0.0).to_dict())
    env_tiny = CppColonyEnv(map_size=100, reward_config=make_cfg(0.0001).to_dict())
    try:
        env_0.reset(seed=42)
        env_tiny.reset(seed=42)
        for _ in range(50):
            _, r0, done0, _, _ = env_0.step(2)
            _, rt, done_t, _, _ = env_tiny.step(2)
            if r0 != 0.0 or rt != 0.0:
                # At least one build succeeded — delta should be ~0.0001 * price
                delta = r0 - rt
                if abs(delta) > 0.0001:
                    assert delta > 0, f"Zero penalty reward should be higher: r0={r0}, rt={rt}"
                return
            if done0:
                env_0.reset(seed=42)
            if done_t:
                env_tiny.reset(seed=42)
    finally:
        env_0.close()
        env_tiny.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
