#!/usr/bin/env python3
"""Simple test for LoopDetector."""

from rl.loop_detector import LoopDetector


def test_basic() -> None:
    """Test basic initialization."""
    detector = LoopDetector()
    assert detector.consecutive_threshold == 3
    print("✓ Basic init passed")


def test_update_batch() -> None:
    """Test batch update with loop detection."""
    detector = LoopDetector({"consecutive_threshold": 3})

    # Simulate 2 environments doing the same action repeatedly
    action_data = [
        {"env_idx": 0, "action": "MOVE_RIGHT"},
        {"env_idx": 0, "action": "MOVE_RIGHT"},
        {"env_idx": 0, "action": "MOVE_RIGHT"},  # Should trigger alert now
        {"env_idx": 1, "action": "BUILD_HOUSE"},
        {"env_idx": 1, "action": "BUILD_HOUSE"},
        {"env_idx": 1, "action": "BUILD_HOUSE"},  # Should trigger alert
    ]

    alerts = detector.update_batch(action_data)

    assert 0 in alerts, "Env 0 should have alert"
    assert alerts[0] == "MOVE_RIGHT", f"Expected MOVE_RIGHT, got {alerts[0]}"
    assert 1 in alerts, "Env 1 should have alert"
    assert alerts[1] == "BUILD_HOUSE", f"Expected BUILD_HOUSE, got {alerts[1]}"

    print("✓ Update batch with loops passed")


def test_no_loops() -> None:
    """Test when no loops detected."""
    detector = LoopDetector({"consecutive_threshold": 3})

    # Different actions each time
    action_data = [
        {"env_idx": 0, "action": "MOVE_RIGHT"},
        {"env_idx": 0, "action": "BUILD_HOUSE"},
        {"env_idx": 0, "action": "GATHER_WOOD"},
    ]

    alerts = detector.update_batch(action_data)
    assert len(alerts) == 0, "No loops should be detected"

    print("✓ No loops detection passed")


def test_stats() -> None:
    """Test statistics calculation."""
    detector = LoopDetector({"consecutive_threshold": 3})

    # Add some activity
    for i in range(10):
        detector.update_batch([
            {"env_idx": j, "action": f"ACTION_{i % 3}", "step": i}
            for j in range(4)
        ])

    stats = detector.get_stats()

    assert "total_envs" in stats
    assert "envs_with_loops" in stats
    assert "max_consecutive" in stats
    assert "avg_consecutive" in stats
    assert "loop_rate_percent" in stats

    print(f"✓ Stats calculation passed: {stats}")


def test_threshold_change() -> None:
    """Test changing threshold."""
    detector = LoopDetector({"consecutive_threshold": 3})

    # Add some data with 3 consecutive actions
    action_data = [
        {"env_idx": 0, "action": "MOVE_RIGHT"},
        {"env_idx": 0, "action": "MOVE_RIGHT"},
        {"env_idx": 0, "action": "MOVE_RIGHT"},
    ]

    alerts1 = detector.update_batch(action_data)
    assert len(alerts1) == 1, "Should detect loop with threshold=3"

    # Change threshold to 2
    detector.consecutive_threshold = 2

    # Same data should now have MORE consecutive (re-evaluated)
    alerts2 = detector.update_batch([])

    print(f"✓ Threshold change passed: {len(alerts1)} -> {len(alerts2)} alerts")


if __name__ == "__main__":
    test_basic()
    test_update_batch()
    test_no_loops()
    test_stats()
    test_threshold_change()
    print("\nAll tests passed!")
