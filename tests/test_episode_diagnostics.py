"""Pure-Python contract tests for the persistent per-episode JSONL artifact."""
from __future__ import annotations

import json
from pathlib import Path

from rl.episode_diagnostics import EpisodeDiagnosticsWriter


def test_episode_diagnostics_jsonl_is_append_only_and_analysis_ready(tmp_path: Path):
    path = tmp_path / "run" / "episode_diagnostics.jsonl"
    writer = EpisodeDiagnosticsWriter(
        path,
        run_id="test-run",
        fsync=False,
        metadata={"obs_mode": "hybrid", "seed": 41},
    )
    assert not writer.append_episode({}, total_timesteps=10, env_index=0,
                                     curriculum_stage=1)
    info = {
        "episode": {
            "r": 12.375,
            "l": 640,
            "seed": 41007,
            "days": 271,
            "people": 49,
            "money": 12345,
            "bases": 8,
            "metrics": {
                "total_reward": 12.375,
                "chains_activated": 3,
                "max_chain_depth": 2,
                "total_builds": 9,
                "unique_build_types": 4,
                "builds_by_type": {"Водоканал": 1, "Farm": 2},
                "reached_resources": 3,
                "reached_resource_ids": ["food", "water", "wood"],
                "priority_reached": 2,
            },
        }
    }
    assert writer.append_episode(info, total_timesteps=1280, env_index=1,
                                 curriculum_stage=2)
    writer.close(total_timesteps=1280, status="completed")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["record_type"] for record in records] == [
        "run_start", "episode", "run_end"
    ]
    start, episode, end = records
    assert start["schema_version"] == 1 and start["run_id"] == "test-run"
    assert start["metadata"] == {"obs_mode": "hybrid", "seed": 41}
    assert episode["total_timesteps"] == 1280
    assert episode["env_index"] == 1 and episode["curriculum_stage"] == 2
    assert episode["episode_return"] == 12.375
    assert episode["episode_length"] == 640
    assert episode["seed"] == 41007
    assert episode["final_state"] == {
        "days": 271, "people": 49, "money": 12345, "bases": 8
    }
    metrics = episode["episode_metrics"]
    assert metrics["chains_activated"] == 3
    assert metrics["max_chain_depth"] == 2
    assert metrics["builds_by_type"] == {"Farm": 2, "Водоканал": 1}
    assert metrics["reached_resource_ids"] == ["food", "water", "wood"]
    assert episode["metrics_available"] is True
    assert end["total_timesteps"] == 1280 and end["episodes_logged"] == 1


def test_episode_without_native_metrics_is_explicitly_marked(tmp_path: Path):
    writer = EpisodeDiagnosticsWriter(tmp_path / "episodes.jsonl", fsync=False)
    assert writer.append_episode(
        {"episode": {"r": 1.0, "l": 4}},
        total_timesteps=4,
        env_index=0,
        curriculum_stage=0,
    )
    row = [json.loads(line) for line in
           (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()][1]
    assert row["metrics_available"] is False
    assert row["episode_metrics"] is None
