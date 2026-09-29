from __future__ import annotations

import json

from train_ui2.demo import aggregate_demo_stats, copy_demo_files, inspect_demo_file, scan_demo_dir


def _row(action: int = 0, terminated: bool = False, obs_size: int = 3, n_actions: int = 4):
    return {
        "obs": [0.0] * obs_size,
        "action": action,
        "action_mask": [1.0] * n_actions,
        "reward": 0.5,
        "terminated": terminated,
    }


def test_inspect_demo_file_and_aggregate(tmp_path):
    path = tmp_path / "human_01.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in [_row(), _row(1, True)]) + "\n",
        encoding="utf-8",
    )

    stat = inspect_demo_file(path)
    assert stat.valid
    assert stat.transitions == 2
    assert stat.episodes == 1
    assert stat.obs_size == 3
    assert stat.n_actions == 4
    assert aggregate_demo_stats([stat])["consistent"] is True


def test_inspect_demo_file_reports_dimension_and_action_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = [_row(), _row(5, False, obs_size=4, n_actions=5)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    stat = inspect_demo_file(path)
    assert not stat.valid
    assert any("вне mask" in error for error in stat.errors)
    assert any("obs имеет размер" in error for error in stat.errors)
    assert any("mask имеет размер" in error for error in stat.errors)


def test_scan_and_copy_demo_files(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    target = tmp_path / "demos"

    copied = copy_demo_files([source], target)
    assert copied == [target / "source.jsonl"]
    assert len(scan_demo_dir(target)) == 1
