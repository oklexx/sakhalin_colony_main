"""Tests for rl/bc_dataset.py (--record-demo JSONL loader).

Pure-Python/numpy: does not need the compiled colony_cpp extension, so these
run in any environment (unlike most tests/test_*.py, which skip when
colony_cpp is unavailable).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from rl.bc_dataset import (
    BCDataset,
    compute_normalization,
    expand_demo_paths,
    infer_obs_version,
    load_demo_file,
    load_demo_files,
    split_by_episode,
    validate_consistent_dims,
)

OBS_SIZE = 6
N_ACTIONS = 4


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _row(action=0, reward=0.0, terminated=False, obs=None, mask=None):
    return {
        "action": action,
        "reward": reward,
        "terminated": terminated,
        "obs": obs if obs is not None else [0.1 * i for i in range(OBS_SIZE)],
        "action_mask": mask if mask is not None else [1.0] * N_ACTIONS,
    }


# ── infer_obs_version ────────────────────────────────────────────────────

def test_infer_obs_version_known_sizes():
    assert infer_obs_version(248) == 0
    assert infer_obs_version(289) == 1
    assert infer_obs_version(299) == 2


def test_infer_obs_version_unknown_raises():
    with pytest.raises(ValueError, match="unrecognized obs_size"):
        infer_obs_version(123)


# ── load_demo_file ────────────────────────────────────────────────────────

def test_load_demo_file_basic(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [
        _row(action=1, reward=1.0, terminated=False),
        _row(action=2, reward=2.0, terminated=True),
        _row(action=0, reward=0.5, terminated=False),
    ])
    steps = load_demo_file(path)
    assert len(steps) == 3
    assert [s.action for s in steps] == [1, 2, 0]
    # First two lines belong to episode 0 (terminated on line 2); the
    # trailing non-terminated line starts episode 1.
    assert [s.episode for s in steps] == [0, 0, 1]
    assert steps[0].obs.shape == (OBS_SIZE,)
    assert steps[0].action_mask.shape == (N_ACTIONS,)
    assert steps[0].source == str(path)


def test_load_demo_file_skips_blank_lines(tmp_path):
    path = tmp_path / "demo.jsonl"
    path.write_text(json.dumps(_row(action=1)) + "\n\n\n" + json.dumps(_row(action=2)) + "\n")
    steps = load_demo_file(path)
    assert len(steps) == 2


def test_load_demo_file_missing_field_raises(tmp_path):
    path = tmp_path / "demo.jsonl"
    bad = _row(action=1)
    del bad["reward"]
    _write_jsonl(path, [bad])
    with pytest.raises(ValueError, match="missing field"):
        load_demo_file(path)


def test_load_demo_file_malformed_json_raises(tmp_path):
    path = tmp_path / "demo.jsonl"
    path.write_text("{not valid json\n")
    with pytest.raises(ValueError, match="malformed demo JSONL line"):
        load_demo_file(path)


def test_load_demo_file_empty(tmp_path):
    path = tmp_path / "demo.jsonl"
    path.write_text("")
    assert load_demo_file(path) == []


# ── load_demo_files (multi-file episode offset) ───────────────────────────

def test_load_demo_files_renumbers_episodes_across_files(tmp_path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    _write_jsonl(p1, [_row(action=0, terminated=False), _row(action=1, terminated=True)])
    _write_jsonl(p2, [_row(action=2, terminated=False), _row(action=3, terminated=True)])
    steps = load_demo_files([p1, p2])
    assert len(steps) == 4
    # File a: episode 0 (both lines); file b starts at episode 1, not 0 again.
    assert [s.episode for s in steps] == [0, 0, 1, 1]


def test_load_demo_files_skips_empty_file(tmp_path):
    p1 = tmp_path / "empty.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text("")
    _write_jsonl(p2, [_row(action=1, terminated=True)])
    steps = load_demo_files([p1, p2])
    assert len(steps) == 1
    assert steps[0].episode == 0


# ── expand_demo_paths ──────────────────────────────────────────────────────

def test_expand_demo_paths_directory(tmp_path):
    (tmp_path / "b.jsonl").write_text(json.dumps(_row()) + "\n")
    (tmp_path / "a.jsonl").write_text(json.dumps(_row()) + "\n")
    (tmp_path / "ignore.txt").write_text("nope")
    found = expand_demo_paths([str(tmp_path)])
    assert found == sorted(found)
    assert all(f.endswith(".jsonl") for f in found)
    assert len(found) == 2


def test_expand_demo_paths_empty_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="no \\*.jsonl demo files"):
        expand_demo_paths([str(empty_dir)])


def test_expand_demo_paths_no_match_raises(tmp_path):
    with pytest.raises(ValueError, match="no files match"):
        expand_demo_paths([str(tmp_path / "nothing_here_*.jsonl")])


def test_expand_demo_paths_explicit_file(tmp_path):
    f = tmp_path / "demo.jsonl"
    f.write_text(json.dumps(_row()) + "\n")
    assert expand_demo_paths([str(f)]) == [str(f)]


# ── validate_consistent_dims ────────────────────────────────────────────────

def test_validate_consistent_dims_ok(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [_row(action=0), _row(action=3, terminated=True)])
    steps = load_demo_file(path)
    obs_size, n_actions = validate_consistent_dims(steps)
    assert obs_size == OBS_SIZE
    assert n_actions == N_ACTIONS


def test_validate_consistent_dims_empty_raises():
    with pytest.raises(ValueError, match="empty dataset"):
        validate_consistent_dims([])


def test_validate_consistent_dims_mismatched_obs_size_raises(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [
        _row(obs=[0.0] * OBS_SIZE),
        _row(obs=[0.0] * (OBS_SIZE + 1)),
    ])
    steps = load_demo_file(path)
    with pytest.raises(ValueError, match="obs_size"):
        validate_consistent_dims(steps)


def test_validate_consistent_dims_mismatched_action_mask_raises(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [
        _row(mask=[1.0] * N_ACTIONS),
        _row(mask=[1.0] * (N_ACTIONS + 1)),
    ])
    steps = load_demo_file(path)
    with pytest.raises(ValueError, match="n_actions"):
        validate_consistent_dims(steps)


def test_validate_consistent_dims_out_of_range_action_raises(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [_row(action=N_ACTIONS)])  # out of [0, N_ACTIONS)
    steps = load_demo_file(path)
    with pytest.raises(ValueError, match="out of range"):
        validate_consistent_dims(steps)


# ── compute_normalization ──────────────────────────────────────────────────

def test_compute_normalization_schema_and_values(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [
        _row(obs=[0.0, 0.0]),
        _row(obs=[2.0, 4.0]),
    ])
    steps = load_demo_file(path)
    for s in steps:
        s.obs = s.obs[:2]  # shrink to a simple 2-dim case for this test
    norm = compute_normalization(steps, clip=5.0)
    assert set(norm.keys()) == {"mean", "var", "count", "obs_size", "clip"}
    np.testing.assert_allclose(norm["mean"], [1.0, 2.0])
    np.testing.assert_allclose(norm["var"], [1.0, 4.0])
    assert norm["count"] == 2
    assert norm["obs_size"] == 2
    assert norm["clip"] == 5.0


# ── split_by_episode ─────────────────────────────────────────────────────

def test_split_by_episode_respects_val_frac(tmp_path):
    path = tmp_path / "demo.jsonl"
    rows = []
    for ep in range(10):
        rows.append(_row(action=ep % N_ACTIONS, terminated=True))
    _write_jsonl(path, rows)
    steps = load_demo_file(path)
    assert len({s.episode for s in steps}) == 10
    train, val = split_by_episode(steps, val_frac=0.2, seed=0)
    train_eps = {s.episode for s in train}
    val_eps = {s.episode for s in val}
    assert len(val_eps) == 2
    assert train_eps.isdisjoint(val_eps)
    assert train_eps | val_eps == {s.episode for s in steps}


def test_split_by_episode_zero_val_frac_returns_all_as_train(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [_row(terminated=True), _row(terminated=True)])
    steps = load_demo_file(path)
    train, val = split_by_episode(steps, val_frac=0.0)
    assert len(train) == len(steps)
    assert val == []


def test_split_by_episode_single_episode_no_split(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [_row(terminated=False), _row(terminated=True)])
    steps = load_demo_file(path)
    assert len({s.episode for s in steps}) == 1
    train, val = split_by_episode(steps, val_frac=0.5)
    assert len(train) == len(steps)
    assert val == []


def test_split_by_episode_deterministic_with_seed(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [_row(terminated=True) for _ in range(20)])
    steps = load_demo_file(path)
    t1, v1 = split_by_episode(steps, val_frac=0.3, seed=42)
    t2, v2 = split_by_episode(steps, val_frac=0.3, seed=42)
    assert {s.episode for s in v1} == {s.episode for s in v2}


# ── BCDataset ──────────────────────────────────────────────────────────────

def test_bc_dataset_len_and_getitem(tmp_path):
    path = tmp_path / "demo.jsonl"
    _write_jsonl(path, [_row(action=1), _row(action=2, terminated=True)])
    steps = load_demo_file(path)
    ds = BCDataset(steps)
    assert len(ds) == 2
    obs, action, mask = ds[0]
    assert obs.shape == (OBS_SIZE,)
    assert int(action) == 1
    assert mask.shape == (N_ACTIONS,)
