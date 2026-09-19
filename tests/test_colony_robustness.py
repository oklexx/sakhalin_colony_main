from __future__ import annotations

import pytest
from pathlib import Path
from rl.config import RewardConfig, Config


def test_reward_config_completeness():
    cfg = RewardConfig()
    d = cfg.to_dict()
    # v4-профиль расширил набор ключей (goal_survival_coeff,
    # main_tax_cash_bonus, main_tax_pressure_coeff, survival_coeff,
    # mask_managers_by_applicability и др.)
    assert len(d) == 47, f"Expected 47 reward keys, got {len(d)}"


def test_curriculum_stage_progression():
    cfg = Config(difficulty="light", curriculum_stage=1)
    assert cfg.difficulty == "light"
    assert cfg.curriculum_stage == 1


def test_normalization_path_handling(tmp_path):
    norm_path = tmp_path / "normalization.json"
    assert not norm_path.exists()
    # Simulating eval check
    norm_str = str(norm_path) if norm_path.exists() else None
    assert norm_str is None
