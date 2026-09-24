from __future__ import annotations

from pathlib import Path
from rl.config import RewardConfig, Config


def test_reward_config_completeness():
    import json

    cfg = RewardConfig()
    d = cfg.to_dict()
    # Золотое правило (RULES.md): дефолты dataclass == канонический профиль наград,
    # сейчас configs/reward_v4.json. Точное число полей здесь не хардкодим — оно
    # растёт с каждым новым полем (было 47, стало 55 после road-shaping P2-9 и
    # двух C++-абляций), а само расхождение ловит эта проверка и
    # tests/test_reward_field_sync.py (он же сверяет C++ RewardConfig и биндинги).
    profile_path = Path(__file__).resolve().parent.parent / "configs" / "reward_v4.json"
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = {k: v for k, v in raw.items() if not k.startswith("_")}
    assert set(d) == set(profile), (
        f"rl.config.RewardConfig и {profile_path.name} разошлись: "
        f"только в dataclass {sorted(set(d) - set(profile))}, "
        f"только в профиле {sorted(set(profile) - set(d))}")
    for k, v in profile.items():
        assert d[k] == v, f"дефолт {k}: dataclass={d[k]!r}, профиль={v!r}"


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
