"""Tests for curriculum scoping: the allowed-buildings set must be identical in
training, evaluation and watching.

Regression: «Поставил только водоканал. Курикулум 0, чекбокс активен» — but the
champion built прииск (Goldmine), because eval/watch envs were created with
stage 0 and no manual set (all 32 buildings unlocked).
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from rl.curriculum import (  # noqa: E402
    ALL_IDS,
    allowed_ids,
    curriculum_from_meta,
    manual_ids_csv,
    read_curriculum_meta,
    resolve_curriculum,
)


# ── SSOT helpers ────────────────────────────────────────────────────────────

def test_manual_set_at_stage_zero_only_allows_that_set():
    """Курикулум 0 + чекбокс активен + только водоканал → доступен только водоканал."""
    allowed = allowed_ids(0, "WaterChannel", True)

    assert allowed == ["WaterChannel"]
    assert "Goldmine" not in allowed
    assert len(allowed) < len(ALL_IDS)


def test_manual_set_is_merged_with_stage_preset():
    allowed = allowed_ids(1, "Goldmine", True)

    assert "WaterChannel" in allowed  # из пресета этапа 1
    assert "Goldmine" in allowed      # ручной набор
    assert "Sawmill" not in allowed   # этап 2 ещё закрыт


def test_checkbox_off_ignores_manual_set():
    """Чекбокс выключен → работает только этап (0 = все здания)."""
    assert allowed_ids(0, "WaterChannel", False) == list(ALL_IDS)
    assert manual_ids_csv("WaterChannel", False) == ""
    assert manual_ids_csv("WaterChannel", True) == "WaterChannel"


def test_no_manual_set_stage_zero_allows_everything():
    assert allowed_ids(0, "", True) == list(ALL_IDS)


def test_parse_manual_ids_dedupes_and_trims():
    assert manual_ids_csv(" WaterChannel , Road ,WaterChannel ", True) == "WaterChannel,Road"


# ── meta.json round-trip ────────────────────────────────────────────────────

def test_curriculum_from_meta_flat_layout():
    meta = {
        "curriculum_stage_at_best": 0,
        "unlock_ids": "WaterChannel",
        "use_curriculum_tab": True,
    }
    cur = curriculum_from_meta(meta)

    assert cur["curriculum_stage"] == 0
    assert cur["unlock_ids"] == "WaterChannel"
    assert cur["use_curriculum_tab"] is True


def test_curriculum_from_meta_nested_config_layout():
    """Run meta.json keeps the whole Config under "config" (worker layout)."""
    meta = {"config": {"unlock_ids": "WaterChannel", "use_curriculum_tab": True,
                       "curriculum_stage": 0}}

    cur = curriculum_from_meta(meta)

    assert cur["unlock_ids"] == "WaterChannel"
    assert cur["use_curriculum_tab"] is True


def test_curriculum_from_meta_missing_is_none():
    cur = curriculum_from_meta({})

    assert cur == {"curriculum_stage": None, "unlock_ids": None,
                     "use_curriculum_tab": None, "resources": None,
                     "obs_version": None, "enabled_mechanics": None,
                     "disabled_mechanics": None,
                     "mechanics_unlock_schedule": None,
                     "mechanics_step": None}


def test_read_curriculum_meta_merges_both_files(tmp_path):
    """best_model.meta.json may carry only the stage — the manual set lives in the
    run meta.json; both must be merged (else an old model silently unlocks all)."""
    (tmp_path / "best_model.meta.json").write_text(
        json.dumps({"best_score": 1.0, "curriculum_stage_at_best": 0}), encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        json.dumps({"config": {"unlock_ids": "WaterChannel", "use_curriculum_tab": True}}),
        encoding="utf-8")

    cur = read_curriculum_meta(tmp_path)

    assert cur["curriculum_stage"] == 0
    assert cur["unlock_ids"] == "WaterChannel"
    assert cur["use_curriculum_tab"] is True


# ── resolver used by eval / watch ───────────────────────────────────────────

def test_resolve_curriculum_uses_meta_when_args_absent(tmp_path):
    (tmp_path / "best_model.meta.json").write_text(
        json.dumps({"curriculum_stage_at_best": 0, "unlock_ids": "WaterChannel",
                    "use_curriculum_tab": True}),
        encoding="utf-8")

    res = resolve_curriculum(tmp_path)

    assert res["curriculum_stage"] == 0
    assert res["unlock_ids"] == "WaterChannel"
    assert res["allowed"] == ["WaterChannel"]


def test_resolve_curriculum_explicit_args_win(tmp_path):
    (tmp_path / "best_model.meta.json").write_text(
        json.dumps({"curriculum_stage_at_best": 3, "unlock_ids": "Goldmine",
                    "use_curriculum_tab": True}),
        encoding="utf-8")

    res = resolve_curriculum(tmp_path, curriculum_stage=0, unlock_ids="WaterChannel",
                             use_curriculum_tab=True)

    assert res["curriculum_stage"] == 0
    assert res["allowed"] == ["WaterChannel"]


def test_resolve_curriculum_without_meta_or_args_allows_everything(tmp_path):
    res = resolve_curriculum(tmp_path)

    assert res["allowed"] == list(ALL_IDS)


def test_resolve_curriculum_honours_unchecked_tab(tmp_path):
    (tmp_path / "best_model.meta.json").write_text(
        json.dumps({"curriculum_stage_at_best": 0, "unlock_ids": "WaterChannel",
                    "use_curriculum_tab": False}),
        encoding="utf-8")

    res = resolve_curriculum(tmp_path)

    assert res["unlock_ids"] == ""
    assert res["allowed"] == list(ALL_IDS)


# ── evaluator / trainer wiring ──────────────────────────────────────────────

def test_run_eval_signature_accepts_curriculum():
    """run_eval must expose curriculum params so the trainer can pass the same
    scenario it trains on (imports torch → skipped when unavailable)."""
    pytest.importorskip("torch")
    import inspect

    from train_ui2.evaluator import run_eval

    sig = inspect.signature(run_eval)
    for name in ("curriculum_stage", "unlock_ids", "use_curriculum_tab"):
        assert name in sig.parameters, f"run_eval must accept {name}"


def test_run_eval_builds_env_with_restored_curriculum(tmp_path, monkeypatch):
    """The eval env must be created with the curriculum stored in the meta files.

    This is the bug in the report: eval/watch ran with stage 0 and no manual set,
    so a WaterChannel-only champion was evaluated while Goldmine was unlocked.
    """
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    import types

    import numpy as np
    import torch

    created: dict = {}

    class FakeCppEnv:
        def __init__(self, **kw):
            created.update(kw)
            self.observation_space = types.SimpleNamespace(shape=(8,))
            self.action_space = types.SimpleNamespace(n=6)
            self._action_names = ["DAY", "WEEK", "BUILD_FARM", "BUILD_WATERCHANNEL",
                                  "BUILD_GOLDMINE", "PAY_TAX"]
            outer = self

            class Norm:
                _obs_size = 8

                def load(self, *_a, **_k):
                    pass

                def set_update(self, *_a, **_k):
                    pass

            self.normalizer = Norm()

        def reset(self, seed=None, options=None):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(8, dtype=np.float32), 0.0, False, False, {"days": 1, "bases": 1}

        def action_mask(self):
            return np.ones(6, dtype=bool)

        def close(self):
            pass

    fake_cpp_env = types.ModuleType("cpp_env")
    fake_cpp_env.CppColonyEnv = FakeCppEnv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cpp_env", fake_cpp_env)

    from rl.actor_critic import ActorCritic
    from train_ui2 import evaluator as ev

    policy = ActorCritic(8, 6, [8], torch.device("cpu"))
    monkeypatch.setattr(ev, "_load_policy", lambda *a, **k: policy)

    model_path = tmp_path / "best_model.pt"
    model_path.write_bytes(b"stub")
    (tmp_path / "best_model.meta.json").write_text(json.dumps({
        "curriculum_stage_at_best": 0,
        "unlock_ids": "WaterChannel",
        "use_curriculum_tab": True,
        "obs_version": 2,
    }), encoding="utf-8")

    ev.run_eval(model_path, episodes=1, max_days=2, seed=1, normalization_path=None)

    # PR 1: the env takes ONE computed state, not (stage, unlock_ids).
    from rl.curriculum import CurriculumState

    st = created["curriculum"]
    assert isinstance(st, CurriculumState)
    assert st.all_builds is False
    assert st.allowed_builds == ("WaterChannel",)
    assert st.stage_report == 0


def test_run_eval_explicit_curriculum_overrides_meta(tmp_path, monkeypatch):
    """Explicit args (trainer path) must win over anything stored in meta."""
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    import types

    import numpy as np
    import torch

    created: dict = {}

    class FakeCppEnv:
        def __init__(self, **kw):
            created.update(kw)
            self.observation_space = types.SimpleNamespace(shape=(8,))
            self.action_space = types.SimpleNamespace(n=6)
            self._action_names = ["DAY", "WEEK"]

            class Norm:
                _obs_size = 8

                def load(self, *_a, **_k):
                    pass

                def set_update(self, *_a, **_k):
                    pass

            self.normalizer = Norm()

        def reset(self, seed=None, options=None):
            return np.zeros(8, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(8, dtype=np.float32), 0.0, False, False, {"days": 1, "bases": 1}

        def action_mask(self):
            return np.ones(6, dtype=bool)

        def close(self):
            pass

    fake_cpp_env = types.ModuleType("cpp_env")
    fake_cpp_env.CppColonyEnv = FakeCppEnv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cpp_env", fake_cpp_env)

    from rl.actor_critic import ActorCritic
    from train_ui2 import evaluator as ev

    monkeypatch.setattr(ev, "_load_policy",
                        lambda *a, **k: ActorCritic(8, 6, [8], torch.device("cpu")))

    model_path = tmp_path / "best_model.pt"
    model_path.write_bytes(b"stub")
    (tmp_path / "best_model.meta.json").write_text(json.dumps({
        "curriculum_stage_at_best": 3,
        "unlock_ids": "Goldmine",
        "use_curriculum_tab": True,
        "obs_version": 2,
    }), encoding="utf-8")

    ev.run_eval(model_path, episodes=1, max_days=2, seed=1, normalization_path=None,
                curriculum_stage=0, unlock_ids="WaterChannel", use_curriculum_tab=True)

    # PR 1: the env takes ONE computed state, not (stage, unlock_ids).
    from rl.curriculum import CurriculumState

    st = created["curriculum"]
    assert isinstance(st, CurriculumState)
    assert st.all_builds is False
    assert st.allowed_builds == ("WaterChannel",)
    assert st.stage_report == 0


def test_trainer_passes_curriculum_to_eval():
    """AsyncTrainer must forward the live curriculum to run_eval."""
    pytest.importorskip("torch")
    import inspect

    from rl.async_trainer import AsyncTrainer

    src = inspect.getsource(AsyncTrainer)
    assert "_curriculum_kwargs" in src
    # Канонический вызов — **self._curriculum_kwargs(); в турнире kwargs
    # сначала копируются в eval_curriculum (чтобы подмешать стадию из meta
    # чекпоинта) и разворачиваются как **eval_curriculum. Оба call-site
    # обязаны передавать курикулум.
    n_calls = src.count("**self._curriculum_kwargs()") + src.count("**eval_curriculum")
    assert n_calls >= 2, \
        "both eval and tournament run_eval calls must pass the curriculum"


def test_config_effective_unlock_ids():
    from rl.config import Config

    on = Config(unlock_ids="WaterChannel", use_curriculum_tab=True)
    off = Config(unlock_ids="WaterChannel", use_curriculum_tab=False)

    assert on.effective_unlock_ids() == "WaterChannel"
    assert off.effective_unlock_ids() == ""
    assert on.curriculum_meta()["unlock_ids"] == "WaterChannel"
    assert on.curriculum_meta()["curriculum_stage_at_best"] == 0


def test_config_effective_unlock_ids_at_stage_zero_restricts():
    """End-to-end: курикулум 0 + чекбокс ON + водоканал → в среде только водоканал."""
    from rl.config import Config

    cfg = Config(curriculum_stage=0, unlock_ids="WaterChannel", use_curriculum_tab=True)
    allowed = allowed_ids(cfg.curriculum_stage, cfg.effective_unlock_ids(), True)

    assert allowed == ["WaterChannel"]


# ── watch_champion action mask ──────────────────────────────────────────────

def test_watch_curriculum_action_mask_locks_foreign_buildings():
    pytest.importorskip("torch")
    pytest.importorskip("colony_cpp")
    from watch_champion import curriculum_action_mask

    names = ["DAY", "WEEK", "BUILD_FARM", "BUILD_WATERCHANNEL", "BUILD_GOLDMINE", "PAY_TAX"]
    mask = curriculum_action_mask(["WaterChannel"], names)

    assert mask is not None
    assert mask[names.index("BUILD_WATERCHANNEL")] == 1.0
    assert mask[names.index("BUILD_GOLDMINE")] == 0.0
    assert mask[names.index("BUILD_FARM")] == 0.0
    assert mask[names.index("PAY_TAX")] == 1.0  # manager actions stay available


def test_watch_curriculum_action_mask_none_when_unrestricted():
    pytest.importorskip("torch")
    pytest.importorskip("colony_cpp")
    from watch_champion import curriculum_action_mask

    assert curriculum_action_mask(list(ALL_IDS), ["DAY", "BUILD_GOLDMINE"]) is None
