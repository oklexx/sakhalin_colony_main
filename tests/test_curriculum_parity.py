"""PR 1: all env-creation paths must apply the identical curriculum.

Covers the 4 paths from the plan: train (CppVecEnv), eval (CppColonyEnv),
watch text (CppColonyEnv) and watch visual (JSON transport to the GUI exe —
here simulated through the transport dict, since the exe needs Windows; the
C++ from_json half is covered by tests/cpp/curriculum_check.cpp).

Needs the compiled extension + gymnasium/torch (importorskip otherwise).
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("colony_cpp")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from rl.curriculum import build_state  # noqa: E402

_MAP = 64  # small island: parity checks don't need 280x280


def _reported(env_cpp) -> dict:
    """Normalise C++ curriculum() for comparison (flag + sorted ids)."""
    got = env_cpp.curriculum()
    return {
        "all_builds": bool(got["all_builds"]),
        "allowed_builds": sorted(got.get("allowed_builds", [])),
    }


def _train_path(st):
    """Train path: batched vec env (as EnvManager._make_vec_env builds it)."""
    pytest.importorskip("stable_baselines3")
    from cpp_vecenv import CppVecEnv

    venv = CppVecEnv(n_envs=2, map_size=_MAP, curriculum=st)
    try:
        return _reported(venv.venv)
    finally:
        venv.close()


def _single_path(st):
    """Eval / watch-text path: single env (evaluator.run_eval, watch_champion)."""
    from cpp_env import CppColonyEnv

    env = CppColonyEnv(map_size=_MAP, curriculum=st)
    try:
        return _reported(env.cpp_env)
    finally:
        env.close()


def _visual_path(st):
    """Watch-visual path: state -> JSON -> transport dict -> env.

    Mirrors watch_champion.launch_visual_watch (json.dumps(state.to_dict()))
    plus the GUI's --curriculum parsing (C++ from_json, covered by the C++
    harness) by feeding the parsed transport dict back into an env.
    """
    from cpp_env import CppColonyEnv

    payload = json.loads(json.dumps(st.to_dict()))
    env = CppColonyEnv(map_size=_MAP, curriculum=payload)
    try:
        return _reported(env.cpp_env)
    finally:
        env.close()


@pytest.mark.parametrize("make_state", [
    lambda: build_state(0, "WaterChannel", True),   # restricted: 1 building
    lambda: build_state(1, "Goldmine", True),       # restricted: preset + manual
    lambda: build_state(0, None, True),             # unrestricted
    lambda: build_state(2, None, False),            # restricted: preset only
])
def test_all_paths_report_identical_curriculum(make_state):
    st = make_state()
    train, single, visual = _train_path(st), _single_path(st), _visual_path(st)
    assert train == single == visual
    if st.all_builds:
        assert train["all_builds"] is True
    else:
        assert train["all_builds"] is False
        assert train["allowed_builds"] == sorted(st.allowed_builds)


def test_locked_building_masked_on_every_path():
    """A restricted state must mask foreign BUILD actions in vec and single envs."""
    pytest.importorskip("stable_baselines3")
    import numpy as np

    from cpp_env import CppColonyEnv
    from cpp_vecenv import CppVecEnv

    st = build_state(0, "WaterChannel", True)
    venv = CppVecEnv(n_envs=2, map_size=_MAP, seed=7, curriculum=st)
    try:
        venv.reset()
        vec_masks = np.asarray(venv.action_masks)
    finally:
        venv.close()
    env = CppColonyEnv(map_size=_MAP, curriculum=st)
    try:
        env.reset(seed=7)
        mask = np.asarray(env.action_mask(), dtype=bool)
        names = list(env._action_names)
    finally:
        env.close()

    # NOTE: CppColonyEnv._action_names keeps raw build ids (no BUILD_ prefix).
    gold = names.index("Goldmine")
    assert vec_masks.shape == (2, len(names))
    assert not vec_masks[:, gold].any(), "Goldmine must be masked in the vec env"
    assert not mask[gold], "Goldmine must be masked in the single env"


def test_env_manager_parity_assert_fires_on_mismatch():
    """EnvManager must reject an env whose curriculum() disagrees (live check)."""
    pytest.importorskip("torch")
    import torch

    from rl.config import Config
    from rl.curriculum import CurriculumState
    from rl.env_manager import EnvManager

    cfg = Config(n_envs=2, map_size=_MAP, curriculum_stage=0,
                 unlock_ids="WaterChannel", use_curriculum_tab=True)
    em = EnvManager(cfg, torch.device("cpu"))
    try:
        # construction above already passed the parity assert (matching state);
        # a fabricated mismatch must raise loudly.
        bad = CurriculumState(False, ("Goldmine",), True, (1.0,) * 9, 0)
        with pytest.raises(AssertionError, match="курикулум не применён"):
            em._assert_curriculum_parity(bad)
    finally:
        em.close()
