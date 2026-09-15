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
        "obs_version": int(got.get("obs_version", -1)),
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
    lambda: build_state(0, None, True, None, 0),      # PR 5: v0 layout
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


# ── PR 6: gated step = pure refusal, degenerate warning ────────────────────

def _single_env(st, map_size=_MAP):
    from cpp_env import CppColonyEnv

    return CppColonyEnv(map_size=map_size, curriculum=st)


def test_gated_step_is_pure_refusal():
    """Locked BUILD: exactly error_penalty, no day passes, nothing built."""
    st = build_state(0, "WaterChannel", True)
    assert not st.all_builds and st.allowed_builds == ("WaterChannel",)
    env = _single_env(st)
    try:
        env.reset(seed=11)
        ep = float(env.cpp_env.reward_config().error_penalty)
        house = 2 + list(env.cpp_env.build_ids()).index("House")
        assert not env.action_mask()[house], "House must be masked (locked)"

        _, _, terminated, truncated, info_day = env.step(0)  # DAY passes
        assert not (terminated or truncated)
        days_after_day = info_day["days"]

        _, reward, terminated, truncated, info = env.step(house)
        assert not (terminated or truncated)
        assert reward == ep, f"gated step must cost exactly {ep}, got {reward}"
        assert info["days"] == days_after_day, "gated step advances no days"
        assert info["bases"] == 1, "gated step builds nothing"
    finally:
        env.close()


def test_degenerate_warning_on_reset(capfd):
    """Seed 7 + WaterChannel-only (280): reset warns with the lot reason."""
    st = build_state(0, "WaterChannel", True)
    env = _single_env(st, map_size=280)
    try:
        env.reset(seed=7)
        out, _ = capfd.readouterr()
        assert "вырожденный сценарий" in out
        assert "WaterChannel" in out and "лота" in out
    finally:
        env.close()


def test_no_degenerate_warning_when_healthy(capfd):
    """Unrestricted scenario: reset stays silent about degeneracy."""
    st = build_state(0, None, False)
    assert st.all_builds
    env = _single_env(st)
    try:
        env.reset(seed=42)
        out, _ = capfd.readouterr()
        assert "вырожденный сценарий" not in out
    finally:
        env.close()


# ── PR 4: resource weights through the stack ──────────────────────────────

def test_env_manager_resource_parity():
    """EnvManager accepts matching weights and rejects mismatched ones."""
    pytest.importorskip("torch")
    import torch

    from rl.config import Config
    from rl.curriculum import ALL_IDS, CurriculumState
    from rl.env_manager import EnvManager

    cfg = Config(n_envs=2, map_size=_MAP, curriculum_stage=0,
                 unlock_ids="", use_curriculum_tab=False,
                 curriculum_resources="water,wood")
    em = EnvManager(cfg, torch.device("cpu"))
    try:
        bad = CurriculumState(True, tuple(ALL_IDS), False,
                              (0.0,) * 8 + (1.0,), 0)  # energy-only ≠ water,wood
        with pytest.raises(AssertionError, match="курикулум не применён"):
            em._assert_curriculum_parity(bad)
    finally:
        em.close()


def test_resource_weights_transport_round_trip():
    """to_dict → C++ → curriculum(): weights survive exactly (doubles)."""
    st = build_state(0, None, False, "water,wood")
    env = _single_env(st)
    try:
        env.reset(seed=11)
        got = env.cpp_env.curriculum()
        assert got["all_resources"] is False
        assert list(got["resource_weights"]) == [0.0] * 6 + [1.0, 1.0, 0.0]
        # live re-apply mid-episode (the schedule path) works too
        env.cpp_env.set_curriculum(build_state(0, None, False, "energy").to_dict())
        got2 = env.cpp_env.curriculum()
        assert got2["resource_weights"][8] == 1.0
        assert sum(got2["resource_weights"]) == 1.0
    finally:
        env.close()


def _autumn_metrics(st, seed=21, days=250):
    """Run AirStation into autumn; return (reached, priority) via raw dicts."""
    env = _single_env(st)
    try:
        env.reset(seed=seed)
        air = 2 + list(env.cpp_env.build_ids()).index("AirStation")
        env.cpp_env.step(air)
        res = {}
        for _ in range(days):
            res = env.cpp_env.step(0)  # DAY (raw: skip obs normalization)
        m = res["metrics"]
        return int(m.reached_resources), int(m.priority_reached)
    finally:
        env.close()


def test_priority_metrics_end_to_end():
    """reached counts every extracted resource; priority only w>0 ones."""
    reached, prio = _autumn_metrics(build_state(0, None, False, "water"))
    assert reached > 0, "energy must be extracted by autumn"
    assert prio == 0, "energy at weight 0 is reached but not priority"
    reached_e, prio_e = _autumn_metrics(build_state(0, None, False, "energy"))
    assert prio_e == reached_e > 0, "energy at weight 1 is priority"


# ── PR 5: obs layout on the real env ─────────────────────────────────────

@pytest.mark.parametrize("obs_version,expected", [(1, 289), (0, 248)])
def test_obs_layout_sizes(obs_version, expected):
    st = build_state(0, None, True, None, obs_version)
    env = _single_env(st)
    try:
        obs, _ = env.reset(seed=7)
        assert len(obs) == expected == env.cpp_env.obs_size()
        assert int(env.cpp_env.curriculum()["obs_version"]) == obs_version
    finally:
        env.close()


def test_v1_frame_visible_in_obs():
    st = build_state(0, "WaterChannel", True, "water", 1)
    env = _single_env(st)
    try:
        obs, _ = env.reset(seed=7)
        assert len(obs) == 289
        # reset() returns the NORMALIZED obs — read the raw frame from C++.
        raw = list(env.cpp_env.obs())
        assert len(raw) == 289
        weights = list(raw[248:257])
        assert weights[6] == pytest.approx(1.0)
        assert sum(weights) == pytest.approx(1.0)
        bits = list(raw[257:289])
        assert sum(bits) == pytest.approx(1.0)
        idx = max(range(32), key=lambda i: bits[i])
        assert env.cpp_env.build_ids()[idx] == "WaterChannel"
    finally:
        env.close()


def test_set_curriculum_refuses_version_change():
    st = build_state(0, None, True, None, 1)
    env = _single_env(st)
    try:
        d = st.to_dict()
        d["obs_version"] = 0
        with pytest.raises(Exception, match="obs_version"):
            env.cpp_env.set_curriculum(d)
        # same-version update still works (schedule path)
        d["obs_version"] = 1
        d["all_builds"] = False
        d["allowed_builds"] = ["WaterChannel"]
        env.cpp_env.set_curriculum(d)
        assert env.cpp_env.curriculum()["all_builds"] is False
    finally:
        env.close()


def test_env_manager_obs_layout_and_buffer():
    """EnvManager on v1: 289-dim obs, version in curriculum(), sized buffer."""
    pytest.importorskip("torch")
    import torch

    from rl.config import Config
    from rl.env_manager import EnvManager

    for ver, size in ((1, 289), (0, 248)):
        cfg = Config(n_envs=2, map_size=_MAP, obs_version=ver)
        em = EnvManager(cfg, torch.device("cpu"))
        try:
            assert em.obs_size == size
            assert int(em.vec_env.venv.curriculum()["obs_version"]) == ver
            assert em.buffer.obs.shape[1] == size
        finally:
            em.close()


def test_water_relative_coordinates():
    st = build_state(0, None, True, None, 0)
    env = _single_env(st)
    try:
        env.reset(seed=7)
        raw = list(env.cpp_env.obs())
        assert len(raw) == 248
        dx = raw[246]
        dy = raw[247]
        assert isinstance(dx, float)
        assert isinstance(dy, float)
        assert -1.0 <= dx <= 1.0
        assert -1.0 <= dy <= 1.0
    finally:
        env.close()
