"""PR 1: the unified curriculum contract (pure Python, no extension needed).

`build_state()` is the only function that turns (stage, manual set, checkbox)
into a CurriculumState; every env creator (train/eval/watch/GUI) must use it.
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl.curriculum import (  # noqa: E402
    ALL_IDS,
    BUILD_ALIASES,
    RESOURCE_NAMES,
    STAGE_MAP,
    CurriculumState,
    parse_resources,
    allowed_ids,
    build_state,
    parse_unlock_ids,
    resolve_state,
)


# ── build_state matrix ───────────────────────────────────────────────────

def test_stage_zero_without_manual_is_unrestricted():
    for tab in (True, False):
        for manual in (None, "", []):
            st = build_state(0, manual, tab)
            assert st.all_builds is True, f"stage=0 manual={manual!r} tab={tab}"
            assert st.stage_report == 0


def test_manual_set_at_stage_zero_restricts():
    st = build_state(0, "WaterChannel", True)
    assert st.all_builds is False
    assert st.allowed_builds == ("WaterChannel",)


def test_checkbox_off_ignores_manual_set():
    st = build_state(0, "WaterChannel", False)
    assert st.all_builds is True
    st = build_state(1, "Goldmine", False)
    assert st.all_builds is False
    assert set(st.allowed_builds) == set(STAGE_MAP[1])
    assert "Goldmine" not in st.allowed_builds


def test_stage_preset_without_manual():
    st = build_state(1, None, False)
    assert set(st.allowed_builds) == set(STAGE_MAP[1])
    st = build_state(2, "", False)
    assert set(st.allowed_builds) == set(STAGE_MAP[1]) | set(STAGE_MAP[2])
    assert len(st.allowed_builds) == 25


def test_manual_merges_with_stage_preset():
    st = build_state(1, "Goldmine", True)
    assert "WaterChannel" in st.allowed_builds  # stage-1 preset
    assert "Goldmine" in st.allowed_builds      # manual
    assert "Sawmill" not in st.allowed_builds   # stage 2 still locked
    assert len(st.allowed_builds) == 15


def test_full_set_normalises_to_unrestricted():
    """All 32 checkboxes ticked == no restriction (explicit all_builds)."""
    st = build_state(0, ",".join(ALL_IDS), True)
    assert st.all_builds is True
    st = build_state(3, None, False)  # presets 1..3 cover everything
    assert st.all_builds is True
    assert len(st.allowed_builds) == 32


def test_stage_report_echoes_stage():
    assert build_state(2, None, False).stage_report == 2
    assert build_state(0, "WaterChannel", True).stage_report == 0


def test_build_state_matches_allowed_ids():
    """build_state and the legacy allowed_ids() must agree on every combo."""
    cases = [
        (0, None, True), (0, "", True), (0, "WaterChannel", True),
        (0, "WaterChannel", False), (1, None, False), (1, "Goldmine", True),
        (2, "Road", True), (2, None, False), (3, None, True),
    ]
    for stage, manual, tab in cases:
        st = build_state(stage, manual, tab)
        legacy = allowed_ids(stage, manual, tab)
        assert set(st.allowed_builds) == set(legacy), f"{stage} {manual} {tab}"
        assert st.all_builds == (len(legacy) == len(ALL_IDS))


# ── aliases & validation ─────────────────────────────────────────────────

def test_aliases_fold_to_canon():
    assert parse_unlock_ids("BigRefinery") == ["BigRefinary"]
    assert parse_unlock_ids("Water_Channel") == ["WaterChannel"]
    st = build_state(0, "BigRefinery, Water_Channel", True)
    assert set(st.allowed_builds) == {"BigRefinary", "WaterChannel"}


def test_alias_targets_are_canonical():
    assert set(BUILD_ALIASES.values()) <= set(ALL_IDS)


def test_unknown_id_raises():
    with pytest.raises(ValueError, match="unknown building"):
        parse_unlock_ids("NoSuchBuilding")
    with pytest.raises(ValueError, match="NoSuchBuilding"):
        parse_unlock_ids("WaterChannel, NoSuchBuilding")
    with pytest.raises(ValueError, match="unknown building"):
        build_state(0, "House, TypoHouse", True)


# ── resolve_state ────────────────────────────────────────────────────────

def _write_meta(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_state_from_meta(tmp_path):
    _write_meta(tmp_path / "best_model.meta.json", {
        "curriculum_stage_at_best": 0, "unlock_ids": "WaterChannel",
        "use_curriculum_tab": True})
    st = resolve_state(tmp_path)
    assert st.all_builds is False
    assert st.allowed_builds == ("WaterChannel",)


def test_resolve_state_explicit_wins(tmp_path):
    _write_meta(tmp_path / "best_model.meta.json", {
        "curriculum_stage_at_best": 3, "unlock_ids": "Goldmine",
        "use_curriculum_tab": True})
    st = resolve_state(tmp_path, curriculum_stage=0, unlock_ids="WaterChannel",
                       use_curriculum_tab=True)
    assert st.allowed_builds == ("WaterChannel",)


def test_resolve_state_without_meta_is_unrestricted(tmp_path):
    st = resolve_state(tmp_path)
    assert st.all_builds is True


def test_resolve_state_unchecked_tab_is_unrestricted(tmp_path):
    """Stored-but-unchecked manual set must not restrict (tab=False)."""
    _write_meta(tmp_path / "best_model.meta.json", {
        "curriculum_stage_at_best": 0, "unlock_ids": "WaterChannel",
        "use_curriculum_tab": False})
    st = resolve_state(tmp_path)
    assert st.all_builds is True


def test_resolve_state_missing_tab_honours_manual(tmp_path):
    """Legacy meta without use_curriculum_tab: a stored manual set applies.

    This is the «not stored» vs «stored as empty» distinction: a missing tab
    (None) defaults to honouring the manual set, while an explicit False
    ignores it (see the test above).
    """
    _write_meta(tmp_path / "best_model.meta.json", {
        "curriculum_stage_at_best": 0, "unlock_ids": "WaterChannel"})
    st = resolve_state(tmp_path)
    assert st.all_builds is False
    assert st.allowed_builds == ("WaterChannel",)


# ── transport form ───────────────────────────────────────────────────────

def test_state_round_trips_through_dict():
    st = build_state(1, "Goldmine", True)
    blob = json.dumps(st.to_dict())  # must be JSON-serialisable (GUI transport)
    back = CurriculumState.from_dict(json.loads(blob))
    assert back == st


def test_state_round_trips_unrestricted():
    st = build_state(0, None, True)
    assert CurriculumState.from_dict(st.to_dict()) == st


def test_from_dict_defaults_to_unrestricted():
    st = CurriculumState.from_dict({})
    assert st.all_builds is True
    assert st.stage_report == 0


def test_transport_keys_match_cpp_schema():
    """Keys must match C++ curriculum_from_dict/from_json (bindings.cpp, env.cpp)."""
    d = build_state(0, "WaterChannel", True).to_dict()
    assert d["all_builds"] is False
    assert d["allowed_builds"] == ["WaterChannel"]
    assert d["stage"] == 0
    assert d["all_resources"] is True
    assert d["resource_weights"] == [1.0] * 9


# ── cross-language set parity ────────────────────────────────────────────

def _build_subset_from_header() -> set:
    text = (ROOT / "include" / "colony" / "constants.h").read_text(encoding="utf-8")
    m = re.search(r"BUILD_SUBSET\[32\] = \{(.*?)\};", text, re.S)
    assert m, "BUILD_SUBSET not found in constants.h"
    return set(re.findall(r"\"([^\"]+)\"", m.group(1)))


def test_all_ids_match_cpp_build_subset():
    """Python ALL_IDS and C++ BUILD_SUBSET must be the same 32 ids.

    The parity assert in EnvManager compares these sets — a drift would fail
    loudly at startup; this test fails even earlier (no extension needed).
    """
    assert set(ALL_IDS) == _build_subset_from_header()


def test_stage_presets_cover_all_ids():
    union = {bid for s in (1, 2, 3) for bid in STAGE_MAP[s]}
    assert union == set(ALL_IDS)


# ── Config integration ───────────────────────────────────────────────────

def test_config_curriculum_state():
    from rl.config import Config

    cfg = Config(curriculum_stage=0, unlock_ids="WaterChannel", use_curriculum_tab=True)
    st = cfg.curriculum_state()
    assert isinstance(st, CurriculumState)
    assert st.allowed_builds == ("WaterChannel",)

    default = Config().curriculum_state()
    assert default.all_builds is True


# ── PR 6: unified mask-fill value ──────────────────────────────────────────

def test_mask_fill_value_all_masked():
    """Rationale for -1e9 (train/eval/watch): finite under a fully-closed mask.

    With float('-inf') a fully-masked softmax is NaN everywhere (it used to
    leak into eval top-3 logs); with -1e9 it degrades to uniform — argmaxable,
    differentiable, and NaN-free.
    """
    torch = pytest.importorskip("torch")
    n = 45
    logits = torch.zeros(n)
    mask = torch.zeros(n)  # everything blocked (degenerate scenario)

    blocked_inf = logits.masked_fill(mask == 0, float("-inf"))
    assert torch.isnan(torch.softmax(blocked_inf, dim=-1)).all()

    blocked_big = logits.masked_fill(mask == 0, -1e9)
    probs = torch.softmax(blocked_big, dim=-1)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs, torch.full((n,), 1.0 / n))


# ── PR 4: resource priorities ────────────────────────────────────────────

def test_parse_resources_none_and_empty_is_all():
    assert parse_resources(None) is None
    assert parse_resources("") is None
    assert parse_resources([]) is None
    assert parse_resources("  ") is None


def test_parse_resources_weights():
    assert parse_resources("water") == [0.0] * 6 + [1.0] + [0.0] * 2
    assert parse_resources(["wood", "water"]) == [0.0] * 6 + [1.0, 1.0, 0.0]
    # case/whitespace/dupes tolerated
    assert parse_resources("WATER, Water ,wood") == [0.0] * 6 + [1.0, 1.0, 0.0]


def test_parse_resources_unknown_raises():
    with pytest.raises(ValueError, match="неизвестные ресурсы"):
        parse_resources("unobtanium")
    with pytest.raises(ValueError, match="unobtanium"):
        parse_resources("water,unobtanium")


def test_build_state_resources():
    st = build_state(0, None, False, "water,wood")
    assert st.all_builds is True
    assert st.all_resources is False
    assert st.resource_weights == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    default = build_state(0, None, False)
    assert default.all_resources is True
    assert default.resource_weights == (1.0,) * 9


def test_build_state_full_nine_normalises_to_all():
    st = build_state(0, None, False, ",".join(RESOURCE_NAMES))
    assert st.all_resources is True
    assert st.resource_weights == (1.0,) * 9


def test_curriculum_meta_carries_resources():
    from rl.config import Config

    m = Config(
        curriculum_stage=0, unlock_ids="WaterChannel", use_curriculum_tab=True,
        curriculum_resources="water,wood").curriculum_meta()
    assert m["curriculum_resources"] == "water,wood"
    assert Config().curriculum_meta()["curriculum_resources"] == ""


def test_resolve_state_restores_resources(tmp_path):
    _write_meta(tmp_path / "best_model.meta.json", {
        "curriculum_stage_at_best": 0, "unlock_ids": "WaterChannel",
        "use_curriculum_tab": True, "curriculum_resources": "water"})
    st = resolve_state(tmp_path)
    assert st.all_resources is False
    assert st.resource_weights == (0.0,) * 6 + (1.0, 0.0, 0.0)


def test_resolve_state_resources_explicit_wins(tmp_path):
    _write_meta(tmp_path / "best_model.meta.json", {
        "curriculum_stage_at_best": 0, "unlock_ids": "",
        "use_curriculum_tab": False, "curriculum_resources": "water"})
    st = resolve_state(tmp_path, resources="wood")
    assert st.resource_weights == (0.0,) * 7 + (1.0, 0.0)


def test_resource_names_match_cpp_and_ui():
    """The sunduk order must never diverge across the three owners."""
    from train_ui2.constants import RESOURCE_IDS

    assert list(RESOURCE_NAMES) == list(RESOURCE_IDS)
    src = (ROOT / "src" / "resources.cpp").read_text(encoding="utf-8")
    m = re.search(r"names\[SUNDUK_SIZE\] = \{(.*?)\};", src, re.S)
    assert m, "C++ names array not found"
    cpp_names = re.findall(r'"(\w+)"', m.group(1))
    assert cpp_names == list(RESOURCE_NAMES)


# ── PR 5: obs layout version ─────────────────────────────────────────────

def test_build_state_obs_version_default_and_explicit():
    # P0 (2026-09-17): дефолт — obs v2 (299): без направлений к ближайшим
    # ресурсам 12 из 32 построек недостижимы (см. docs/RL_DIAGNOSIS_2026_09.md).
    assert build_state(0, None, True).obs_version == 2
    assert build_state(0, None, True, None, 0).obs_version == 0
    assert build_state(0, "WaterChannel", True, "water", 0).obs_version == 0


def test_build_state_obs_version_invalid_raises():
    with pytest.raises(ValueError, match="obs_version must be 0, 1 or 2"):
        build_state(0, None, True, None, 3)


def test_curriculum_state_obs_roundtrip():
    st = build_state(1, "Goldmine", True, "water,wood", 0)
    d = st.to_dict()
    assert d["obs_version"] == 0
    assert CurriculumState.from_dict(d).to_dict() == d
    # absent key = current default (tolerant read)
    d2 = dict(d)
    del d2["obs_version"]
    assert CurriculumState.from_dict(d2).obs_version == 2
    assert CurriculumState.all().obs_version == 2


def test_curriculum_meta_carries_obs_version():
    from rl.config import Config

    assert Config().curriculum_meta()["obs_version"] == 2
    assert Config(obs_version=0).curriculum_meta()["obs_version"] == 0
    assert Config(obs_version=0).curriculum_state().obs_version == 0


def test_curriculum_from_meta_picks_obs_version(tmp_path):
    from rl.curriculum import curriculum_from_meta, read_curriculum_meta

    assert curriculum_from_meta({"obs_version": 0})["obs_version"] == 0
    assert curriculum_from_meta({"config": {"obs_version": 1}})["obs_version"] == 1
    assert curriculum_from_meta({})["obs_version"] is None
    _write_meta(tmp_path / "meta.json", {"obs_version": 1, "unlock_ids": ""})
    assert read_curriculum_meta(tmp_path)["obs_version"] == 1


def test_stored_obs_version_cases(tmp_path):
    from rl.curriculum import stored_obs_version

    assert stored_obs_version(tmp_path) is None  # no meta files: unknown
    _write_meta(tmp_path / "meta.json", {"unlock_ids": ""})  # legacy: no key
    assert stored_obs_version(tmp_path) == 0
    _write_meta(tmp_path / "meta.json", {"obs_version": 1})
    assert stored_obs_version(tmp_path) == 1
    assert stored_obs_version(None, {"obs_version": 0}) == 0
    assert stored_obs_version(None, {"unlock_ids": "x"}) == 0  # legacy meta
    assert stored_obs_version(None) is None
    assert stored_obs_version(None, {}) is None


def test_check_obs_version_compat():
    from rl.curriculum import check_obs_version_compat

    check_obs_version_compat(1, 1, ckpt_path="m.pt")  # match: silent
    check_obs_version_compat(None, 1, ckpt_path="m.pt")  # unknown: silent
    with pytest.raises(RuntimeError,
                       match=r"obs v0.*248.*obs v1.*289.*--obs-version 0"):
        check_obs_version_compat(0, 1, ckpt_path="m.pt")
    with pytest.raises(RuntimeError,
                       match=r"obs v1.*289.*obs v0.*248.*--obs-version 1"):
        check_obs_version_compat(1, 0, ckpt_path="m.pt")


def test_check_policy_obs_compat():
    from rl.curriculum import check_policy_obs_compat

    check_policy_obs_compat(289, 289, ckpt_path="m.pt")
    with pytest.raises(RuntimeError,
                       match=r"expects a 248-dim.*serves 289.*--obs-version 0"):
        check_policy_obs_compat(248, 289, ckpt_path="m.pt")
    with pytest.raises(RuntimeError, match=r"expects a 203-dim"):
        check_policy_obs_compat(203, 289, ckpt_path="m.pt")  # no hint


def test_ckpt_flat_width():
    from types import SimpleNamespace

    from rl.curriculum import ckpt_flat_width

    assert ckpt_flat_width(
        {"trunk.0.weight": SimpleNamespace(shape=(64, 248))}) == 248
    assert ckpt_flat_width(
        {"flat_trunk.0.weight": SimpleNamespace(shape=(64, 289))}) == 289
    assert ckpt_flat_width(
        {"cnn.0.weight": SimpleNamespace(shape=(16, 8, 3, 3))}) is None
    assert ckpt_flat_width({}) is None


def test_resolve_state_obs_version_explicit_only(tmp_path):
    # A stored v0 must NOT silently rebuild the eval env — explicit wins.
    _write_meta(tmp_path / "best_model.meta.json", {
        "curriculum_stage_at_best": 0, "obs_version": 0})
    assert resolve_state(tmp_path).obs_version == 1
    assert resolve_state(tmp_path, obs_version=0).obs_version == 0
