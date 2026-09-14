"""PR 3: colony_cpp handshake — stale binaries raise, not warn.

Runs without the compiled extension and without torch: `colony_cpp` is faked
via sys.modules. A live-binary check (real extension_info from a fresh build)
is covered separately — see the PR 3 verification notes.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from colony_cpp_api import (  # noqa: E402
    EXTENSION_MIN_VERSION,
    REQUIRED_FEATURES,
    STALE_ENV_VAR,
    StaleExtensionError,
    extension_info,
    require_colony,
    stale_allowed,
)


def _fake_module(**attrs):
    mod = types.ModuleType("colony_cpp")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


@pytest.fixture
def fake_colony(monkeypatch):
    """Install a fake `colony_cpp` module into sys.modules."""
    def install(**attrs):
        monkeypatch.setitem(sys.modules, "colony_cpp", _fake_module(**attrs))
    return install


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(STALE_ENV_VAR, raising=False)


# ── failure modes: must raise ────────────────────────────────────────────

def test_legacy_binary_without_extension_info_raises(fake_colony):
    """A pre-PR-3 .pyd (no extension_info attr) must raise, not warn."""
    fake_colony()  # legacy binary: no extension_info at all
    assert extension_info() is None
    with pytest.raises(StaleExtensionError, match="predates extension_info"):
        require_colony()


def test_missing_feature_raises(fake_colony):
    fake_colony(extension_info=lambda: {
        "version": EXTENSION_MIN_VERSION, "features": [], "src_sha": "abc1234"})
    with pytest.raises(StaleExtensionError, match="missing features"):
        require_colony()


def test_old_version_raises(fake_colony):
    fake_colony(extension_info=lambda: {
        "version": 0, "features": list(REQUIRED_FEATURES), "src_sha": "abc1234"})
    with pytest.raises(StaleExtensionError, match="version 0 < required"):
        require_colony()


def test_unimportable_extension_raises(monkeypatch):
    """No colony_cpp importable at all → clear rebuild error."""
    monkeypatch.setitem(sys.modules, "colony_cpp", None)
    with pytest.raises(StaleExtensionError, match="not found"):
        require_colony()


def test_custom_required_subset_checked(fake_colony):
    fake_colony(extension_info=lambda: {
        "version": EXTENSION_MIN_VERSION,
        "features": ["set_unlock_ids"], "src_sha": "abc1234"})
    # caller may demand more than the default set
    with pytest.raises(StaleExtensionError, match="nope"):
        require_colony(required=("set_unlock_ids", "nope"))


# ── happy path ───────────────────────────────────────────────────────────

def test_fresh_binary_returns_info(fake_colony):
    fake_colony(extension_info=lambda: {
        "version": EXTENSION_MIN_VERSION,
        "features": ["set_unlock_ids", "minimap"],
        "src_sha": "deadbee"})
    info = require_colony()
    assert info["version"] == EXTENSION_MIN_VERSION
    assert info["src_sha"] == "deadbee"
    assert "set_unlock_ids" in info["features"]


def test_required_features_contract():
    # set_unlock_ids is THE capability whose absence broke the curriculum
    # silently with stale binaries — it must stay required.
    assert "set_unlock_ids" in REQUIRED_FEATURES
    assert EXTENSION_MIN_VERSION >= 1


# ── escape hatch ─────────────────────────────────────────────────────────

def test_allow_stale_flag_downgrades_to_warning(fake_colony, capsys):
    fake_colony()
    info = require_colony(allow_stale=True)
    assert info["version"] == 0
    assert "WARNING" in capsys.readouterr().out


def test_allow_stale_env_var(fake_colony, monkeypatch, capsys):
    fake_colony(extension_info=lambda: {
        "version": 0, "features": [], "src_sha": "old"})
    monkeypatch.setenv(STALE_ENV_VAR, "1")
    assert stale_allowed() is True
    info = require_colony()  # no explicit flag — env var decides
    assert info["version"] == 0
    assert "WARNING" in capsys.readouterr().out


def test_stale_not_allowed_by_default():
    assert stale_allowed() is False
    assert stale_allowed(explicit=True) is True


# ── live binary (only when the real extension is importable) ─────────────

def test_live_extension_reports_info():
    colony_cpp = pytest.importorskip("colony_cpp")
    if not callable(getattr(colony_cpp, "extension_info", None)):
        pytest.skip("loaded colony_cpp predates extension_info (stale binary)")
    info = require_colony()
    assert info["version"] >= EXTENSION_MIN_VERSION
    assert set(REQUIRED_FEATURES) <= set(info["features"])
    assert isinstance(info["src_sha"], str) and info["src_sha"]
