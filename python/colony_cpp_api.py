"""Handshake with the compiled C++ extension (PR 3).

The repo used to commit `colony_cpp.pyd` binaries whose behaviour silently
shadowed fixed C++ sources: whoever cloned the repo got OLD behaviour with
CORRECT sources. Every entry point must call `require_colony()` before
creating environments — a binary that predates `extension_info()` (or lacks
a required feature) now raises instead of running with wrong behaviour.

Escape hatch (debugging only): `--allow-stale-pyd` CLI flag (train.py,
watch_champion.py) or the `COLONY_ALLOW_STALE_PYD=1` env var.

When bindings gain features that Python depends on, advertise them in
`extension_info()` (src/bindings.cpp), bump the C++ version, and extend
`REQUIRED_FEATURES` / `EXTENSION_MIN_VERSION` here.
"""
from __future__ import annotations

import os

#: Extension API version Python expects.
EXTENSION_MIN_VERSION = 3

#: Capability flags (see `colony_cpp.extension_info()["features"]`) that must
#: be present. PR 1 replaced the (stage, unlock_ids) pair with set_curriculum();
#: PR 4 needs resource weights + priority_reached (a binary that silently
#: ignores them would resurrect the dead-setting bug). P0 (2026-09-17) needs
#: obs_v2 (направления к ближайшим ресурсам) и tax_to_debt (налог → долг,
#: иначе календарь замирает на 365-й день).
REQUIRED_FEATURES = ("set_curriculum", "resource_curriculum", "obs_v2", "tax_to_debt")

#: Env var escape hatch (same effect as --allow-stale-pyd).
STALE_ENV_VAR = "COLONY_ALLOW_STALE_PYD"


class StaleExtensionError(RuntimeError):
    """Raised when colony_cpp is missing, outdated, or lacks features."""


def stale_allowed(explicit: bool = False) -> bool:
    """True if the stale-binary escape hatch is active (flag or env var)."""
    if explicit:
        return True
    return os.environ.get(STALE_ENV_VAR, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def extension_info() -> dict | None:
    """Return `colony_cpp.extension_info()` as a plain dict.

    Returns None when the loaded binary predates the handshake (PR 3).
    Raises StaleExtensionError when the extension cannot be imported at all.
    """
    try:
        import colony_cpp  # lazy: the extension may be absent (Linux CI, tests)
    except ImportError as e:
        raise StaleExtensionError(
            "colony_cpp extension not found — build it with build_pyext.bat "
            f"(import error: {e})"
        ) from e

    info_fn = getattr(colony_cpp, "extension_info", None)
    if not callable(info_fn):
        return None
    try:
        return dict(info_fn())
    except (TypeError, ValueError):
        return {"version": 0, "features": [], "src_sha": "unknown"}


def require_colony(required=REQUIRED_FEATURES, allow_stale: bool = False) -> dict:
    """Verify the loaded colony_cpp binary is fresh enough; return its info.

    Raises StaleExtensionError (not a warning) when the binary predates
    `extension_info()`, is older than EXTENSION_MIN_VERSION, or lacks any of
    `required`. Pass allow_stale=True (or set COLONY_ALLOW_STALE_PYD=1) to
    downgrade to a warning — debugging only, behaviour may be wrong.
    """
    info = extension_info()  # raises StaleExtensionError if not importable
    if info is None:
        msg = ("colony_cpp binary predates extension_info() — rebuild it with "
               "build_pyext.bat (binaries are no longer committed to git, a "
               "stale .pyd silently shadows fixed C++ sources).")
        if stale_allowed(allow_stale):
            print(f"[colony_cpp_api] WARNING: {msg} Continuing anyway (--allow-stale-pyd).",
                  flush=True)
            return {"version": 0, "features": [], "src_sha": "unknown"}
        raise StaleExtensionError(msg)

    problems: list[str] = []
    try:
        ver = int(info.get("version", 0))
    except (TypeError, ValueError):
        ver = 0
    if ver < EXTENSION_MIN_VERSION:
        problems.append(f"version {ver} < required {EXTENSION_MIN_VERSION}")
    have = set(info.get("features", []) or [])
    missing = [f for f in (required or ()) if f not in have]
    if missing:
        problems.append(f"missing features {missing} (binary has {sorted(have)})")
    if problems:
        msg = ("stale colony_cpp binary (src_sha={sha}): {probs} — rebuild "
               "with build_pyext.bat.".format(
                   sha=info.get("src_sha", "?"),
                   probs="; ".join(problems)))
        if stale_allowed(allow_stale):
            print(f"[colony_cpp_api] WARNING: {msg} Continuing anyway (--allow-stale-pyd).",
                  flush=True)
            return info
        raise StaleExtensionError(msg)
    return info
