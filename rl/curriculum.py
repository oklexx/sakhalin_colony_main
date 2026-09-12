"""Curriculum definitions — single source of truth.

Both C++ (env.cpp) and Python (EnvManager, UI) should mirror this.
Keeping it here ensures UI presets and env allowed lists stay in sync.
"""
from __future__ import annotations

from typing import Dict, List

# Stage → building ids (as in configs/bases.json without City)
STAGE_MAP: Dict[int, List[str]] = {
    1: ["House", "SmallHouse", "Farm", "Garden", "Mushroom", "WaterChannel",
        "Refinery", "Fish", "HuntingLand", "CowFarm", "Apiary", "Hothouse",
        "Puerperal", "Road"],
    2: ["Sawmill", "Coalmine", "CoalCut", "Ironmine", "PowerStation",
        "HydroStation", "AirStation", "Torchlight", "Goldmine", "BigHouse",
        "BigFarm"],
    3: ["BigSawmill", "BigRefinary", "BigIronmine", "WaterMill",
        "SmallAtomStation", "AtomStation", "SuperHouse"],
}

# All 32 buildable ids in canonical order
ALL_IDS: List[str] = [
    "Farm", "Garden", "WaterChannel", "Sawmill", "Coalmine", "Ironmine", "Refinery", "Goldmine",
    "PowerStation", "HydroStation", "Road", "House", "SmallHouse", "Fish", "CoalCut",
    "HuntingLand", "CowFarm", "Mushroom", "BigHouse", "BigFarm", "Apiary", "Torchlight", "Hothouse",
    "SuperHouse", "BigSawmill", "WaterMill", "BigRefinary", "Puerperal", "BigIronmine",
    "AirStation", "SmallAtomStation", "AtomStation",
]

def parse_unlock_ids(unlock_ids: str | List[str] | None) -> List[str]:
    """Normalise a manual unlock set (CSV string or list) to an ordered list."""
    if not unlock_ids:
        return []
    if isinstance(unlock_ids, str):
        raw = unlock_ids.split(",")
    else:
        raw = [str(s) for s in unlock_ids]
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        bid = item.strip()
        if bid and bid not in seen:
            seen.add(bid)
            out.append(bid)
    return out


def manual_ids_csv(
    unlock_ids: str | List[str] | None,
    use_curriculum_tab: bool = True,
) -> str:
    """Manual curriculum set as CSV, honouring the «Ручной набор» checkbox.

    ``use_curriculum_tab=False`` (checkbox off) means the manual set is ignored
    and only the stage preset applies — exactly the semantics the UI documents.
    """
    if not use_curriculum_tab:
        return ""
    return ",".join(parse_unlock_ids(unlock_ids))


def ids_for_stage(stage: int, unlock_ids: str | List[str] | None = None) -> List[str]:
    """Cumulative ids up to stage (0 = all), plus explicit manual `unlock_ids`.

    Mirrors C++ `ColonyEnvCpp` logic: если задан непустой ручной набор,
    он объединяется с набором этапа и становится единственным разрешённым
    списком (даже на этапе 0, где иначе доступны все здания).
    """
    manual = parse_unlock_ids(unlock_ids)

    if stage == 0 and not manual:
        return list(ALL_IDS)
    out: List[str] = []
    for s in range(1, stage + 1):
        out.extend(STAGE_MAP.get(s, []))
    out.extend(manual)
    # de-duplicate while preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for bid in out:
        if bid not in seen:
            seen.add(bid)
            uniq.append(bid)
    return uniq


def allowed_ids(
    stage: int,
    unlock_ids: str | List[str] | None = None,
    use_curriculum_tab: bool = True,
) -> List[str]:
    """Single entry point: building ids allowed by stage + manual set.

    This is THE function every env instance must use (training, eval, watch) so
    a curriculum never depends on which code path created the environment.
    """
    return ids_for_stage(stage, unlock_ids=manual_ids_csv(unlock_ids, use_curriculum_tab))


def curriculum_from_meta(meta: Dict[str, object] | None) -> Dict[str, object]:
    """Extract curriculum settings from a model/run meta dict.

    Accepts both layouts:
      * ``best_model.meta.json`` — flat keys (``curriculum_stage_at_best``,
        ``unlock_ids``, ``use_curriculum_tab``);
      * run ``meta.json`` written by the worker — the whole Config nested under
        the ``config`` key.

    Missing values come back as ``None`` so callers can tell "not stored" from
    "stored as empty".
    """
    if not isinstance(meta, dict):
        return {"curriculum_stage": None, "unlock_ids": None, "use_curriculum_tab": None}
    nested = meta.get("config")
    if not isinstance(nested, dict):
        nested = {}

    def pick(*keys: str):
        for src in (meta, nested):
            for key in keys:
                if key in src and src[key] is not None:
                    return src[key]
        return None

    stage = pick("curriculum_stage_at_best", "curriculum_stage")
    manual = pick("unlock_ids")
    use_tab = pick("use_curriculum_tab")
    return {
        "curriculum_stage": int(stage) if stage is not None else None,
        "unlock_ids": None if manual is None else str(manual),
        "use_curriculum_tab": None if use_tab is None else bool(use_tab),
    }


_META_NAMES = ("best_model.meta.json", "meta.json")


def read_curriculum_meta(model_dir) -> Dict[str, object]:
    """Read curriculum settings from the meta files in a model directory.

    ``best_model.meta.json`` usually carries only the stage, while the run
    ``meta.json`` stores the whole Config (incl. ``unlock_ids`` /
    ``use_curriculum_tab``) under ``config`` — so fields are merged across files.
    Values that are not stored stay ``None``.
    """
    import json
    from pathlib import Path

    out: Dict[str, object] = {
        "curriculum_stage": None,
        "unlock_ids": None,
        "use_curriculum_tab": None,
    }
    model_dir = Path(model_dir)
    for meta_name in _META_NAMES:
        meta_path = model_dir / meta_name
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        for key, value in curriculum_from_meta(meta).items():
            if out[key] is None and value is not None:
                out[key] = value
    return out


def resolve_curriculum(
    model_dir,
    meta: Dict[str, object] | None = None,
    curriculum_stage: int | None = None,
    unlock_ids: str | List[str] | None = None,
    use_curriculum_tab: bool | None = None,
) -> Dict[str, object]:
    """Resolve the effective curriculum for a model (explicit args win).

    Returns ``{"curriculum_stage", "unlock_ids", "use_curriculum_tab",
    "allowed"}`` where ``unlock_ids`` is the effective CSV ("" = no manual set)
    and ``allowed`` is the resulting building-id list.
    """
    stored = read_curriculum_meta(model_dir) if model_dir is not None else {}
    if meta:
        for key, value in curriculum_from_meta(meta).items():
            if stored.get(key) is None and value is not None:
                stored[key] = value

    stage = curriculum_stage if curriculum_stage is not None else stored.get("curriculum_stage")
    manual_raw = unlock_ids if unlock_ids is not None else stored.get("unlock_ids")
    tab = use_curriculum_tab if use_curriculum_tab is not None else stored.get("use_curriculum_tab")

    stage_i = int(stage or 0)
    manual_csv = manual_ids_csv(manual_raw, True if tab is None else bool(tab))
    return {
        "curriculum_stage": stage_i,
        "unlock_ids": manual_csv,
        "use_curriculum_tab": tab,
        "allowed": allowed_ids(stage_i, manual_csv, True),
    }


def allowed_buildings_for_stage(stage: int) -> List[str]:
    """Legacy helper returning BUILD_ prefixed names (EnvManager compatibility)."""
    ids = ids_for_stage(stage)
    # EnvManager expects BUILD_* + managers + DAY/WEEK etc. Keep that expansion there.
    return ids
