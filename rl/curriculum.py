"""Curriculum definitions — single source of truth.

PR 1 contract: Python COMPUTES the allowed set, C++ stores and applies it.
`build_state()` is the only function that turns (stage, manual set, checkbox)
into a `CurriculumState`; the C++ side never interprets stages or lists again
(no more «empty list means all» fail-open, no more stage wiping the manual set).
"""
from __future__ import annotations

from dataclasses import dataclass
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

# All 32 buildable ids in canonical order (same SET as C++ BUILD_SUBSET)
ALL_IDS: List[str] = [
    "Farm", "Garden", "WaterChannel", "Sawmill", "Coalmine", "Ironmine", "Refinery", "Goldmine",
    "PowerStation", "HydroStation", "Road", "House", "SmallHouse", "Fish", "CoalCut",
    "HuntingLand", "CowFarm", "Mushroom", "BigHouse", "BigFarm", "Apiary", "Torchlight", "Hothouse",
    "SuperHouse", "BigSawmill", "WaterMill", "BigRefinary", "Puerperal", "BigIronmine",
    "AirStation", "SmallAtomStation", "AtomStation",
]
_ALL_IDS_SET = frozenset(ALL_IDS)

# Канон — configs/bases.json: BigRefinary (не BigRefinery!), WaterChannel
# (не Water_Channel). Частые опечатки нормализуются, остальное — ошибка.
BUILD_ALIASES: Dict[str, str] = {
    "BigRefinery": "BigRefinary",
    "Water_Channel": "WaterChannel",
}

_FULL_WEIGHTS = (1.0,) * 9  # PR 4: neutral resource weights (all resources)


@dataclass(frozen=True)
class CurriculumState:
    """Computed curriculum: the single object every env creator passes to C++.

    `all_builds=True` means «no building restriction» explicitly; in that case
    `allowed_builds` is informational only (conventionally ALL_IDS) and the C++
    gate ignores it. `stage_report` is report-only (obs feature, dumps).
    `all_resources` / `resource_weights` are the PR 4 payload (always neutral
    until the resource curriculum lands).
    """

    all_builds: bool
    allowed_builds: tuple[str, ...] = ()
    all_resources: bool = True
    resource_weights: tuple[float, ...] = _FULL_WEIGHTS
    stage_report: int = 0

    @classmethod
    def all(cls, stage_report: int = 0) -> "CurriculumState":
        """Unrestricted state (conventionally carries ALL_IDS for logging)."""
        return cls(True, tuple(ALL_IDS), True, _FULL_WEIGHTS, stage_report)

    def to_dict(self) -> dict:
        """Transport form for C++ set_curriculum() and the GUI --curriculum JSON."""
        return {
            "all_builds": bool(self.all_builds),
            "allowed_builds": list(self.allowed_builds),
            "all_resources": bool(self.all_resources),
            "resource_weights": [float(w) for w in self.resource_weights],
            "stage": int(self.stage_report),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurriculumState":
        """Tolerant read of the transport form (unknown keys ignored)."""
        allowed = d.get("allowed_builds", ())
        weights = d.get("resource_weights", None)
        return cls(
            all_builds=bool(d.get("all_builds", True)),
            allowed_builds=tuple(str(b) for b in (allowed or ())),
            all_resources=bool(d.get("all_resources", True)),
            resource_weights=tuple(float(w) for w in weights)
            if weights is not None else _FULL_WEIGHTS,
            stage_report=int(d.get("stage", d.get("stage_report", 0))),
        )


def parse_unlock_ids(unlock_ids: str | List[str] | None) -> List[str]:
    """Normalise a manual unlock set (CSV string or list) to an ordered list.

    Aliases (BigRefinery→BigRefinary, Water_Channel→WaterChannel) are folded to
    the bases.json canon. Unknown ids RAISE — a typo must fail the run loudly
    instead of silently locking everything (fail-closed) or, worse, being
    dropped into an «allow all» state.
    """
    if not unlock_ids:
        return []
    if isinstance(unlock_ids, str):
        raw = unlock_ids.split(",")
    else:
        raw = [str(s) for s in unlock_ids]
    out: List[str] = []
    seen: set[str] = set()
    unknown: List[str] = []
    for item in raw:
        bid = item.strip()
        if not bid:
            continue
        bid = BUILD_ALIASES.get(bid, bid)
        if bid not in _ALL_IDS_SET:
            if bid not in unknown:
                unknown.append(bid)
            continue
        if bid not in seen:
            seen.add(bid)
            out.append(bid)
    if unknown:
        raise ValueError(
            f"unknown building id(s): {unknown}; "
            f"check --unlock-ids / the «Курикулум» tab "
            f"(available: {', '.join(ALL_IDS)})"
        )
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

    Mirrors the C++ `ColonyEnvCpp` logic it replaced: если задан непустой ручной
    набор, он объединяется с набором этапа и становится единственным разрешённым
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


def build_state(
    stage: int,
    unlock_ids: str | List[str] | None = None,
    use_curriculum_tab: bool = True,
    resources: str | List[str] | None = None,
) -> CurriculumState:
    """Compute the CurriculumState from (stage, manual set, checkbox).

    THE function every env instance must use (training, eval, watch) so a
    curriculum never depends on which code path created the environment. A full
    32-id set normalises to `all_builds=True` (same behaviour, explicit form).

    `resources`: reserved for the PR 4 resource curriculum; currently ignored.
    """
    _ = resources  # PR 4 wires this; the setting is dead until then (as before)
    stage_i = max(0, int(stage))
    manual = parse_unlock_ids(unlock_ids)
    if use_curriculum_tab:
        ids = ids_for_stage(stage_i, manual)
    elif stage_i == 0:
        ids = list(ALL_IDS)
    else:
        ids = ids_for_stage(stage_i, None)
    if set(ids) == _ALL_IDS_SET:
        return CurriculumState.all(stage_i)
    return CurriculumState(False, tuple(ids), True, _FULL_WEIGHTS, stage_i)


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


def resolve_state(
    model_dir,
    meta: Dict[str, object] | None = None,
    curriculum_stage: int | None = None,
    unlock_ids: str | List[str] | None = None,
    use_curriculum_tab: bool | None = None,
    resources: str | List[str] | None = None,
) -> CurriculumState:
    """resolve_curriculum + build_state in one call (eval/watch paths).

    Returns the CurriculumState to hand to the env constructor — «not stored»
    and «stored as empty» both resolve through the same code as training, so
    the eval/watch scenario can never silently differ from training.
    """
    resolved = resolve_curriculum(
        model_dir,
        meta=meta,
        curriculum_stage=curriculum_stage,
        unlock_ids=unlock_ids,
        use_curriculum_tab=use_curriculum_tab,
    )
    # manual_csv is already checkbox-filtered, so the tab is trivially True here.
    return build_state(
        int(resolved["curriculum_stage"]),  # type: ignore[arg-type]
        str(resolved["unlock_ids"]),
        True,
        resources,
    )


def allowed_buildings_for_stage(stage: int) -> List[str]:
    """Legacy helper returning BUILD_ prefixed names (EnvManager compatibility)."""
    ids = ids_for_stage(stage)
    # EnvManager expects BUILD_* + managers + DAY/WEEK etc. Keep that expansion there.
    return ids
