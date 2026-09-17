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

# Канон порядка сундука: обязан совпадать с Sunduk::resource_name
# (src/resources.cpp) и train_ui2/constants.py RESOURCE_IDS (регресс-тест).
RESOURCE_NAMES = ("gold", "food", "coal", "iron", "oil", "stone", "water", "wood", "energy")
_RESOURCE_SET = frozenset(RESOURCE_NAMES)


@dataclass(frozen=True)
class CurriculumState:
    """Computed curriculum: the single object every env creator passes to C++.

    `all_builds=True` means «no building restriction» explicitly; in that case
    `allowed_builds` is informational only (conventionally ALL_IDS) and the C++
    gate ignores it. `stage_report` is report-only (obs feature, dumps).
    `all_resources` / `resource_weights` are the PR 4 soft priority weights
    (neutral = all 1.0, i.e. legacy behaviour bit-for-bit).
    `obs_version` selects the obs layout: 0 = 248-dim, 1 = 289-dim (frame),
    2 = 299-dim (+ направления к ближайшим ресурсам)
    (PR 5 «frame»: +9 effective weights +32 build_allowed bits). The version
    is fixed at env construction — a mid-run change is refused by C++.
    """

    all_builds: bool
    allowed_builds: tuple[str, ...] = ()
    all_resources: bool = True
    resource_weights: tuple[float, ...] = _FULL_WEIGHTS
    stage_report: int = 0
    obs_version: int = 2

    @classmethod
    def all(cls, stage_report: int = 0) -> "CurriculumState":
        """Unrestricted state (conventionally carries ALL_IDS for logging)."""
        return cls(True, tuple(ALL_IDS), True, _FULL_WEIGHTS, stage_report, 1)

    def to_dict(self) -> dict:
        """Transport form for C++ set_curriculum() and the GUI --curriculum JSON."""
        return {
            "all_builds": bool(self.all_builds),
            "allowed_builds": list(self.allowed_builds),
            "all_resources": bool(self.all_resources),
            "resource_weights": [float(w) for w in self.resource_weights],
            "stage": int(self.stage_report),
            "obs_version": int(self.obs_version),
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
            obs_version=int(d.get("obs_version", 2)),
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


def parse_resources(csv_or_list: str | List[str] | None) -> List[float] | None:
    """Normalise a resource-priority set to 9 soft weights (PR 4).

    ``None``/``""`` ⇒ ``None`` (all_resources: full legacy behaviour). Otherwise
    the listed ids get 1.0, the rest 0.0 (no extraction bonuses for them).
    Unknown names RAISE — same fail-loud rule as parse_unlock_ids.
    """
    if csv_or_list is None:
        return None
    if isinstance(csv_or_list, str):
        raw = csv_or_list.split(",")
    else:
        raw = [str(x) for x in csv_or_list]
    ids = [x.strip().lower() for x in raw if x.strip()]
    if not ids:
        return None
    unknown = sorted({x for x in ids if x not in _RESOURCE_SET})
    if unknown:
        raise ValueError(
            f"неизвестные ресурсы: {unknown}; доступны {list(RESOURCE_NAMES)}"
        )
    picked = set(ids)
    return [1.0 if r in picked else 0.0 for r in RESOURCE_NAMES]


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
    obs_version: int = 2,
) -> CurriculumState:
    """Compute the CurriculumState from (stage, manual set, checkbox).

    THE function every env instance must use (training, eval, watch) so a
    curriculum never depends on which code path created the environment. A full
    32-id set normalises to `all_builds=True` (same behaviour, explicit form).

    `resources`: PR 4 priority set (CSV/list/None). None/"" ⇒ all_resources
    (legacy behaviour); otherwise soft weights (1.0 listed, 0.0 rest). A full
    9-weight set normalises to all_resources, mirroring the building rule.
    `obs_version`: obs layout (0 = 248-dim, 1 = 289-dim frame, 2 = 299-dim:
    frame + dx/dy к ближайшим wood/coal/iron/oil/gold). Only 0/1/2 are accepted
    — anything else fails fast here, not in C++.
    """
    if int(obs_version) not in (0, 1, 2):
        raise ValueError(f"obs_version must be 0, 1 or 2, got {obs_version!r}")
    obs_i = int(obs_version)
    stage_i = max(0, int(stage))
    manual = parse_unlock_ids(unlock_ids)
    if use_curriculum_tab:
        ids = ids_for_stage(stage_i, manual)
    elif stage_i == 0:
        ids = list(ALL_IDS)
    else:
        ids = ids_for_stage(stage_i, None)
    weights = parse_resources(resources)
    all_res = weights is None or all(w == 1.0 for w in weights)
    res_tuple = _FULL_WEIGHTS if all_res else tuple(weights)  # type: ignore[arg-type]
    if set(ids) == _ALL_IDS_SET:
        return CurriculumState(True, tuple(ALL_IDS), all_res, res_tuple, stage_i, obs_i)
    return CurriculumState(False, tuple(ids), all_res, res_tuple, stage_i, obs_i)


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
        return {"curriculum_stage": None, "unlock_ids": None,
                "use_curriculum_tab": None, "resources": None,
                "obs_version": None}
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
    resources = pick("curriculum_resources")
    obs_version = pick("obs_version")
    return {
        "curriculum_stage": int(stage) if stage is not None else None,
        "unlock_ids": None if manual is None else str(manual),
        "use_curriculum_tab": None if use_tab is None else bool(use_tab),
        "resources": None if resources is None else str(resources),
        "obs_version": int(obs_version) if obs_version is not None else None,
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
        "resources": None,
        "obs_version": None,
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
    resources: str | List[str] | None = None,
) -> Dict[str, object]:
    """Resolve the effective curriculum for a model (explicit args win).

    Returns ``{"curriculum_stage", "unlock_ids", "use_curriculum_tab",
    "allowed", "resources"}`` where ``unlock_ids`` is the effective CSV
    ("" = no manual set), ``allowed`` is the resulting building-id list and
    ``resources`` is the effective priority set (None = all, legacy).
    """
    stored = read_curriculum_meta(model_dir) if model_dir is not None else {}
    if meta:
        for key, value in curriculum_from_meta(meta).items():
            if stored.get(key) is None and value is not None:
                stored[key] = value

    stage = curriculum_stage if curriculum_stage is not None else stored.get("curriculum_stage")
    manual_raw = unlock_ids if unlock_ids is not None else stored.get("unlock_ids")
    res_raw = resources if resources is not None else stored.get("resources")
    tab = use_curriculum_tab if use_curriculum_tab is not None else stored.get("use_curriculum_tab")

    stage_i = int(stage or 0)
    manual_csv = manual_ids_csv(manual_raw, True if tab is None else bool(tab))
    return {
        "curriculum_stage": stage_i,
        "unlock_ids": manual_csv,
        "use_curriculum_tab": tab,
        "allowed": allowed_ids(stage_i, manual_csv, True),
        "resources": res_raw,
    }


def resolve_state(
    model_dir,
    meta: Dict[str, object] | None = None,
    curriculum_stage: int | None = None,
    unlock_ids: str | List[str] | None = None,
    use_curriculum_tab: bool | None = None,
    resources: str | List[str] | None = None,
    obs_version: int = 2,
) -> CurriculumState:
    """resolve_curriculum + build_state in one call (eval/watch paths).

    Returns the CurriculumState to hand to the env constructor — «not stored»
    and «stored as empty» both resolve through the same code as training, so
    the eval/watch scenario can never silently differ from training.

    `obs_version` is explicit-only (never restored from meta): a stored v0
    must ERROR against a v1 env, not silently rebuild it — see
    check_obs_version_compat().
    """
    resolved = resolve_curriculum(
        model_dir,
        meta=meta,
        curriculum_stage=curriculum_stage,
        unlock_ids=unlock_ids,
        use_curriculum_tab=use_curriculum_tab,
        resources=resources,
    )
    # manual_csv is already checkbox-filtered, so the tab is trivially True here.
    # resources fall back to the stored run scenario (explicit args win).
    return build_state(
        int(resolved["curriculum_stage"]),  # type: ignore[arg-type]
        str(resolved["unlock_ids"]),
        True,
        resolved["resources"],  # type: ignore[arg-type]
        obs_version,
    )


# ── PR 5: obs-version compatibility ──────────────────────────────────────────
# Flat-obs size by layout version (n_build=32). Message-only: the real sizes
# come from C++ ColonyEnvCpp::obs_size(), this map only names mismatches.
_OBS_SIZE_BY_VERSION = {0: 248, 1: 289, 2: 299}

#: obs v2, последние 10 float в кадре — (dx, dy) к ближайшему тайлу каждого
#: типа из этого списка, делённые на размер карты. Порядок = C++
#: `NEAREST_LOT_TYPES` (include/colony/constants.h): wood, coal, iron, oil, gold.
#: Вода как была в хвосте v0 (индексы 246/247), так и осталась.
NEAREST_LOT_ORDER = (3, 4, 5, 6, 7)  # LT_WOOD, LT_COAL, LT_IRON, LT_OIL, LT_GOLD


def obs_size_for_version(version: int) -> int:
    """Flat-obs dimensionality: 0 → 248, 1 → 289, 2 → 299 (n_build=32)."""
    try:
        return _OBS_SIZE_BY_VERSION[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unknown obs_version: {version!r}") from None


def stored_obs_version(model_dir, meta: Dict[str, object] | None = None) -> int | None:
    """Obs version a checkpoint was trained with, from its meta files.

    Returns None when there is nothing to read (no meta files, no meta dict);
    meta files (or a non-empty meta dict) that predate obs_version mean a
    legacy v0 run and come back as 0.
    """
    stored = read_curriculum_meta(model_dir) if model_dir is not None else {}
    if meta:
        for key, value in curriculum_from_meta(meta).items():
            if stored.get(key) is None and value is not None:
                stored[key] = value
    if stored.get("obs_version") is not None:
        return int(stored["obs_version"])  # type: ignore[arg-type]
    if model_dir is not None:
        from pathlib import Path

        model_dir = Path(model_dir)
        if any((model_dir / name).exists() for name in _META_NAMES):
            return 0
    if meta:
        return 0
    return None


def check_obs_version_compat(
    stored: int | None, env_version: int, *, ckpt_path: str = "checkpoint"
) -> None:
    """Fail fast when a checkpoint's obs layout differs from the env's.

    A v0 policy fed v1 frames (or vice versa) would die in a cryptic matmul
    — or worse, run with silently misaligned features. None (unknown) skips
    the check; the policy-vs-env size check below still guards that case.
    """
    if stored is None or int(stored) == int(env_version):
        return
    raise RuntimeError(
        f"obs mismatch: checkpoint '{ckpt_path}' was trained on obs v{stored} "
        f"({obs_size_for_version(stored)}-dim), but the env serves obs v{env_version} "
        f"({obs_size_for_version(env_version)}-dim); re-run with "
        f"--obs-version {stored} or retrain the model"
    )


def ckpt_flat_width(state_dict) -> int | None:
    """First-layer flat input width of a checkpoint state_dict.

    None for CNN-only policies (no flat input at all). Used by the resume
    paths to compare the checkpoint's real tensor width against the env.
    """
    for key in ("trunk.0.weight", "flat_trunk.0.weight", "flat_proj.weight"):
        w = state_dict.get(key) if hasattr(state_dict, "get") else None
        if w is not None:
            try:
                return int(w.shape[1])
            except (AttributeError, IndexError, TypeError):
                return None
    return None


def check_policy_obs_compat(
    policy_flat_dim: int, env_obs_size: int, *, ckpt_path: str = "checkpoint"
) -> None:
    """Fail fast when a flat policy's input width differs from the env's obs.

    Catches the no-meta case (a bare .pt with no version recorded) by
    comparing the actual tensor widths instead of version tags.
    """
    if int(policy_flat_dim) == int(env_obs_size):
        return
    hint = ""
    for ver, size in sorted(_OBS_SIZE_BY_VERSION.items()):
        if size == int(policy_flat_dim):
            hint = f" (the policy looks like obs v{ver}: try --obs-version {ver})"
            break
    raise RuntimeError(
        f"obs mismatch: policy '{ckpt_path}' expects a {policy_flat_dim}-dim flat "
        f"observation, but the env serves {env_obs_size}{hint}"
    )


def allowed_buildings_for_stage(stage: int) -> List[str]:
    """Legacy helper returning BUILD_ prefixed names (EnvManager compatibility)."""
    ids = ids_for_stage(stage)
    # EnvManager expects BUILD_* + managers + DAY/WEEK etc. Keep that expansion there.
    return ids
