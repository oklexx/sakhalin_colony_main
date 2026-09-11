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

def ids_for_stage(stage: int) -> List[str]:
    """Cumulative ids up to stage (0 = all)."""
    if stage == 0:
        return list(ALL_IDS)
    out: List[str] = []
    for s in range(1, stage + 1):
        out.extend(STAGE_MAP.get(s, []))
    # de-duplicate while preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for bid in out:
        if bid not in seen:
            seen.add(bid)
            uniq.append(bid)
    return uniq

def allowed_buildings_for_stage(stage: int) -> List[str]:
    """Legacy helper returning BUILD_ prefixed names (EnvManager compatibility)."""
    ids = ids_for_stage(stage)
    # EnvManager expects BUILD_* + managers + DAY/WEEK etc. Keep that expansion there.
    return ids
