#include "colony/rewards.h"
#include "colony/bases.h"
#include "colony/constants.h"
#include "colony/game.h"

#include <algorithm>

namespace colony {

// Proximity bonus: building id → nearby land type that gives bonus.
//
// NOTE: the search in env.cpp is a ±3 window that excludes the building's own
// cell, so a building placed ON its required lot type still needs a *neighbour*
// of that type to score. That is why WaterChannel (need_earth=LT_WATER) is
// deliberately absent: it always sits on water, so a LT_WATER proximity check
// would fire unconditionally and carry no information. Its siting incentive is
// water_need_bonus in env.cpp instead.
//
// KNOWN ODDITY (left as-is, reward balance): WaterMill and PowerStation map to
// LT_OIL. WaterMill wanting oil rather than water looks like a copy-paste, but
// changing it moves the reward landscape, so it needs a measured A/B first.
int proximity_land_for(const std::string& id) {
    if (id == "Farm" || id == "Garden" || id == "CowFarm" || id == "Hothouse" ||
        id == "Goldmine" || id == "Apiary" || id == "Puerperal")
        return LT_WATER;
    if (id == "Coalmine" || id == "CoalCut" || id == "HuntingLand" ||
        id == "Mushroom" || id == "BigFarm")
        return LT_WOOD;
    // BigFarm is matched by the LT_WOOD branch above, so a second BigFarm case
    // here (returning LT_IRON) was unreachable dead code. Removed.
    if (id == "Ironmine" || id == "Sawmill" || id == "BigSawmill" ||
        id == "PowerStation" || id == "WaterMill")
        return LT_OIL;
    return LT_NONE;
}

double provider_bonus(const Game& g, const BaseData& d) {
    int beneficiaries = 0;
    for (const Base& b : g.bases) {
        if (!b.need_sunduk) continue;
        for (int r = 0; r < SUNDUK_SIZE; ++r) {
            if (d.profit[r] > 0 && b.data->consume[r] > 0) {
                ++beneficiaries;
                break;
            }
        }
    }
    return std::min(3.0, 0.5 * beneficiaries);
}

double prerequisite_bonus(const Game& g, const BaseData& d,
                          const std::vector<const BaseData*>& all_build_data) {
    (void)g;  // currently unused — kept for future Game-aware logic
    int dependents = 0;
    for (const BaseData* other : all_build_data) {
        if (other->id == d.id) continue;
        for (int r = 0; r < SUNDUK_SIZE; ++r) {
            if (d.profit[r] > 0 && other->consume[r] > 0) {
                ++dependents;
                break;
            }
        }
    }
    return std::min(2.0, 0.3 * dependents);
}

}  // namespace colony
