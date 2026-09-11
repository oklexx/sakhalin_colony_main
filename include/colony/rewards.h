#pragma once
/**
 * @file rewards.h
 * @brief Reward helpers extracted from env.cpp for maintainability.
 *
 * Contains pure functions that compute proximity/provider/prerequisite
 * bonuses. No state, no side effects — easy to unit-test.
 * Part of the C++ refactoring (Phase 6) — keeps env.cpp focused on
 * step/obs/minimap control flow.
 */
#include <string>
#include <vector>

namespace colony {

struct BaseData;
class Game;
struct Base;

// Returns land-type constant (LT_*) that gives proximity bonus for building id,
// or LT_NONE if no bonus.
int proximity_land_for(const std::string& id);

// Bonus for building a producer that current idle buildings need.
// 0.5 per beneficiary, capped at 3.0. Returns 0 if no provider bonus.
double provider_bonus(const Game& g, const BaseData& d);

// Bonus for building infrastructure that future building types depend on.
// 0.3 per dependent type, capped at 2.0.
double prerequisite_bonus(const Game& g, const BaseData& d,
                          const std::vector<const BaseData*>& all_build_data);

}  // namespace colony
