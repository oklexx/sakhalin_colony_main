// Regression checks for episode diagnostics: metric semantics, build/resource
// details, and terminal-info transport through the native vector environment.
#include "colony/data.h"
#include "colony/env.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <json.hpp>

using namespace colony;
using nlohmann::json;

static int failures = 0;
static void check(bool ok, const char* label) {
    std::printf("%s %s\n", ok ? "[ok]  " : "[FAIL]", label);
    if (!ok) ++failures;
}

static int find_build(const ColonyEnvCpp& env, const std::string& id) {
    for (int i = 0; i < env.n_build(); ++i)
        if (env.build_ids()[(size_t)i] == id) return i;
    return -1;
}

static BaseData make_test_base(const std::string& id, int consume, int produce) {
    BaseData d;
    d.id = id;
    d.caption = id;
    d.price = 1;
    d.build_time = 0;
    d.need_workers = 0;
    d.need_earth = LT_EVERYWHERE;
    for (int s = 0; s < 4; ++s) d.work_seasons[s] = true;
    if (consume >= 0) d.consume[consume] = 1;
    if (produce >= 0) d.profit[produce] = 1;
    return d;
}

static bool build_first_legal(Game& game, const BaseData& d) {
    for (int y = 0; y < game.map_size(); ++y) {
        for (int x = 0; x < game.map_size(); ++x) {
            if (!game.can_build_at(d, x, y).first) continue;
            if (game.build(d.id, x, y).first) return true;
        }
    }
    return false;
}

int main() {
    auto base_data = load_base_data("configs/bases.json");
    auto events = load_events("configs/events.json");

    // An ordinary successful env action must preserve exact floating return and
    // expose a type-level build count (Road counts as an actual build).
    ColonyEnvCpp built_env(base_data, events, 42, 200);
    built_env.reset(42);
    const int road_idx = find_build(built_env, "Road");
    const auto road_mask = built_env.action_mask();
    bool road_built = road_idx >= 0 && road_mask[(size_t)(A_BUILD0 + road_idx)] != 0.0f;
    ColonyEnvCpp::StepOut road_out;
    if (road_built) {
        road_out = built_env.step(A_BUILD0 + road_idx);
        road_built = road_out.metrics.builds_by_type.count("Road") == 1 &&
                     road_out.metrics.builds_by_type.at("Road") == 1;
    }
    check(road_built && road_out.metrics.total_builds == 1,
          "successful build is counted by type and in total_builds");
    check(std::fabs(road_out.metrics.total_reward - road_out.ep_return) < 1e-9,
          "EpisodeMetrics.total_reward preserves fractional return");

    // A known three-resource chain: water -> wood -> food. It verifies that
    // chains_activated is a count of active consumer/resource pairs (2), while
    // max_chain_depth is the longest simple resource-conversion path (2 edges).
    auto chain_data = base_data;
    const BaseData source = make_test_base("DiagWaterSource", -1, 6); // water
    const BaseData converter = make_test_base("DiagWoodConverter", 6, 7); // water -> wood
    const BaseData sink = make_test_base("DiagFoodConverter", 7, 1); // wood -> food
    chain_data.push_back(source);
    chain_data.push_back(converter);
    chain_data.push_back(sink);
    ColonyEnvCpp chain_env(chain_data, events, 42, 120);
    chain_env.reset(42);
    Game& game = chain_env.game();
    game.money = 1000000;
    game.sunduk[6] = 1000;
    game.sunduk[7] = 1000;
    const bool chain_bases_built = build_first_legal(game, source) &&
                                   build_first_legal(game, converter) &&
                                   build_first_legal(game, sink);
    const auto chain_out = chain_env.step(A_DAY);
    check(chain_bases_built, "diagnostic producer/consumer fixtures are legal");
    check(chain_out.metrics.chains_activated == 2,
          "chains_activated counts unique active consumer/resource pairs");
    check(chain_out.metrics.max_chain_depth == 2,
          "max_chain_depth measures the two active resource conversions");
    check(chain_out.metrics.reached_resources == 3 &&
              chain_out.metrics.reached_resource_ids ==
                  std::vector<std::string>({"food", "water", "wood"}),
          "reached resources include canonical resource ids, not only a count");

    // Exercise the serialized terminal info used by Python CppVecEnv. A no-op
    // episode ends by game-over or the hard step limit; either path must retain
    // the final episode metrics before C++ auto-resets the native env.
    ColonyVecEnvCpp vec(base_data, events, 1, 7, 64, Curriculum(),
                        RewardConfig(), 1, "normal", true);
    vec.reset_batch({7});
    bool got_episode = false;
    for (int t = 0; t < MAX_STEPS + 1 && !got_episode; ++t) {
        vec.step_async_batch({A_DAY});
        StepBatchResult batch = vec.step_wait_batch();
        if (batch.terminateds[0] || batch.trunceds[0]) {
            const json info = json::parse(batch.infos[0]);
            const json& episode = info.at("episode");
            const json& metrics = episode.at("metrics");
            got_episode = episode.contains("seed") && episode.contains("days") &&
                          episode.contains("people") && episode.contains("money") &&
                          episode.contains("bases") && metrics.contains("chains_activated") &&
                          metrics.contains("max_chain_depth") &&
                          metrics.contains("builds_by_type") &&
                          metrics.contains("reached_resource_ids");
        }
    }
    check(got_episode,
          "terminal VecEnv info serializes seed, chains, builds and reached resources");

    std::printf("EPISODE METRICS CHECKS: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
