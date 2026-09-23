// W6 diagnostic: separately verify what an ordinary rollout did and what the
// placement mechanics allow. The first part steers a synthetic policy towards
// water (NOT a trained model); after WaterChannel becomes legal, the second part
// intentionally asks Game::build to put a Road on that same legal water cell.
// No Road validator or action mask is modified by this probe.
#include "colony/data.h"
#include "colony/env.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int failures = 0;
static void check(bool ok, const char* label) {
    std::printf("%s %s\n", ok ? "[ok]  " : "[FAIL]", label);
    if (!ok) ++failures;
}

static const BaseData* find_data(const ColonyEnvCpp& env, const char* id) {
    for (const BaseData* d : env.build_data()) if (d->id == id) return d;
    return nullptr;
}
static int find_idx(const ColonyEnvCpp& env, const char* id) {
    for (int i = 0; i < env.n_build(); ++i)
        if (env.build_ids()[(size_t)i] == id) return i;
    return -1;
}
static int legal_cells(const ColonyEnvCpp& env, const BaseData& d) {
    const Game& g = env.game();
    int n = 0;
    for (int y = 0; y < g.map_size(); ++y)
        for (int x = 0; x < g.map_size(); ++x)
            if (g.can_build_at(d, x, y).first) ++n;
    return n;
}

int main() {
    const auto base_data = load_base_data("configs/bases.json");
    const auto events = load_events("configs/events.json");
    int reached = 0, road_allowed = 0, road_built = 0, target_removed = 0;
    int aggregate_decreased = 0, aggregate_same = 0, aggregate_increased = 0;
    int roads_during_steering = 0, water_roads_during_steering = 0;

    for (int64_t seed : {1, 7, 21, 42, 100, 777, 31337, 2026}) {
        ColonyEnvCpp env(base_data, events, seed, 200);
        env.reset(seed);
        const BaseData* wc = find_data(env, "WaterChannel");
        const BaseData* road = find_data(env, "Road");
        const int road_idx = find_idx(env, "Road");
        const int road_dir_base = env.road_dir_base();
        bool measured = false;

        for (int step = 0; step < 400 && !measured; ++step) {
            const Game& g = env.game();
            const int wc_sites = legal_cells(env, *wc);
            if (wc_sites > 0) {
                int wx = -1, wy = -1;
                for (int y = 0; y < g.map_size() && wx < 0; ++y) {
                    for (int x = 0; x < g.map_size(); ++x) {
                        if (g.earth.lot(x, y) == LT_WATER &&
                            g.can_build_at(*wc, x, y).first) {
                            wx = x;
                            wy = y;
                            break;
                        }
                    }
                }
                if (wx < 0) break;
                ++reached;
                const bool can_road = g.can_build_at(*road, wx, wy).first;
                if (can_road) ++road_allowed;
                const int before = wc_sites;
                const auto built = env.game().build("Road", wx, wy);
                if (built.first) ++road_built;
                const int after = legal_cells(env, *wc);
                const bool removed = !env.game().can_build_at(*wc, wx, wy).first;
                if (removed) ++target_removed;
                if (after < before) ++aggregate_decreased;
                else if (after == before) ++aggregate_same;
                else ++aggregate_increased;
                std::printf("seed=%-5lld roads_to_water=%-3d site=(%d,%d) "
                            "road_legal=%d build=%d legal_WC=%d->%d target_removed=%d\n",
                            (long long)seed, roads_during_steering, wx, wy,
                            (int)can_road, (int)built.first, before, after, (int)removed);
                measured = true;
                continue;
            }

            // Deliberately synthetic water-seeking controller matching the
            // W9 probe's direction choice. This establishes reachability only;
            // it says nothing about a learned policy.
            const auto mask = env.action_mask();
            int best = -1;
            double best_score = -1e18;
            const int ms = g.map_size();
            const int bx = g.earth.init_sel_x, by = g.earth.init_sel_y;
            int wx = -1, wy = -1;
            long long best_d2 = -1;
            for (int y = 0; y < ms; ++y) for (int x = 0; x < ms; ++x) {
                if (g.earth.lot(x, y) != LT_WATER) continue;
                const long long dx = x - bx, dy = y - by;
                const long long d2 = dx * dx + dy * dy;
                if (best_d2 < 0 || d2 < best_d2) { best_d2 = d2; wx = x; wy = y; }
            }
            const double wdx = best_d2 < 0 ? 0.0 : (wx - bx) / (double)ms;
            const double wdy = best_d2 < 0 ? 0.0 : (wy - by) / (double)ms;
            static const int dir_x[4] = {1, -1, 0, 0};
            static const int dir_y[4] = {0, 0, 1, -1};
            for (int d = 0; d < 4; ++d) {
                if (mask[(size_t)(road_dir_base + d)] == 0.0f) continue;
                const double score = -(std::fabs(wdx - dir_x[d] * 0.05) +
                                       std::fabs(wdy - dir_y[d] * 0.05));
                if (score > best_score) { best_score = score; best = d; }
            }
            int action = A_DAY;
            if (best >= 0) action = road_dir_base + best;
            else if (road_idx >= 0 && mask[(size_t)(A_BUILD0 + road_idx)] != 0.0f)
                action = A_BUILD0 + road_idx;

            const int roads_before = (int)std::count_if(
                g.bases.begin(), g.bases.end(), [](const Base& b) { return b.data->id == ROAD_ID; });
            std::vector<int8_t> lots_before(g.earth.lots().begin(), g.earth.lots().end());
            env.step(action);
            const Game& after = env.game();
            const int roads_after = (int)std::count_if(
                after.bases.begin(), after.bases.end(), [](const Base& b) { return b.data->id == ROAD_ID; });
            if (roads_after > roads_before) {
                ++roads_during_steering;
                const Base& placed = after.bases.back();
                if (lots_before[(size_t)placed.y * ms + placed.x] == LT_WATER)
                    ++water_roads_during_steering;
            }
        }
    }

    std::printf("\nsteering baseline: roads=%d, placed_on_water=%d\n",
                roads_during_steering, water_roads_during_steering);
    std::printf("targeted sites: reached=%d road_allowed=%d road_built=%d target_removed=%d\n",
                reached, road_allowed, road_built, target_removed);
    std::printf("aggregate legal WaterChannel sites: decreased=%d same=%d increased=%d\n",
                aggregate_decreased, aggregate_same, aggregate_increased);
    check(reached == 8, "synthetic controller reaches an eligible WaterChannel site on all W9 seeds");
    check(road_allowed == 8 && road_built == 8,
          "Road is mechanically legal and can be built on that water site (all 8 seeds)");
    check(target_removed == 8,
          "a Road occupies and invalidates the exact water cell legal for WaterChannel");
    check(aggregate_decreased > 0 && aggregate_increased > 0,
          "net legal-water count can decrease or increase as connectivity changes");
    std::printf("ROAD WATER OCCUPANCY: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
