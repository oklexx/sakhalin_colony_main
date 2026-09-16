// Standalone probe: does the env actually let an agent discover & build the
// WaterChannel, and is there any reward gradient pointing at water?
//
// Build (repo root):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/probe \
//       tests/cpp/water_probe.cpp src/env.cpp src/data.cpp src/resources.cpp \
//       src/game.cpp src/rewards.cpp src/rng.cpp src/earth.cpp
#include "colony/env.h"
#include "colony/data.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int find_id(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[i] == id) return i;
    return -1;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("=== Q1: how far is water from the start, and is WaterChannel buildable? ===\n");
    printf("%-6s %-10s %-9s %-11s %-11s %-10s\n", "seed", "map", "wc_earth", "wc_masked", "road_mask", "water_dist");
    for (int64_t seed : {1, 7, 21, 42, 100, 777}) {
        ColonyEnvCpp e(bd, ed, seed, 200);
        e.reset(seed);
        e.game().money = 5'000'000;  // affordability out of the equation
        auto mask = e.action_mask();
        int wc = find_id(e, "WaterChannel");
        int rd = find_id(e, "Road");
        const Game& g = e.game();
        int ms = g.map_size();
        int bx = g.earth.init_sel_x, by = g.earth.init_sel_y;
        double best = 1e9;
        const int8_t* lots = g.earth.lots().data();
        for (int y = 0; y < ms; y++)
            for (int x = 0; x < ms; x++)
                if (lots[(size_t)y * ms + x] == LT_WATER)
                    best = std::min(best, std::hypot((double)(x - bx), (double)(y - by)));
        printf("%-6lld %-10d %-9d %-11s %-11s %-10.2f\n", (long long)seed, ms,
               e.build_data()[wc]->need_earth,
               mask[A_BUILD0 + wc] != 0.0f ? "OPEN" : "MASKED",
               mask[A_BUILD0 + rd] != 0.0f ? "OPEN" : "MASKED", best);
    }

    printf("\n=== Q2: reward economics of the only road to water ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().money = 5'000'000;
        int rd = find_id(e, "Road");
        auto s0 = e.step(A_BUILD0 + rd);
        printf("Road build step reward            = %+.4f\n", s0.rew);
        auto s1 = e.step(A_BUILD0 + rd);
        printf("Road build step reward (2nd)      = %+.4f\n", s1.rew);
        int wc = find_id(e, "WaterChannel");
        // force-select WaterChannel while masked, to see the penalty path
        auto s2 = e.step(A_BUILD0 + wc);
        printf("WaterChannel while unreachable    = %+.4f\n", s2.rew);
        auto s3 = e.step(0);  // DAY
        printf("DAY step reward                   = %+.4f\n", s3.rew);
    }

    printf("\n=== Q3: can a road even be placed on a WATER tile? ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().money = 5'000'000;
        const Game& g = e.game();
        int ms = g.map_size();
        const int8_t* lots = g.earth.lots().data();
        // does lot_ok() accept a water cell for Road (need_earth=LT_EVERYWHERE)?
        int wx = -1, wy = -1;
        for (int y = 0; y < ms && wx < 0; y++)
            for (int x = 0; x < ms; x++)
                if (lots[(size_t)y * ms + x] == LT_WATER) { wx = x; wy = y; break; }
        printf("first water tile at (%d,%d)\n", wx, wy);
        bool ok = e.lot_ok(wx, wy, LT_EVERYWHERE, false);
        printf("lot_ok(water, LT_EVERYWHERE) [ignores connectivity] = %s\n", ok ? "true" : "false");
        bool okn = e.lot_ok(wx, wy, LT_NORMAL, false);
        printf("lot_ok(water, LT_NORMAL)                          = %s\n", okn ? "true" : "false");
        bool okw = e.lot_ok(wx, wy, LT_WATER, false);
        printf("lot_ok(water, LT_WATER)                           = %s\n", okw ? "true" : "false");
    }

    printf("\n=== Q4: minimap 32x32 over a 200 map — water localisation error ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        auto mm = e.minimap();
        const int G = 32;
        printf("minimap vector size = %zu (expect %d = 8*32*32)\n", mm.size(), 8 * G * G);
        int ms = e.game().map_size();
        double scale = (double)ms / G;
        printf("map_size=%d -> 1 minimap cell = %.2f map cells\n", ms, scale);
        // count water-marked cells on channel 1
        int n_water_cells = 0;
        for (int i = 0; i < G * G; i++) if (mm[(size_t)G * G + i] > 0.0f) n_water_cells++;
        // true water tiles
        const int8_t* lots = e.game().earth.lots().data();
        int n_true_water = 0;
        for (int i = 0; i < ms * ms; i++) if (lots[i] == LT_WATER) n_true_water++;
        printf("true water tiles = %d ; minimap cells flagged water = %d\n", n_true_water, n_water_cells);
        printf("=> each flagged cell covers %.0f map tiles; centroid error up to %.1f cells\n",
               scale * scale, scale);
    }

    printf("\n=== Q5: water_dx / water_dy values (flat obs tail) ===\n");
    {
        for (int64_t seed : {1, 42}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            auto o = e.obs();
            printf("seed %lld obs_size=%zu ", (long long)seed, o.size());
            printf("declared obs_size()=%d\n", e.obs_size());
            // water_dx/dy are pushed 2 before the v1 frame (9+32) when obs_version>=1
            int n = (int)o.size();
            int frame = 9 + e.n_build();
            printf("  tail[-2-frame]=%.4f  tail[-1-frame]=%.4f  (water_dx, water_dy)\n",
                   o[n - frame - 2], o[n - frame - 1]);
        }
    }

    printf("\n=== Q6: is WaterChannel reachable at all via roads? greedy road walk ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().money = 5'000'000;
        int wc = find_id(e, "WaterChannel");
        int rd = find_id(e, "Road");
        bool ever_open = false;
        int steps = 0;
        for (int k = 0; k < 400; k++) {
            auto m = e.action_mask();
            if (m[A_BUILD0 + wc] != 0.0f) { ever_open = true; steps = k; break; }
            if (m[A_BUILD0 + rd] == 0.0f) { printf("Road got masked at k=%d\n", k); break; }
            e.step(A_BUILD0 + rd);
        }
        printf("WaterChannel became OPEN after %d blind Road builds: %s\n",
               steps, ever_open ? "YES" : "NO");
        printf("NOTE: find_lot picks the cell, the agent cannot aim the road.\n");
    }
    return 0;
}
