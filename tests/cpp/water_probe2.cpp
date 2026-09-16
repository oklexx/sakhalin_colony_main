// Probe 2: can the WaterChannel actually be BUILT once unmasked, and what is
// the true cost/benefit of the 83 blind roads needed to get there?
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

    printf("=== A: correct water_dx / water_dy (obs_version=0 -> last 2 slots) ===\n");
    {
        for (int64_t seed : {1, 42}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            auto o = e.obs();
            int n = (int)o.size();
            printf("seed %-4lld obs_size=%d  water_dx=%+.4f water_dy=%+.4f\n",
                   (long long)seed, n, o[n - 2], o[n - 1]);
            const Game& g = e.game();
            printf("           init_sel=(%d,%d) map=%d  dx,dy in cells = (%.0f,%.0f)\n",
                   g.earth.init_sel_x, g.earth.init_sel_y, g.map_size(),
                   o[n - 2] * g.map_size(), o[n - 1] * g.map_size());
        }
        // and with obs_version = 1 (what Python CurriculumState defaults to)
        Curriculum c; c.obs_version = 1;
        ColonyEnvCpp e(bd, ed, 42, 200, c);
        e.reset(42);
        auto o = e.obs();
        int n = (int)o.size();
        int frame = 9 + e.n_build();
        printf("obs_version=1: obs_size=%d (declared %d) water_dx=%+.4f water_dy=%+.4f\n",
               n, e.obs_size(), o[n - frame - 2], o[n - frame - 1]);
    }

    printf("\n=== B: build the roads, then actually BUILD the WaterChannel ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().money = 5'000'000;
        int wc = find_id(e, "WaterChannel");
        int rd = find_id(e, "Road");
        double road_rew = 0.0;
        int k = 0;
        for (; k < 400; k++) {
            auto m = e.action_mask();
            if (m[A_BUILD0 + wc] != 0.0f) break;
            if (m[A_BUILD0 + rd] == 0.0f) { printf("road masked at %d\n", k); break; }
            auto s = e.step(A_BUILD0 + rd);
            road_rew += s.rew;
        }
        printf("roads built = %d, cumulative road reward = %+.3f\n", k, road_rew);
        auto st = e.stats();
        printf("after roads: days=%lld people=%lld bases=%lld money=%lld\n",
               (long long)st.days, (long long)st.people, (long long)st.bases, (long long)st.money);
        int64_t water0 = e.game().sunduk[WATER];
        auto s = e.step(A_BUILD0 + wc);
        printf("WaterChannel step reward = %+.4f\n", s.rew);
        bool built = false; int wx = -1, wy = -1;
        for (const Base& b : e.game().bases)
            if (b.data->id == "WaterChannel") { built = true; wx = b.x; wy = b.y; }
        printf("WaterChannel actually built = %s at (%d,%d)\n", built ? "YES" : "NO", wx, wy);
        if (built) {
            printf("  built on lot type = %d (LT_WATER=2)\n", e.game().earth.lot(wx, wy));
            printf("  water stock before=%lld\n", (long long)water0);
            for (int d = 0; d < 120; d++) e.step(0);
            printf("  water stock after 120 days=%lld (WaterChannel works only summer/autumn/spring)\n",
                   (long long)e.game().sunduk[WATER]);
        }
    }

    printf("\n=== C: is road placement water-directed at all? ===\n");
    {
        // Two maps with water in different places -> same road sequence?
        for (int64_t seed : {1, 999}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            e.game().money = 5'000'000;
            int rd = find_id(e, "Road");
            std::vector<std::pair<int,int>> road_cells;
            for (int k = 0; k < 12; k++) {
                int before = (int)e.game().bases.size();
                e.step(A_BUILD0 + rd);
                for (const Base& b : e.game().bases)
                    if (b.data->id == "Road" && (int)road_cells.size() < before) {}
                // find newest road
                for (const Base& b : e.game().bases)
                    if (b.data->id == "Road") road_cells.push_back({b.x, b.y});
            }
            printf("seed %lld: first 12 road placements end near (%d,%d)\n",
                   (long long)seed, road_cells.empty()?-1:road_cells.back().first,
                   road_cells.empty()?-1:road_cells.back().second);
        }
        printf("=> find_lot() is a BFS from the colony; it never looks at water.\n");
    }

    printf("\n=== D: how many roads on average, across seeds? ===\n");
    {
        for (int64_t seed : {1, 7, 21, 42, 100, 777}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            e.game().money = 5'000'000;
            int wc = find_id(e, "WaterChannel"), rd = find_id(e, "Road");
            int k = 0; bool ok = false;
            for (; k < 600; k++) {
                auto m = e.action_mask();
                if (m[A_BUILD0 + wc] != 0.0f) { ok = true; break; }
                if (m[A_BUILD0 + rd] == 0.0f) break;
                e.step(A_BUILD0 + rd);
            }
            auto st = e.stats();
            printf("seed %-4lld roads_to_water=%-4d ok=%s  days=%-6lld people=%-4lld alive=%s\n",
                   (long long)seed, k, ok ? "Y" : "N", (long long)st.days,
                   (long long)st.people, st.people > 0 ? "yes" : "NO");
        }
    }
    return 0;
}
