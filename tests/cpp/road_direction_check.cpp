// Standalone harness for the directional road actions (no Python, no raylib).
//
// Why: the plain BUILD:Road action places a road at the BFS-first legal cell,
// which the agent cannot aim. Water sat 7-14 cells from the start but the
// WaterChannel action was masked on 0/2000 random steps, so no policy could
// ever reach it. ROAD_E/W/S/N give the frontier a steerable direction.
//
// Build (from the repo root, Linux/macOS):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/road_check \
//       tests/cpp/road_direction_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/road_check        # run from the repo root: configs/*.json are loaded
#include "colony/env.h"
#include "colony/data.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    printf("%s  %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) failures++;
}
static int find_id(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[i] == id) return i;
    return -1;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    // ── D1: action space grew by exactly N_ROAD_DIRS, layout intact ──────────
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        int expected = A_BUILD0 + e.n_build() + N_MANAGERS + N_ROAD_DIRS;
        printf("[info] n_actions=%d expected=%d road_dir_base=%d\n",
               e.n_actions(), expected, e.road_dir_base());
        check(e.n_actions() == expected, "D1 n_actions == 2 + n_build + managers + 4");
        check(e.road_dir_base() == A_BUILD0 + e.n_build() + N_MANAGERS,
              "D1 road directions sit after the manager block");
        check(e.action_mask().size() == (size_t)e.n_actions(),
              "D1 action_mask covers the new actions");
        // The A_BUILD0+i <-> build_ids_[i] invariant must be untouched.
        check(e.build_ids()[find_id(e, "Road")] == ROAD_ID,
              "D1 build index of Road is unchanged");
    }

    // ── D2: the four directions are legal at reset and actually differ ──────
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().money = 5'000'000;
        int rdb = e.road_dir_base();
        int open = 0;
        for (int d = 0; d < N_ROAD_DIRS; d++)
            if (e.action_mask()[rdb + d] != 0.0f) open++;
        printf("[info] directional road actions open at reset: %d/%d\n", open, N_ROAD_DIRS);
        check(open == N_ROAD_DIRS, "D2 all four directions are legal at reset");

        // Each direction must place its road somewhere different.
        std::vector<std::pair<int, int>> placed;
        for (int d = 0; d < N_ROAD_DIRS; d++) {
            ColonyEnvCpp f(bd, ed, 42, 200);
            f.reset(42);
            f.game().money = 5'000'000;
            f.step(f.road_dir_base() + d);
            for (const Base& b : f.game().bases)
                if (b.data->id == ROAD_ID) placed.push_back({b.x, b.y});
        }
        check(placed.size() == N_ROAD_DIRS, "D2 each direction placed exactly one road");
        bool all_distinct = true;
        for (size_t i = 0; i < placed.size(); i++)
            for (size_t j = i + 1; j < placed.size(); j++)
                if (placed[i] == placed[j]) all_distinct = false;
        for (size_t i = 0; i < placed.size(); i++)
            printf("[info]   dir %d -> road at (%d,%d)\n", (int)i, placed[i].first, placed[i].second);
        check(all_distinct, "D2 the four directions pick different cells");
    }

    // ── D3: the curriculum still gates the directional roads ────────────────
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        Curriculum c;
        c.all_builds = false;
        c.allowed_builds = {"Farm"};  // Road NOT allowed
        e.set_curriculum(c);
        int rdb = e.road_dir_base();
        bool any_open = false;
        for (int d = 0; d < N_ROAD_DIRS; d++)
            if (e.action_mask()[rdb + d] != 0.0f) any_open = true;
        check(!any_open, "D3 directional roads are masked when Road is locked");
    }

    // ── D4: walking water_dx/water_dy reaches the water and builds it ───────
    {
        int reached = 0, total_seeds = 0, min_roads = 1 << 30, max_roads = 0;
        for (int64_t seed : {1, 7, 21, 42, 777}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            int wc = find_id(e, "WaterChannel");
            int rdb = e.road_dir_base();
            int roads = 0, built = 0;
            for (int k = 0; k < 400; k++) {
                auto o = e.obs();
                int n = (int)o.size();
                double wdx = o[n - 2], wdy = o[n - 1];
                auto m = e.action_mask();
                int act = 0;
                if (m[A_BUILD0 + wc] != 0.0f) {
                    act = A_BUILD0 + wc;
                } else {
                    int best = -1; double bs = -1e18;
                    for (int d = 0; d < N_ROAD_DIRS; d++) {
                        if (m[rdb + d] == 0.0f) continue;
                        double sc = -(std::fabs(wdx - ROAD_DIR_DX[d] * 0.05) +
                                      std::fabs(wdy - ROAD_DIR_DY[d] * 0.05));
                        if (sc > bs) { bs = sc; best = d; }
                    }
                    if (best < 0) break;
                    act = rdb + best;
                    roads++;
                }
                auto s = e.step(act);
                if (act == A_BUILD0 + wc) {
                    for (const Base& b : e.game().bases)
                        if (b.data->id == "WaterChannel") built++;
                    break;
                }
                if (s.terminated || s.truncated) break;
            }
            total_seeds++;
            if (built > 0) {
                reached++;
                min_roads = std::min(min_roads, roads);
                max_roads = std::max(max_roads, roads);
            }
            printf("[info] seed %-4lld roads=%-4d waterchannel_built=%d\n",
                   (long long)seed, roads, built);
        }
        printf("[info] reached water on %d/%d seeds, roads used %d..%d\n",
               reached, total_seeds, reached ? min_roads : -1, reached ? max_roads : -1);
        check(reached >= 4, "D4 a water_dx/dy-greedy policy reaches the water on >=4/5 seeds");
        check(reached == 0 || max_roads < 83,
              "D4 directional roads need fewer steps than the 83 blind BFS roads");
    }

    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
