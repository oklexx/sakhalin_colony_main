// Probe 6: with the ROAD_E/W/S/N actions, can a policy that simply reads
// water_dx/water_dy reach the water and build the WaterChannel?
#include "colony/env.h"
#include "colony/data.h"
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

    printf("=== n_actions before/after ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        printf("n_build=%d n_actions=%d (was 45), road_dir_base=%d\n",
               e.n_build(), e.n_actions(), e.road_dir_base());
        auto m = e.action_mask();
        int open_dirs = 0;
        for (int d = 0; d < N_ROAD_DIRS; d++)
            if (m[e.road_dir_base() + d] != 0.0f) open_dirs++;
        printf("directional road actions open at reset: %d/4\n", open_dirs);
    }

    printf("\n=== greedy 'walk to water' policy using water_dx/water_dy ===\n");
    printf("%-6s %-8s %-8s %-7s %-8s %-9s %-8s\n",
           "seed", "steps", "roads", "wc?", "water", "food", "reward");
    for (int64_t seed : {1, 7, 21, 42, 100, 777}) {
        ColonyEnvCpp e(bd, ed, seed, 200);
        e.reset(seed);
        int wc = find_id(e, "WaterChannel"), f = find_id(e, "Farm"), g = find_id(e, "Garden");
        int rdb = e.road_dir_base();
        double total = 0.0;
        int steps = 0, roads = 0, built_wc = 0;

        for (int k = 0; k < 1200; k++) {
            steps = k + 1;
            auto o = e.obs();
            int n = (int)o.size();
            double wdx = o[n - 2], wdy = o[n - 1];   // water_dx, water_dy
            auto m = e.action_mask();

            int act = 0;  // DAY
            if (m[A_BUILD0 + wc] != 0.0f) {
                act = A_BUILD0 + wc;                   // build the canal ASAP
            } else if (m[A_BUILD0 + f] != 0.0f && e.game().sunduk[WATER] > 50) {
                act = A_BUILD0 + f;                    // farm once water exists
            } else if (m[A_BUILD0 + g] != 0.0f && e.game().sunduk[WATER] > 50) {
                act = A_BUILD0 + g;
            } else {
                // walk towards water: pick the compass direction that closes
                // the gap, falling back to any open direction.
                int best = -1;
                double best_score = -1e18;
                for (int d = 0; d < N_ROAD_DIRS; d++) {
                    if (m[rdb + d] == 0.0f) continue;
                    double score = -(std::fabs(wdx - ROAD_DIR_DX[d] * 0.05) +
                                     std::fabs(wdy - ROAD_DIR_DY[d] * 0.05));
                    if (score > best_score) { best_score = score; best = d; }
                }
                if (best >= 0) act = rdb + best;
            }

            const bool dir_road = (act >= rdb && act < rdb + N_ROAD_DIRS);
            auto s = e.step(act);
            total += s.rew;
            if (dir_road) roads++;
            if (s.terminated || s.truncated) {
                printf("  [seed %lld ended: terminated=%d truncated=%d day=%lld pop=%lld money=%lld tax_grace_expired=%d]\n",
                       (long long)seed, (int)s.terminated, (int)s.truncated,
                       (long long)s.days, (long long)s.people, (long long)s.money,
                       (int)s.tax_grace_expired);
                break;
            }
        }
        for (const Base& b : e.game().bases) if (b.data->id == "WaterChannel") built_wc++;
        printf("%-6lld %-8d %-8d %-7d %-8lld %-9lld %-8.1f\n", (long long)seed, steps,
               roads, built_wc, (long long)e.game().sunduk[WATER],
               (long long)e.game().sunduk[FOOD], total);
    }
    return 0;
}
