// Эталонный «ручной план» из RL_DIAGNOSIS §3.6: дороги К ВОДЕ (направленные
// ROAD_*) → 2 водоканала → огороды → дом, потом спокойная жизнь с продажами.
// Замер: когда срабатывает chain, какой день-в-день return у устойчивой экономики.
//
// Build (repo root):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/manual_plan \
//       tests/cpp/manual_plan_probe.cpp src/env.cpp src/data.cpp \
//       src/resources.cpp src/game.cpp src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/manual_plan
#include "colony/env.h"
#include "colony/data.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace colony;

static int idx_of(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[i] == id) return i;
    return -1;
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");
    ColonyEnvCpp e(bd, ed, 42, 200);
    e.reset(42);
    e.set_step_log("/tmp/manual_plan.log");

    // nearest water tile to the city -> dominant road direction
    const auto& g0 = e.game();
    const int ms = g0.map_size();
    const int8_t* lots = g0.earth.lots().data();
    int cx = g0.earth.init_sel_x, cy = g0.earth.init_sel_y;
    int best = 1 << 30, bx = 0, by = 0;
    for (int y = 0; y < ms; y++)
        for (int x = 0; x < ms; x++) {
            if (lots[(size_t)y * ms + x] != LT_WATER) continue;
            int d = std::abs(x - cx) + std::abs(y - cy);
            if (d < best) { best = d; bx = x; by = y; }
        }
    int dir = (std::abs(bx - cx) > std::abs(by - cy))
                  ? (bx > cx ? 0 : 1)   // E / W
                  : (by > cy ? 2 : 3);  // S / N
    printf("[plan] nearest water at (%d,%d), dist=%d, dir=%d (E/W/S/N)\n",
           bx, by, best, dir);

    int water_built = 0, garden_built = 0, house_built = 0;
    int chain_step = -1, first_food_day = -1;
    for (int s = 0; s < 4300; ++s) {
        auto mask = e.action_mask();
        int mgr = e.road_dir_base() - N_MANAGERS;
        int rdb = e.road_dir_base();
        int act = A_DAY;

        bool need_house = house_built < 4 &&
            (int64_t)e.game().now_home_places() < (int64_t)e.game().people * 105 / 100 &&
            e.game().money > 18000;
        if (water_built < 2 && mask[A_BUILD0 + idx_of(e, "WaterChannel")] != 0.0f)
            act = A_BUILD0 + idx_of(e, "WaterChannel");
        else if (garden_built < 3 && water_built >= 1 &&
                 mask[A_BUILD0 + idx_of(e, "Garden")] != 0.0f)
            act = A_BUILD0 + idx_of(e, "Garden");
        else if (need_house && mask[A_BUILD0 + idx_of(e, "House")] != 0.0f)
            act = A_BUILD0 + idx_of(e, "House");
        else if ((water_built < 2 || garden_built < 3) &&
                 mask[rdb + dir] != 0.0f)
            act = rdb + dir;
        else if (mask[mgr + 6] != 0.0f)  // SELL whenever surplus
            act = mgr + 6;

        ColonyEnvCpp::StepOut out = e.step(act);

        // count real bases by id
        int wc = 0, gd = 0, hs = 0;
        for (const Base& bb : e.game().bases) {
            if (bb.data->id == "WaterChannel") wc++;
            else if (bb.data->id == "Garden") gd++;
            else if (bb.data->id == "House") hs++;
        }
        water_built = wc; garden_built = gd; house_built = hs;

        if (chain_step < 0 && e.game().check_milestones(0, 0, 0, 0) >= 0 &&
            garden_built > 0 && water_built > 0) {
            // chain fires lazily inside step; detect via step log component
        }
        if (first_food_day < 0 && e.game().sunduk[FOOD] > 0) first_food_day = (int)out.days;
        if (out.days % 500 == 0 || out.terminated) {
            printf("day=%4lld money=%7lld credit=%7lld pop=%3lld home=%3lld "
                   "water=%4lld food=%4lld | rew=%8.2f ep=%9.1f wc=%d gd=%d hs=%d\n",
                   (long long)out.days, (long long)out.money,
                   (long long)e.game().credit, (long long)out.people,
                   (long long)e.game().now_home_places(),
                   (long long)e.game().sunduk[WATER],
                   (long long)e.game().sunduk[FOOD], out.rew, out.ep_return,
                   wc, gd, hs);
        }
        if (out.terminated) { printf("TERMINATED at step %d\n", s + 1); break; }
    }
    const auto& g = e.game();
    printf("FINAL: day=%lld money=%lld credit=%lld pop=%lld bases=%d first_food_day=%d\n",
           (long long)g.days_alive, (long long)g.money, (long long)g.credit,
           (long long)g.people, (int)g.bases.size(), first_food_day);
    return 0;
}
