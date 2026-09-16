// Probe 4: the causal chain. (a) episode length cap, (b) days needed to reach
// water, (c) do Farms idle because water==0, (d) does granting water fix it?
#include "colony/env.h"
#include "colony/data.h"
#include <algorithm>
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

    printf("=== (a) episode length cap ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        int k = 0;
        for (; k < 100000; k++) { auto s = e.step(0); if (s.terminated || s.truncated) break; }
        printf("DAY-only episode ended after %d steps, days_alive=%lld, truncated/terminated\n",
               k, (long long)e.game().days_alive);
    }

    printf("\n=== (b) days required to connect water vs the cap ===\n");
    {
        for (int64_t seed : {1, 42, 777}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            e.game().money = 5'000'000;
            int wc = find_id(e, "WaterChannel"), rd = find_id(e, "Road");
            int k = 0;
            for (; k < 1000; k++) {
                auto m = e.action_mask();
                if (m[A_BUILD0 + wc] != 0.0f) break;
                if (m[A_BUILD0 + rd] == 0.0f) break;
                e.step(A_BUILD0 + rd);
            }
            printf("seed %-4lld roads=%-4d days_elapsed=%-6lld (cap ~365) -> %s\n",
                   (long long)seed, k, (long long)e.game().days_alive,
                   e.game().days_alive <= 365 ? "fits" : "EXCEEDS EPISODE");
        }
    }

    printf("\n=== (c) do Farms idle because water == 0? ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        int f = find_id(e, "Farm"), g = find_id(e, "Garden");
        for (int k = 0; k < 30; k++) {
            auto m = e.action_mask();
            int a = 0;
            if (m[A_BUILD0 + f] != 0.0f) a = A_BUILD0 + f;
            else if (m[A_BUILD0 + g] != 0.0f) a = A_BUILD0 + g;
            e.step(a);
        }
        int idle = 0, farms = 0;
        for (const Base& b : e.game().bases) {
            if (b.data->id == "Farm" || b.data->id == "Garden") { farms++; if (b.need_sunduk) idle++; }
        }
        printf("farms/gardens=%d  idle(need_sunduk)=%d  water_stock=%lld food_stock=%lld\n",
               farms, idle, (long long)e.game().sunduk[WATER], (long long)e.game().sunduk[FOOD]);
        printf("Farm consumes water=%d, Garden consumes water=%d (configs/bases.json)\n",
               bd[1].consume[WATER], bd[2].consume[WATER]);
    }

    printf("\n=== (d) grant water and re-run the same farm policy ===\n");
    {
        for (int64_t seed : {1, 42, 777}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            e.game().sunduk[WATER] = 100000;  // cheat: infinite water
            int f = find_id(e, "Farm"), g = find_id(e, "Garden"), h = find_id(e, "SmallHouse");
            double total = 0;
            for (int k = 0; k < 1500; k++) {
                auto m = e.action_mask();
                int a = 0;
                if (m[A_BUILD0 + f] != 0.0f) a = A_BUILD0 + f;
                else if (m[A_BUILD0 + g] != 0.0f) a = A_BUILD0 + g;
                else if (m[A_BUILD0 + h] != 0.0f) a = A_BUILD0 + h;
                auto s = e.step(a); total += s.rew;
                if (s.terminated || s.truncated) break;
            }
            printf("seed %-4lld WITH water: reward=%+9.1f people=%-4lld bases=%-4lld food=%-6lld\n",
                   (long long)seed, total, (long long)e.game().people,
                   (long long)e.game().bases.size(), (long long)e.game().sunduk[FOOD]);
        }
        for (int64_t seed : {1, 42, 777}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            int f = find_id(e, "Farm"), g = find_id(e, "Garden"), h = find_id(e, "SmallHouse");
            double total = 0;
            for (int k = 0; k < 1500; k++) {
                auto m = e.action_mask();
                int a = 0;
                if (m[A_BUILD0 + f] != 0.0f) a = A_BUILD0 + f;
                else if (m[A_BUILD0 + g] != 0.0f) a = A_BUILD0 + g;
                else if (m[A_BUILD0 + h] != 0.0f) a = A_BUILD0 + h;
                auto s = e.step(a); total += s.rew;
                if (s.terminated || s.truncated) break;
            }
            printf("seed %-4lld NO water  : reward=%+9.1f people=%-4lld bases=%-4lld food=%-6lld\n",
                   (long long)seed, total, (long long)e.game().people,
                   (long long)e.game().bases.size(), (long long)e.game().sunduk[FOOD]);
        }
    }
    return 0;
}
