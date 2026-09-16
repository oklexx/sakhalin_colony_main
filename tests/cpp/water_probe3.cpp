// Probe 3: is "spam roads" actually the reward-optimal degenerate policy, and
// how much of the action space is even legal early on?
#include "colony/env.h"
#include "colony/data.h"
#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace colony;

static int find_id(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[i] == id) return i;
    return -1;
}

struct Summary { double rew; int64_t days, people, bases, water, food; int roads, wc; };

// Greedy policy: pick the first legal action from a preference list.
static Summary run(const std::vector<BaseData>& bd, const std::vector<BaseEvent>& ed,
                   int64_t seed, const std::vector<std::string>& pref, int max_steps) {
    ColonyEnvCpp e(bd, ed, seed, 200);
    e.reset(seed);
    double total = 0.0;
    int roads = 0, wc = 0;
    std::vector<int> idx;
    for (auto& p : pref) { int i = find_id(e, p); if (i >= 0) idx.push_back(i); }
    for (int k = 0; k < max_steps; k++) {
        auto m = e.action_mask();
        int chosen = -1;
        for (int i : idx) if (m[A_BUILD0 + i] != 0.0f) { chosen = i; break; }
        int act = (chosen >= 0) ? A_BUILD0 + chosen : 0;  // else DAY
        auto s = e.step(act);
        total += s.rew;
        if (s.terminated || s.truncated) break;
    }
    int nr = 0, nw = 0;
    for (const Base& b : e.game().bases) {
        if (b.data->id == "Road") nr++;
        if (b.data->id == "WaterChannel") nw++;
    }
    return {total, e.game().days_alive, e.game().people, (int64_t)e.game().bases.size(),
            e.game().sunduk[WATER], e.game().sunduk[FOOD], nr, nw};
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("=== Policy comparison over 1500 steps (realistic starting money) ===\n");
    printf("%-34s %-10s %-6s %-7s %-6s %-6s %-6s %-6s\n",
           "policy", "reward", "days", "people", "bases", "water", "food", "roads");
    struct P { const char* name; std::vector<std::string> pref; };
    std::vector<P> pols = {
        {"ROADS ONLY",            {"Road"}},
        {"Farm,Garden,Road",      {"Farm", "Garden", "Road"}},
        {"Road then Water then F",{"Road", "WaterChannel", "Farm", "Garden"}},
        {"Water-first (ideal)",   {"WaterChannel", "Farm", "Garden", "Road"}},
        {"House,Farm,Garden,Road",{"House", "SmallHouse", "Farm", "Garden", "Road"}},
    };
    for (auto& p : pols) {
        double r = 0; Summary last{};
        int nseed = 0;
        for (int64_t s : {1, 42, 777}) { last = run(bd, ed, s, p.pref, 1500); r += last.rew; nseed++; }
        printf("%-34s %-10.1f %-6lld %-7lld %-6lld %-6lld %-6lld %-6lld  (wc=%d)\n",
               p.name, r / nseed, (long long)last.days, (long long)last.people,
               (long long)last.bases, (long long)last.water, (long long)last.food,
               (long long)last.roads, last.wc);
    }

    printf("\n=== How often is WaterChannel legal during a random rollout? ===\n");
    {
        std::mt19937 rng(12345);
        for (int64_t seed : {1, 42}) {
            ColonyEnvCpp e(bd, ed, seed, 200);
            e.reset(seed);
            int wc = find_id(e, "WaterChannel");
            int open_steps = 0, total_steps = 0;
            for (int k = 0; k < 2000; k++) {
                auto m = e.action_mask();
                total_steps++;
                if (m[A_BUILD0 + wc] != 0.0f) open_steps++;
                // random legal action
                std::vector<int> legal;
                for (int a = 0; a < e.n_actions(); a++) if (m[a] != 0.0f) legal.push_back(a);
                if (legal.empty()) break;
                auto s = e.step(legal[rng() % legal.size()]);
                if (s.terminated || s.truncated) { e.reset(seed + k); }
            }
            printf("seed %-4lld WaterChannel legal in %d / %d steps (%.2f%%)\n",
                   (long long)seed, open_steps, total_steps,
                   100.0 * open_steps / std::max(1, total_steps));
        }
    }

    printf("\n=== n_actions / n_build ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        printf("n_build=%d n_actions=%d obs_size=%d\n", e.n_build(), e.n_actions(), e.obs_size());
    }
    return 0;
}
