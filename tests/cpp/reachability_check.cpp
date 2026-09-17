// Standalone diagnostic #3: action reachability, random-policy damage and the
// long-run economy.
//
// Questions answered on the real C++ core:
//   1. Of the 32 building actions, how many are EVER legal during a long
//      rollout? Which land types are reachable without a directional signal?
//   2. How much of a masked-random rollout's reward is `error_penalty` (i.e.
//      how big is the "legal but self-destructive" action minefield)?
//   3. Does any hand-crafted policy beat "do nothing" over a long horizon
//      (i.e. is the economy net-positive for the agent at all)?
//
// Build (from the repo root):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/reach_check \
//       tests/cpp/reachability_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
// Run from the repo root:  /tmp/reach_check
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace colony;

namespace {

int find_id(const ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

struct Run {
    double total = 0.0;
    int steps = 0;
    int64_t days = 0, money = 0, people = 0, bases = 0;
    std::map<std::string, int> built;
    std::map<std::string, double> comp;
    std::vector<long long> acted;   // per action index
    int terminated = 0;
};

// Parse the env step-log component line.
void parse_components(const std::string& path, std::map<std::string, double>& out) {
    std::ifstream f(path);
    std::string line;
    const std::vector<std::string> keys = {
        "err", "tax", "taxb", "build", "cost", "div", "prox", "prov", "preq",
        "pres", "sale", "mtax", "chain", "daily", "nov", "mile", "surv",
        "idle", "gover", "extr", "loan"};
    while (std::getline(f, line)) {
        if (line.rfind("STEP ", 0) != 0) continue;
        for (const auto& k : keys) {
            std::string tag = k + "=";
            size_t pos = line.find(" " + tag);
            if (pos == std::string::npos) continue;
            if (k == "tax" && line.compare(pos + 1, 5, "taxb=") == 0) continue;
            size_t v = pos + 1 + tag.size();
            size_t e = line.find(' ', v);
            out[k] += std::stod(line.substr(v, e - v));
        }
    }
}

Run roll(ColonyEnvCpp& e, int n_steps, const std::function<int(ColonyEnvCpp&)>& policy,
         const std::string& log_path = "", bool collect_mask = false,
         std::vector<int>* ever_legal = nullptr) {
    if (!log_path.empty()) { std::remove(log_path.c_str()); e.set_step_log(log_path); }
    Run r;
    r.acted.assign((size_t)e.n_actions(), 0);
    for (int i = 0; i < n_steps; i++) {
        auto mask = e.action_mask();
        if (collect_mask && ever_legal) {
            for (size_t k = 0; k < mask.size(); k++)
                if (mask[k] > 0.5f) (*ever_legal)[k] = 1;
        }
        int a = policy(e);
        if (a < 0 || a >= (int)mask.size() || mask[(size_t)a] < 0.5f) a = A_DAY;
        r.acted[(size_t)a]++;
        if (a >= A_BUILD0 && a < A_BUILD0 + e.n_build())
            r.built[e.build_ids()[(size_t)(a - A_BUILD0)]]++;
        auto out = e.step(a);
        r.total += out.rew;
        r.steps++;
        r.days = out.days;
        r.money = out.money;
        r.people = out.people;
        r.bases = out.n_bases;
        if (out.terminated) { r.terminated = 1; break; }
        if (out.truncated) { r.terminated = 2; break; }
    }
    if (!log_path.empty()) {
        e.set_step_log("");
        parse_components(log_path, r.comp);
    }
    return r;
}

void show(const std::string& label, const Run& r) {
    printf("%-30s steps=%4d total=%9.2f days=%5lld money=%7lld people=%3lld bases=%3lld %s\n",
           label.c_str(), r.steps, r.total, (long long)r.days, (long long)r.money,
           (long long)r.people, (long long)r.bases,
           r.terminated == 1 ? "[TERMINATED]" : r.terminated == 2 ? "[truncated]" : "[budget]");
    if (!r.built.empty()) {
        printf("%-30s placed:", "");
        for (const auto& kv : r.built) printf(" %s=%d", kv.first.c_str(), kv.second);
        printf("\n");
    }
    if (!r.comp.empty()) {
        printf("%-30s comp:", "");
        for (const auto& kv : r.comp)
            if (std::fabs(kv.second) > 0.5) printf(" %s=%+.0f", kv.first.c_str(), kv.second);
        printf("\n");
    }
}

}  // namespace

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("=========== 1. which building actions are reachable? (road spam) ===========\n");
    {
        std::vector<int> ever((size_t)49, 0);
        ColonyEnvCpp e(bd, ed, 42, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
        e.reset(42);
        int road = find_id(e, "Road");
        // greedy: extend the road frontier east every step (worst case for
        // reachability: a single arm), 1200 steps
        Run r = roll(e, 1200,
                     [&](ColonyEnvCpp& env) {
                         auto m = env.action_mask();
                         int a = env.road_dir_base() + 0;  // ROAD_E
                         if (m[(size_t)a] > 0.5f) return a;
                         if (road >= 0 && m[(size_t)(A_BUILD0 + road)] > 0.5f)
                             return A_BUILD0 + road;
                         return A_DAY;
                     },
                     "", true, &ever);
        printf("after %d steps of ROAD_E spam (seed 42, map 200): days=%lld roads/bases=%lld\n",
               r.steps, (long long)r.days, (long long)r.bases);
        printf("building actions legal at least once:\n");
        int n_reach = 0;
        for (int i = 0; i < e.n_build(); i++) {
            bool ok = ever[(size_t)(A_BUILD0 + i)] != 0;
            if (ok) n_reach++;
            printf("   %-18s %s\n", e.build_ids()[(size_t)i].c_str(), ok ? "yes" : "NO");
        }
        printf("   -> %d / %d buildings ever legal\n\n", n_reach, e.n_build());
    }

    printf("=========== 2. uniform masked-random policy (5 seeds, 2000 steps) ===========\n");
    {
        for (int64_t seed : {1, 42, 777, 1234, 20260}) {
            ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
            e.reset(seed);
            unsigned s = 987654321u ^ (unsigned)seed;
            std::string log = "/tmp/reach_random_" + std::to_string(seed) + ".txt";
            Run rr = roll(e, 2000,
                          [&](ColonyEnvCpp& env) {
                              auto m = env.action_mask();
                              std::vector<int> legal;
                              for (size_t k = 0; k < m.size(); k++)
                                  if (m[k] > 0.5f) legal.push_back((int)k);
                              s = s * 1664525u + 1013904223u;
                              return legal[s % legal.size()];
                          },
                          log);
            show("masked-random seed " + std::to_string(seed), rr);
            double err = rr.comp.count("err") ? rr.comp["err"] : 0.0;
            printf("%-30s err-share of total reward: %.1f%% (err=%.0f of %.0f)\n\n", "",
                   100.0 * err / (std::fabs(rr.total) + 1e-9), err, rr.total);
        }
    }

    printf("=========== 3. long horizon: does investing ever pay? ===========\n");
    for (int64_t seed : {1, 42}) {
        auto mk = [&]() {
            auto e = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                    RewardConfig(), DIFFICULTY_NORMAL);
            e->reset(seed);
            return e;
        };

        auto e1 = mk();
        show("DAY only", roll(*e1, 4000, [](ColonyEnvCpp&) { return A_DAY; }));

        auto e2 = mk();
        // Greedy "water first, then whatever is legal, keep 20k tax buffer".
        show("water greedy + invest",
             roll(*e2, 4000, [](ColonyEnvCpp& e) {
                 auto m = e.action_mask();
                 auto obs = e.obs();
                 int64_t money = (int64_t)std::llround((double)obs[4] * 2e5);
                 int wc = find_id(e, "WaterChannel");
                 if (wc >= 0 && m[(size_t)(A_BUILD0 + wc)] > 0.5f) return A_BUILD0 + wc;
                 if (money > 30000) {
                     for (int b = 0; b < e.n_build(); b++)
                         if (m[(size_t)(A_BUILD0 + b)] > 0.5f &&
                             e.build_ids()[(size_t)b] != "Road")
                             return A_BUILD0 + b;
                 }
                 float dx = obs[obs.size() - 2], dy = obs[obs.size() - 1];
                 int dir = std::fabs(dx) >= std::fabs(dy) ? (dx > 0 ? 0 : 1) : (dy > 0 ? 2 : 3);
                 int a = e.road_dir_base() + dir;
                 if (m[(size_t)a] > 0.5f) return a;
                 return A_DAY;
             }));

        auto e3 = mk();
        // Same, but never spend below 20k (survive the tax).
        show("water greedy, 20k buffer",
             roll(*e3, 4000, [](ColonyEnvCpp& e) {
                 auto m = e.action_mask();
                 auto obs = e.obs();
                 int64_t money = (int64_t)std::llround((double)obs[4] * 2e5);
                 int wc = find_id(e, "WaterChannel");
                 if (wc >= 0 && m[(size_t)(A_BUILD0 + wc)] > 0.5f && money > 25000)
                     return A_BUILD0 + wc;
                 if (money > 30000) {
                     for (int b = 0; b < e.n_build(); b++)
                         if (m[(size_t)(A_BUILD0 + b)] > 0.5f &&
                             e.build_ids()[(size_t)b] != "Road")
                             return A_BUILD0 + b;
                 }
                 float dx = obs[obs.size() - 2], dy = obs[obs.size() - 1];
                 int dir = std::fabs(dx) >= std::fabs(dy) ? (dx > 0 ? 0 : 1) : (dy > 0 ? 2 : 3);
                 int a = e.road_dir_base() + dir;
                 if (m[(size_t)a] > 0.5f) return a;
                 return A_DAY;
             }));
        printf("\n");
    }
    return 0;
}
