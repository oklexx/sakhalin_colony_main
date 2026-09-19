// Наградной аудит: замер вклада каждого члена награды (через step_log) у
// нескольких репрезентативных политик. Дефолтный профиль reward_v4
// (канонический), tax_to_debt=true (политика RL).
//
// Build (repo root):
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/reward_audit \
//       tests/cpp/reward_audit_probe.cpp src/env.cpp src/data.cpp \
//       src/resources.cpp src/game.cpp src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/reward_audit /tmp/audit
#include "colony/env.h"
#include "colony/data.h"
#include <cstdio>
#include <string>
#include <vector>

using namespace colony;

static int idx_of(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[i] == id) return i;
    return -1;
}

// first legal action from a priority list of building ids, else fallback
static int first_legal_build(ColonyEnvCpp& e,
                             const std::vector<int>& prio, int fallback) {
    auto mask = e.action_mask();
    for (int b : prio) {
        if (b < 0) continue;
        if (mask[A_BUILD0 + b] != 0.0f) return A_BUILD0 + b;
    }
    return fallback;
}

static void run_policy(const std::string& tag,
                       const std::vector<BaseData>& bd,
                       const std::vector<BaseEvent>& ed,
                       std::function<int(ColonyEnvCpp&)> policy,
                       int max_steps) {
    ColonyEnvCpp e(bd, ed, 42, 200);
    e.reset(42);
    e.set_step_log("/tmp/audit_" + tag + ".log");
    int steps = 0, clipped = 0;
    double max_abs_raw = 0.0, sum_raw = 0.0;
    bool done = false;
    for (int s = 0; s < max_steps && !done; ++s) {
        ColonyEnvCpp::StepOut out = e.step(policy(e));
        ++steps;
        sum_raw += out.rew;
        // raw vs clip: compare against clip bounds (±50)
        if (out.rew > 50.0 - 1e-9 || out.rew < -50.0 + 1e-9) ++clipped;
        double a = out.rew > 0 ? out.rew : -out.rew;
        if (a > max_abs_raw) max_abs_raw = a;
        done = out.terminated;
    }
    const auto& g = e.game();
    printf("POLICY %s\n", tag.c_str());
    printf("  steps=%d clipped_steps=%d max_abs_raw=%.2f sum_raw=%.1f\n",
           steps, clipped, max_abs_raw, sum_raw);
    printf("  day=%lld people=%lld money=%lld credit=%lld bases=%d\n",
           (long long)g.days_alive, (long long)g.people, (long long)g.money,
           (long long)g.credit, (int)g.bases.size());
    fflush(stdout);
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "/tmp";
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");
    if (dir.empty()) dir = "/tmp";

    // 1. DAY only
    run_policy("day", bd, ed,
               [](ColonyEnvCpp&) { return A_DAY; }, 2000);

    // 2. WEEK only
    run_policy("week", bd, ed,
               [](ColonyEnvCpp&) { return A_WEEK; }, 300);

    // 3. Road spam
    run_policy("roadspam", bd, ed,
               [&bd](ColonyEnvCpp& e) {
                   int rd = idx_of(e, "Road");
                   auto mask = e.action_mask();
                   if (mask[A_BUILD0 + rd] != 0.0f) return A_BUILD0 + rd;
                   return A_DAY;
               }, 3000);

    // 4. Loan spam: TAKE_LOAN whenever legal, else DAY
    run_policy("loan", bd, ed,
               [](ColonyEnvCpp& e) {
                   auto mask = e.action_mask();
                   if (mask[e.road_dir_base() - N_MANAGERS + 8] != 0.0f)
                       return e.road_dir_base() - N_MANAGERS + 8;
                   return A_DAY;
               }, 3000);

    // 5. Buy-food spam: BUY_FOOD whenever applicable, else DAY
    run_policy("foodspam", bd, ed,
               [](ColonyEnvCpp& e) {
                   auto mask = e.action_mask();
                   if (mask[e.road_dir_base() - N_MANAGERS + 7] != 0.0f)
                       return e.road_dir_base() - N_MANAGERS + 7;
                   return A_DAY;
               }, 3000);

    // 6. Greedy "reasonable": water channel -> roads -> houses -> farms ->
    //    gardens; SELL when legal; else DAY
    run_policy("greedy", bd, ed,
               [&bd](ColonyEnvCpp& e) {
                   auto mask = e.action_mask();
                   if (mask[e.road_dir_base() - N_MANAGERS + 6] != 0.0f)  // SELL
                       return e.road_dir_base() - N_MANAGERS + 6;
                   std::vector<int> prio = {
                       idx_of(e, "WaterChannel"), idx_of(e, "Road"),
                       idx_of(e, "House"), idx_of(e, "SmallHouse"),
                       idx_of(e, "Farm"), idx_of(e, "Garden")};
                   return first_legal_build(e, prio, A_DAY);
               }, 3000);

    // 7. Novelty harvester: always the first legal NOT-yet-built type
    {
        std::vector<std::string> done_types;
        run_policy("novelty", bd, ed,
                   [&bd, &done_types](ColonyEnvCpp& e) {
                       auto mask = e.action_mask();
                       for (int i = 0; i < e.n_build(); ++i) {
                           if (mask[A_BUILD0 + i] == 0.0f) continue;
                           const std::string& id = e.build_ids()[i];
                           bool seen = false;
                           for (const std::string& d : done_types)
                               if (d == id) seen = true;
                           if (!seen) { done_types.push_back(id); return A_BUILD0 + i; }
                       }
                       // all types built — fall back to any legal building
                       for (int i = 0; i < e.n_build(); ++i)
                           if (mask[A_BUILD0 + i] != 0.0f &&
                               e.build_ids()[i] != "Road")
                               return A_BUILD0 + i;
                       return A_DAY;
                   }, 3000);
    }

    printf("AUDIT PROBE DONE\n");
    return 0;
}
