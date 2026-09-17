// Standalone diagnostic #2: the tax freeze and what it does to the MDP.
//
// Findings this harness is meant to pin down on the real C++ core:
//   * `Game::check_advance()` blocks *time itself* when the annual tax is due
//     and unpaid: `advance_day()` returns !ok, so no day passes, no production
//     happens, and the only thing the agent can do is burn steps (-2 error
//     penalty for DAY/WEEK) until `tax_due_days_ >= TAX_GRACE_DAYS` (60) kills
//     the episode at day 365 — the first tax date.
//   * Consequence: every trajectory that spends the starting capital dies at
//     the first March 1st, while "do nothing" survives for years.
//   * Rescue: TAKE_LOAN (+50k) makes the tax payable again.
//
// Build (from the repo root):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/tax_trap \
//       tests/cpp/tax_trap_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
// Run from the repo root:  /tmp/tax_trap
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

#include <algorithm>
#include <cmath>
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

std::vector<int> legal_actions(ColonyEnvCpp& e) {
    std::vector<float> m = e.action_mask();
    std::vector<int> out;
    for (size_t i = 0; i < m.size(); i++)
        if (m[i] > 0.5f) out.push_back((int)i);
    return out;
}

struct Ep {
    int steps = 0;
    double total = 0.0;
    int64_t days = 0, money = 0, bases = 0;
    std::map<std::string, int> built;
    std::string end_reason = "?";
};

Ep roll(ColonyEnvCpp& e, int n_steps, const std::function<int(ColonyEnvCpp&)>& policy) {
    Ep ep;
    for (int i = 0; i < n_steps; i++) {
        std::vector<float> m = e.action_mask();
        int a = policy(e);
        if (a < 0 || a >= (int)m.size() || m[(size_t)a] < 0.5f) a = A_DAY;
        if (a >= A_BUILD0 && a < A_BUILD0 + e.n_build())
            ep.built[e.build_ids()[(size_t)(a - A_BUILD0)]]++;
        auto out = e.step(a);
        ep.total += out.rew;
        ep.steps++;
        ep.days = out.days;
        ep.money = out.money;
        ep.bases = out.n_bases;
        if (out.terminated) { ep.end_reason = "terminated"; break; }
        if (out.truncated) { ep.end_reason = "truncated(MAX_STEPS)"; break; }
    }
    if (ep.end_reason == "?") ep.end_reason = "ran out of budget";
    return ep;
}

void show(const std::string& label, const Ep& ep) {
    printf("%-26s steps=%4d total=%9.2f days=%5lld money=%7lld bases=%3lld  [%s]\n",
           label.c_str(), ep.steps, ep.total, (long long)ep.days, (long long)ep.money,
           (long long)ep.bases, ep.end_reason.c_str());
    if (!ep.built.empty()) {
        printf("%-26s built:", "");
        for (const auto& kv : ep.built) printf(" %s=%d", kv.first.c_str(), kv.second);
        printf("\n");
    }
}

}  // namespace

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("================ 1. tax freeze: what DAY does when broke ================\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
        e.reset(42);
        const int farm = find_id(e, "Farm");
        // Spend the capital: 4 farms then nothing.
        for (int k = 0; k < 4; k++) {
            auto out = e.step(A_BUILD0 + farm);
            (void)out;
        }
        printf("after 4 farms: is Farm legal now? ");
        {
            std::vector<float> m = e.action_mask();
            printf("%s\n", m[(size_t)(A_BUILD0 + farm)] > 0.5f ? "yes" : "no");
        }
        printf("  day | money  | steps | DAY advances the calendar?\n");
        int64_t prev_days = -1;
        for (int i = 0; i < 80; i++) {
            std::vector<float> m = e.action_mask();
            auto out = e.step(A_DAY);
            if (i < 8 || i % 10 == 0 || i > 74) {
                printf("  %5lld | %6lld | %5lld | %s%s\n", (long long)out.days,
                       (long long)out.money, (long long)i,
                       (prev_days >= 0 && out.days == prev_days) ? "NO (frozen)" : "yes",
                       out.terminated ? "   <- episode TERMINATED" : "");
            }
            if (out.terminated) break;
            prev_days = out.days;
        }
    }

    printf("\n================ 2. the loan trap: -0.5 now, -5/day forever ==========\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
        e.reset(42);
        const int MGR = A_BUILD0 + e.n_build();
        const int TAKE_LOAN = MGR + 8;
        std::vector<float> m = e.action_mask();
        printf("TAKE_LOAN=%d legal at reset? %s\n", TAKE_LOAN,
               m[(size_t)TAKE_LOAN] > 0.5f ? "yes" : "no");
        auto before = e.step(A_DAY);
        printf("DAY before loan: reward=%+.3f money=%lld\n", before.rew,
               (long long)before.money);
        auto loan = e.step(TAKE_LOAN);
        printf("TAKE_LOAN:       reward=%+.3f money=%lld  <- 50k arrives\n", loan.rew,
               (long long)loan.money);
        for (int i = 0; i < 5; i++) {
            auto o = e.step(A_DAY);
            printf("  DAY #%d:        reward=%+.3f money=%lld   (tax_daily_bonus +0.3, "
                   "debt term hidden, step log does NOT show it)\n",
                   i + 1, o.rew, (long long)o.money);
        }
        // 365 days later the debt term has grown
        for (int i = 0; i < 360; i++) { auto o = e.step(A_DAY); if (o.terminated) break; }
        auto late = e.step(A_DAY);
        printf("DAY ~+365:       reward=%+.3f money=%lld  (0.2%%/day interest on 50k -> "
               "0.1*credit/1000 per day)\n", late.rew, (long long)late.money);
    }

    printf("\n============ 3. do-nothing vs invest: 3000 steps, 3 seeds ==============\n");
    for (int64_t seed : {1, 42, 777}) {
        auto e1 = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                 RewardConfig(), DIFFICULTY_NORMAL);
        e1->reset(seed);
        show("DAY only", roll(*e1, 3000, [](ColonyEnvCpp&) { return A_DAY; }));

        auto e2 = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                 RewardConfig(), DIFFICULTY_NORMAL);
        e2->reset(seed);
        show("cheapest build, no buffer",
             roll(*e2, 3000, [](ColonyEnvCpp& e) {
                 for (int b = 0; b < e.n_build(); b++) {
                     std::vector<float> m = e.action_mask();
                     if (m[(size_t)(A_BUILD0 + b)] > 0.5f) return A_BUILD0 + b;
                 }
                 return A_DAY;
             }));

        auto e3 = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                 RewardConfig(), DIFFICULTY_NORMAL);
        e3->reset(seed);
        show("build, keep 25k buffer",
             roll(*e3, 3000, [](ColonyEnvCpp& e) {
                 std::vector<float> m = e.action_mask();
                 std::vector<float> obs = e.obs();
                 int64_t money = (int64_t)std::llround((double)obs[4] * 2e5);
                 for (int b = 0; b < e.n_build(); b++) {
                     if (m[(size_t)(A_BUILD0 + b)] < 0.5f) continue;
                     // keep a tax buffer; pick the cheapest legal build
                     const std::string& id = e.build_ids()[(size_t)b];
                     double price = 0;
                     for (int k = 0; k < e.n_build(); k++)
                         if (e.build_ids()[(size_t)k] == id) price = 0;
                     (void)price;
                     if (money < 25000) break;  // too poor to invest
                     return A_BUILD0 + b;
                 }
                 return A_DAY;
             }));

        auto e4 = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                 RewardConfig(), DIFFICULTY_NORMAL);
        e4->reset(seed);
        // What a human does: one producer that pays for itself, then idle.
        show("Mushroom + hoard",
             roll(*e4, 3000, [](ColonyEnvCpp& e) {
                 std::vector<float> m = e.action_mask();
                 int mid = find_id(e, "Mushroom");
                 if (mid >= 0 && m[(size_t)(A_BUILD0 + mid)] > 0.5f) return A_BUILD0 + mid;
                 return A_DAY;
             }));
        printf("\n");
    }
    return 0;
}
