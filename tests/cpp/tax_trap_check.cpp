// Standalone diagnostic #2: the tax freeze and what it does to the MDP.
//
// HISTORY (2026-09 diagnostic): `Game::check_advance()` used to block *time
// itself* when the annual tax was due and unpaid — `advance_day()` returned
// !ok, so no day passed, no production happened, and the only thing a policy
// could do was burn steps (-2 error for DAY/WEEK) until
// `tax_due_days_ >= TAX_GRACE_DAYS` (60) killed the episode at day 365, the
// first tax date. Every trajectory that spent the starting capital died there
// while "do nothing" survived for years; TAKE_LOAN (+50k) was the only rescue.
//
// FIX (P0, 2026-09-17): the env now runs a *debt policy* by default
// (`tax_to_debt=true`): the unpaid remainder is settled into bank credit and
// the calendar keeps moving. This harness therefore runs each scenario twice:
//   * `legacy`  = tax_to_debt=false → the old freeze (kept for the GUI/dialog
//                 path and as the control that the bug is really gone);
//   * `debt`    = tax_to_debt=true  → the RL default (see tests/cpp/p0_p1_check.cpp
//                 for the focused regression checks).
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

    printf("=========== 1. broke with tax due: legacy freeze vs debt policy ===========\n");
    for (bool to_debt : {false, true}) {
        printf("  --- tax_to_debt=%s ---\n", to_debt ? "true" : "false");
        ColonyEnvCpp e(bd, ed, 42, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL,
                       /*no_city_game_over=*/false, /*no_people_days=*/365, to_debt);
        e.reset(42);
        const int farm = find_id(e, "Farm");
        // Spend the capital: 4 farms then nothing.
        for (int k = 0; k < 4; k++) {
            auto out = e.step(A_BUILD0 + farm);
            (void)out;
        }
        printf("  after 4 farms: money=%lld, DAY advances the calendar?\n",
               (long long)e.game().money);
        int64_t prev_days = -1;
        int frozen_run = 0;
        for (int i = 0; i < 430; i++) {
            auto out = e.step(A_DAY);
            bool frozen = prev_days >= 0 && out.days == prev_days;
            frozen_run = frozen ? frozen_run + 1 : 0;
            if (i < 3 || i % 100 == 0 || (frozen && frozen_run <= 2) || out.terminated)
                printf("    step %4d | day %5lld | money %6lld | credit %6lld | %s%s\n",
                       i, (long long)out.days, (long long)out.money,
                       (long long)e.game().credit,
                       frozen ? "FROZEN" : "advances",
                       out.terminated ? "  <- TERMINATED" : "");
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
            printf("  DAY #%d:        reward=%+.3f money=%lld   (tax_daily_bonus +0.3; "
                   "debt term now visible in the step log as debt=/taxdebt=)\n",
                   i + 1, o.rew, (long long)o.money);
        }
        // 365 days later the debt term has grown
        for (int i = 0; i < 360; i++) { auto o = e.step(A_DAY); if (o.terminated) break; }
        auto late = e.step(A_DAY);
        printf("DAY ~+365:       reward=%+.3f money=%lld  (0.2%%/day interest on 50k -> "
               "0.1*credit/1000 per day)\n", late.rew, (long long)late.money);
    }

    printf("\n============ 3. do-nothing vs invest: 3000 steps, 3 seeds ==============\n"
           "  (legacy = old freeze, debt = P0 default: calendar always moves)\n");
    for (int64_t seed : {1, 42, 777}) {
        auto e1 = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                 RewardConfig(), DIFFICULTY_NORMAL);
        e1->reset(seed);
        show("DAY only", roll(*e1, 3000, [](ColonyEnvCpp&) { return A_DAY; }));

        auto spend = [](ColonyEnvCpp& e) {
            for (int b = 0; b < e.n_build(); b++) {
                std::vector<float> m = e.action_mask();
                if (m[(size_t)(A_BUILD0 + b)] > 0.5f) return A_BUILD0 + b;
            }
            return A_DAY;
        };
        auto e2 = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                 RewardConfig(), DIFFICULTY_NORMAL,
                                                 false, 365, /*tax_to_debt=*/false);
        e2->reset(seed);
        show("cheapest build (legacy)", roll(*e2, 3000, spend));

        auto e2d = std::make_unique<ColonyEnvCpp>(bd, ed, seed, 200, Curriculum(),
                                                  RewardConfig(), DIFFICULTY_NORMAL);
        e2d->reset(seed);
        show("cheapest build (debt)", roll(*e2d, 3000, spend));

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
