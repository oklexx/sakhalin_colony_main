// Standalone diagnostic #4: what does PPO actually optimise here?
//
// PPO maximises a *discounted* return. With one decision = one game day:
//   gamma=0.99  -> horizon ln(0.5)/ln(0.99)  ~ 69 days
//   gamma=0.999 -> ~ 693 days
// The economy pays back in 174..523 days and the tax cycle is 365 days, so the
// discount factor decides whether investing can ever look profitable.
//
// This harness replays several hand-made policies and reports, for each:
//   * the undiscounted sum (what the UI prints),
//   * the discounted sum at gamma = 0.99 and 0.999 (what PPO actually maximises).
//
// Build (from the repo root):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/discount_check \
//       tests/cpp/discount_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
// Run from the repo root:  /tmp/discount_check
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
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

struct Out {
    int steps = 0;
    int64_t days = 0, money = 0, bases = 0;
    double sum = 0.0, g99 = 0.0, g999 = 0.0;
    int64_t built = 0;
    std::string note;
};

Out run(int64_t seed, int n_steps, const std::function<int(ColonyEnvCpp&)>& policy,
        const std::vector<BaseData>& bd, const std::vector<BaseEvent>& ed,
        const std::string& note) {
    ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
    e.reset(seed);
    Out o;
    o.note = note;
    double d99 = 1.0, d999 = 1.0;
    for (int i = 0; i < n_steps; i++) {
        auto m = e.action_mask();
        int a = policy(e);
        if (a < 0 || a >= (int)m.size() || m[(size_t)a] < 0.5f) a = A_DAY;
        if (a >= A_BUILD0 && a < A_BUILD0 + e.n_build()) o.built++;
        auto s = e.step(a);
        o.sum += s.rew;
        o.g99 += d99 * s.rew;
        o.g999 += d999 * s.rew;
        d99 *= 0.99;
        d999 *= 0.999;
        o.steps++;
        o.days = s.days;
        o.money = s.money;
        o.bases = s.n_bases;
        if (s.terminated || s.truncated) break;
    }
    return o;
}

void show(const Out& o) {
    printf("  %-32s steps=%4d days=%5lld sum=%9.1f  G(0.99)=%8.1f  G(0.999)=%8.1f  bases=%3lld money=%7lld\n",
           o.note.c_str(), o.steps, (long long)o.days, o.sum, o.g99, o.g999,
           (long long)o.bases, (long long)o.money);
}

}  // namespace

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("== discounted objective by policy (map 200, seed 42, max 4000 steps) ==\n");
    for (int64_t seed : {42, 777}) {
        printf("seed %lld\n", (long long)seed);
        show(run(seed, 4000, [](ColonyEnvCpp&) { return A_DAY; }, bd, ed, "DAY only (do nothing)"));
        show(run(seed, 4000, [](ColonyEnvCpp& e) { return A_WEEK; }, bd, ed, "WEEK only"));
        show(run(seed, 4000,
                 [](ColonyEnvCpp& e) {
                     for (int b = 0; b < e.n_build(); b++) {
                         auto m = e.action_mask();
                         if (m[(size_t)(A_BUILD0 + b)] > 0.5f) return A_BUILD0 + b;
                     }
                     return A_DAY;
                 },
                 bd, ed, "cheapest legal build"));
        show(run(seed, 4000,
                 [](ColonyEnvCpp& e) {
                     // most expensive legal build: maximise one-off build bonuses
                     auto m = e.action_mask();
                     int best = -1;
                     for (int b = 0; b < e.n_build(); b++)
                         if (m[(size_t)(A_BUILD0 + b)] > 0.5f) best = b;
                     return best >= 0 ? A_BUILD0 + best : A_DAY;
                 },
                 bd, ed, "priciest legal build"));
        show(run(seed, 4000,
                 [](ColonyEnvCpp& e) {
                     // no buildings at all, just roads (cheapest way to farm
                     // "new building" bookkeeping) -> tests milestone farming
                     auto m = e.action_mask();
                     int rd = find_id(e, "Road");
                     if (rd >= 0 && m[(size_t)(A_BUILD0 + rd)] > 0.5f) return A_BUILD0 + rd;
                     return A_DAY;
                 },
                 bd, ed, "road spam"));
        show(run(seed, 4000,
                 [](ColonyEnvCpp& e) {
                     // water first, then producers, keeping a tax buffer
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
                 },
                 bd, ed, "water-first + invest"));
        printf("\n");
    }
    return 0;
}
