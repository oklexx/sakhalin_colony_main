// Standalone diagnostic #5: a hand-played "human" strategy, 6 years, and the
// reward breakdown of the degenerate policies PPO is drawn to.
//
// Build (from the repo root):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/longrun_check \
//       tests/cpp/longrun_check.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
// Run from the repo root:  /tmp/longrun_check
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
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

struct Out {
    int steps = 0;
    int64_t days = 0, money = 0, bases = 0;
    double sum = 0.0, g99 = 0.0, g999 = 0.0;
    std::map<std::string, int> built;
    std::map<std::string, double> comp;
    std::string status = "budget";
};

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

Out run(int64_t seed, int n_steps, const std::function<int(ColonyEnvCpp&)>& policy,
        const std::vector<BaseData>& bd, const std::vector<BaseEvent>& ed,
        const std::string& log_path) {
    if (!log_path.empty()) std::remove(log_path.c_str());
    ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), RewardConfig(), DIFFICULTY_NORMAL);
    e.reset(seed);
    if (!log_path.empty()) e.set_step_log(log_path);
    Out o;
    double d99 = 1.0, d999 = 1.0;
    for (int i = 0; i < n_steps; i++) {
        auto m = e.action_mask();
        int a = policy(e);
        if (a < 0 || a >= (int)m.size() || m[(size_t)a] < 0.5f) a = A_DAY;
        if (a >= A_BUILD0 && a < A_BUILD0 + e.n_build())
            o.built[e.build_ids()[(size_t)(a - A_BUILD0)]]++;
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
        if (s.terminated) { o.status = "TERMINATED"; break; }
        if (s.truncated) { o.status = "truncated"; break; }
    }
    if (!log_path.empty()) { e.set_step_log(""); parse_components(log_path, o.comp); }
    return o;
}

void show(const std::string& label, const Out& o) {
    printf("%-30s steps=%4d days=%5lld sum=%8.1f G(.99)=%7.1f G(.999)=%8.1f money=%7lld bases=%3lld [%s]\n",
           label.c_str(), o.steps, (long long)o.days, o.sum, o.g99, o.g999,
           (long long)o.money, (long long)o.bases, o.status.c_str());
    if (!o.built.empty()) {
        printf("%-30s built:", "");
        for (const auto& kv : o.built) printf(" %s=%d", kv.first.c_str(), kv.second);
        printf("\n");
    }
    if (!o.comp.empty()) {
        printf("%-30s comp:", "");
        for (const auto& kv : o.comp)
            if (std::fabs(kv.second) > 0.5) printf(" %s=%+.0f", kv.first.c_str(), kv.second);
        printf("\n");
    }
}

// A human-ish plan: walk roads to water, build the water chain, then food
// production, sell the surplus, and always keep one tax payment in the bank.
struct Plan {
    int roads = 0, channels = 0, gardens = 0, sells = 0;
};

int human_policy(ColonyEnvCpp& e, Plan& st) {
    auto m = e.action_mask();
    auto obs = e.obs();
    const int MGR = A_BUILD0 + e.n_build();
    const int SELL_SURPLUS = MGR + 6;   // managers: ... PRESERVE, UNPRESERVE, SELL
    int64_t money = (int64_t)std::llround((double)obs[4] * 2e5);
    int64_t food = (int64_t)std::llround(std::min(1.0f, obs[9]) * 100.0);
    int wc = find_id(e, "WaterChannel");
    int gd = find_id(e, "Garden");

    // 0. cash in the warehouse when the barn is full
    if (food >= 90 && m[(size_t)SELL_SURPLUS] > 0.5f) { st.sells++; return SELL_SURPLUS; }
    // 1. never invest the tax buffer away
    if (money < 20000) return A_DAY;
    // 2. water chain
    if (wc >= 0 && st.channels < 2 && m[(size_t)(A_BUILD0 + wc)] > 0.5f) {
        st.channels++;
        return A_BUILD0 + wc;
    }
    if (st.channels < 2 && st.roads < 14) {
        float dx = obs[obs.size() - 2], dy = obs[obs.size() - 1];
        int dir = std::fabs(dx) >= std::fabs(dy) ? (dx > 0 ? 0 : 1) : (dy > 0 ? 2 : 3);
        int a = e.road_dir_base() + dir;
        if (m[(size_t)a] > 0.5f) { st.roads++; return a; }
    }
    // 3. food production
    if (gd >= 0 && st.gardens < 6 && money > 40000 && m[(size_t)(A_BUILD0 + gd)] > 0.5f) {
        st.gardens++;
        return A_BUILD0 + gd;
    }
    return A_DAY;
}

}  // namespace

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("== 1. reward breakdown of the two degenerate attractors (seed 42) ==\n");
    {
        Out o = run(42, 1500,
                    [](ColonyEnvCpp& e) {
                        auto m = e.action_mask();
                        int best = -1;
                        for (int b = 0; b < e.n_build(); b++)
                            if (m[(size_t)(A_BUILD0 + b)] > 0.5f) best = b;
                        return best >= 0 ? A_BUILD0 + best : A_DAY;
                    },
                    bd, ed, "/tmp/longrun_pricy.txt");
        show("priciest legal build", o);
        Out o2 = run(42, 1500,
                     [](ColonyEnvCpp& e) {
                         auto m = e.action_mask();
                         for (int b = 0; b < e.n_build(); b++)
                             if (m[(size_t)(A_BUILD0 + b)] > 0.5f) return A_BUILD0 + b;
                         return A_DAY;
                     },
                     bd, ed, "/tmp/longrun_cheap.txt");
        show("cheapest legal build", o2);
    }

    printf("\n== 2. six-year horizon: manual road->water->food->sell plan ==\n");
    for (int64_t seed : {1, 42, 777}) {
        Plan st;
        Out o = run(seed, 2400,
                    [&](ColonyEnvCpp& e) { return human_policy(e, st); },
                    bd, ed, "");
        show("human plan seed " + std::to_string(seed), o);
        printf("%-30s roads_toward_water=%d water_channels=%d gardens=%d sells=%d\n", "",
               st.roads, st.channels, st.gardens, st.sells);
    }

    printf("\n== 3. same plan, but with the whole capital (no tax buffer) ==\n");
    for (int64_t seed : {1, 42}) {
        Out o = run(seed, 2400,
                    [](ColonyEnvCpp& e) {
                        auto m = e.action_mask();
                        auto obs = e.obs();
                        int wc = find_id(e, "WaterChannel");
                        int gd = find_id(e, "Garden");
                        if (wc >= 0 && m[(size_t)(A_BUILD0 + wc)] > 0.5f) return A_BUILD0 + wc;
                        if (gd >= 0 && m[(size_t)(A_BUILD0 + gd)] > 0.5f) return A_BUILD0 + gd;
                        float dx = obs[obs.size() - 2], dy = obs[obs.size() - 1];
                        int dir = std::fabs(dx) >= std::fabs(dy) ? (dx > 0 ? 0 : 1) : (dy > 0 ? 2 : 3);
                        int a = e.road_dir_base() + dir;
                        if (m[(size_t)a] > 0.5f) return a;
                        return A_DAY;
                    },
                    bd, ed, "");
        show("all-in seed " + std::to_string(seed), o);
    }
    return 0;
}
