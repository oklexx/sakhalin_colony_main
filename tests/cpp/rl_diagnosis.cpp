// Standalone diagnostic harness: why does PPO not learn this economy?
//
// Measures, on the real C++ core (no Python, no raylib):
//   A) action-mask accessibility — how many actions are legal per step, which
//      building actions are ever reachable at all, cost of computing the mask;
//   B) return of degenerate policies (DAY spam / WEEK spam / ROAD spam /
//      masked-random / water-greedy / cheap-build-greedy) with the full
//      reward-component breakdown from the env's own step log;
//   C) per-step cost of action_mask() vs step() (the mask runs find_lot() for
//      every one of the 32 buildings on every single step);
//   D) terminal checks: does a do-nothing policy beat a building policy?
//
// Build (from the repo root):
//   g++ -std=c++17 -O2 -Iinclude -Iinclude/third_party -o /tmp/rl_diag \
//       tests/cpp/rl_diagnosis.cpp \
//       src/env.cpp src/data.cpp src/resources.cpp src/game.cpp \
//       src/rewards.cpp src/rng.cpp src/earth.cpp
// Run from the repo root (configs/*.json are loaded by relative path):
//   /tmp/rl_diag
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace colony;

namespace {

int find_id(const ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

struct RunStats {
    double total = 0.0;
    int steps = 0;
    int64_t days = 0, money = 0, people = 0, bases = 0;
    int64_t food = 0, water = 0;
    int build_actions = 0;
    int failed_builds = 0;
    std::map<std::string, double> comp;
    std::map<std::string, int> build_counts;  // id -> count placed
    std::vector<double> mask_sizes;           // available actions per step
    std::vector<double> mask_us;              // time to compute mask, µs
};

// Parse the env step log (append mode) for the final component sums.
void parse_components(const std::string& path, std::map<std::string, double>& out,
                      std::map<std::string, int>& build_counts,
                      int& build_actions, int& failed_builds) {
    std::ifstream f(path);
    std::string line;
    const std::vector<std::string> keys = {
        "err", "tax", "taxb", "build", "cost", "div", "prox", "prov", "preq",
        "pres", "sale", "mtax", "chain", "daily", "nov", "mile", "surv",
        "idle", "gover", "extr", "loan"};
    static const std::vector<std::string> anchors = {"tax=", "taxb="};  // avoid tax inside taxb
    std::string last_build;
    while (std::getline(f, line)) {
        if (line.rfind("  ACTION:", 0) == 0) {
            // "  ACTION: 12 (BUILD:Road price=400 ...)"
            auto p = line.find("(BUILD:");
            if (p != std::string::npos) {
                size_t b = p + 7;
                size_t e = line.find_first_of(" )", b);
                last_build = line.substr(b, e - b);
            } else {
                last_build.clear();
            }
        }
        if (line.rfind("  BUILD FAILED:", 0) == 0) failed_builds++;
        if (line.rfind("STEP ", 0) != 0) continue;
        if (!last_build.empty()) {
            build_counts[last_build]++;
            build_actions++;
            last_build.clear();
        }
        for (const auto& k : keys) {
            std::string tag = k + "=";
            size_t pos = line.find(" " + tag);
            if (pos == std::string::npos) continue;
            // "tax=" must not match the tail of "taxb="
            if (k == "tax" && line.compare(pos + 1, 5, "taxb=") == 0) continue;
            size_t v = pos + 1 + tag.size();
            size_t e = line.find(' ', v);
            out[k] += std::stod(line.substr(v, e - v));
        }
    }
}

struct Env {
    std::vector<BaseData> bd;
    std::vector<BaseEvent> ed;
    RewardConfig cfg;
    std::unique_ptr<ColonyEnvCpp> env;

    Env(int64_t seed, int map_size = 200, const Curriculum& c = Curriculum()) {
        bd = load_base_data("configs/bases.json");
        ed = load_events("configs/events.json");
        env = std::make_unique<ColonyEnvCpp>(bd, ed, seed, map_size, c, cfg,
                                             DIFFICULTY_NORMAL);
        env->reset(seed);
    }
};

using Policy = std::function<int(ColonyEnvCpp&, const std::vector<float>&)>;

RunStats run_policy(const std::string& label, int64_t seed, int steps,
                    const Policy& policy, int map_size = 200,
                    const Curriculum& c = Curriculum()) {
    const std::string log = "/tmp/rl_diag_" + label + "_" + std::to_string(seed) + ".txt";
    std::remove(log.c_str());
    Env e(seed, map_size, c);
    e.env->set_step_log(log);

    const int wc = find_id(*e.env, "WaterChannel");
    (void)wc;
    RunStats st;
    for (int i = 0; i < steps; i++) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> mask = e.env->action_mask();
        auto t1 = std::chrono::steady_clock::now();
        st.mask_us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
        double navail = 0;
        for (float m : mask) navail += m > 0.5f ? 1.0 : 0.0;
        st.mask_sizes.push_back(navail);

        std::vector<float> obs = e.env->obs();
        int a = policy(*e.env, obs);
        if (a >= 0 && a < (int)mask.size() && mask[(size_t)a] < 0.5f) a = A_DAY;  // keep it legal
        auto out = e.env->step(a);
        st.total += out.rew;
        st.steps++;
        st.days = out.days;
        st.money = out.money;
        st.people = out.people;
        st.bases = out.n_bases;
        if (out.terminated || out.truncated) break;
    }
    e.env->set_step_log("");
    parse_components(log, st.comp, st.build_counts, st.build_actions, st.failed_builds);
    // food/water from the last OBS log line
    {
        std::ifstream f(log);
        std::string line, last;
        while (std::getline(f, line))
            if (line.rfind("STEP ", 0) == 0) last = line;
        for (const char* key : {" food=", " water="}) {
            size_t p = last.find(key);
            if (p == std::string::npos) continue;
            size_t v = p + std::string(key).size();
            size_t en = last.find(' ', v);
            int64_t val = std::stoll(last.substr(v, en - v));
            (std::string(key) == " food=" ? st.food : st.water) = val;
        }
    }
    return st;
}

void print_run(const std::string& label, const RunStats& st) {
    double avg_mask = 0;
    for (double v : st.mask_sizes) avg_mask += v;
    if (!st.mask_sizes.empty()) avg_mask /= (double)st.mask_sizes.size();
    double avg_us = 0;
    for (double v : st.mask_us) avg_us += v;
    if (!st.mask_us.empty()) avg_us /= (double)st.mask_us.size();

    printf("%-22s steps=%4d total=%9.2f | bases=%3lld people=%3lld money=%7lld "
           "days=%5lld food=%6lld water=%5lld | builds=%3d fail=%3d | mask=%.1f/%d %.0fus\n",
           label.c_str(), st.steps, st.total, (long long)st.bases, (long long)st.people,
           (long long)st.money, (long long)st.days, (long long)st.food,
           (long long)st.water, st.build_actions, st.failed_builds, avg_mask,
           (int)st.mask_sizes.empty() ? 0 : 49, avg_us);
    printf("    components:");
    for (const auto& kv : st.comp)
        if (std::fabs(kv.second) > 0.01) printf(" %s=%+.1f", kv.first.c_str(), kv.second);
    printf("\n");
    if (!st.build_counts.empty()) {
        printf("    placed:");
        for (const auto& kv : st.build_counts) printf(" %s=%d", kv.first.c_str(), kv.second);
        printf("\n");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 1500;
    const std::vector<int64_t> seeds = {1, 42, 777};

    printf("======================= A. mask accessibility =======================\n");
    {
        // Which actions are legal on a fresh env, and how often they stay legal
        // under a uniform masked-random policy.
        Env e(42, 200);
        std::vector<float> m = e.env->action_mask();
        int total_avail = 0;
        std::vector<std::string> names = {"DAY", "WEEK"};
        for (const auto& b : e.env->build_ids()) names.push_back("BUILD:" + b);
        names.push_back("IMPROVE");
        names.push_back("REPAIR");
        names.push_back("REPAIR_ALL");
        names.push_back("DEMOLISH");
        names.push_back("PRESERVE");
        names.push_back("UNPRESERVE");
        names.push_back("SELL");
        names.push_back("BUY_FOOD");
        names.push_back("TAKE_LOAN");
        names.push_back("REPAY_LOAN");
        names.push_back("PAY_TAX");
        names.push_back("ROAD_E");
        names.push_back("ROAD_W");
        names.push_back("ROAD_S");
        names.push_back("ROAD_N");
        printf("available at t=0 (seed 42):");
        for (size_t i = 0; i < m.size(); i++)
            if (m[i] > 0.5f) {
                printf(" %s", i < names.size() ? names[i].c_str() : "?");
                total_avail++;
            }
        printf("\n  -> %d / %d actions legal on the first decision\n\n", total_avail,
               (int)m.size());
    }

    printf("===================== B. policy returns (%d steps) =====================\n", steps);
    // WaterChannel availability over a uniform masked-random rollout
    {
        for (int64_t seed : seeds) {
            int wc_steps = 0, any_build_steps = 0, n = 0;
            double sum_avail = 0;
            Env e(seed, 200);
            int wc = find_id(*e.env, "WaterChannel");
            unsigned rng = 12345u + (unsigned)seed;
            for (int i = 0; i < 2000; i++) {
                std::vector<float> mask = e.env->action_mask();
                std::vector<int> legal;
                for (size_t k = 0; k < mask.size(); k++)
                    if (mask[k] > 0.5f) legal.push_back((int)k);
                for (size_t k = 0; k < mask.size(); k++) sum_avail += mask[k];
                bool any_b = false;
                for (int b = 0; b < e.env->n_build(); b++)
                    if (mask[(size_t)(A_BUILD0 + b)] > 0.5f) any_b = true;
                if (any_b) any_build_steps++;
                if (wc >= 0 && mask[(size_t)(A_BUILD0 + wc)] > 0.5f) wc_steps++;
                n++;
                rng = rng * 1664525u + 1013904223u;
                int pick = legal[rng % legal.size()];
                auto out = e.env->step(pick);
                if (out.terminated || out.truncated) { e.env->reset(seed + i); }
            }
            printf("seed %5lld  water_channel_legal=%4d/%4d (%.2f%%)  any_build_legal=%4d  "
                   "avg_legal_actions=%.1f/49\n",
                   (long long)seed, wc_steps, n, 100.0 * wc_steps / n, any_build_steps,
                   sum_avail / n);
        }
        printf("\n");
    }

    auto day_only = [](ColonyEnvCpp&, const std::vector<float>&) { return A_DAY; };
    auto week_only = [](ColonyEnvCpp&, const std::vector<float>&) { return A_WEEK; };

    auto road_only = [](ColonyEnvCpp& e, const std::vector<float>&) {
        int r = find_id(e, "Road");
        return r >= 0 ? A_BUILD0 + r : A_DAY;
    };

    auto road_dir_only = [](ColonyEnvCpp&, const std::vector<float>&) { return 45; };  // ROAD_E

    auto cheap_build = [](ColonyEnvCpp& e, const std::vector<float>&) {
        // cheapest legal building action, else DAY
        static std::vector<float> mask;
        // recompute via env (mask is deterministic, called right before step)
        mask = const_cast<ColonyEnvCpp&>(e).action_mask();
        int best = -1;
        for (int b = 0; b < e.n_build(); b++) {
            if (mask[(size_t)(A_BUILD0 + b)] < 0.5f) continue;
            if (best < 0) best = b;
        }
        return best >= 0 ? A_BUILD0 + best : A_DAY;
    };

    auto water_greedy = [](ColonyEnvCpp& e, const std::vector<float>& obs) {
        std::vector<float> mask = const_cast<ColonyEnvCpp&>(e).action_mask();
        int wc = find_id(e, "WaterChannel");
        if (wc >= 0 && mask[(size_t)(A_BUILD0 + wc)] > 0.5f) return A_BUILD0 + wc;
        float dx = obs[obs.size() - 2], dy = obs[obs.size() - 1];
        int dir = -1;
        if (std::fabs(dx) >= std::fabs(dy)) dir = dx > 0 ? 0 : (dx < 0 ? 1 : -1);
        else dir = dy > 0 ? 2 : (dy < 0 ? 3 : -1);
        if (dir >= 0) {
            int a = e.road_dir_base() + dir;
            if (mask[(size_t)a] > 0.5f) return a;
        }
        return A_DAY;
    };

    for (int64_t seed : seeds) {
        printf("-- seed %lld --\n", (long long)seed);
        print_run("DAY only", run_policy("day", seed, steps, day_only));
        print_run("WEEK only", run_policy("week", seed, steps, week_only));
        print_run("BUILD Road only", run_policy("road", seed, steps, road_only));
        print_run("ROAD_E only", run_policy("roade", seed, steps, road_dir_only));
        print_run("cheapest legal build", run_policy("cheap", seed, steps, cheap_build));
        print_run("water greedy", run_policy("watergreedy", seed, steps, water_greedy));
        printf("\n");
    }

    printf("===================== C. per-step cost =====================\n");
    {
        Env e(42, 200);
        const int N = 3000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; i++) e.env->action_mask();
        auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; i++) {
            e.env->action_mask();
            auto out = e.env->step(A_DAY);
            (void)out;
            if (out.terminated || out.truncated) e.env->reset(42 + i);
        }
        auto t2 = std::chrono::steady_clock::now();
        double mask_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        double full_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / N;
        printf("action_mask(): %.3f ms/call   (mask + step): %.3f ms/call\n", mask_ms, full_ms);
        printf("10000 steps -> %.1f s/env  (mask alone %.1f s)\n", full_ms * 10.0,
               mask_ms * 10.0);
    }
    return 0;
}
