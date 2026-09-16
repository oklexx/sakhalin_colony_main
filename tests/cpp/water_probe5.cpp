// Probe 5: reward-component breakdown for "road spam" vs "farm economy".
#include "colony/env.h"
#include "colony/data.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace colony;

static int find_id(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); i++)
        if (e.build_ids()[i] == id) return i;
    return -1;
}

// Parse "key=value" pairs out of the STEP lines and accumulate per component.
static std::map<std::string, double> accumulate(const std::string& path, double& total) {
    std::map<std::string, double> acc;
    std::ifstream in(path);
    std::string line;
    total = 0.0;
    while (std::getline(in, line)) {
        if (line.rfind("STEP ", 0) != 0) continue;
        std::istringstream ss(line);
        std::string tok;
        while (ss >> tok) {
            auto eq = tok.find('=');
            if (eq == std::string::npos) continue;
            std::string k = tok.substr(0, eq);
            double v = atof(tok.c_str() + eq + 1);
            acc[k] += v;
            if (k == "total") total += v;
        }
    }
    return acc;
}

static void run_policy(const std::vector<BaseData>& bd, const std::vector<BaseEvent>& ed,
                       const std::vector<std::string>& pref, const char* logpath,
                       const char* label) {
    ColonyEnvCpp e(bd, ed, 42, 200);
    e.reset(42);
    e.set_step_log(logpath);
    std::vector<int> idx;
    for (auto& p : pref) { int i = find_id(e, p); if (i >= 0) idx.push_back(i); }
    for (int k = 0; k < 1500; k++) {
        auto m = e.action_mask();
        int chosen = -1;
        for (int i : idx) if (m[A_BUILD0 + i] != 0.0f) { chosen = i; break; }
        auto s = e.step(chosen >= 0 ? A_BUILD0 + chosen : 0);
        if (s.terminated || s.truncated) break;
    }
    double total = 0;
    auto acc = accumulate(logpath, total);
    printf("\n--- %s (seed 42, 1500 steps) total=%+.1f ---\n", label, total);
    const char* keys[] = {"build","cost","div","prox","prov","preq","daily","nov","mile",
                          "surv","idle","err","tax","taxb","sale","chain","extr","gover"};
    for (const char* k : keys)
        if (acc.count(k)) printf("   %-6s %+10.2f\n", k, acc[k]);
}

int main() {
    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");
    run_policy(bd, ed, {"Road"}, "/tmp/log_road.txt", "ROADS ONLY");
    run_policy(bd, ed, {"Farm","Garden","SmallHouse"}, "/tmp/log_farm.txt", "FARM ECONOMY (no water)");

    printf("\n=== same farm policy but with water granted ===\n");
    {
        ColonyEnvCpp e(bd, ed, 42, 200);
        e.reset(42);
        e.game().sunduk[WATER] = 100000;
        e.set_step_log("/tmp/log_farmw.txt");
        int f = find_id(e,"Farm"), g = find_id(e,"Garden"), h = find_id(e,"SmallHouse");
        for (int k = 0; k < 1500; k++) {
            auto m = e.action_mask();
            int a = 0;
            if (m[A_BUILD0+f] != 0.0f) a = A_BUILD0+f;
            else if (m[A_BUILD0+g] != 0.0f) a = A_BUILD0+g;
            else if (m[A_BUILD0+h] != 0.0f) a = A_BUILD0+h;
            auto s = e.step(a);
            if (s.terminated || s.truncated) break;
        }
        double total = 0;
        auto acc = accumulate("/tmp/log_farmw.txt", total);
        printf("--- FARM ECONOMY (water granted) total=%+.1f ---\n", total);
        const char* keys[] = {"build","cost","div","prox","prov","preq","daily","nov","mile",
                              "surv","idle","err","tax","taxb","sale","chain","extr","gover"};
        for (const char* k : keys)
            if (acc.count(k)) printf("   %-6s %+10.2f\n", k, acc[k]);
    }
    return 0;
}
