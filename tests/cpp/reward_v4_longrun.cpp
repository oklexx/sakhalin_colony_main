// Standalone harness: ломает ли reward_v4 road-spam-аргмакс?
//
// Открытый вопрос P0-2 (docs/REMAINING_WORK_2026_09.md): на профиле v3 спам
// дорогами давал +898 против -233 у работающей экономики
// (docs/REVIEW_WATER_MODES_2026_09.md). Профиль v4 (docs/AUDIT_REWARD_2026_09.md)
// должен это переломить. Проверка — длинный прогон одних и тех же политик на
// одном и том же C++ ядре, три профиля наград:
//
//   v3_file  configs/reward_v3.json как его грузит текущий загрузчик
//            (ВНИМАНИЕ: ключей v4 в файле нет, поэтому goal_survival_coeff и
//             main_tax_* остаются дефолтами структуры — см. вывод [warn])
//   v3_pure  тот же v3, но v4-терминалы явно обнулены (честный «старый» профиль)
//   v4_file  configs/reward_v4.json — канон
//
// Политики: DAY-only, road-spam (обычный и направленный), «экономика»
// (водоканал → фермы/сады/дома + продажа излишка), жадная смешанная.
//
// Вердикт: на v4 road-spam НЕ должен быть максимумом среди политик.
//
// Build (from the repo root, Linux/macOS) — одна строка, без продолжений:
//   g++ -std=c++17 -O1 -Iinclude -Iinclude/third_party -o /tmp/reward_v4
//       tests/cpp/reward_v4_longrun.cpp src/env.cpp src/data.cpp
//       src/resources.cpp src/game.cpp src/rewards.cpp src/rng.cpp src/earth.cpp
//   /tmp/reward_v4 [steps] [seed,seed,...]
#include "colony/env.h"
#include "colony/data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <json.hpp>

using namespace colony;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    printf("%s  %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) failures++;
}

// ── Загрузчик профиля наград: явная таблица полей, строгая к неизвестным ────
#define RCF_D(name) {#name, &RewardConfig::name}
#define RCF_I(name) {#name, &RewardConfig::name}

struct FieldSpec {
    const char* name;
    double RewardConfig::*ptr;
};
struct IntFieldSpec {
    const char* name;
    int RewardConfig::*ptr;
};

static RewardConfig load_reward_profile(const std::string& path, RewardConfig base) {
    std::ifstream f(path);
    if (!f) {
        printf("[FAIL] не удалось открыть профиль наград: %s\n", path.c_str());
        failures++;
        return base;
    }
    nlohmann::json j = nlohmann::json::parse(f);

    // Таблица полей собирается из тех же имён, что и в rl/config.py /
    // include/colony/reward_config.h. Числовые и целочисленные поля.
    static const FieldSpec kFields[] = {
        RCF_D(build_bonus), RCF_D(chain_bonus), RCF_D(chain_daily),
        RCF_D(first_extraction_bonus), RCF_D(extraction_daily), RCF_D(need_fill_bonus),
        RCF_D(loan_penalty), RCF_D(tax_debt_penalty), RCF_D(novelty),
        RCF_D(daily_income), RCF_D(sale_bonus), RCF_D(tax_daily_bonus),
        RCF_D(survival_bonus), RCF_D(game_over_penalty), RCF_D(diversity_bonus),
        RCF_D(error_penalty), RCF_D(preserve_penalty), RCF_D(demolish_penalty),
        RCF_D(manual_tax_penalty), RCF_D(build_cost_penalty), RCF_D(idle_build_penalty),
        RCF_D(survival_coeff),
        RCF_D(milestone_base_bonus), RCF_D(milestone_people_bonus),
        RCF_D(milestone_day_bonus), RCF_D(milestone_year_bonus),
        RCF_D(proximity_bonus), RCF_D(clip_reward_min), RCF_D(clip_reward_max),
        // P2-9: road-shaping к воде (вынесен из хардкода src/env.cpp)
        RCF_D(road_shaping_cap), RCF_D(road_shaping_per_cell), RCF_D(water_reach_bonus),
        RCF_D(water_reach_radius), RCF_D(road_no_progress_penalty),
        RCF_D(road_progress_epsilon),
        RCF_D(tax_fail_penalty), RCF_D(death_penalty), RCF_D(base_lost_penalty),
        RCF_D(born_bonus), RCF_D(debt_coeff), RCF_D(home_overflow_penalty),
        RCF_D(housing_need_bonus), RCF_D(food_need_bonus), RCF_D(water_need_bonus),
        RCF_D(buy_food_penalty), RCF_D(goal_survival_coeff),
        RCF_D(main_tax_cash_bonus), RCF_D(main_tax_pressure_coeff),
    };
    static const IntFieldSpec kIntFields[] = {RCF_I(idle_build_threshold_days)};

    // Флаги-абляции — отдельно (bool).
    static bool RewardConfig::*const kFlags[] = {
        &RewardConfig::disable_net_worth, &RewardConfig::disable_daily_income,
        &RewardConfig::disable_provider_bonus, &RewardConfig::mask_managers_by_applicability,
        &RewardConfig::priority_count_over_allowed, &RewardConfig::obs_mask_locked_catalog,
    };
    static const char* kFlagNames[] = {
        "disable_net_worth", "disable_daily_income", "disable_provider_bonus",
        "mask_managers_by_applicability", "priority_count_over_allowed",
        "obs_mask_locked_catalog",
    };

    std::map<std::string, bool> known;
    for (const FieldSpec& s : kFields) known[s.name] = true;
    for (const IntFieldSpec& s : kIntFields) known[s.name] = true;
    for (const char* n : kFlagNames) known[n] = true;
    known["_comment"] = true;

    std::vector<std::string> unknown, unset;
    for (auto it = j.begin(); it != j.end(); ++it)
        if (!known.count(it.key())) unknown.push_back(it.key());
    for (const FieldSpec& s : kFields)
        if (!j.contains(s.name)) unset.push_back(s.name);
    for (const IntFieldSpec& s : kIntFields)
        if (!j.contains(s.name)) unset.push_back(s.name);
    for (const char* n : kFlagNames)
        if (!j.contains(n)) unset.push_back(n);

    if (!unknown.empty()) {
        printf("[FAIL] %s: неизвестные ключи (загрузчик их молча проигнорирует):", path.c_str());
        for (const std::string& k : unknown) printf(" %s", k.c_str());
        printf("\n");
        failures++;
    }
    if (!unset.empty()) {
        printf("[warn] %s: НЕ заданы (останутся дефолтами структуры = v4):", path.c_str());
        for (const std::string& k : unset) printf(" %s", k.c_str());
        printf("\n");
    }

    for (const FieldSpec& s : kFields) {
        if (!j.contains(s.name)) continue;
        base.*(s.ptr) = j[s.name].get<double>();
    }
    for (const IntFieldSpec& s : kIntFields) {
        if (!j.contains(s.name)) continue;
        base.*(s.ptr) = j[s.name].get<int>();
    }
    for (size_t i = 0; i < sizeof(kFlags) / sizeof(kFlags[0]); ++i)
        if (j.contains(kFlagNames[i])) base.*(kFlags[i]) = j[kFlagNames[i]].get<bool>();
    return base;
}

static int find_idx(ColonyEnvCpp& e, const std::string& id) {
    for (int i = 0; i < e.n_build(); ++i)
        if (e.build_ids()[(size_t)i] == id) return i;
    return -1;
}

// Менеджерские действия лежат перед блоком направлений.
static int mgr_act(ColonyEnvCpp& e, int manager_index) {
    return e.road_dir_base() - N_MANAGERS + manager_index;
}

struct PolicyResult {
    double reward = 0.0;
    int steps = 0;
    int64_t days = 0, people = 0, money = 0;
    int bases = 0, roads = 0, waterchannels = 0, farms = 0;
    int64_t food = 0, water = 0;
    bool game_over = false;
};

static PolicyResult run_policy(const std::vector<BaseData>& bd,
                               const std::vector<BaseEvent>& ed,
                               const RewardConfig& cfg, int64_t seed, int max_steps,
                               const std::function<int(ColonyEnvCpp&)>& policy) {
    ColonyEnvCpp e(bd, ed, seed, 200, Curriculum(), cfg);
    e.reset(seed);
    PolicyResult r;
    for (int s = 0; s < max_steps; ++s) {
        auto out = e.step(policy(e));
        r.reward += out.rew;
        r.steps++;
        r.game_over = out.terminated;
        if (out.terminated || out.truncated) break;
    }
    const Game& g = e.game();
    r.days = g.days_alive;
    r.people = g.people;
    r.money = g.money;
    r.bases = (int)g.bases.size();
    r.food = g.sunduk[FOOD];
    r.water = g.sunduk[WATER];
    for (const Base& b : g.bases) {
        if (b.data->id == ROAD_ID) r.roads++;
        else if (b.data->id == "WaterChannel") r.waterchannels++;
        else if (b.data->id == "Farm" || b.data->id == "BigFarm" || b.data->id == "Garden")
            r.farms++;
    }
    return r;
}

struct PolicySet {
    std::vector<std::string> names;
    std::vector<std::function<int(ColonyEnvCpp&)>> fns;
};

static PolicySet make_policies() {
    PolicySet ps;
    // 1. DAY-only: нижняя граница (ничего не делать).
    ps.names.push_back("day_only");
    ps.fns.push_back([](ColonyEnvCpp&) { return A_DAY; });

    // 2. Road spam — исторический аргмакс (+898 на v3).
    ps.names.push_back("road_spam");
    ps.fns.push_back([](ColonyEnvCpp& e) {
        auto m = e.action_mask();
        int rd = find_idx(e, "Road");
        if (rd >= 0 && m[A_BUILD0 + rd] != 0.0f) return A_BUILD0 + rd;
        return A_DAY;
    });

    // 3. Направленный road spam (ROAD_E), т.е. то же самое после 45→49.
    ps.names.push_back("road_spam_dir");
    ps.fns.push_back([](ColonyEnvCpp& e) {
        auto m = e.action_mask();
        if (m[e.road_dir_base()] != 0.0f) return e.road_dir_base();
        return A_DAY;
    });

    // 4. «Работающая экономика»: водоканал → дома → фермы/сады, продажа излишка.
    //    Дорога — только направленная, к воде, и только пока водоканала нет.
    ps.names.push_back("economy");
    ps.fns.push_back([](ColonyEnvCpp& e) {
        auto m = e.action_mask();
        if (m[mgr_act(e, 6)] != 0.0f) return mgr_act(e, 6);  // SELL_SURPLUS
        static const char* prio[] = {"WaterChannel", "Farm", "Garden", "House",
                                     "SmallHouse", "CowFarm", "Sawmill", "Coalmine"};
        bool has_channel = false;
        for (const Base& b : e.game().bases)
            if (b.data->id == "WaterChannel") has_channel = true;
        for (const char* id : prio) {
            int idx = find_idx(e, id);
            if (idx >= 0 && m[A_BUILD0 + idx] != 0.0f) return A_BUILD0 + idx;
        }
        if (!has_channel) {
            // тянем дорогу к воде по water_dx/dy (хвост obs v0)
            const auto& obs = e.obs();
            int n0 = 27 + e.n_build() + 7 + 9 + 4 * e.n_build() + 9 + e.n_build();
            double dx = obs[(size_t)n0], dy = obs[(size_t)n0 + 1];
            int dir = (std::fabs(dx) >= std::fabs(dy)) ? (dx >= 0 ? 0 : 1)
                                                       : (dy >= 0 ? 2 : 3);
            if (m[e.road_dir_base() + dir] != 0.0f) return e.road_dir_base() + dir;
        }
        return A_DAY;
    });

    // 5. Смешанная жадная (как «greedy» в reward_audit_probe): сначала продажи,
    //    потом водоканал, дорога, дома, фермы.
    ps.names.push_back("greedy_mix");
    ps.fns.push_back([](ColonyEnvCpp& e) {
        auto m = e.action_mask();
        if (m[mgr_act(e, 6)] != 0.0f) return mgr_act(e, 6);
        static const char* prio[] = {"WaterChannel", "Road", "House", "SmallHouse",
                                     "Farm", "Garden"};
        for (const char* id : prio) {
            int idx = find_idx(e, id);
            if (idx >= 0 && m[A_BUILD0 + idx] != 0.0f) return A_BUILD0 + idx;
        }
        return A_DAY;
    });
    return ps;
}

int main(int argc, char** argv) {
    const int max_steps = argc > 1 ? std::atoi(argv[1]) : 4000;
    std::vector<int64_t> seeds = {1, 42, 777};
    if (argc > 2) {
        seeds.clear();
        std::stringstream ss(argv[2]);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) seeds.push_back(std::stoll(tok));
    }

    auto bd = load_base_data("configs/bases.json");
    auto ed = load_events("configs/events.json");

    printf("=== Профили наград ===\n");
    RewardConfig v3_file = load_reward_profile("configs/reward_v3.json", RewardConfig());
    RewardConfig v3_pure = v3_file;
    v3_pure.goal_survival_coeff = 0.0;
    v3_pure.main_tax_cash_bonus = 0.0;
    v3_pure.main_tax_pressure_coeff = 0.0;
    v3_pure.survival_coeff = 0.0;
    RewardConfig v4_file = load_reward_profile("configs/reward_v4.json", RewardConfig());
    printf("[info] v3_file: milestone_base=%.1f build_bonus=%.1f clip=[%.0f,%.0f] "
           "goal_survival=%.1f main_tax_cash=%.1f debt_coeff=%.2f\n",
           v3_file.milestone_base_bonus, v3_file.build_bonus, v3_file.clip_reward_min,
           v3_file.clip_reward_max, v3_file.goal_survival_coeff,
           v3_file.main_tax_cash_bonus, v3_file.debt_coeff);
    printf("[info] v3_pure: goal_survival=%.1f main_tax_cash=%.1f main_tax_pressure=%.4f\n",
           v3_pure.goal_survival_coeff, v3_pure.main_tax_cash_bonus,
           v3_pure.main_tax_pressure_coeff);
    printf("[info] v4_file: milestone_base=%.1f build_bonus=%.1f clip=[%.0f,%.0f] "
           "goal_survival=%.1f main_tax_cash=%.1f debt_coeff=%.2f\n",
           v4_file.milestone_base_bonus, v4_file.build_bonus, v4_file.clip_reward_min,
           v4_file.clip_reward_max, v4_file.goal_survival_coeff,
           v4_file.main_tax_cash_bonus, v4_file.debt_coeff);

    PolicySet ps = make_policies();
    struct Profile {
        const char* tag;
        const RewardConfig* cfg;
    };
    const Profile profiles[] = {{"v3_pure", &v3_pure}, {"v3_file", &v3_file},
                                {"v4_file", &v4_file}};

    std::map<std::string, std::map<std::string, double>> mean_reward;
    std::map<std::string, std::map<std::string, PolicyResult>> last_result;

    for (const Profile& prof : profiles) {
        printf("\n=== Профиль %s: %d шагов × %zu семян ===\n", prof.tag, max_steps,
               seeds.size());
        printf("%-14s %-9s %-7s %-7s %-8s %-7s %-7s %-6s %-6s %-7s %-7s\n",
               "policy", "reward", "steps", "gover", "days", "people", "money",
               "bases", "roads", "wc", "food");
        for (size_t p = 0; p < ps.names.size(); ++p) {
            double sum = 0.0;
            PolicyResult agg;
            int govers = 0;
            for (int64_t seed : seeds) {
                PolicyResult r = run_policy(bd, ed, *prof.cfg, seed, max_steps, ps.fns[p]);
                sum += r.reward;
                if (r.game_over) govers++;
                agg = r;  // для последней строки — показатели последнего сида
            }
            double mean = sum / (double)seeds.size();
            mean_reward[prof.tag][ps.names[p]] = mean;
            last_result[prof.tag][ps.names[p]] = agg;
            printf("%-14s %-9.1f %-7d %-7d %-8lld %-7lld %-7lld %-6d %-6d %-7d %-7lld\n",
                   ps.names[p].c_str(), mean, agg.steps, govers, (long long)agg.days,
                   (long long)agg.people, (long long)agg.money, agg.bases, agg.roads,
                   agg.waterchannels, (long long)agg.food);
        }
    }

    // ── Сводка: кто аргмакс ─────────────────────────────────────────────────
    printf("\n=== Сводка: средняя награда по политикам ===\n");
    printf("%-14s %-12s %-12s %-12s\n", "policy", "v3_pure", "v3_file", "v4_file");
    for (const std::string& name : ps.names)
        printf("%-14s %-12.1f %-12.1f %-12.1f\n", name.c_str(),
               mean_reward["v3_pure"][name], mean_reward["v3_file"][name],
               mean_reward["v4_file"][name]);

    auto best_of = [&](const char* prof, const std::string& except) {
        double best = -1e18;
        std::string who;
        for (const std::string& name : ps.names) {
            if (name == except) continue;
            if (mean_reward[prof][name] > best) { best = mean_reward[prof][name]; who = name; }
        }
        return std::make_pair(best, who);
    };

    const double v3_road = mean_reward["v3_pure"]["road_spam"];
    auto v3_best = best_of("v3_pure", "road_spam");
    const double v4_road = mean_reward["v4_file"]["road_spam"];
    auto v4_best = best_of("v4_file", "road_spam");
    const double v4_road_dir = mean_reward["v4_file"]["road_spam_dir"];

    printf("\n=== ВЕРДИКТ по P0-2 ===\n");
    printf("v3_pure: road_spam=%.1f, лучшая остальная политика — %s (%.1f) → спам %s\n",
           v3_road, v3_best.second.c_str(), v3_best.first,
           v3_road > v3_best.first ? "ВЫИГРЫВАЕТ (исторический баг)" : "проигрывает");
    printf("v4_file: road_spam=%.1f, лучшая остальная политика — %s (%.1f) → спам %s\n",
           v4_road, v4_best.second.c_str(), v4_best.first,
           v4_road > v4_best.first ? "ВЫИГРЫВАЕТ (v4 НЕ чинит аргмакс)"
                                   : "проигрывает (v4 ломает road-spam-аргмакс)");
    printf("v4_file: направленный road_spam_dir=%.1f (тоже должен проигрывать экономике)\n",
           v4_road_dir);

    check(v4_road < v4_best.first,
          "P0-2 на reward_v4 спам дорогами НЕ является максимумом среди политик");
    check(v4_road_dir < v4_best.first,
          "P0-2 на reward_v4 направленный спам дорогами тоже не максимум");
    check(v3_road < v3_best.first,
          "P0-2 и на v3_pure спам дорогами уже не максимум — дыру закрыл R1 "
          "(дороги исключены из milestone_base_bonus), а не профиль v4");
    check(v3_road < 500.0,
          "P0-2 исторический road-spam +898 на текущем ядре не воспроизводится (<500)");
    check(last_result["v4_file"]["greedy_mix"].waterchannels >= 1,
          "P0-2 на v4 не-спам политика доходит до воды и ставит водоканал");
    check(mean_reward["v4_file"]["day_only"] != mean_reward["v3_pure"]["day_only"],
          "P0-2 профили действительно применяются (day_only на v4 != v3_pure)");
    // configs/reward_v3.json теперь задаёт v4-терминалы явно нулями, поэтому
    // «файл v3» и «чистый v3» совпадают: A/B больше не сравнивает гибрид.
    bool same = true;
    for (const std::string& name : ps.names)
        if (mean_reward["v3_file"][name] != mean_reward["v3_pure"][name]) same = false;
    check(same, "P0-2 configs/reward_v3.json задаёт v4-терминалы явно (v3_file == v3_pure)");

    printf("\n%s (%d failure(s))\n", failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
