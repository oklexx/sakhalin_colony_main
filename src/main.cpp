// Sakhalin Colony — консольная версия (без pybind11)
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "colony/bases.h"
#include "colony/data.h"
#include "colony/env.h"
#include "colony/game.h"

using namespace colony;

static void print_header() {
    std::cout << "========================================\n"
              << "   SAHALIN COLONY — Survival Game\n"
              << "   Year " << START_YEAR << ", Sakhalin Island\n"
              << "========================================\n\n";
}

static void print_status(const ColonyEnvCpp& env) {
    const Game& g = env.game();
    std::cout << "\n--- STATUS ---\n";
    std::cout << "Date:    " << g.year << "-" << g.month << "-" << g.day;
    std::cout << "  Season: " << SEASON_NAMES[g.season] << "\n";
    std::cout << "Money:   " << thousands(g.money)
              << "  Credit: " << thousands(g.credit) << "\n";
    std::cout << "People:  " << g.people
              << "  Free: " << g.free_people()
              << "  Homes: " << g.now_home_places() << "\n";
    std::cout << "Bases:   " << g.bases.size() << "\n";
    std::cout << "Resources: ";
    for (int i = 0; i < SUNDUK_SIZE; i++) {
        std::cout << SUNDUK_CAPTIONS[i] << "=" << g.sunduk[i] << " ";
    }
    std::cout << "\n";
    if (g.annual_tax_due())
        std::cout << "  [!] Annual tax due: " << thousands(g.annual_tax_amount()) << "\n";
    if (g.main_tax_due())
        std::cout << "  [!] Main tax due: " << thousands(g.main_tax_amount()) << "\n";
    auto ov = g.game_over();
    if (ov) std::cout << "  [GAME OVER] " << ov->reason << "\n";
    std::cout << "------------------\n";
}

static void print_menu(const ColonyEnvCpp& env) {
    std::cout << "\nACTIONS:\n";
    std::cout << "  [1]  Advance day\n";
    std::cout << "  [2]  Advance week\n";
    std::cout << "\nBUILD:\n";
    for (int i = 0; i < env.n_build(); i++) {
        int key = i + 3;
        const BaseData& d = *env.build_data()[i];
        bool locked = !env.build_allowed(d.id);  // PR 2
        std::cout << "  [" << key << "]  " << d.caption << " (" << d.id << ")"
                  << "  $" << thousands(d.price)
                  << "  workers=" << d.need_workers;
        if (locked) std::cout << "  [locked]";
        if (d.live_years)
            std::cout << "  life=" << d.live_years << "y";
        std::cout << "\n";
    }
    int mgr_base = 3 + env.n_build();
    const char* mgr_names[] = {
        "Improve earth", "Repair worst", "Repair all",
        "Demolish worst", "Preserve oldest", "Unpreserve oldest",
        "Sell surplus", "Buy food", "Take loan (50k)",
        "Repay loan (50k)", "Pay tax (manual)",
    };
    std::cout << "\nMANAGE:\n";
    for (int i = 0; i < N_MANAGERS; i++) {
        std::cout << "  [" << mgr_base + i << "]  " << mgr_names[i] << "\n";
    }
    int quit_key = mgr_base + N_MANAGERS;
    std::cout << "\n  [" << quit_key << "]  Quit\n";
}

static void print_day_results(const std::vector<DayResult>& results) {
    for (const auto& r : results) {
        if (r.season_changed)
            std::cout << "  >> Season changed to: " << SEASON_NAMES[r.season_new] << "\n";
        if (r.born) std::cout << "  + Born: " << r.born << "\n";
        if (r.died) std::cout << "  - Died: " << r.died << "\n";
        if (r.people_arrived) std::cout << "  + People arrived: " << r.people_arrived << "\n";
        if (r.base_lost) std::cout << "  ! Base lost: " << r.base_lost << "\n";
        if (r.home_overflow) std::cout << "  ! Home overflow\n";
        for (const auto& ev : r.events) {
            std::cout << "  EVENT: " << ev.first << "\n";
            if (!ev.second.empty()) std::cout << "    " << ev.second << "\n";
        }
    }
}

static void print_bases(const ColonyEnvCpp& env) {
    const Game& g = env.game();
    std::cout << "\nBASES (" << g.bases.size() << "):\n";
    for (const auto& b : g.bases) {
        std::cout << "  " << b.data->caption << " [" << b.x << "," << b.y << "]";
        if (b.build_days > 0) std::cout << "  (building: " << b.build_days << "d left)";
        if (b.preserved) std::cout << "  (preserved)";
        if (b.data->live_years && b.live_time > 0)
            std::cout << "  (" << b.live_time << "/" << b.data->live_time_total() << " days)";
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    int64_t seed = 42;
    int map_size = 280;
    Curriculum curriculum;  // default: everything allowed

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) seed = std::stoll(argv[++i]);
        else if (arg == "--map-size" && i + 1 < argc) map_size = std::stoi(argv[++i]);
        else if (arg == "--curriculum" && i + 1 < argc) {
            try {
                curriculum = Curriculum::from_json(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "ERROR: " << e.what() << "\n";
                return 1;
            }
        }
        else if (arg == "--curriculum-all") curriculum = Curriculum();
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Sakhalin Colony\n"
                      << "Usage: sakhalin_colony.exe [--seed N] [--map-size N]\n"
                      << "                           [--curriculum JSON | --curriculum-all]\n"
                      << "  JSON: {\"all_builds\":bool,\"allowed_builds\":[...],\"stage\":int}\n";
            return 0;
        }
    }

    std::cout << "\x1b[?25l";  // hide cursor
    print_header();
    std::cout << "Loading data...\n";

    std::string base_path = "configs/bases.json";
    std::string events_path = "configs/events.json";

    auto bases_data = load_base_data(base_path);
    auto events_data = load_events(events_path);

    std::cout << "  bases: " << bases_data.size() << " types\n";
    std::cout << "  events: " << events_data.size() << " types\n";

    // P0 (2026-09-17): у консоли нет диалога налогов (в отличие от GUI) —
    // берём долговую политику: время идёт всегда, неуплаченный остаток → credit.
    ColonyEnvCpp env(bases_data, events_data, seed, map_size, curriculum,
                     RewardConfig(), "normal", false, GAME_OVER_NO_PEOPLE_DAYS,
                     /*tax_to_debt=*/true);

    std::cout << "  env created: obs_size=" << env.obs_size()
              << " n_actions=" << env.n_actions()
              << " n_build=" << env.n_build() << "\n\n";

    env.reset(seed);
    print_status(env);

    int max_key = 2 + env.n_build() + N_MANAGERS + 1;
    std::vector<int> action_map;
    action_map.push_back(A_DAY);
    action_map.push_back(A_WEEK);
    for (int i = 0; i < env.n_build(); i++) action_map.push_back(A_BUILD0 + i);
    for (int i = 0; i < N_MANAGERS; i++) action_map.push_back(env.n_actions() - N_MANAGERS + i);

    while (true) {
        print_menu(env);
        std::cout << "\nChoose [1-" << max_key << "]: ";
        std::string line;
        if (!std::getline(std::cin, line)) break;

        int choice = 0;
        try { choice = std::stoi(line); } catch (...) {}

        if (choice < 1 || choice > max_key) {
            std::cout << "  Invalid choice.\n";
            continue;
        }
        if (choice == max_key) {
            std::cout << "  Quitting. Final stats:\n";
            print_status(env);
            std::cout << "  Total steps: " << env.steps() << "\n";
            std::cout << "  Episode return: " << env.ep_return() << "\n";
            std::cout << "\x1b[?25h";
            return 0;
        }

        int action = action_map[choice - 1];

        if (choice <= 2) {
            auto out = env.step(action);
            if (out.terminated) {
                std::cout << "\n  *** GAME OVER ***\n";
                auto ov = env.game().game_over();
                if (ov) std::cout << "  Reason: " << ov->reason << " — " << ov->message << "\n";
            }
        } else if (choice <= 2 + env.n_build()) {
            int build_idx = choice - 3;
            const BaseData& d = *env.build_data()[build_idx];
            std::cout << "  Building: " << d.caption << " $" << thousands(d.price) << "\n";
            auto out = env.step(action);
            if (out.terminated) {
                std::cout << "\n  *** GAME OVER ***\n";
                auto ov = env.game().game_over();
                if (ov) std::cout << "  Reason: " << ov->reason << " — " << ov->message << "\n";
            }
        } else {
            int mgr_idx = choice - 3 - env.n_build();
            const char* mgr_names[] = {
                "Improve earth", "Repair worst", "Repair all",
                "Demolish worst", "Preserve oldest", "Unpreserve oldest",
                "Sell surplus", "Buy food", "Take loan (50k)",
                "Repay loan (50k)", "Pay tax (manual)",
            };
            std::cout << "  " << mgr_names[mgr_idx] << "\n";
            auto out = env.step(action);
            if (out.terminated) {
                std::cout << "\n  *** GAME OVER ***\n";
                auto ov = env.game().game_over();
                if (ov) std::cout << "  Reason: " << ov->reason << " — " << ov->message << "\n";
            }
        }

        std::cout << "  reward=" << env.last_reward()
                  << "  ep_return=" << env.ep_return() << "\n";
        print_status(env);

        if (env.steps() >= MAX_STEPS) {
            std::cout << "\n  Max steps reached. Game over.\n";
            std::cout << "\x1b[?25h";
            return 0;
        }
    }

    std::cout << "\x1b[?25h";
    return 0;
}
