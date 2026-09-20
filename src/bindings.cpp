#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "colony/bases.h"
#include "colony/data.h"
#include "colony/env.h"
#include "colony/running_mean_std.h"
#include "colony/events.h"
#include "colony/game.h"
#include "colony/resources.h"
#include "colony/rng.h"

namespace py = pybind11;
using namespace colony;

// PR 3 (handshake): расширение сообщает свою версию, чтобы Python мог отклонить
// протухший бинарь вместо молчаливого старого поведения.
// COLONY_GIT_SHA подставляет CMake; для ручных сборок — "unknown".
#ifndef COLONY_GIT_SHA
#define COLONY_GIT_SHA "unknown"
#endif
#ifndef COLONY_EXTENSION_VERSION
#define COLONY_EXTENSION_VERSION 4
#endif

namespace {

py::dict day_result_to_dict(const DayResult& r) {
    py::dict d;
    d["season_changed"] = r.season_changed;
    d["season_new"] = std::string(SEASON_NAMES[r.season_new]);
    d["base_lost"] = r.base_lost;
    d["born"] = r.born;
    d["died"] = r.died;
    d["home_overflow"] = r.home_overflow;
    d["people_arrived"] = r.people_arrived;
    d["reminder_may"] = r.reminder_may;
    d["stop_week"] = (bool)r.stop_week;
    py::list evs;
    for (const auto& e : r.events) {
        py::list item;
        item.append(e.first);
        item.append(e.second);
        evs.append(item);
    }
    d["events"] = evs;
    return d;
}

py::list sunduk_to_list(const Sunduk& s) {
    py::list l;
    for (int i = 0; i < SUNDUK_SIZE; i++) l.append(s[i]);
    return l;
}

Sunduk sunduk_from_list(const py::list& l) {
    if ((int)l.size() != SUNDUK_SIZE)
        throw std::runtime_error("Sunduk list must contain exactly 9 items");
    Sunduk s;
    for (int i = 0; i < SUNDUK_SIZE; i++) s[i] = l[i].cast<int64_t>();
    return s;
}

py::dict base_to_dict(const Base& b) {
    py::dict d;
    d["id"] = b.data->id;
    d["x"] = b.x;
    d["y"] = b.y;
    d["live_time"] = b.live_time;
    d["build_days"] = b.build_days;
    d["preserved"] = b.preserved;
    d["need_sunduk"] = b.need_sunduk;
    d["need_workers"] = b.need_workers;
    return d;
}

// PR 1: транспортный dict курикулума Python <-> C++:
//   {"all_builds": bool, "allowed_builds": [str], "stage": int}
// Неизвестные ключи игнорируются (forward compat для PR 4/5).
Curriculum curriculum_from_dict(const py::dict& d) {
    Curriculum c;
    if (d.contains("all_builds")) c.all_builds = d["all_builds"].cast<bool>();
    if (d.contains("allowed_builds")) {
        c.allowed_builds.clear();
        for (const auto& v : d["allowed_builds"]) c.allowed_builds.insert(v.cast<std::string>());
    }
    if (d.contains("stage")) c.stage_report = d["stage"].cast<int>();
    // PR 4: ресурсные веса (ровно SUNDUK_SIZE чисел, иначе fail-fast).
    if (d.contains("all_resources")) c.all_resources = d["all_resources"].cast<bool>();
    if (d.contains("resource_weights")) {
        std::vector<double> w;
        for (const auto& v : d["resource_weights"]) w.push_back(v.cast<double>());
        if ((int)w.size() != SUNDUK_SIZE)
            throw std::runtime_error("curriculum dict: resource_weights must hold 9 numbers");
        for (int j = 0; j < SUNDUK_SIZE; j++) c.resource_weights[(size_t)j] = w[(size_t)j];
    }
    // PR 5: obs layout version (absent = 0, legacy layout).
    if (d.contains("obs_version")) c.obs_version = d["obs_version"].cast<int>();
    // Mechanic allow-list: absent keeps legacy/all-enabled behaviour.
    if (d.contains("enabled_mechanics")) {
        c.enabled_mechanics = {false, false, false};
        for (const auto& v : d["enabled_mechanics"]) {
            const std::string name = v.cast<std::string>();
            if (name == "all") c.enabled_mechanics = {true, true, true};
            else if (name == "improve_land") c.enabled_mechanics[0] = true;
            else if (name == "preservation") c.enabled_mechanics[1] = true;
            else if (name == "credit") c.enabled_mechanics[2] = true;
            else throw std::runtime_error("curriculum dict: unknown mechanic " + name);
        }
    }
    return c;
}

py::dict curriculum_to_dict(const Curriculum& c) {
    py::dict d;
    d["all_builds"] = c.all_builds;
    std::vector<std::string> ids(c.allowed_builds.begin(), c.allowed_builds.end());
    std::sort(ids.begin(), ids.end());
    d["allowed_builds"] = ids;
    d["stage"] = c.stage_report;
    d["all_resources"] = c.all_resources;
    std::vector<double> w(c.resource_weights.begin(), c.resource_weights.end());
    d["resource_weights"] = w;
    d["obs_version"] = c.obs_version;
    std::vector<std::string> mechanics;
    if (c.enabled_mechanics[0]) mechanics.emplace_back("improve_land");
    if (c.enabled_mechanics[1]) mechanics.emplace_back("preservation");
    if (c.enabled_mechanics[2]) mechanics.emplace_back("credit");
    d["enabled_mechanics"] = mechanics;
    return d;
}

py::dict env_step_to_dict(const ColonyEnvCpp::StepOut& s) {
    py::dict d;
    d["obs"] = s.obs;
    d["reward"] = s.rew;
    d["terminated"] = s.terminated;
    d["truncated"] = s.truncated;
    d["days"] = s.days;
    d["people"] = s.people;
    d["money"] = s.money;
    d["bases"] = s.n_bases;
    d["seed"] = s.seed;
    d["tax_due_days"] = s.tax_due_days;
    d["tax_grace_expired"] = s.tax_grace_expired;
    d["ep_return"] = s.ep_return;
    d["steps"] = s.steps;
    d["tax_borrowed"] = s.tax_borrowed;
    d["metrics"] = s.metrics;
    return d;
}

}  // namespace

PYBIND11_MODULE(colony_cpp, m) {
    m.doc() = "C++ ядро «Сахалинская колония» (бит-в-бит порт)";

    // ---------------- handshake (PR 3) ----------------
    // Python вызывает extension_info() при старте (см. python/colony_cpp_api.py)
    // и отказывается работать с бинарём без нужных фич. При добавлении фич,
    // от которых зависит Python, — расширить список и поднять версию.
    m.def("extension_info", [] {
        py::dict d;
        d["version"] = COLONY_EXTENSION_VERSION;
        d["features"] = std::vector<std::string>{
            "set_curriculum",      // set_curriculum() + curriculum() обеих сред (PR 1)
            "curriculum",
            "resource_curriculum", // веса ресурсов + priority_reached (PR 4)
            "minimap",             // minimap()/minimap_batch()/set_minimap_radius()
            "action_masks_batch",  // ColonyVecEnvCpp.action_masks_batch()
            "obs_v2",              // P0: obs v2 = 299-dim (+dx/dy ближайших ресурсов)
            "tax_to_debt",         // P0: налог → долг, календарь не замирает
            "mechanic_curriculum", // fixed manager logits, monotonic allow-list
        };
        d["src_sha"] = COLONY_GIT_SHA;
        return d;
    });

    // ---------------- RNG ----------------
    py::class_<MtRandom>(m, "MtRandom")
        .def(py::init<int64_t>())
        .def("random", &MtRandom::random)
        .def("getrandbits", &MtRandom::getrandbits)
        .def("randrange", &MtRandom::randrange)
        .def("randint", &MtRandom::randint)
        .def("uniform", &MtRandom::uniform);

    py::class_<PCG64>(m, "PCG64")
        .def(py::init<uint64_t>())
        .def("next64", &PCG64::next64)
        .def("next32", &PCG64::next32)
        .def("next_double", &PCG64::next_double)
        .def("randrange", &PCG64::randrange)
        .def("randint", &PCG64::randint)
        .def("uniform", &PCG64::uniform)
        .def("binomial", &PCG64::binomial);

    py::class_<Earth>(m, "Earth")
        .def_readonly("init_sel_x", &Earth::init_sel_x)
        .def_readonly("init_sel_y", &Earth::init_sel_y)
        .def_property_readonly("size", &Earth::size)
        .def("lot", &Earth::lot)
        .def("sub", &Earth::sub)
        .def("in_bounds", &Earth::in_bounds);

    // ---------------- данные ----------------
    m.def("load_base_data", &load_base_data, py::arg("path"));
    m.def("load_events", &load_events, py::arg("path"));

    py::class_<BaseEvent>(m, "BaseEvent")
        .def_readonly("id", &BaseEvent::id)
        .def_readonly("target", &BaseEvent::target)
        .def_readonly("message", &BaseEvent::message)
        .def_readonly("live_years", &BaseEvent::live_years)
        .def_readonly("live_years_range", &BaseEvent::live_years_range)
        .def_readonly("people", &BaseEvent::people)
        .def_readonly("people_range", &BaseEvent::people_range);

    py::class_<BaseData>(m, "BaseData")
        .def_readonly("id", &BaseData::id)
        .def_readonly("caption", &BaseData::caption)
        .def_readonly("price", &BaseData::price)
        .def_readonly("build_time", &BaseData::build_time)
        .def_readonly("live_years", &BaseData::live_years)
        .def_readonly("home_places", &BaseData::home_places)
        .def_readonly("need_workers", &BaseData::need_workers)
        .def_readonly("need_earth", &BaseData::need_earth)
        .def_readonly("consume", &BaseData::consume)
        .def_readonly("profit", &BaseData::profit)
        .def_readonly("profit_range", &BaseData::profit_range)
        .def("restore_price_per_day", &BaseData::restore_price_per_day)
        .def("restore_price", &BaseData::restore_price);

    // ---------------- игра ----------------
    py::class_<Sunduk>(m, "Sunduk")
        .def(py::init<>())
        .def_property("items",
                      [](const Sunduk& s) { return sunduk_to_list(s); },
                      [](Sunduk& s, const py::list& l) { s = sunduk_from_list(l); })
        .def("add", [](Sunduk& s, const Sunduk& o) { s.add(o); })
        .def("remove", [](Sunduk& s, const Sunduk& o) { s.remove(o); })
        .def("clear", &Sunduk::clear)
        .def("empty", &Sunduk::empty);

    py::class_<Game>(m, "Game")
        .def(py::init<const std::vector<BaseData>&, const std::vector<BaseEvent>&,
                      int64_t, int, const std::string&, bool, int64_t>(),
             py::arg("base_data"), py::arg("events_data"), py::arg("seed"),
             py::arg("map_size") = 280, py::arg("difficulty") = "normal",
             py::arg("no_city_game_over") = true, py::arg("no_people_days") = 365)
        .def_property("year", [](const Game& g) { return g.year; },
                      [](Game& g, int v) { g.year = v; })
        .def_property("month", [](const Game& g) { return g.month; },
                      [](Game& g, int v) { g.month = v; })
        .def_property("day", [](const Game& g) { return g.day; },
                      [](Game& g, int v) { g.day = v; })
        .def_property("season",
                      [](const Game& g) { return std::string(SEASON_NAMES[g.season]); },
                      [](Game& g, const std::string& s) {
                          for (int i = 0; i < 4; i++)
                              if (s == SEASON_NAMES[i]) g.season = (Season)i;
                      })
        .def_property("money", [](const Game& g) { return g.money; },
                      [](Game& g, int64_t v) { g.money = v; })
        .def_property("credit", [](const Game& g) { return g.credit; },
                      [](Game& g, int64_t v) { g.credit = v; })
        .def_property("people", [](const Game& g) { return g.people; },
                      [](Game& g, int64_t v) { g.people = v; })
        .def_property("busy_people", [](const Game& g) { return g.busy_people; },
                      [](Game& g, int64_t v) { g.busy_people = v; })
        .def_property("days_alive", [](const Game& g) { return g.days_alive; },
                      [](Game& g, int64_t v) { g.days_alive = v; })
        .def_property("sunduk",
                      [](Game& g) { return sunduk_to_list(g.sunduk); },
                      [](Game& g, const py::list& l) { g.sunduk = sunduk_from_list(l); })
        .def("map_size", &Game::map_size)
        .def("light", &Game::light)
        .def("depot_exists", &Game::depot_exists)
        .def("max_credit", &Game::max_credit)
        .def("now_home_places", &Game::now_home_places)
        .def("now_need_workers", &Game::now_need_workers)
        .def("free_people", &Game::free_people)
        .def_property_readonly("init_sel_x", [](const Game& g) { return g.earth.init_sel_x; })
        .def_property_readonly("init_sel_y", [](const Game& g) { return g.earth.init_sel_y; })
        .def_property_readonly("earth", [](const Game& g) -> const Earth& { return g.earth; })
        .def("annual_tax_due", &Game::annual_tax_due)
        .def("main_tax_due", &Game::main_tax_due)
        .def("check_advance", [](const Game& g) {
            auto r = g.check_advance();
            // py::object cast: py::none and py::str have no common type, so a
            // bare ternary does not compile on some toolchains (MSVC/GCC).
            py::object reason = r.second.empty()
                                    ? py::object(py::none())
                                    : py::object(py::str(r.second));
            return py::make_tuple(r.first, reason);
        })
        .def("annual_tax_amount", &Game::annual_tax_amount)
        .def("main_tax_amount", &Game::main_tax_amount)
        .def("pay_annual_tax", &Game::pay_annual_tax)
        .def("pay_main_tax", &Game::pay_main_tax)
        .def_property("tax_postponed",
                      [](const Game& g) { return g.tax_postponed(); },
                      [](Game& g, bool v) { g.set_tax_postponed(v); })
        .def("new_day", [](Game& g) { return day_result_to_dict(g.new_day()); })
        .def("advance_day", [](Game& g) -> py::object {
            auto r = g.advance_day();
            if (r.ok) return day_result_to_dict(r.res);
            return py::str(r.blocker);
        })
        .def("advance_week", [](Game& g) {
            py::list out;
            for (const DayResult& r : g.advance_week()) out.append(day_result_to_dict(r));
            return out;
        })
        .def("build", [](Game& g, const std::string& id, int x, int y) {
            auto r = g.build(id, x, y);
            return py::make_tuple(r.first, r.second);
        })
        .def("restore", [](Game& g, int x, int y) {
            auto r = g.restore(x, y);
            return py::make_tuple(r.ok, r.msg, r.price, r.days);
        })
        .def("restore_all", [](Game& g) {
            auto r = g.restore_all();
            return py::make_tuple(r.ok, r.msg, r.price, r.days);
        })
        .def("destroy", [](Game& g, int x, int y) {
            auto r = g.destroy(x, y);
            return py::make_tuple(r.first, r.second);
        })
        .def("preserve", [](Game& g, int x, int y) {
            auto r = g.preserve(x, y);
            return py::make_tuple(r.first, r.second);
        })
        .def("good_earth", [](Game& g, int x, int y) {
            auto r = g.good_earth(x, y);
            return py::make_tuple(r.first, r.second);
        })
        .def("market_buy", [](Game& g, const py::list& counts) {
            auto r = g.market_buy(sunduk_from_list(counts));
            return py::make_tuple(r.ok, r.msg, r.total);
        })
        .def("market_sell", [](Game& g, const py::list& counts) {
            auto r = g.market_sell(sunduk_from_list(counts));
            return py::make_tuple(r.ok, r.msg, r.total);
        })
        .def("bank_take", [](Game& g, int64_t v) {
            auto r = g.bank_take(v);
            return py::make_tuple(r.first, r.second);
        })
        .def("bank_give", [](Game& g, int64_t v) {
            auto r = g.bank_give(v);
            return py::make_tuple(r.first, r.second);
        })
        .def("save_undo", &Game::save_undo)
        .def("undo", &Game::undo)
        .def("snapshot", [](const Game& g) { return g.snapshot(); })
        .def("game_over", [](const Game& g) -> py::object {
            auto ov = g.game_over();
            if (!ov) return py::none();
            py::dict d;
            d["reason"] = ov->reason;
            d["days"] = ov->days;
            d["message"] = ov->message;
            return d;
        })
        .def("bases", [](const Game& g) {
            py::list out;
            for (const Base& b : g.bases) out.append(base_to_dict(b));
            return out;
        })
        .def("is_good", &Game::is_good)
        .def("earth_lots", [](const Game& g) {
            py::list out;
            const auto& lots = g.earth.lots();
            for (int8_t v : lots) out.append((int)v);
            return out;
        })
        .def("earth_subtypes", [](const Game& g) {
            py::list out;
            const auto& st = g.earth.subtype();
            for (int8_t v : st) out.append((int)v);
            return out;
        })
        .def("good_lots", [](const Game& g) {
            py::list out;
            for (uint8_t v : g.good_lots) out.append((int)v);
            return out;
        })
        .def("destroyed_lots", [](const Game& g) {
            py::list out;
            for (int8_t v : g.destroyed_lots) out.append((int)v);
            return out;
        });

    // ---------------- среда ----------------
    py::class_<RewardConfig>(m, "RewardConfig")
        .def(py::init<>())
        .def_readwrite("build_bonus", &RewardConfig::build_bonus)
        .def_readwrite("chain_bonus", &RewardConfig::chain_bonus)
        .def_readwrite("chain_daily", &RewardConfig::chain_daily)
        .def_readwrite("first_extraction_bonus", &RewardConfig::first_extraction_bonus)
        .def_readwrite("extraction_daily", &RewardConfig::extraction_daily)
        .def_readwrite("need_fill_bonus", &RewardConfig::need_fill_bonus)
        .def_readwrite("loan_penalty", &RewardConfig::loan_penalty)
        .def_readwrite("novelty", &RewardConfig::novelty)
        .def_readwrite("daily_income", &RewardConfig::daily_income)
        .def_readwrite("sale_bonus", &RewardConfig::sale_bonus)
        .def_readwrite("tax_daily_bonus", &RewardConfig::tax_daily_bonus)
        .def_readwrite("survival_bonus", &RewardConfig::survival_bonus)
        .def_readwrite("game_over_penalty", &RewardConfig::game_over_penalty)
        .def_readwrite("diversity_bonus", &RewardConfig::diversity_bonus)
        .def_readwrite("error_penalty", &RewardConfig::error_penalty)
        .def_readwrite("preserve_penalty", &RewardConfig::preserve_penalty)
        .def_readwrite("demolish_penalty", &RewardConfig::demolish_penalty)
        .def_readwrite("manual_tax_penalty", &RewardConfig::manual_tax_penalty)
        .def_readwrite("build_cost_penalty", &RewardConfig::build_cost_penalty)
        .def_readwrite("idle_build_penalty", &RewardConfig::idle_build_penalty)
        .def_readwrite("idle_build_threshold_days", &RewardConfig::idle_build_threshold_days)
        .def_readwrite("survival_coeff", &RewardConfig::survival_coeff)
        .def_readwrite("milestone_base_bonus", &RewardConfig::milestone_base_bonus)
        .def_readwrite("milestone_people_bonus", &RewardConfig::milestone_people_bonus)
        .def_readwrite("milestone_day_bonus", &RewardConfig::milestone_day_bonus)
        .def_readwrite("milestone_year_bonus", &RewardConfig::milestone_year_bonus)
        .def_readwrite("proximity_bonus", &RewardConfig::proximity_bonus)
        .def_readwrite("clip_reward_min", &RewardConfig::clip_reward_min)
        .def_readwrite("clip_reward_max", &RewardConfig::clip_reward_max)
        .def_readwrite("disable_net_worth", &RewardConfig::disable_net_worth)
        .def_readwrite("disable_daily_income", &RewardConfig::disable_daily_income)
        .def_readwrite("disable_provider_bonus", &RewardConfig::disable_provider_bonus)
        .def_readwrite("priority_count_over_allowed", &RewardConfig::priority_count_over_allowed)
        .def_readwrite("obs_mask_locked_catalog", &RewardConfig::obs_mask_locked_catalog)
        // formerly-hardcoded weights (env.cpp) — exposed so Python/UI can tune them
        .def_readwrite("tax_fail_penalty", &RewardConfig::tax_fail_penalty)
        .def_readwrite("death_penalty", &RewardConfig::death_penalty)
        .def_readwrite("base_lost_penalty", &RewardConfig::base_lost_penalty)
        .def_readwrite("born_bonus", &RewardConfig::born_bonus)
        .def_readwrite("debt_coeff", &RewardConfig::debt_coeff)
        .def_readwrite("home_overflow_penalty", &RewardConfig::home_overflow_penalty)
        .def_readwrite("housing_need_bonus", &RewardConfig::housing_need_bonus)
        .def_readwrite("food_need_bonus", &RewardConfig::food_need_bonus)
        .def_readwrite("water_need_bonus", &RewardConfig::water_need_bonus)
        .def_readwrite("buy_food_penalty", &RewardConfig::buy_food_penalty)
        // P0/P1 (2026-09-17)
        .def_readwrite("tax_debt_penalty", &RewardConfig::tax_debt_penalty)
        .def_readwrite("mask_managers_by_applicability",
                       &RewardConfig::mask_managers_by_applicability)
        // v4 (2026-09)
        .def_readwrite("goal_survival_coeff", &RewardConfig::goal_survival_coeff)
        .def_readwrite("main_tax_cash_bonus", &RewardConfig::main_tax_cash_bonus)
        .def_readwrite("main_tax_pressure_coeff", &RewardConfig::main_tax_pressure_coeff)
        // P2-9 (2026-09-19): road-shaping к воде — раньше хардкод в step()
        .def_readwrite("road_shaping_cap", &RewardConfig::road_shaping_cap)
        .def_readwrite("road_shaping_per_cell", &RewardConfig::road_shaping_per_cell)
        .def_readwrite("water_reach_bonus", &RewardConfig::water_reach_bonus)
        .def_readwrite("water_reach_radius", &RewardConfig::water_reach_radius)
        .def_readwrite("road_no_progress_penalty", &RewardConfig::road_no_progress_penalty)
        .def_readwrite("road_progress_epsilon", &RewardConfig::road_progress_epsilon);

    py::class_<ColonyEnvCpp::EpisodeMetrics>(m, "EpisodeMetrics")
        .def_readonly("total_reward", &ColonyEnvCpp::EpisodeMetrics::total_reward)
        .def_readonly("days_survived", &ColonyEnvCpp::EpisodeMetrics::days_survived)
        .def_readonly("total_builds", &ColonyEnvCpp::EpisodeMetrics::total_builds)
        .def_readonly("unique_build_types", &ColonyEnvCpp::EpisodeMetrics::unique_build_types)
        .def_readonly("chains_activated", &ColonyEnvCpp::EpisodeMetrics::chains_activated)
        .def_readonly("max_chain_depth", &ColonyEnvCpp::EpisodeMetrics::max_chain_depth)
        .def_readonly("reached_resources", &ColonyEnvCpp::EpisodeMetrics::reached_resources)
        .def_readonly("priority_reached", &ColonyEnvCpp::EpisodeMetrics::priority_reached)
        .def_readonly("deaths", &ColonyEnvCpp::EpisodeMetrics::deaths)
        .def_readonly("births", &ColonyEnvCpp::EpisodeMetrics::births)
        .def_readonly("base_count_peak", &ColonyEnvCpp::EpisodeMetrics::base_count_peak)
        .def_readonly("net_worth", &ColonyEnvCpp::EpisodeMetrics::net_worth)
        .def_readonly("population_peak", &ColonyEnvCpp::EpisodeMetrics::population_peak);

    py::class_<ColonyEnvCpp>(m, "ColonyEnvCpp")
        .def(py::init([](const std::vector<BaseData>& base_data,
                         const std::vector<BaseEvent>& events_data,
                         int64_t seed, int map_size, const py::dict& curriculum,
                         const RewardConfig& reward, const std::string& difficulty,
                         bool no_city_game_over, int64_t no_people_days,
                         bool tax_to_debt) {
                 return new ColonyEnvCpp(base_data, events_data, seed, map_size,
                                         curriculum_from_dict(curriculum), reward,
                                         difficulty, no_city_game_over, no_people_days,
                                         tax_to_debt);
             }),
             py::arg("base_data"), py::arg("events_data"), py::arg("seed"),
             py::arg("map_size") = 280, py::arg("curriculum") = py::dict(),
             py::arg("reward") = RewardConfig(),
             py::arg("difficulty") = "normal",
             py::arg("no_city_game_over") = false,
             py::arg("no_people_days") = 365,
             // P0: RL-дефолт — календарь не замирает на налоге, остаток → долг.
             py::arg("tax_to_debt") = true)
        .def("reset", &ColonyEnvCpp::reset)
        .def("set_step_log", &ColonyEnvCpp::set_step_log, py::arg("path"))
        .def("dump_obs", &ColonyEnvCpp::dump_obs)
        .def("action_mask", [](ColonyEnvCpp& env) { return env.action_mask(); })
        .def("step", [](ColonyEnvCpp& env, int action) {
            return env_step_to_dict(env.step(action));
        })
        .def("obs", [](ColonyEnvCpp& env) { return env.obs(); })
        .def("minimap", [](const ColonyEnvCpp& env) {
            std::vector<float> mm = env.minimap();
            py::array_t<float> arr({8, 32, 32});
            std::memcpy(arr.mutable_data(), mm.data(), mm.size() * sizeof(float));
            return arr;
        })
        .def("minimap_radius", &ColonyEnvCpp::minimap_radius)
        // P2-8: setter оставлен только ради совместимости — миникарта глобальная
        // 32x32 (ColonyEnvCpp::minimap_grid_size()), значение ни на что не влияет.
        .def("set_minimap_radius", [](ColonyEnvCpp& env, int r) {
            if (PyErr_WarnEx(PyExc_DeprecationWarning,
                             "set_minimap_radius() is a no-op: the minimap is a global "
                             "32x32 grid (minimap_grid_size()); the value is stored for "
                             "compatibility only and does not change the observation.",
                             1) < 0)
                throw py::error_already_set();
            env.set_minimap_radius(r);
        })
        .def("n_build", &ColonyEnvCpp::n_build)
        .def("tax_to_debt", &ColonyEnvCpp::tax_to_debt)
        .def("last_tax_borrowed", &ColonyEnvCpp::last_tax_borrowed)
        .def("n_bases", &ColonyEnvCpp::n_bases)
        .def("n_actions", &ColonyEnvCpp::n_actions)
        .def("obs_size", &ColonyEnvCpp::obs_size)
        .def("build_ids", &ColonyEnvCpp::build_ids)
        .def("last_daily_value", &ColonyEnvCpp::last_daily_value)
        .def("last_chain_daily", &ColonyEnvCpp::last_chain_daily)
        .def("steps", &ColonyEnvCpp::steps)
        .def("ep_return", &ColonyEnvCpp::ep_return)
        .def("tax_grace_expired", &ColonyEnvCpp::tax_grace_expired)
        .def("tax_grace_days", &ColonyEnvCpp::tax_grace_days)
        .def("game", [](ColonyEnvCpp& env) -> Game& { return env.game(); }, py::return_value_policy::reference)
        .def("debug_find_lot", [](ColonyEnvCpp& env, int need_earth, bool no_near_base) -> py::object {
            auto cell = env.find_lot(need_earth, no_near_base);
            if (cell) return py::make_tuple(cell->first, cell->second);
            return py::none();
        })
        .def("debug_lot_ok", [](ColonyEnvCpp& env, int x, int y, int need_earth, bool no_near_base) {
            return env.lot_ok(x, y, need_earth, no_near_base);
        })
        .def("debug_build_data", [](const ColonyEnvCpp& env) {
            py::list out;
            for (const auto* d : env.build_data()) {
                out.append(d->id);
            }
            return out;
        })
        .def("debug_net_worth", &ColonyEnvCpp::debug_net_worth)
        .def("stats", [](const ColonyEnvCpp& env) {
            auto s = env.stats();
            py::dict d;
            d["days"] = s.days;
            d["people"] = s.people;
            d["bases"] = s.bases;
            d["money"] = s.money;
            return d;
        })
        .def("set_curriculum", [](ColonyEnvCpp& env, const py::dict& d) {
            env.set_curriculum(curriculum_from_dict(d));
        }, py::arg("curriculum"))
        .def("curriculum", [](const ColonyEnvCpp& env) { return curriculum_to_dict(env.curriculum()); })
        .def("build_allowed", &ColonyEnvCpp::build_allowed, py::arg("id"))
        .def("set_rewards", &ColonyEnvCpp::set_rewards, py::arg("cfg"))
        .def("reward_config", &ColonyEnvCpp::reward_config);

    // ---------------- RunningMeanStd ----------------
    py::class_<RunningMeanStd>(m, "RunningMeanStd")
        .def(py::init<int>(), py::arg("size"))
        .def(py::init<>())
        .def("size", &RunningMeanStd::size)
        .def("count", &RunningMeanStd::count)
        .def("mean", [](const RunningMeanStd& r) {
            return std::vector<double>(r.mean(), r.mean() + r.size());
        })
        .def("var", [](const RunningMeanStd& r) {
            return std::vector<double>(r.var(), r.var() + r.size());
        })
        .def("update", [](RunningMeanStd& r, py::array_t<float> data, int n, int stride) {
            r.update(data.data(), n, stride);
        }, py::arg("data"), py::arg("n_rows"), py::arg("row_stride"))
        .def("update_scalar", &RunningMeanStd::update_scalar, py::arg("val"))
        .def("normalize", [](RunningMeanStd& r, py::array_t<float> obs, int n, int stride, double clip) {
            r.normalize(obs.mutable_data(), n, stride, clip);
        }, py::arg("obs"), py::arg("n_rows"), py::arg("row_stride"), py::arg("clip_val") = 10.0)
        .def("set_mean", &RunningMeanStd::set_mean, py::arg("m"))
        .def("set_var", &RunningMeanStd::set_var, py::arg("v"))
        .def("set_count", &RunningMeanStd::set_count, py::arg("c"));

    // ---------------- batched vectorized environment ----------------
    py::class_<StepBatchResult>(m, "StepBatchResult")
        .def_readonly("obs", &StepBatchResult::obs)
        .def_readonly("rewards", &StepBatchResult::rewards)
        .def_readonly("terminateds", &StepBatchResult::terminateds)
        .def_readonly("trunceds", &StepBatchResult::trunceds)
        .def_readonly("infos", &StepBatchResult::infos);

    py::class_<ColonyVecEnvCpp>(m, "ColonyVecEnvCpp")
        .def(py::init([](const std::vector<BaseData>& base_data,
                         const std::vector<BaseEvent>& events_data,
                         int n_envs, int64_t base_seed, int map_size,
                         const py::dict& curriculum, const RewardConfig& reward,
                         int n_threads, const std::string& difficulty,
                         bool tax_to_debt) {
                 return new ColonyVecEnvCpp(base_data, events_data, n_envs, base_seed,
                                            map_size, curriculum_from_dict(curriculum),
                                            reward, n_threads, difficulty, tax_to_debt);
             }),
             py::arg("base_data"), py::arg("events_data"),
             py::arg("n_envs"), py::arg("base_seed"),
             py::arg("map_size") = 280,
             py::arg("curriculum") = py::dict(),
             py::arg("reward") = RewardConfig(),
             py::arg("n_threads") = 0,
             py::arg("difficulty") = "normal",
             py::arg("tax_to_debt") = true)
        .def("reset_batch", [](ColonyVecEnvCpp& v, const std::vector<int64_t>& seeds) {
            v.reset_batch(seeds);
        }, py::arg("seeds"), py::call_guard<py::gil_scoped_release>())
        .def("step_async_batch", [](ColonyVecEnvCpp& v, const std::vector<int>& actions) {
            v.step_async_batch(actions);
        }, py::arg("actions"), py::call_guard<py::gil_scoped_release>())
        .def("step_wait_batch", [](ColonyVecEnvCpp& v) {
            return v.step_wait_batch();
        }, py::call_guard<py::gil_scoped_release>())
        .def("save_normalization", &ColonyVecEnvCpp::save_normalization, py::arg("path"))
        .def("load_normalization", &ColonyVecEnvCpp::load_normalization, py::arg("path"))
        .def("n_envs", &ColonyVecEnvCpp::n_envs)
        .def("obs_size", &ColonyVecEnvCpp::obs_size)
        .def("n_actions", &ColonyVecEnvCpp::n_actions)
        .def("n_build", &ColonyVecEnvCpp::n_build)
        .def("build_ids", &ColonyVecEnvCpp::build_ids)
        .def("n_bases", &ColonyVecEnvCpp::n_bases)
        .def("n_people", &ColonyVecEnvCpp::n_people)
        .def("total_daily_value", &ColonyVecEnvCpp::total_daily_value)
        .def("total_chain_daily", &ColonyVecEnvCpp::total_chain_daily)
        .def("mean_net_worth", &ColonyVecEnvCpp::mean_net_worth)
        .def("set_curriculum", [](ColonyVecEnvCpp& v, const py::dict& d) {
            v.set_curriculum(curriculum_from_dict(d));
        }, py::arg("curriculum"))
        .def("curriculum", [](const ColonyVecEnvCpp& v) { return curriculum_to_dict(v.curriculum()); })
        .def("set_rewards", &ColonyVecEnvCpp::set_rewards, py::arg("cfg"))
        .def("set_step_log", &ColonyVecEnvCpp::set_step_log, py::arg("env_idx"), py::arg("path"))
        .def("clear_step_log", &ColonyVecEnvCpp::clear_step_log, py::arg("env_idx"))
        .def("dump_obs", &ColonyVecEnvCpp::dump_obs, py::arg("env_idx"))
        .def("minimap_batch", [](const ColonyVecEnvCpp& v) {
            std::vector<float> mm = v.minimap_batch();
            py::array_t<float> arr({(int)v.n_envs(), 8, 32, 32});
            std::memcpy(arr.mutable_data(), mm.data(), mm.size() * sizeof(float));
            return arr;
        })
        .def("action_masks_batch", [](ColonyVecEnvCpp& v) {
            std::vector<float> masks = v.action_masks_batch();
            py::array_t<float> arr({(int)v.n_envs(), v.n_actions()});
            std::memcpy(arr.mutable_data(), masks.data(), masks.size() * sizeof(float));
            return arr;
        })
        .def("minimap_radius", &ColonyVecEnvCpp::minimap_radius)
        // P2-8: то же, что и для одиночной среды — no-op с DeprecationWarning.
        .def("set_minimap_radius", [](ColonyVecEnvCpp& v, int r) {
            if (PyErr_WarnEx(PyExc_DeprecationWarning,
                             "set_minimap_radius() is a no-op: the minimap is a global "
                             "32x32 grid; the value is stored for compatibility only.",
                             1) < 0)
                throw py::error_already_set();
            v.set_minimap_radius(r);
        })
        .def("obs_buffer", [](const ColonyVecEnvCpp& v) {
            const size_t n = (size_t)v.n_envs() * (size_t)v.obs_size();
            const float* src = v.obs_buffer();
            // Return a COPY: the underlying buffer is overwritten on the next
            // step_wait_batch, and sharing memory would silently corrupt
            // previously returned observations in Python.
            py::array_t<float> arr({(int)v.n_envs(), v.obs_size()});
            std::memcpy(arr.mutable_data(), src, n * sizeof(float));
            return arr;
        });

    // ---------------- отладка (временные хелперы) ----------------
    m.def("debug_earth_stages", [](int64_t seed, int size) {
        MtRandom rng((int64_t)seed);
        int coarse = std::max(8, size / 16);
        int n = size / coarse + 2;
        std::vector<double> grid((size_t)n * n);
        for (int i = 0; i < n * n; i++) grid[i] = rng.random();
        std::vector<double> xs(size);
        if (size > 1) {
            double step = (double)(n - 1) / (double)(size - 1);
            for (int i = 0; i < size; i++) xs[i] = (double)i * step;
        }
        std::vector<int> gx(size);
        std::vector<double> fx(size);
        for (int i = 0; i < size; i++) {
            gx[i] = (int)std::floor(xs[i]);
            if (gx[i] < 0) gx[i] = 0;
            if (gx[i] > n - 2) gx[i] = n - 2;
            fx[i] = xs[i] - (double)gx[i];
            fx[i] = fx[i] * fx[i] * (3 - 2 * fx[i]);
        }
        std::vector<double> noise((size_t)size * size);
        for (int y = 0; y < size; y++) {
            int gy = (size > 1) ? (int)((double)(y * (n - 1)) / (double)(size - 1)) : 0;
            if (gy < 0) gy = 0;
            if (gy > n - 2) gy = n - 2;
            double fy = (double)(y * (n - 1)) / (double)(size - 1) - (double)gy;
            fy = fy * fy * (3 - 2 * fy);
            double* row = &noise[(size_t)y * size];
            for (int x = 0; x < size; x++) {
                double a = grid[(size_t)gy * n + gx[x]] * (1 - fx[x]) +
                           grid[(size_t)gy * n + gx[x] + 1] * fx[x];
                double b = grid[(size_t)(gy + 1) * n + gx[x]] * (1 - fx[x]) +
                           grid[(size_t)(gy + 1) * n + gx[x] + 1] * fx[x];
                row[x] = a * (1 - fy) + b * fy;
            }
        }
        // стадии после шума (порядок RNG-вызовов идентичен earth.cpp)
        std::vector<double> score((size_t)size * size);
        std::vector<double> dist((size_t)size * size);
        double denom = size * 0.62;
        double cy = size / 2.0, cx = size / 2.0;
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) {
                double ddx = (double)x - cx, ddy = (double)y - cy;
                double d = std::sqrt(std::pow(ddx, 2.0) + std::pow(ddy, 2.0)) / denom;
                dist[(size_t)y * size + x] = d;
                score[(size_t)y * size + x] = noise[(size_t)y * size + x] * (1.0 - 0.55 * d);
            }
        double threshold = 0.30;
        std::vector<char> land((size_t)size * size);
        long long land_cnt = 0;
        for (int iter = 0; iter < 5; iter++) {
            land_cnt = 0;
            for (int i = 0; i < size * size; i++) {
                if (score[i] > threshold) { land[i] = 1; land_cnt++; } else land[i] = 0;
            }
            if ((double)land_cnt / (double)(size * size) >= 0.35) break;
            threshold -= 0.03;
        }
        py::dict d;
        py::list dl;
        for (double v : dist) dl.append(v);
        d["dist"] = dl;
        d["land_frac"] = (double)land_cnt / (double)(size * size);
        d["threshold"] = threshold;
        int lakes = (int)rng.randint(2, 5);
        py::list lakes_list;
        for (int i = 0; i < lakes; i++) {
            int px = (int)rng.randint(10, size - 11);
            int py = (int)rng.randint(10, size - 11);
            int r = (int)rng.randint(6, 16);
            py::list item; item.append(px); item.append(py); item.append(r);
            lakes_list.append(item);
        }
        d["lakes"] = lakes_list;
        return d;
    });
}