#include "colony/data.h"

#include <fstream>

#include <json.hpp>

namespace colony {

using nlohmann::json;

namespace {

Sunduk sunduk_from_json(const json& j) {
    static const char* names[SUNDUK_SIZE] = {
        "gold", "food", "coal", "iron", "oil", "stone", "water", "wood", "energy"};
    Sunduk s;
    if (j.is_null()) return s;
    for (int i = 0; i < SUNDUK_SIZE; i++) {
        if (j.contains(names[i])) s[i] = j[names[i]].get<int64_t>();
    }
    return s;
}

bool style_has(const json& j, const char* name) {
    if (!j.is_array()) return false;
    for (const auto& item : j)
        if (item.is_string() && item.get<std::string>() == name) return true;
    return false;
}

}  // namespace

std::vector<BaseData> load_base_data(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j = json::parse(f);
    std::vector<BaseData> out;
    out.reserve(j.size());
    for (const auto& d : j) {
        BaseData b;
        b.id = d.at("id").get<std::string>();
        b.caption = d.at("caption").get<std::string>();
        b.price = d.at("price").get<int64_t>();
        b.build_time = d.at("build_time").get<int64_t>();
        b.live_years = d.at("live_years").get<int64_t>();
        b.home_places = d.at("home_places").get<int64_t>();
        b.need_workers = d.at("need_workers").get<int64_t>();
        b.need_earth = d.value("need_earth", LT_EVERYWHERE);
        for (int s = 0; s < 4; s++) b.work_seasons[s] = false;
        if (d.contains("work_seasons") && d["work_seasons"].is_array()) {
            for (const auto& s : d["work_seasons"]) {
                std::string name = s.get<std::string>();
                for (int i = 0; i < 4; i++)
                    if (name == SEASON_NAMES[i]) b.work_seasons[i] = true;
            }
        }
        b.consume = sunduk_from_json(d.value("consume", json()));
        b.profit = sunduk_from_json(d.value("profit", json()));
        b.profit_range = sunduk_from_json(d.value("profit_range", json()));
        json style = d.value("style", json());
        b.no_near_base = style_has(style, "no_near_base");
        b.no_preserve = style_has(style, "no_preserve");
        b.plan = style_has(style, "plan");
        b.image_index = d.value("image_index", 0);
        out.push_back(b);
    }
    return out;
}

std::vector<BaseEvent> load_events(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    json j = json::parse(f);
    std::vector<BaseEvent> out;
    out.reserve(j.size());
    for (const auto& d : j) {
        BaseEvent e;
        e.id = d.at("id").get<std::string>();
        e.target = d.at("target").get<std::string>();
        e.message = d.at("message").get<std::string>();
        e.live_years = d.value("live_years", 0);
        e.live_years_range = d.value("live_years_range", 0);
        e.people = d.value("people", 0);
        e.people_range = d.value("people_range", 0);
        e.sunduk = sunduk_from_json(d.value("sunduk", json()));
        e.sunduk_range = sunduk_from_json(d.value("sunduk_range", json()));
        out.push_back(e);
    }
    return out;
}

}  // namespace colony