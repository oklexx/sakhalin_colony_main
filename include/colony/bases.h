#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "colony/constants.h"
#include "colony/resources.h"

namespace colony {

class Game;  // forward

// Статические параметры постройки (аналог TBaseData)
class BaseData {
public:
    std::string id;
    std::string caption;
    int64_t price = 0;
    int64_t build_time = 0;
    int64_t live_years = 0;
    int64_t home_places = 0;
    int64_t need_workers = 0;
    int need_earth = LT_EVERYWHERE;
    bool work_seasons[4] = {false, false, false, false};  // summer, autumn, winter, spring
    Sunduk consume;
    Sunduk profit;
    Sunduk profit_range;
    bool no_near_base = false;
    bool no_preserve = false;
    bool plan = false;
    int image_index = 0;

    int64_t live_time_total() const { return live_years * 364; }

    int64_t restore_price_per_day() const {
        if (!live_years) return 0;
        return (price * 75 / 100) / live_time_total();
    }

    int64_t restore_price(int64_t live_time_left) const {
        int64_t max_live = live_time_total();
        if (live_time_left >= max_live || !live_years) return 0;
        return (max_live - live_time_left) * restore_price_per_day();
    }

    bool season_works(Season s) const { return work_seasons[s]; }
};

// Экземпляр постройки на карте (аналог TBase)
class Base {
public:
    Base() = default;
    Base(const BaseData* d, int x, int y, bool built)
        : data(d), x(x), y(y), build_days(built ? 0 : d->build_time),
          live_time(d->live_time_total()) {}

    int64_t uid = 0;  // уникальный id экземпляра (аналог id() объекта в Python-референсе)
    const BaseData* data = nullptr;
    int x = 0, y = 0;
    int birth_year = 0;
    int64_t build_days = 0;
    int64_t live_time = 0;
    bool preserved = false;
    // state: два бита (need_sunduk / need_workers) — полный аналог set-а строк
    bool need_sunduk = false;
    bool need_workers = false;

    bool state_empty() const { return !need_sunduk && !need_workers; }

    bool can_work(Season season) const {
        return build_days == 0 && !preserved && data->season_works(season);
    }

    // Распределение рабочих. Возвращает число занятых рабочих.
    int64_t begin_day(Season season, Sunduk& sunduk, int64_t free_workers);

    // Производство и износ. Точная копия TBase::EndDay.
    void end_day(Game& game);

    int64_t home_places(Season season) const {
        return can_work(season) ? data->home_places : 0;
    }
    int64_t need_workers_now(Season season) const {
        return can_work(season) ? data->need_workers : 0;
    }
    bool is_alarm() const {
        return data->live_years && live_time > 0 && live_time <= 60;
    }
};

}  // namespace colony